"""
real_backend.py — executor.ArmBackend on the physical SO-101.

The same run_goal() that drives Isaac drives this. Only the four primitives
change: send(), measured(), state_for(), dwell().

    python3 real_backend.py --probe        read joints, move nothing
    python3 real_backend.py --map          discover sign and offset per joint
    python3 real_backend.py --home         ramp slowly to the rest pose
    python3 real_backend.py --pick X Y     one pick-and-place, live

SAFETY, in the order the mistakes actually happen:

  1. Our IK speaks the digital twin's radians. lerobot speaks calibrated
     DEGREES for the five body joints and 0..100 for the gripper. These are
     different conventions with different zeros and, per joint, possibly
     different signs. Sending one into the other is how an arm folds itself
     into the table. JOINT_MAP below starts as all-unknown on purpose, and
     --map fills it in by measurement.
  2. Every run is a dry run until --live is passed. A dry run exercises the
     whole path with sends suppressed, which is the house rule from Guide 1.
  3. max_relative_target caps how far a single command may move a joint from
     where it is. The executor already interpolates, so this costs nothing and
     turns "wrong number" into "small twitch" rather than "full-speed swing".
  4. connect() enables torque: the arm stiffens on connect and DROPS if torque
     is cut while it holds a pose. Ramp to rest, then disconnect.

Keep one hand near the power switch on first motion.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

from arm_ik import JAW_CLOSED, JAW_OPEN, JOINT_ORDER, LIMITS, forward
from executor import ArmBackend, run_goal

# Our joint -> lerobot motor. Order matters: it is JOINT_ORDER.
MOTOR_OF = {
    "Rotation": "shoulder_pan",
    "Pitch": "shoulder_lift",
    "Elbow": "elbow_flex",
    "Wrist_Pitch": "wrist_flex",
    "Wrist_Roll": "wrist_roll",
    "Jaw": "gripper",
}

# degrees_real = SIGN * degrees(radians_twin) + OFFSET, per joint.
# Both start unknown. --map measures them by commanding a small, bounded move
# on ONE joint at a time and watching which way the encoder actually went.
# Filling these in by eye from a photograph is exactly the shortcut that ends
# with a servo against its stop.
JOINT_MAP: dict[str, tuple[float, float]] = {
    # "Rotation": (1.0, 0.0),
}

# Gripper is 0..100, not an angle. Measured jaw range in the twin is
# -0.1745..1.7453 rad; map the useful part of it onto the driver's scale.
JAW_SCALE = (JAW_CLOSED, JAW_OPEN, 10.0, 60.0)  # rad_closed, rad_open, pct_closed, pct_open

SAFE_STEP_DEG = 4.0     # max_relative_target: one command may move this far
SEND_HZ = 30.0


def _rad_to_driver(name: str, rad: float) -> float:
    """One joint, our radians -> the driver's units."""
    if name == "Jaw":
        r0, r1, p0, p1 = JAW_SCALE
        t = 0.0 if r1 == r0 else (rad - r0) / (r1 - r0)
        return p0 + max(0.0, min(1.0, t)) * (p1 - p0)
    if name not in JOINT_MAP:
        raise RuntimeError(
            f"no measured mapping for {name}. Run --map before commanding the arm; "
            "guessing this number is how the arm hits the table."
        )
    sign, offset = JOINT_MAP[name]
    return sign * math.degrees(rad) + offset


def _driver_to_rad(name: str, val: float) -> float:
    if name == "Jaw":
        r0, r1, p0, p1 = JAW_SCALE
        t = 0.0 if p1 == p0 else (val - p0) / (p1 - p0)
        return r0 + t * (r1 - r0)
    if name not in JOINT_MAP:
        return 0.0
    sign, offset = JOINT_MAP[name]
    return math.radians((val - offset) / sign)


class RealBackend(ArmBackend):
    """The physical arm. Object poses come from perception or from the operator,
    never from the robot — a real block does not report where it is."""

    def __init__(self, port: str, live: bool = False, zone_rect=(0.15, 0.23, 0.055, 0.14),
                 objects: dict | None = None, robot_id: str = "interx_follower"):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        self.live = live
        self.zone_rect = zone_rect
        self.objects = dict(objects or {})
        self.held: str | None = None
        self.ticks = 0
        self.sent_log: list[dict] = []
        self.step_rad = 0.03      # small steps: these are real servos
        self.robot = SO101Follower(SO101FollowerConfig(
            id=robot_id,   # calibration profiles are keyed by this
            port=port,
            use_degrees=True,
            max_relative_target=SAFE_STEP_DEG,
            disable_torque_on_disconnect=False,   # ramp to rest first, then cut
        ))

    # -- lifecycle -----------------------------------------------------------
    def connect(self):
        self.robot.connect(calibrate=False)
        # A calibration FILE is not a calibrated ARM: lerobot compares the file
        # against registers inside each servo, and without pushing it there the
        # arm reports uncalibrated and reads in whatever frame it booted with.
        # calibrate=True would do this too, but only after an interactive
        # prompt, which is no use inside a run.
        if not _push_calibration(self.robot):
            raise RuntimeError(
                "arm is not calibrated. Run:  lerobot-calibrate "
                "--robot.type=so101_follower --robot.port=<port> --robot.id=<id>"
            )
        return self

    def close(self):
        """Ramp to rest before cutting torque, or the arm drops."""
        try:
            rest = {n: 0.0 for n in JOINT_ORDER}
            rest["Jaw"] = JAW_OPEN
            self.send([rest[n] for n in JOINT_ORDER])
        finally:
            self.robot.disconnect()

    # -- ArmBackend ----------------------------------------------------------
    def send(self, q):
        action = {}
        for name, rad in zip(JOINT_ORDER, q):
            lo, hi = LIMITS[name]
            action[f"{MOTOR_OF[name]}.pos"] = _rad_to_driver(name, max(lo, min(hi, rad)))
        self.sent_log.append(action)
        self.ticks += 1
        if self.live:
            self.robot.send_action(action)
            time.sleep(1.0 / SEND_HZ)
        else:
            print("  [dry] " + "  ".join(f"{k.split('.')[0]}={v:7.2f}"
                                         for k, v in action.items()))

    def measured(self):
        obs = self.robot.get_observation()
        return [_driver_to_rad(n, float(obs[f"{MOTOR_OF[n]}.pos"])) for n in JOINT_ORDER]

    def state_for(self, name):
        """Physics gates need object poses. On hardware those come from
        perception, not from the simulator — until that lands, the operator
        supplies them and the gates report on what they were told."""
        obj = self.objects.get(name, (0.0, 0.0, 0.0))
        return {
            "obj_xyz": obj,
            "gripper_xyz": forward(dict(zip(JOINT_ORDER, self.measured()))),
            "obj_quat": (1.0, 0.0, 0.0, 0.0),
            "obj_speed": 0.0,
            "zone_rect": getattr(self, "place_zone", None) or self.zone_rect,
            "table_z": obj[2],
            "ticks_in_phase": self.ticks,
        }

    def dwell(self, seconds):
        if self.live:
            time.sleep(seconds)
        self.ticks += int(seconds * SEND_HZ)

    def attach(self, name):
        self.held = name

    def detach(self, name=None):
        self.held = None


# ------------------------------------------------------------------ tooling --


def _push_calibration(robot) -> bool:
    """A calibration FILE is not a calibrated ARM.

    lerobot compares the file against registers inside each servo; without
    pushing it there the arm reports uncalibrated and reads in whatever frame
    it booted with. connect(calibrate=True) also does this, but only behind an
    interactive prompt, which is no use inside a script.
    """
    if robot.calibration and not robot.is_calibrated:
        robot.bus.write_calibration(robot.calibration)
    return robot.is_calibrated


def cmd_probe(port):
    """Read the arm. Commands nothing."""
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(
        id="interx_follower", port=port, use_degrees=True))
    robot.connect(calibrate=False)
    print(f"connected: {robot.is_connected}   calibrated: {_push_calibration(robot)}")
    try:
        for i in range(10):
            obs = robot.get_observation()
            row = "  ".join(f"{k.split('.')[0]:>13}={obs[k]:7.2f}"
                            for k in sorted(obs) if k.endswith(".pos"))
            print(f"{i}: {row}")
            time.sleep(0.4)
    finally:
        robot.disconnect()
    print("\nMove each joint by hand and re-run: the numbers should track "
          "smoothly, one joint at a time, and repeat when you let go.")


def cmd_map(port, live):
    """Measure sign and offset per joint, one small move at a time."""
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(
        id="interx_follower", port=port, use_degrees=True,
        max_relative_target=SAFE_STEP_DEG))
    robot.connect(calibrate=False)
    if not _push_calibration(robot):
        print("arm is not calibrated — run calibrate_sweep.py --write first")
        robot.disconnect()
        return
    print("Each joint gets a +3 deg nudge; the encoder says which way that is.")
    print("DRY RUN — nothing moves." if not live else "LIVE — hand on the switch.")
    try:
        for name in ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"):
            motor = MOTOR_OF[name]
            obs = robot.get_observation()
            before = float(obs[f"{motor}.pos"])
            target = dict((k, float(v)) for k, v in obs.items() if k.endswith(".pos"))
            target[f"{motor}.pos"] = before + 3.0
            if live:
                robot.send_action(target)
                time.sleep(1.0)
                after = float(robot.get_observation()[f"{motor}.pos"])
                robot.send_action({k: (before if k == f"{motor}.pos" else v)
                                   for k, v in target.items()})
                time.sleep(0.5)
            else:
                after = before
            print(f"  {name:<12} {motor:<14} {before:7.2f} -> {after:7.2f}")
    finally:
        robot.disconnect()
    print("\nFill JOINT_MAP with (sign, offset) per joint, then re-run --home.")


def cmd_home(port, live):
    be = RealBackend(port, live=live).connect()
    print("ramping to rest" + ("" if live else "  [dry run]"))
    try:
        be.close()
    except Exception:
        raise


def main():
    ap = argparse.ArgumentParser(description="Drive the physical SO-101")
    ap.add_argument("--port", default=None, help="defaults to the single usbmodem port")
    ap.add_argument("--live", action="store_true",
                    help="actually move the arm; without this every send is printed only")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--map", action="store_true")
    ap.add_argument("--home", action="store_true")
    args = ap.parse_args()

    port = args.port
    if port is None:
        import glob

        ports = sorted(glob.glob("/dev/tty.usbmodem*"))
        if len(ports) != 1:
            print(f"expected exactly one usbmodem port, found {ports}")
            return 1
        port = ports[0]
    print(f"port: {port}\n")

    if args.probe:
        cmd_probe(port)
    elif args.map:
        cmd_map(port, args.live)
    elif args.home:
        cmd_home(port, args.live)
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
