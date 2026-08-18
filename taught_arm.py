"""
taught_arm.py — the physical SO-101, driven by taught poses.

    python3 taught_arm.py --list
    python3 taught_arm.py --teach pick_cup --seconds 6
    python3 taught_arm.py --replay pick_cup
    python3 taught_arm.py --demo

WHY NOT IK. real_backend.py drives the arm from arm_ik, which needs a measured
correspondence between the twin's radians and the driver's calibrated degrees.
Establishing that safely takes known physical poses and careful sign work, and
every mistake in it is a full-speed swing into the table. This module skips it:
poses are taught by hand and replayed verbatim, so no number is ever converted
between two conventions. It gives up arbitrary Cartesian targets and keeps the
part that the project is actually about.

WHAT IS STILL REAL. planner.py is untouched and still decides, from the
geometry of the scene, that a blocker has to be relocated before the goal can
be reached. Taught poses carry a nominal (x, y) label so the planner can do
that reasoning; the label never drives a motor. Same reasoning, same goal
stack, a different executor underneath — which is the claim the demo makes.

SAFETY
  * Torque is off while teaching. THE ARM WILL FALL if you let go.
  * Every move is interpolated and clamped to the calibrated range. A joint
    found outside its range is walked back in slowly, because letting the
    driver clip a goal produced a 54 degree lunge on the first live command.
  * Nothing moves without --live.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import time

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper")
ROBOT_ID = "interx_follower"
POSES_PATH = pathlib.Path(__file__).with_name("arm_poses.json")

GRIP_DEG = 25.0         # how far the jaws close past the taught pose
STEP_DEG = 2.5          # per tick; the servos are geared and unloaded here
TICK_HZ = 30.0
SAFE_STEP_DEG = 12.0    # driver-side cap, measured from the arm's real position
ARRIVE_DEG = 2.0        # close enough to call a pose reached
MOVE_TIMEOUT_S = 12.0   # a pose the arm cannot reach must not hang the demo
BUS_RETRY = 3           # this arm's wiring drops packets under motion


def _port(explicit=None):
    if explicit:
        return explicit
    ports = sorted(glob.glob("/dev/tty.usbmodem*"))
    if len(ports) != 1:
        raise SystemExit(f"expected exactly one usbmodem port, found {ports}")
    return ports[0]


def load_poses() -> dict:
    if POSES_PATH.exists():
        return json.loads(POSES_PATH.read_text())
    return {}


def save_poses(poses: dict) -> None:
    POSES_PATH.write_text(json.dumps(poses, indent=2))


class TaughtArm:
    def __init__(self, port, live=False):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        self.live = live
        self.robot = SO101Follower(SO101FollowerConfig(
            id=ROBOT_ID, port=port, use_degrees=True,
            max_relative_target=SAFE_STEP_DEG,
            disable_torque_on_disconnect=False,
        ))

    def connect(self):
        self.robot.connect(calibrate=False)
        if self.robot.calibration and not self.robot.is_calibrated:
            # The file is not the arm: lerobot compares it against registers
            # inside each servo, and reads in the boot frame until it is pushed.
            self.robot.bus.write_calibration(self.robot.calibration)
        if not self.robot.is_calibrated:
            raise SystemExit("arm is not calibrated — run calibrate_sweep.py --write")
        self.limits = {m: (c.range_min, c.range_max)
                       for m, c in self.robot.calibration.items()}
        return self

    def close(self):
        self.robot.disconnect()

    def read(self) -> dict:
        # The bus on this arm drops packets while it moves — a loose connector
        # somewhere in the chain. A retry costs milliseconds; an exception ends
        # the demo mid-reach with the gripper loaded.
        last = None
        for _ in range(BUS_RETRY):
            try:
                obs = self.robot.get_observation()
                return {m: float(obs[f"{m}.pos"]) for m in MOTORS}
            except Exception as exc:
                last = exc
                time.sleep(0.05)
        raise last

    def _range_deg(self, motor):
        """Calibrated travel in degrees either side of centre."""
        lo, hi = self.limits[motor]
        half = (hi - lo) / 2.0 * 360.0 / 4096.0
        return -half, half

    def clamp(self, pose: dict) -> dict:
        out = {}
        for m, v in pose.items():
            lo, hi = self._range_deg(m)
            out[m] = max(lo, min(hi, v))
        return out

    def bring_into_range(self):
        """Walk any out-of-range joint back inside before anything else.

        Measured the hard way: the first live command found wrist_flex at 100.7
        deg against an 82 deg limit, the driver clipped the goal to the border,
        and the joint travelled 54 degrees in one step instead of the three it
        was asked for.
        """
        now = self.read()
        target = self.clamp(now)
        drift = {m: round(target[m] - now[m], 1)
                 for m in MOTORS if abs(target[m] - now[m]) > 0.5}
        if not drift:
            return
        # A small excursion is worth easing in. A large one is not: the arm is
        # somewhere the calibration does not describe, the direction back is a
        # guess, and driving 44 degrees on a guess is how a demo becomes a
        # repair. Ask for hands instead.
        far = {m: d for m, d in drift.items() if abs(d) > 15}
        if far:
            raise SystemExit(
                f"arm is far outside its calibrated range: {far}\n"
                "Move those joints back by hand — roughly to one of the taught "
                "poses — and run again. Nothing has moved."
            )
        print(f"  slightly outside range {drift} — easing in")
        self.move_to(target, label="into range")

    def move_to(self, pose: dict, label="") -> None:
        """Interpolate to a pose. Never a step command: a real servo answers a
        step by racing there at full speed."""
        pose = self.clamp(pose)
        self._dry_ticks = 0
        deadline = time.time() + MOVE_TIMEOUT_S
        while True:
            # Step from where the arm actually IS, not from an integrated
            # command. max_relative_target measures against the measured
            # position, so an open-loop stream of targets runs away from a
            # loaded arm and every command gets clipped — which is exactly what
            # the first live run did, crawling toward 80 deg through a wall of
            # "clamped to be safe".
            cur = self.read() if self.live else pose
            delta = {m: pose[m] - cur[m] for m in pose}
            span = max(abs(d) for d in delta.values())
            if span <= ARRIVE_DEG or not self.live:
                self._send(dict(pose))
                if label:
                    how = "" if self.live else f"  ({self._dry_ticks} ticks, dry)"
                    print(f"  -> {label:<24} " +
                          " ".join(f"{m[:9]}={pose[m]:7.1f}" for m in MOTORS) + how)
                return
            if time.time() > deadline:
                stuck = {m: round(delta[m], 1) for m in delta if abs(delta[m]) > ARRIVE_DEG}
                print(f"  !! '{label}' not reached in {MOVE_TIMEOUT_S:.0f}s, still off by {stuck}")
                return
            k = min(1.0, STEP_DEG / span)
            self._send({m: cur[m] + delta[m] * k for m in pose})

    def _send(self, pose: dict):
        action = {f"{m}.pos": float(v) for m, v in pose.items()}
        if self.live:
            for attempt in range(BUS_RETRY):
                try:
                    self.robot.send_action(action)
                    break
                except Exception:
                    if attempt == BUS_RETRY - 1:
                        raise
                    time.sleep(0.05)
            time.sleep(1.0 / TICK_HZ)
        else:
            self._dry_ticks = getattr(self, "_dry_ticks", 0) + 1
            self._dry_last = pose


# ------------------------------------------------------------------ commands --


def cmd_list():
    poses = load_poses()
    if not poses:
        print("no poses taught yet")
        return
    for name, p in poses.items():
        xy = p.get("xy")
        tag = f"   nominal xy={xy}" if xy else ""
        print(f"{name:<18}" + " ".join(f"{m[:9]}={p['joints'][m]:7.2f}" for m in MOTORS) + tag)


def cmd_teach(port, name, seconds, xy):
    """Record where the operator puts the arm. Torque off; hold the arm."""
    arm = TaughtArm(port, live=False).connect()
    try:
        arm.robot.bus.disable_torque()
        print(f"teaching '{name}': torque OFF — hold the arm, pose it, keep still at the end")
        end = time.time() + seconds
        last = None
        while time.time() < end:
            last = arm.read()
            left = end - time.time()
            print(f"  {left:4.1f}s  " + " ".join(f"{m[:9]}={last[m]:7.2f}" for m in MOTORS),
                  end="\r", flush=True)
            time.sleep(0.2)
        print()
        # Refuse to save a pose the arm cannot actually be commanded to. A
        # taught joint outside its calibrated range gets clipped on replay, and
        # the clip arrives as a lunge — 90 degrees, measured, on the first set
        # of poses taught here. Better to catch it while the operator is still
        # holding the arm.
        outside = {}
        for m, v in last.items():
            lo, hi = arm._range_deg(m)
            if v < lo - 0.5 or v > hi + 0.5:
                outside[m] = f"{v:.0f}deg vs limit {hi:.0f}"
        if outside:
            print(f"NOT SAVED — outside the calibrated range: {outside}")
            print("Move that joint back into its working range and teach again.")
            return

        poses = load_poses()
        entry = {"joints": last}
        if xy:
            entry["xy"] = [float(xy[0]), float(xy[1])]
        poses[name] = entry
        save_poses(poses)
        print(f"saved '{name}': " + " ".join(f"{m[:9]}={last[m]:7.2f}" for m in MOTORS))
    finally:
        arm.close()


def cmd_replay(port, name, live):
    poses = load_poses()
    if name not in poses:
        raise SystemExit(f"no pose named '{name}' — taught: {list(poses)}")
    arm = TaughtArm(port, live=live).connect()
    try:
        arm.bring_into_range()
        arm.move_to(poses[name]["joints"], label=name)
        print("done" if live else "dry run complete")
    finally:
        arm.close()


def cmd_demo(port, live):
    """The planner decides the order; taught poses carry it out."""
    from planner import Obj, Planner

    poses = load_poses()
    need = ["home", "pick_cup", "place_cup", "pick_medicine", "place_medicine"]
    missing = [n for n in need if n not in poses]
    if missing:
        raise SystemExit(f"teach these first: {missing}")

    # Nominal coordinates for the planner's geometry only. They describe where
    # the taught poses put the gripper on the table; no motor ever sees them.
    def xy(n, default):
        return tuple(poses[n].get("xy", default))

    goal_xy = xy("place_medicine", (0.19, 0.10))
    gx, gy = goal_xy
    scene = {
        "medicine": Obj("medicine", *xy("pick_medicine", (0.24, -0.08)), 0.018),
        "cup": Obj("cup", *xy("pick_cup", (0.17, -0.04)), 0.030),
        # The anchor of "put the medicine next to the water" has to sit beside
        # the delivery spot, or the planner treats it as one more obstacle
        # sitting in open table. Derived from where the medicine is actually
        # delivered rather than fixed, since that spot is taught, not chosen.
        "water": Obj("water", gx + 0.02, gy - 0.06, 0.030),
    }
    plan = Planner(scene, (0.135, 0.270, -0.160, 0.160), (0.075, 0.0), naive=False)
    plan.request("medicine", goal_xy, anchor="water")

    order = []
    guard = 0
    while (g := plan.next_action()) is not None and guard < 6:
        guard += 1
        order.append(g)
        plan.on_result(g, success=True, new_pose=g.to_xy)

    # Taught poses give each object exactly ONE parking spot, so a second
    # relocation of the same object is a no-op with nothing to show. In sim the
    # planner picks a fresh free spot each time and the repeat is meaningful;
    # here it is not, so consecutive repeats collapse.
    collapsed = []
    for g in order:
        if collapsed and collapsed[-1].obj == g.obj:
            continue
        collapsed.append(g)
    order = collapsed

    print("planner decided:")
    for i, g in enumerate(order, 1):
        print(f"  {i}. move {g.obj:<9} [{g.reason}]")
    for line in plan.log:
        print("   ", line)

    arm = TaughtArm(port, live=live).connect()
    try:
        arm.bring_into_range()
        arm.move_to(poses["home"]["joints"], label="home")
        for g in order:
            pick = f"pick_{g.obj}" if f"pick_{g.obj}" in poses else "pick_cup"
            place = f"place_{g.obj}" if f"place_{g.obj}" in poses else "place_cup"
            # Home at the top of every goal too: the previous goal ends with
            # the gripper down at a place pose, and going straight from there
            # to the next pick sweeps across the table at object height.
            arm.move_to(poses["home"]["joints"], label="home")
            arm.move_to(poses[pick]["joints"], label=f"{pick} (approach)")
            closed = poses[pick]["joints"]["gripper"] - GRIP_DEG
            grip = dict(poses[pick]["joints"], gripper=closed)
            arm.move_to(grip, label="close gripper")
            # Up before across. Poses are interpolated in JOINT space, so a
            # direct pick->place move drags the gripper along a straight line
            # through whatever is between them. Routing through the taught home
            # pose lifts first and costs nothing extra to teach — which is why
            # home must be taught HIGH.
            # Every taught pose carries the gripper value it was taught with,
            # and home and place were both taught OPEN. Replaying them verbatim
            # while carrying re-opens the jaws and drops the object on the way
            # — visible in the dry run as gripper going -25 -> 0 the moment the
            # arm lifts. The grip is held explicitly until the release step.
            arm.move_to(dict(poses["home"]["joints"], gripper=closed),
                        label="lift (via home)")
            arm.move_to(dict(poses[place]["joints"], gripper=closed), label=place)
            arm.move_to(dict(poses[place]["joints"],
                             gripper=poses[place]["joints"]["gripper"] + GRIP_DEG),
                        label="release")
        arm.move_to(poses["home"]["joints"], label="home")
    finally:
        arm.close()


def main():
    ap = argparse.ArgumentParser(description="SO-101 driven by taught poses")
    ap.add_argument("--port")
    ap.add_argument("--live", action="store_true", help="actually move the arm")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--teach", metavar="NAME")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--xy", nargs=2, metavar=("X", "Y"),
                    help="nominal table coordinates, for the planner's geometry")
    ap.add_argument("--replay", metavar="NAME")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    if a.list:
        return cmd_list() or 0
    port = _port(a.port)
    print(f"port: {port}\n")
    if a.teach:
        cmd_teach(port, a.teach, a.seconds, a.xy)
    elif a.replay:
        cmd_replay(port, a.replay, a.live)
    elif a.demo:
        cmd_demo(port, a.live)
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
