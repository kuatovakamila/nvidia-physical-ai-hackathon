"""
executor.py — the one seam between reasoning and action.

planner.py decides WHAT to move. This file decides HOW, once, and then hands
the same six joint targets to whichever arm is listening:

    ArmBackend (protocol)
      ├── SimBackend    Isaac articulation      -> eval batches, the number
      ├── RealBackend   SO101Follower over USB  -> the live demo
      └── FakeBackend   pure Python             -> this file's self-test

The phase sequence and the monitor calls live HERE, shared by every backend,
because "approach, grasp, lift, carry, place, check after each" is a property
of the task and not of the hardware. Only the four primitives differ. Getting
this boundary right is what lets the demo say: same planner, same phases, same
gates — one runs on a GPU in the cloud, one runs on the table in front of you.

Run `python3 executor.py` for the self-test. No sim, no hardware needed.

Owner: P1. Depends on arm_ik.py (geometry) and monitor.py (gates).
"""

from dataclasses import dataclass, field
import math
import time

from arm_ik import (
    JAW_CLOSED,
    JAW_OPEN,
    JOINT_ORDER,
    Pose,
    Unreachable,
    forward,
    solve,
    to_vector,
)
from monitor import Event, monitor

# --------------------------------------------------------------- constants --

APPROACH_H = 0.070      # m above the object to hover before descending
LIFT_H = 0.090          # m above the table while carrying
GRASP_SETTLE_S = 0.35   # let the jaw close and the servos catch up
STEP_HZ = 40.0          # interpolation rate; servos slam if we jump targets
MAX_STEP_RAD = 0.03     # per-tick joint delta, so motion is smooth on hardware


@dataclass
class Waypoint:
    phase: str
    pose: Pose
    jaw: float
    dwell_s: float = 0.0


# ------------------------------------------------------------- the backends --


class ArmBackend:
    """Four primitives. Everything else is shared."""

    def send(self, q: list[float]) -> None:
        """Command six joint targets, radians, in JOINT_ORDER."""
        raise NotImplementedError

    def measured(self) -> list[float]:
        """Six measured joint positions, radians."""
        raise NotImplementedError

    def state_for(self, name: str) -> dict:
        """Ground truth for monitor.monitor(): obj_xyz, gripper_xyz, obj_quat,
        obj_speed, zone_rect, table_z, ticks_in_phase."""
        raise NotImplementedError

    def dwell(self, seconds: float) -> None:
        raise NotImplementedError


class FakeBackend(ArmBackend):
    """A perfect arm over a dict of object poses. Exists so the sequencing can
    be tested without booting Isaac or powering a servo — and so a failing
    self-test here always means the logic broke, never the hardware."""

    def __init__(self, objects: dict[str, tuple[float, float, float]],
                 zone_rect=(0.16, 0.26, 0.06, 0.16)):
        self.objects = dict(objects)
        self.zone_rect = zone_rect
        self.q = [0.0] * 6
        self.held: str | None = None
        self.ticks = 0
        self.sent: list[list[float]] = []

    def send(self, q):
        self.q = list(q)
        self.sent.append(list(q))
        self.ticks += 1
        # A held object rides the tool tip: that is what carrying means.
        if self.held:
            self.objects[self.held] = forward(dict(zip(JOINT_ORDER, q)))

    def measured(self):
        return list(self.q)

    def state_for(self, name):
        tip = forward(dict(zip(JOINT_ORDER, self.q)))
        obj = self.objects.get(name, (0.0, 0.0, 0.0))
        return {
            "obj_xyz": obj,
            "gripper_xyz": tip,
            "obj_quat": (1.0, 0.0, 0.0, 0.0),
            "obj_speed": 0.0,
            "zone_rect": self.zone_rect,
            "table_z": 0.0,
            "ticks_in_phase": self.ticks,
        }

    def dwell(self, seconds):
        self.ticks += int(seconds * STEP_HZ)

    # Test hooks for what the real backends express through physics. attach()
    # honours a grasp tolerance rather than always succeeding, so a biased
    # approach misses here for the same reason it misses on the table — and a
    # green self-test means the bias study is real, not assumed.
    GRASP_TOL_M = 0.012

    def attach(self, name):
        tip = forward(dict(zip(JOINT_ORDER, self.q)))
        if math.dist(tip, self.objects[name]) <= self.GRASP_TOL_M:
            self.held = name

    def detach(self, name=None):
        self.held = None


# ------------------------------------------------------------- the sequence --


def plan_waypoints(pick_xyz, place_xyz, bias_m: float = 0.0) -> list[Waypoint]:
    """The five phases as poses. `bias_m` displaces the grasp sideways — the
    manufactured failure that makes the supervisor comparison honest. It is
    applied to the grasp only, so approach and lift stay truthful and the
    monitor catches the miss at LIFT exactly as it would in the real world."""
    px, py, pz = pick_xyz
    gx, gy, gz = place_xyz

    # Bias perpendicular to the reach direction: a radial offset would just
    # grasp short and still catch the object, which is not the failure we want
    # to study.
    heading = math.atan2(py, px)
    bx = px + bias_m * math.sin(heading)
    by = py - bias_m * math.cos(heading)

    return [
        Waypoint("approach", Pose(px, py, pz + APPROACH_H), JAW_OPEN),
        Waypoint("grasp", Pose(bx, by, pz), JAW_CLOSED, dwell_s=GRASP_SETTLE_S),
        Waypoint("lift", Pose(px, py, pz + LIFT_H), JAW_CLOSED, dwell_s=0.2),
        Waypoint("carry", Pose(gx, gy, gz + LIFT_H), JAW_CLOSED),
        Waypoint("place", Pose(gx, gy, gz), JAW_OPEN, dwell_s=GRASP_SETTLE_S),
    ]


def move_through(backend: ArmBackend, target_q: dict[str, float],
                 realtime: bool = False) -> None:
    """Interpolate to a joint target instead of jumping. On hardware a step
    command makes every servo race at full speed to its goal, which is how
    arms knock over the object they were about to pick up."""
    goal = to_vector(target_q)
    current = backend.measured()
    while True:
        delta = [g - c for g, c in zip(goal, current)]
        span = max(abs(d) for d in delta)
        if span <= MAX_STEP_RAD:
            backend.send(goal)
            return
        scale = MAX_STEP_RAD / span
        current = [c + d * scale for c, d in zip(current, delta)]
        backend.send(current)
        if realtime:
            time.sleep(1.0 / STEP_HZ)


@dataclass
class GoalReport:
    ok: bool
    failed_phase: str | None = None
    event: Event | None = None
    unreachable: str | None = None
    waypoints_run: int = 0


def run_goal(backend: ArmBackend, obj_name: str, pick_xyz, place_xyz,
             bias_m: float = 0.0, realtime: bool = False,
             on_grasp=None, on_release=None) -> GoalReport:
    """Drive one pick-and-place, gating on physics after every phase.

    Returns rather than raises, because a failed goal is data the planner
    consumes, not an exception the run should die on."""
    try:
        waypoints = plan_waypoints(pick_xyz, place_xyz, bias_m)
    except Unreachable as exc:
        return GoalReport(False, unreachable=str(exc))

    ran = 0
    for wp in waypoints:
        try:
            # solve(), not inverse(): it keeps the approach as vertical as the
            # geometry allows instead of demanding vertical and giving up.
            q = solve(wp.pose.x, wp.pose.y, wp.pose.z, wp.pose.pitch)
        except Unreachable as exc:
            # Refusing to move beats moving somewhere wrong: the planner can
            # pick another spot, but it cannot undo a swipe across the table.
            return GoalReport(False, failed_phase=wp.phase,
                              unreachable=f"{wp.phase}: {exc}", waypoints_run=ran)
        q["Jaw"] = wp.jaw
        move_through(backend, q, realtime=realtime)
        if wp.dwell_s:
            backend.dwell(wp.dwell_s)

        # The jaw closing IS the grasp; tell the backend so physics (or the
        # fake) can attach the object.
        if wp.phase == "grasp" and on_grasp is not None:
            on_grasp(obj_name)
        if wp.phase == "place" and on_release is not None:
            on_release(obj_name)

        ran += 1
        ev = monitor(wp.phase, backend.state_for(obj_name))
        if ev.failed:
            return GoalReport(False, failed_phase=wp.phase, event=ev, waypoints_run=ran)

    return GoalReport(True, waypoints_run=ran)


# --------------------------------------------------------------- self-test --

if __name__ == "__main__":
    ZONE = (0.16, 0.26, 0.06, 0.16)
    PLACE = (0.21, 0.11, 0.02)

    # 1. A clean pick-and-place succeeds and ends in the zone.
    be = FakeBackend({"medicine": (0.22, -0.06, 0.02)}, zone_rect=ZONE)
    rep = run_goal(be, "medicine", (0.22, -0.06, 0.02), PLACE,
                   on_grasp=be.attach, on_release=be.detach)
    assert rep.ok, f"clean run should succeed, got {rep}"
    assert rep.waypoints_run == 5
    x, y, _z = be.objects["medicine"]
    assert ZONE[0] <= x <= ZONE[1] and ZONE[2] <= y <= ZONE[3], \
        f"object ended at ({x:.3f},{y:.3f}), outside {ZONE}"
    print(f"clean run: OK, object delivered to ({x:.3f}, {y:.3f})")

    # 2. Motion is interpolated, not stepped — the property hardware needs.
    steps = be.sent
    worst = max(
        max(abs(b - a) for a, b in zip(prev, nxt))
        for prev, nxt in zip(steps, steps[1:])
    )
    assert worst <= MAX_STEP_RAD + 1e-9, f"joint jumped {worst:.4f} rad in one tick"
    print(f"interpolation: {len(steps)} ticks, largest joint step "
          f"{math.degrees(worst):.2f} deg")

    # 3. A biased grasp misses: the object never rides the tool, so LIFT sees
    #    it still on the table and the monitor says GRASP_MISS.
    be2 = FakeBackend({"medicine": (0.22, -0.06, 0.02)}, zone_rect=ZONE)
    rep2 = run_goal(be2, "medicine", (0.22, -0.06, 0.02), PLACE, bias_m=0.015,
                    on_grasp=be2.attach, on_release=be2.detach)
    assert be2.held is None, "a 15 mm offset must miss the 12 mm grasp tolerance"
    assert not rep2.ok, "a 15 mm bias with nothing grasped must fail"
    assert rep2.failed_phase == "lift", f"expected failure at lift, got {rep2.failed_phase}"
    assert rep2.event is not None and rep2.event.kind == "GRASP_MISS"
    print(f"biased grasp: caught at {rep2.failed_phase} — "
          f"{rep2.event.kind} ({rep2.event.detail})")

    # 4. Out-of-reach targets are refused before anything moves.
    be3 = FakeBackend({"medicine": (0.48, 0.0, 0.02)}, zone_rect=ZONE)
    rep3 = run_goal(be3, "medicine", (0.48, 0.0, 0.02), PLACE)
    assert not rep3.ok and rep3.unreachable, "0.48 m must be refused"
    assert be3.sent == [], "nothing may move toward an unreachable target"
    print(f"unreachable: refused before moving — {rep3.unreachable}")

    print("\nexecutor self-test OK: same sequence will drive Isaac and the SO-101.")
