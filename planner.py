"""
planner.py — Goal-stack planner for the Accessible Kitchen Assistant.

Pure Python, zero dependencies (no Antioch, no Isaac, no numpy).
Run `python planner.py` right now to see the emergent behavior:
a blocked goal pushes "relocate the blocker" on top of itself, then resumes.

Owner: P2. Wire into kitchen_suite.py via Planner.next_action() / on_result().
Coordinates: table plane, metres. Poses come from sim ground truth.
"""

from dataclasses import dataclass, field
import math
import random


# ---------------------------------------------------------------- geometry --

@dataclass
class Obj:
    name: str
    x: float
    y: float
    radius: float           # footprint radius for clearance checks
    graspable: bool = True


def dist_point_segment(px, py, ax, ay, bx, by) -> float:
    """Distance from point P to segment A-B."""
    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby
    if ab2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab2))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def find_blockers(target: Obj, scene: list[Obj], approach_from: tuple[float, float],
                  goal_xy: tuple[float, float] | None = None,
                  anchor: str | None = None,
                  corridor_halfwidth: float = 0.06,
                  goal_clearance: float = 0.08) -> list[Obj]:
    """An object blocks if it sits in the approach->target corridor, or too
    close to the placement point. `anchor` is the reference object of a
    "next to X" command: it is *supposed* to be near the goal, so it is
    exempt from the near_goal check (but still counts in the corridor).
    approach_from ~ gripper home is a deliberate approximation: the arm
    returns toward home between picks."""
    ax, ay = approach_from
    out = []
    for o in scene:
        if o.name == target.name or not o.graspable:
            continue
        in_corridor = dist_point_segment(o.x, o.y, ax, ay, target.x, target.y) \
            < corridor_halfwidth + o.radius
        near_goal = goal_xy is not None and o.name != anchor and \
            math.hypot(o.x - goal_xy[0], o.y - goal_xy[1]) < goal_clearance + o.radius
        if in_corridor or near_goal:
            out.append(o)
    # nearest-to-gripper first: clear the front of the corridor first
    out.sort(key=lambda o: math.hypot(o.x - ax, o.y - ay))
    return out


def find_free_spot(scene: list[Obj], table_bounds: tuple[float, float, float, float],
                   avoid: list[tuple[float, float, float]] | None = None,
                   clearance: float = 0.07, tries: int = 300,
                   rng: random.Random | None = None) -> tuple[float, float] | None:
    """Sample a placement point with clearance from every object and every
    (x, y, r) avoid-circle. Returns None if the table is genuinely full."""
    rng = rng or random.Random(0)
    xmin, xmax, ymin, ymax = table_bounds
    for _ in range(tries):
        x, y = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
        if any(math.hypot(x - o.x, y - o.y) < clearance + o.radius for o in scene):
            continue
        if any(math.hypot(x - cx, y - cy) < cr for (cx, cy, cr) in (avoid or [])):
            continue
        return (x, y)
    return None


# -------------------------------------------------------------- goal stack --

@dataclass
class Goal:
    obj: str
    to_xy: tuple[float, float]
    anchor: str | None = None   # reference object of "next to X", exempt at goal
    reason: str = "user"        # "user" | "unblock <name>"
    attempts: int = 0           # failed executions (monitor said no)
    expansions: int = 0         # times we pushed a blocker-goal for this goal


class Planner:
    """LIFO goal stack.

    next_action() returns the next *executable* Goal: if the top goal is
    blocked, it pushes a relocate-goal for the blocker and loops. The executor
    runs the goal, the monitor judges it, on_result() updates the world model
    and requeues on failure. Baseline mode (naive=True) skips expansion and
    retries entirely — that is the supervisor_off condition for the eval.
    """
    MAX_ATTEMPTS = 2       # retries per goal after a failure event
    MAX_EXPANSIONS = 2     # blocker-goals we will spawn per goal (loop guard)
    MAX_DEPTH = 3          # how deep the "clear the blocker" chain may nest

    def __init__(self, scene: dict[str, Obj],
                 table_bounds: tuple[float, float, float, float],
                 gripper_home: tuple[float, float], naive: bool = False):
        self.scene = scene
        self.bounds = table_bounds
        self.home = gripper_home
        self.naive = naive
        self.stack: list[Goal] = []
        self.log: list[str] = []

    # -- public API ----------------------------------------------------------
    def request(self, obj: str, to_xy: tuple[float, float],
                anchor: str | None = None):
        """Entry point for the language layer: one structured target."""
        self.stack.append(Goal(obj, to_xy, anchor=anchor))

    def next_action(self) -> Goal | None:
        while self.stack:
            g = self.stack[-1]
            if self.naive:
                return self.stack.pop()
            target = self.scene[g.obj]
            blockers = find_blockers(target, list(self.scene.values()),
                                     self.home, goal_xy=g.to_xy, anchor=g.anchor)
            # Two objects can each sit in the other's corridor, and relocating
            # one puts it back in the way of the other. MAX_EXPANSIONS cannot
            # catch that: it is counted per Goal, and every expansion mints a
            # fresh Goal with the counter back at zero, so the chain
            # "clear A -> clear B -> clear A" nests forever. Measured: this
            # hung the whole episode loop with no error, in pure Python.
            # Anything already queued is therefore not a blocker worth chasing,
            # and the chain is capped outright.
            pending = {q.obj for q in self.stack}
            blockers = [b for b in blockers if b.name not in pending]
            if (not blockers
                    or g.expansions >= self.MAX_EXPANSIONS
                    or len(self.stack) >= self.MAX_DEPTH):
                return self.stack.pop()
            b = blockers[0]
            mx, my = (self.home[0] + target.x) / 2, (self.home[1] + target.y) / 2
            spot = find_free_spot(
                list(self.scene.values()), self.bounds,
                avoid=[(target.x, target.y, 0.10), (*g.to_xy, 0.10), (mx, my, 0.08)])
            if spot is None:                       # table full: try anyway
                return self.stack.pop()
            g.expansions += 1
            self.log.append(f"[plan] {g.obj} blocked by {b.name} -> relocate it")
            self.stack.append(Goal(b.name, spot, reason=f"unblock {g.obj}"))
        return None

    def on_result(self, goal: Goal, success: bool,
                  new_pose: tuple[float, float] | None = None):
        """Executor reports back. new_pose = sim ground truth after the move
        (also call this after a sabotage teleport to resync the world model)."""
        if new_pose is not None:
            self.scene[goal.obj].x, self.scene[goal.obj].y = new_pose
        if success or self.naive:
            return
        if goal.attempts < self.MAX_ATTEMPTS:
            goal.attempts += 1
            self.log.append(f"[plan] {goal.obj} failed -> retry {goal.attempts}")
            self.stack.append(goal)               # executor perturbs approach
        else:
            self.log.append(f"[plan] {goal.obj} gave up after "
                            f"{goal.attempts} retries -> rest pose, next goal")


# --------------------------------------------------------------- self-test --

if __name__ == "__main__":
    # Scene: medicine vial sits behind a cup relative to the gripper.
    # "Move the medicine next to the water container" must first move the cup.
    scene = {
        "medicine": Obj("medicine", 0.42, 0.00, 0.015),
        "cup":      Obj("cup",      0.26, 0.01, 0.040),   # in the corridor
        "water":    Obj("water",    0.30, 0.22, 0.045),
        "block_a":  Obj("block_a",  0.15, -0.18, 0.025),  # innocent bystander
    }
    REACH_ZONE = (0.30, 0.20)                              # next to water
    p = Planner(scene, table_bounds=(0.10, 0.50, -0.25, 0.25),
                gripper_home=(0.0, 0.0))
    p.request("medicine", REACH_ZONE, anchor="water")

    print("executed order:")
    step = 1
    while (g := p.next_action()) is not None:
        print(f"  {step}. move {g.obj:<9} -> ({g.to_xy[0]:.2f}, {g.to_xy[1]:.2f})"
              f"   [{g.reason}]")
        p.on_result(g, success=True, new_pose=g.to_xy)     # pretend-perfect exec
        step += 1
    print("\nplanner log:")
    for line in p.log:
        print(" ", line)
    assert step == 3, "expected exactly: relocate cup, then move medicine"
    print("\nself-test OK: the robot decided to move the cup on its own.")

    # Regression: mutually-blocking objects must not nest forever. Two blocks
    # sitting on top of each other's corridors is the natural way to trigger
    # it, and before MAX_DEPTH this loop never returned at all.
    tight = {
        "medicine": Obj("medicine", 0.24, -0.05, 0.018),
        "cup":      Obj("cup",      0.17, -0.03, 0.030),
        "water":    Obj("water",    0.16, 0.12, 0.030),
    }
    p2 = Planner(tight, table_bounds=(0.135, 0.270, -0.160, 0.160),
                 gripper_home=(0.075, 0.0))
    p2.request("medicine", (0.19, 0.10), anchor="water")
    issued = 0
    while (g := p2.next_action()) is not None and issued < 20:
        issued += 1
        p2.on_result(g, success=True, new_pose=g.to_xy)
    assert issued < 20, "goal stack failed to terminate on mutual blockers"
    print(f"cycle guard OK: terminated after {issued} goals, depth cap "
          f"{Planner.MAX_DEPTH}")
