"""
bridge.py — the part of the ROS node that does not need ROS.

Everything here is plain Python over planner.py and executor.py: it decides,
executes, and hands finished messages to three callbacks. kitchen_node.py wires
those callbacks to rclpy publishers.

The split is the point. A node that imports rclpy at module scope can only be
tested where ROS is installed, which on this project is nowhere — no rclpy on
the laptop, no /opt/ros. Keeping the logic ROS-free means it is exercised by
`python3 bridge.py` on any machine, and the ROS layer stays thin enough to read
in one screen. It is also the honest shape of the claim: the executor already
has three interchangeable backends behind one interface, and ROS is a fourth
consumer of the same seam rather than a rewrite.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The package runs against a checkout of the project rather than vendoring it:
# same planner, same executor, no second copy to drift. Finding that checkout
# by counting parent directories breaks the moment colcon copies this file into
# install/, so search upward for the marker instead, and let an environment
# variable win when the workspace lives somewhere else entirely.
import os


def _find_root() -> Path:
    env = os.environ.get("WITHIN_REACH_ROOT")
    if env and (Path(env) / "planner.py").is_file():
        return Path(env)
    for base in (Path(__file__).resolve(), Path.cwd().resolve()):
        for parent in [base, *base.parents]:
            if (parent / "planner.py").is_file() and (parent / "executor.py").is_file():
                return parent
    raise ImportError(
        "cannot locate the project checkout (planner.py, executor.py). "
        "Set WITHIN_REACH_ROOT to it."
    )


_ROOT = _find_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arm_ik import JOINT_ORDER                      # noqa: E402
from executor import FakeBackend, run_goal          # noqa: E402
from planner import Obj, Planner                    # noqa: E402

TABLE = (0.135, 0.270, -0.160, 0.160)
ZONE = (0.150, 0.230, 0.055, 0.140)
HOME = (0.075, 0.0)
OBJ_Z = 0.02


def default_scene() -> dict[str, Obj]:
    """Medicine behind a cup, water beside the delivery zone."""
    return {
        "medicine": Obj("medicine", 0.245, -0.075, 0.018),
        "cup": Obj("cup", 0.170, -0.040, 0.030),
        "water": Obj("water", 0.165, 0.125, 0.030),
    }


def zone_centre() -> tuple[float, float]:
    xmin, xmax, ymin, ymax = ZONE
    return ((xmin + xmax) / 2, (ymin + ymax) / 2)


class _PublishingBackend(FakeBackend):
    """FakeBackend that reports every commanded pose.

    Joint states are published per tick rather than per goal because that is
    what a ROS consumer expects on /joint_states — a stream it can plot or
    record, not a summary after the fact.
    """

    def __init__(self, *args, on_joints=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_joints = on_joints

    def send(self, q):
        super().send(q)
        if self._on_joints is not None:
            self._on_joints(list(JOINT_ORDER), [float(v) for v in q])


class KitchenBridge:
    """Turn a command into planner decisions and executed motion."""

    def __init__(self, publish_plan=None, publish_event=None, publish_joints=None):
        self.publish_plan = publish_plan or (lambda text: None)
        self.publish_event = publish_event or (lambda text: None)
        self.publish_joints = publish_joints or (lambda names, pos: None)

    def handle_command(self, text: str) -> dict:
        """`text` names the object to deliver, e.g. "medicine" or
        "medicine next to water"."""
        words = text.lower().split()
        target = words[0] if words else "medicine"
        anchor = words[-1] if "next" in words else "water"

        scene = default_scene()
        if target not in scene:
            self.publish_event(f"REJECTED unknown object '{target}'")
            return {"ok": False, "reason": f"unknown object '{target}'"}

        rest = {n: (o.x, o.y, OBJ_Z) for n, o in scene.items()}
        backend = _PublishingBackend(rest, zone_rect=ZONE, home=HOME,
                                     on_joints=self.publish_joints)
        plan = Planner(scene, TABLE, HOME, naive=False)
        plan.request(target, zone_centre(), anchor=anchor)

        goals, guard = [], 0
        while (g := plan.next_action()) is not None and guard < 8:
            guard += 1
            self.publish_plan(f"goal {guard}: move {g.obj} -> "
                              f"({g.to_xy[0]:.3f}, {g.to_xy[1]:.3f}) [{g.reason}]")
            obj = scene[g.obj]
            report = run_goal(backend, g.obj, (obj.x, obj.y, OBJ_Z),
                              (g.to_xy[0], g.to_xy[1], OBJ_Z),
                              on_grasp=backend.attach, on_release=backend.detach)
            if report.event is not None:
                self.publish_event(f"{report.event.kind} at {report.failed_phase}: "
                                   f"{report.event.detail}")
            elif report.unreachable:
                self.publish_event(f"UNREACHABLE {report.unreachable}")
            else:
                self.publish_event(f"OK {g.obj} placed")
            x, y, _z = backend.objects[g.obj]
            scene[g.obj].x, scene[g.obj].y = x, y
            plan.on_result(g, report.ok, new_pose=(x, y))
            goals.append({"obj": g.obj, "reason": g.reason, "ok": report.ok})

        for line in plan.log:
            self.publish_plan(line)

        mx, my, _ = backend.objects[target]
        delivered = ZONE[0] <= mx <= ZONE[1] and ZONE[2] <= my <= ZONE[3]
        self.publish_event(f"{'DELIVERED' if delivered else 'FAILED'} {target} "
                           f"at ({mx:.3f}, {my:.3f})")
        return {"ok": delivered, "goals": goals, "log": list(plan.log)}


if __name__ == "__main__":
    plans, events, ticks = [], [], []
    b = KitchenBridge(publish_plan=plans.append,
                      publish_event=events.append,
                      publish_joints=lambda n, p: ticks.append(p))
    result = b.handle_command("medicine next to water")

    print("plan:")
    for line in plans:
        print("  ", line)
    print("events:")
    for line in events:
        print("  ", line)
    print(f"joint states published: {len(ticks)}")

    assert any("cup" in p for p in plans), "the blocker was never relocated"
    assert result["ok"], "medicine was not delivered"
    assert len(ticks) > 100, "no joint stream to publish"
    print("\nbridge self-test OK — no ROS required")
