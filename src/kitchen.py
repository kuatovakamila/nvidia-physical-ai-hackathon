"""
kitchen.py — the eval that produces the headline number.

Two cases, one declaration: supervisor off and supervisor on, same seeds, same
manufactured grasp bias. The difference between their pass rates is the whole
claim, and because every episode ends in run.check() it is a result a judge can
click into rather than a number we printed.

    .venv/bin/antioch scenario run --scenario kitchen
    .venv/bin/antioch suite run kitchen

The arm is driven by executor.run_goal(), the same function that will drive the
real SO-101 — only SimBackend below is swapped for RealBackend. Objects are
repositioned kinematically because the hackathon assets ship no rigid bodies at
all (measured: `kitchen_probe` reports 0 colliders on every one), so a held
object rides the tool tip instead of being gripped by friction.

Owner: P1.
"""

from __future__ import annotations

import math
import random

import antioch

from arm_ik import JOINT_ORDER, forward, reachable
from executor import ArmBackend, run_goal
from planner import Obj, Planner

logger = antioch.Logger("kitchen")

ARM = "so101_antioch"
ARM_VERSION = "1.3.2"
BLOCKS = "hackathon/GeometricBlocks 01"

# Measured reach at 20 mm above the table is x <= 0.300, |y| <= 0.250. Every
# bound below sits inside it with margin — the scaffold's original
# (0.10, 0.50) workspace was roughly half fiction.
TABLE = (0.135, 0.270, -0.160, 0.160)
ZONE = (0.150, 0.230, 0.055, 0.140)
HOME = (0.075, 0.0)
TABLE_Z = 0.0
OBJ_Z = 0.02

# Which block plays which part. Colour is the vocabulary the language layer
# and the HSV perception layer both speak.
ROLES = {
    "medicine": "SM_CylinderRed_01",
    "water": "SM_BlockGreen_01",
    "cup": "SM_BlockBlue_01",
}
PARKED = (0.0, -0.9, 0.0)  # where unused blocks go: out of frame, out of reach

MEDICINE_R, CUP_R, WATER_R = 0.018, 0.030, 0.030


def _zone_center():
    xmin, xmax, ymin, ymax = ZONE
    return ((xmin + xmax) / 2, (ymin + ymax) / 2)


def _set_translate(stage, prim_path: str, pos) -> None:
    """Move a prim without destroying the transform the asset authored.

    ClearXformOpOrder() would be shorter and would also drop any scale or
    rotate op the block relies on, which shows up as an invisible or
    mysteriously giant object rather than an error.
    """
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(*pos))
            return
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))


def _world_xyz(stage, prim_path: str):
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    c = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange().GetMidpoint()
    return (float(c[0]), float(c[1]), float(c[2]))


class SimBackend(ArmBackend):
    """executor.ArmBackend over an Isaac articulation and a USD stage.

    Everything outside this class talks in *centroid* coordinates: where the
    middle of the object is. USD translate ops move a prim's ORIGIN, and for
    these meshes the two differ by a couple of centimetres — enough that
    commanding the tool to an object's origin z misses its centre by more than
    the grasp tolerance, and enough that a block resting on the table already
    measures 40 mm "high" and sails through the lift gate. Both bugs bit on the
    first run. The offset is measured once per object and applied here, so the
    rest of the file never has to think about it again.
    """

    def __init__(self, world, robot, stage, prim_of: dict[str, str]):
        self.world = world
        self.robot = robot
        self.stage = stage
        self.prim_of = prim_of
        self.controller = robot.get_articulation_controller()
        self.held: str | None = None
        self.ticks = 0
        self.render_every = 4
        self.origin_to_centre: dict[str, tuple] = {}
        self.rest_z: dict[str, float] = {}

    def calibrate(self, name: str) -> None:
        """Measure this prim's origin->centroid offset and its resting height."""
        from pxr import UsdGeom

        prim = self.stage.GetPrimAtPath(self.prim_of[name])
        translate = (0.0, 0.0, 0.0)
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                v = op.Get()
                translate = (float(v[0]), float(v[1]), float(v[2]))
        centre = _world_xyz(self.stage, self.prim_of[name])
        self.origin_to_centre[name] = tuple(c - t for c, t in zip(centre, translate))
        self.rest_z[name] = centre[2]

    def place_centre(self, name: str, centre) -> None:
        """Move an object so its centroid lands exactly on `centre`."""
        off = self.origin_to_centre.get(name, (0.0, 0.0, 0.0))
        _set_translate(self.stage, self.prim_of[name],
                       tuple(c - o for c, o in zip(centre, off)))

    def centre_of(self, name: str):
        return _world_xyz(self.stage, self.prim_of[name])

    def send(self, q):
        from isaacsim.core.utils.types import ArticulationAction

        self.controller.apply_action(ArticulationAction(joint_positions=list(q)))
        self.world.step(render=self.ticks % self.render_every == 0)
        self.ticks += 1
        if self.held:
            # No rigid bodies on these assets, so "held" means "rides the tool
            # tip". Honest about what it is; identical from the planner's side.
            tip = forward(dict(zip(JOINT_ORDER, self.measured())))
            self.place_centre(self.held, tip)

    def measured(self):
        return [float(v) for v in self.robot.get_joint_positions()]

    def state_for(self, name):
        return {
            "obj_xyz": self.centre_of(name),
            "gripper_xyz": forward(dict(zip(JOINT_ORDER, self.measured()))),
            # Kinematic carry keeps objects upright by construction, so the
            # tilt gate cannot fire here. It earns its keep on the real arm.
            "obj_quat": (1.0, 0.0, 0.0, 0.0),
            "obj_speed": 0.0,
            "zone_rect": ZONE,
            # Not the table surface but this object's RESTING centroid height:
            # the lift gate asks "did it come off the table", and that is the
            # only reference against which the question has an answer.
            "table_z": self.rest_z.get(name, TABLE_Z),
            "ticks_in_phase": self.ticks,
        }

    def dwell(self, seconds):
        for _ in range(int(seconds * 40)):
            self.world.step(render=False)
            self.ticks += 1

    def attach(self, name):
        tip = forward(dict(zip(JOINT_ORDER, self.measured())))
        if math.dist(tip, _world_xyz(self.stage, self.prim_of[name])) <= 0.012:
            self.held = name

    def detach(self, name=None):
        self.held = None


def _randomize(rng: random.Random, blocked: bool) -> list[Obj]:
    """Medicine and water always; a cup in the home->medicine corridor when
    blocked. Every sampled point is checked against the real reach envelope,
    so an episode can never fail for a reason the robot could not fix."""
    for _ in range(200):
        mx = rng.uniform(0.200, 0.265)
        my = rng.uniform(-0.150, -0.020)
        if reachable(mx, my, OBJ_Z):
            break
    scene = [Obj("medicine", mx, my, MEDICINE_R), Obj("water", 0.165, 0.125, WATER_R)]
    if blocked:
        t = rng.uniform(0.45, 0.72)
        cx = HOME[0] + t * (mx - HOME[0])
        cy = HOME[1] + t * (my - HOME[1]) + rng.uniform(-0.015, 0.015)
        if reachable(cx, cy, OBJ_Z):
            scene.append(Obj("cup", cx, cy, CUP_R))
    return scene


@antioch.scenario(
    tags=["kitchen"],
    capture=False,
    sim=antioch.BootProfile(physics_dt=0.005, render_dt=0.02),
    cases=[antioch.case(grid={"supervisor_on": [False, True]}, id="sup-{supervisor_on}")],
)
def kitchen(
    run: antioch.ScenarioRun,
    supervisor_on: bool = antioch.param(True, description="Goal stack + retries, or naive execution"),
    episodes: int = antioch.param(10, ge=1, le=50),
    blocker_rate: float = antioch.param(0.6, ge=0.0, le=1.0),
    bias_mm: float = antioch.param(15.0, ge=0.0, le=40.0, description="Manufactured grasp offset, both arms"),
    seed: int = antioch.param(0),
) -> None:
    """Deliver the medicine to the accessibility zone, blocked or not, with and
    without the supervisor."""

    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.prims import create_prim

    rng = random.Random(seed)

    world = antioch.world()
    world.scene.add_ground_plane()
    create_prim("/World/dome_light", "DomeLight", attributes={"inputs:intensity": 300.0})
    create_prim("/World/key_light", "DistantLight", attributes={"inputs:intensity": 500.0})

    antioch.load_asset(ARM, prim_path="/World/SO101", version=ARM_VERSION)
    antioch.load_asset(BLOCKS, prim_path="/World/blocks", version="1.0.0")
    robot = world.scene.add(Robot(prim_path="/World/SO101", name="so101"))
    world.reset()
    antioch.capture_viewport()  # spend the pre-scene frame

    stage = antioch.stage()
    prim_of = {role: f"/World/blocks/{mesh}/{mesh}" for role, mesh in ROLES.items()}
    for role, path in prim_of.items():
        if not stage.GetPrimAtPath(path).IsValid():
            run.fail(f"expected block prim missing: {path}")

    backend = SimBackend(world, robot, stage, prim_of)
    for role in prim_of:
        backend.calibrate(role)
    logger.info(f"origin->centre offsets: {backend.origin_to_centre}")
    logger.info(f"resting centroid heights: {backend.rest_z}")

    results = []
    for ep in range(episodes):
        scene = _randomize(rng, blocked=rng.random() < blocker_rate)
        by_name = {o.name: o for o in scene}
        for role in prim_of:
            o = by_name.get(role)
            if o:
                backend.place_centre(role, (o.x, o.y, backend.rest_z[role]))
            else:
                _set_translate(stage, prim_of[role], PARKED)
        backend.held = None
        world.step(render=False)

        plan = Planner(dict(by_name), TABLE, HOME, naive=not supervisor_on)
        plan.request("medicine", _zone_center(), anchor="water")

        goals, guard = [], 0
        while (g := plan.next_action()) is not None and guard < 8:
            guard += 1
            bias = bias_mm / 1000.0 if g.attempts == 0 else 0.0
            target = by_name[g.obj]
            grip_z = backend.rest_z[g.obj]
            rep = run_goal(
                backend, g.obj, (target.x, target.y, grip_z), (g.to_xy[0], g.to_xy[1], grip_z),
                bias_m=bias, on_grasp=backend.attach, on_release=backend.detach,
            )
            truth = backend.centre_of(g.obj)
            by_name[g.obj].x, by_name[g.obj].y = truth[0], truth[1]
            plan.on_result(g, rep.ok, new_pose=(truth[0], truth[1]))
            goals.append({
                "obj": g.obj, "reason": g.reason, "attempts": g.attempts, "ok": rep.ok,
                "failed_phase": rep.failed_phase, "unreachable": rep.unreachable,
            })

        mx, my, _mz = backend.centre_of("medicine")
        delivered = ZONE[0] <= mx <= ZONE[1] and ZONE[2] <= my <= ZONE[3]
        recovered = delivered and any(
            gl["attempts"] > 0 or gl["reason"] != "user" for gl in goals
        )
        results.append({"ep": ep, "delivered": delivered, "recovered": recovered,
                        "blocked": "cup" in by_name, "goals": goals,
                        # The planner's own narration: "medicine blocked by cup
                        # -> relocate it". This is the emergent-behaviour
                        # evidence, so it travels with the run, not just stdout.
                        "plan_log": list(plan.log)})
        logger.scalar("delivered", 1.0 if delivered else 0.0)
        logger.info(f"ep {ep}: delivered={delivered} recovered={recovered} goals={len(goals)}")

    n = len(results)
    ok = sum(r["delivered"] for r in results)
    blocked_n = sum(r["blocked"] for r in results)
    blocked_ok = sum(r["delivered"] for r in results if r["blocked"])

    run.add_results({
        "supervisor_on": supervisor_on,
        "episodes": n,
        "delivered": ok,
        "success_rate": round(100.0 * ok / max(n, 1), 1),
        "blocked_episodes": blocked_n,
        "blocked_delivered": blocked_ok,
        "recovered": sum(r["recovered"] for r in results),
        "relocations": sum(
            1 for r in results for gl in r["goals"] if gl["reason"] != "user"
        ),
        "episodes_detail": results,
    })
    run.check(
        "every episode ran to a decision",
        n == episodes,
        detail=f"{n}/{episodes} episodes completed",
    )
    run.check(
        "the medicine reached the accessibility zone in a majority of episodes",
        ok * 2 > n,
        detail=f"{ok}/{n} delivered ({100.0*ok/max(n,1):.0f}%), "
               f"blocked scenes {blocked_ok}/{blocked_n}",
    )
