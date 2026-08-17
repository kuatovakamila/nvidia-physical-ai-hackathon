"""
Asset recon for the Kitchen Assistant.

Answers the questions that decide the build, and that no amount of local
reading can settle:

  1. What is actually inside `hackathon/GeometricBlocks 01`? The bounding box
     is 0.38 x 0.29 m, which is a *set*, not a block. Can we address one block
     at a time, and do they carry distinct colors for the HSV layer?
  2. Do the hackathon assets ship physics, or are they visual props? A
     "placeable" tag is not a RigidBodyAPI. If they are inert we author
     colliders and rigid bodies ourselves, and that is a task, not a detail.
  3. What are the SO-101's DOF names, limits, and jaw range? Every grasp
     target we ever command is expressed in those units.
  4. What does the scene look like? One picture is worth an hour of guessing.

Run:  .venv/bin/antioch scenario run --scenario kitchen_probe
"""

from __future__ import annotations

import antioch

logger = antioch.Logger("probe")

ARM = "so101_antioch"
ARM_VERSION = "1.3.2"  # 1.3.1 is the "jaw trap" per the asset's own description

# name -> (prim_path, position)
PROPS = {
    "hackathon/Vial 2ml 01": ("/World/vial", (0.30, 0.00, 0.05)),
    "hackathon/GeometricBlocks 01": ("/World/blocks", (0.30, 0.30, 0.05)),
    "hackathon/BlueRack 01": ("/World/rack", (0.30, -0.30, 0.05)),
}


@antioch.scenario(tags=["probe"], capture=False)
def arm_kinematics(run: antioch.ScenarioRun) -> None:
    """
    Measure the SO-101's link geometry from the calibrated twin.

    Analytic IK needs link lengths. Taking them from a vendor drawing invites
    a demo where sim and hardware disagree by a centimetre; taking them from
    the twin that Antioch calibrated means the numbers we solve against are
    the numbers the simulator actually integrates — and, because the twin is
    of this exact arm, the numbers the servos will honour too.
    """

    from pxr import Usd, UsdGeom, UsdPhysics

    antioch.load_asset(ARM, prim_path="/World/SO101", version=ARM_VERSION)
    world = antioch.world()
    world.reset()

    stage = antioch.stage()
    root = stage.GetPrimAtPath("/World/SO101")

    joints = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        body0 = [t.pathString for t in joint.GetBody0Rel().GetTargets()]
        body1 = [t.pathString for t in joint.GetBody1Rel().GetTargets()]
        joints.append(
            {
                "path": prim.GetPath().pathString,
                "axis": joint.GetAxisAttr().Get(),
                "body0": body0[0] if body0 else None,
                "body1": body1[0] if body1 else None,
                # Anchor in each body's frame: the offset between consecutive
                # anchors along the chain IS the link length we need.
                "local_pos0": [round(float(v), 5) for v in (joint.GetLocalPos0Attr().Get() or (0, 0, 0))],
                "local_pos1": [round(float(v), 5) for v in (joint.GetLocalPos1Attr().Get() or (0, 0, 0))],
            }
        )

    # World positions of every rigid link in the default pose, so the chain can
    # be reconstructed independently of how the joints declare their anchors.
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    links = {}
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        m = xform_cache.GetLocalToWorldTransform(prim)
        t = m.ExtractTranslation()
        links[prim.GetPath().pathString] = [round(float(t[0]), 5), round(float(t[1]), 5), round(float(t[2]), 5)]

    logger.info(f"joints: {joints}")
    logger.info(f"links: {links}")
    run.add_result("joints", joints)
    run.add_result("links", links)
    run.check("the arm exposes revolute joints", len(joints) >= 5, detail=f"{len(joints)} revolute joints")
    run.check("the arm exposes rigid links", len(links) >= 5, detail=f"{len(links)} rigid bodies")


def _describe_subtree(stage, root_path: str, max_prims: int = 40) -> dict:
    """Prim structure, physics APIs, and world bounds under one asset root."""
    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return {"error": f"no prim at {root_path}"}

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )

    meshes, rigid_bodies, colliders, materials = [], [], [], []
    total = 0
    for prim in Usd.PrimRange(root):
        total += 1
        path = prim.GetPath().pathString
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(path)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(path)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append(path)
        # Distinct bound materials are how we tell "one prop" from "a set of
        # differently-coloured blocks we can address individually"
        rel = prim.GetRelationship("material:binding")
        if rel:
            for t in rel.GetTargets():
                if t.pathString not in materials:
                    materials.append(t.pathString)

    rng = cache.ComputeWorldBound(root).ComputeAlignedRange()
    size = rng.GetSize()

    return {
        "prims_total": total,
        "meshes": len(meshes),
        "mesh_paths": meshes[:max_prims],
        "rigid_bodies": len(rigid_bodies),
        "rigid_body_paths": rigid_bodies[:max_prims],
        "colliders": len(colliders),
        "materials": len(materials),
        "material_paths": materials[:max_prims],
        "world_bbox_m": [round(float(v), 4) for v in (size[0], size[1], size[2])],
    }


@antioch.scenario(
    tags=["probe"],
    capture=False,
    sim=antioch.BootProfile(physics_dt=0.005, render_dt=0.02),
)
def kitchen_probe(
    run: antioch.ScenarioRun,
    settle_steps: int = antioch.param(
        120, ge=0, description="Physics steps to run, to see what is dynamic"
    ),
) -> None:
    """Load the arm and every hackathon prop; report structure, physics, and a picture."""

    import numpy as np
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.viewports import set_camera_view

    world = antioch.world()
    world.scene.add_ground_plane()
    create_prim("/World/dome_light", "DomeLight", attributes={"inputs:intensity": 300.0})
    create_prim("/World/key_light", "DistantLight", attributes={"inputs:intensity": 500.0})

    antioch.load_asset(ARM, prim_path="/World/SO101", version=ARM_VERSION)
    loaded, load_errors = [], {}
    for name, (prim_path, _pos) in PROPS.items():
        try:
            antioch.load_asset(name, prim_path=prim_path, version="1.0.0")
            loaded.append(name)
        except Exception as exc:  # recon: one bad asset must not hide the rest
            load_errors[name] = f"{type(exc).__name__}: {exc}"
            logger.error(f"load_asset failed for {name}: {exc}")

    robot = world.scene.add(Robot(prim_path="/World/SO101", name="so101"))
    world.reset()

    # The first capture of a run hands back a pre-scene frame regardless of
    # cadence; spend it before any picture we intend to keep.
    antioch.capture_viewport()

    stage = antioch.stage()

    # -- 1..2: what is in each prop, and does it carry physics -----------------
    report = {}
    for name, (prim_path, _pos) in PROPS.items():
        if name in loaded:
            report[name] = _describe_subtree(stage, prim_path)
            logger.info(f"{name}: {report[name]}")
    run.add_result("assets", report)
    run.add_result("load_errors", load_errors)

    # -- 3: the arm's actual joint contract -----------------------------------
    arm = {}
    try:
        dof_names = list(robot.dof_names or [])
        # get_joint_limits() is (n_dof, 2) — one [min, max] row per joint, not
        # a (lower[], upper[]) pair. Indexing it as the latter silently reports
        # joints 0 and 1 and hides the Jaw, which is the one we actually need.
        limits = np.asarray(robot.get_articulation_controller().get_joint_limits())
        arm = {
            "dof_count": len(dof_names),
            "limits_shape": list(limits.shape),
            "range_rad": {
                name: [round(float(limits[i][0]), 4), round(float(limits[i][1]), 4)]
                for i, name in enumerate(dof_names)
            },
            "default_pose": [round(float(v), 4) for v in robot.get_joint_positions()],
            # So the next round-trip never guesses an API name again
            "api": [n for n in dir(robot) if "dof" in n.lower() or "joint" in n.lower()],
        }
    except Exception as exc:
        arm = {"error": f"{type(exc).__name__}: {exc}"}
    logger.info(f"arm: {arm}")
    run.add_result("arm", arm)

    # -- what actually moves when physics runs --------------------------------
    def _pos(prim_path: str):
        from pxr import Usd, UsdGeom

        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        c = cache.ComputeWorldBound(prim).ComputeAlignedRange().GetMidpoint()
        return (float(c[0]), float(c[1]), float(c[2]))

    before = {p: _pos(p) for _n, (p, _x) in PROPS.items()}
    for step in range(settle_steps):
        world.step(render=step % 4 == 0)
    after = {p: _pos(p) for _n, (p, _x) in PROPS.items()}

    moved = {
        p: round(float(np.linalg.norm(np.array(after[p]) - np.array(before[p]))), 4)
        for p in before
        if before[p] is not None and after[p] is not None
    }
    logger.info(f"displacement after {settle_steps} steps: {moved}")
    run.add_result("displacement_m", moved)

    # -- 4: a picture, so we can see what we are talking about ----------------
    set_camera_view(
        eye=[0.9, 0.9, 0.7],
        target=[0.28, 0.0, 0.05],
        camera_prim_path="/OmniverseKit_Persp",
    )
    published = 0
    for _ in range(3):
        world.step(render=True)
        frame = antioch.capture_viewport()
        if frame is None:
            continue
        rgb = np.asarray(frame)[:, :, :3]
        if 10.0 <= float(rgb.mean()) <= 220.0:
            logger.image("camera/rgb", rgb)
            published += 1
            if published == 1:
                # Also as a downloadable file: the .rrd needs the console, and
                # a PNG we can open locally settles "do the colours survive the
                # missing textures" in one look.
                try:
                    from PIL import Image

                    Image.fromarray(rgb.astype(np.uint8)).save("/tmp/scene.png")
                    run.add_artifact("/tmp/scene.png", name="scene.png")
                except Exception as exc:
                    logger.warning(f"could not save PNG artifact: {exc}")
    run.add_result("review_frames", published)

    run.check(
        "every hackathon asset loaded",
        len(loaded) == len(PROPS),
        detail=f"{len(loaded)}/{len(PROPS)} loaded; errors={list(load_errors)}",
    )
    run.check(
        "the arm reports a jointed articulation",
        arm.get("dof_count", 0) >= 5,
        detail=f"dof_names={arm.get('dof_names')}",
    )
    run.check(
        "the scene published a usable picture",
        published > 0,
        detail=f"{published} frames passed the exposure gate",
    )
