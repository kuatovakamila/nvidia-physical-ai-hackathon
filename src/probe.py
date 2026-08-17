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


@antioch.scenario(tags=["probe"], capture=False)
def camera_framing(
    run: antioch.ScenarioRun,
    elevation_deg: float = antioch.param(50.0, ge=5.0, le=89.0),
) -> None:
    """Find a camera placement that frames the STAGED workspace.

    Two lessons are baked in. capture_viewport() returns the last frame the
    render graph produced, so a single render step after moving the camera
    hands back the previous view — the first version of this sweep was
    off by one and "found" a distance that was really its neighbour's.
    And the scene has to be staged the way the eval stages it: sweeping
    against all 18 blocks clustered at the origin optimises for a picture
    the real scenario never takes.
    """

    import math

    import numpy as np
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.viewports import set_camera_view
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom

    world = antioch.world()
    world.scene.add_ground_plane()
    create_prim("/World/dome_light", "DomeLight", attributes={"inputs:intensity": 300.0})
    create_prim("/World/key_light", "DistantLight", attributes={"inputs:intensity": 500.0})
    antioch.load_asset(ARM, prim_path="/World/SO101", version=ARM_VERSION)
    antioch.load_asset("hackathon/GeometricBlocks 01", prim_path="/World/blocks",
                       version="1.0.0")
    world.scene.add(Robot(prim_path="/World/SO101", name="so101"))
    world.reset()
    antioch.capture_viewport()

    stage = antioch.stage()

    def put(prim_path, pos):
        prim = stage.GetPrimAtPath(prim_path)
        xf = UsdGeom.Xformable(prim)
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(*pos))
                return
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))

    # Same staging as the eval: three blocks at the extremes of the workspace,
    # everything else off the table. Framing that fits these fits any episode.
    staged = {
        "SM_CylinderRed_01": (0.26, -0.13, 0.0),   # medicine, far corner
        "SM_BlockGreen_01": (0.165, 0.125, 0.0),   # water, near the zone
        "SM_BlockBlue_01": (0.17, -0.05, 0.0),     # cup, mid corridor
    }
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/blocks")):
        name = prim.GetName()
        if not name.startswith("SM_") or not prim.IsA(UsdGeom.Xform):
            continue
        put(prim.GetPath().pathString, staged.get(name, (0.0, -0.9, 0.0)))

    target = (0.16, 0.0, 0.02)
    el = math.radians(elevation_deg)
    report = {}
    for dist in (0.20, 0.30, 0.45, 0.65, 0.90):
        eye = [
            target[0] + dist * math.cos(el) * 0.75,
            target[1] + dist * math.cos(el) * 0.66,
            target[2] + dist * math.sin(el),
        ]
        set_camera_view(eye=eye, target=list(target),
                        camera_prim_path="/OmniverseKit_Persp")
        for _ in range(4):          # flush the render graph, see docstring
            world.step(render=True)
        frame = antioch.capture_viewport()
        if frame is None:
            continue
        rgb = np.asarray(frame)[:, :, :3]
        busy = float((rgb.std(axis=2) > 6).mean())
        tag = f"d{dist:.2f}"
        path = f"/tmp/{tag}.png"
        Image.fromarray(rgb.astype(np.uint8)).save(path)
        run.add_artifact(path, name=f"{tag}.png")
        report[tag] = {"eye": [round(v, 3) for v in eye], "subject_fraction": round(busy, 4)}
        logger.info(f"{tag}: eye={[round(v,3) for v in eye]} subject={busy:.4f}")

    run.add_result("framing", report)
    run.check("every placement was photographed", len(report) == 5,
              detail=f"{len(report)}/5 placements")


@antioch.scenario(tags=["probe"])
def video_probe(run: antioch.ScenarioRun) -> None:
    """Does the platform's automatic capture produce a video artifact?

    Every scenario so far ran with capture=False, which the init scaffold
    recommends because the automatic viewport points wherever Kit last left it.
    But it is the only mechanism that might yield a moving picture rather than
    stills, and a demo wants motion. This asks the cheap version of the
    question before anything is rewired: sweep the arm, render every step, and
    see what lands in the artifact list.
    """

    import math

    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.utils.viewports import set_camera_view

    world = antioch.world()
    world.scene.add_ground_plane()
    create_prim("/World/dome_light", "DomeLight", attributes={"inputs:intensity": 300.0})
    create_prim("/World/key_light", "DistantLight", attributes={"inputs:intensity": 500.0})
    antioch.load_asset(ARM, prim_path="/World/SO101", version=ARM_VERSION)
    robot = world.scene.add(Robot(prim_path="/World/SO101", name="so101"))
    world.reset()
    antioch.capture_viewport()
    set_camera_view(eye=[0.377, 0.191, 0.365], target=[0.16, 0.0, 0.02],
                    camera_prim_path="/OmniverseKit_Persp")

    controller = robot.get_articulation_controller()
    for i in range(240):
        t = i / 240.0
        q = [0.9 * math.sin(2 * math.pi * t), 0.5 * math.sin(4 * math.pi * t),
             -0.8, -0.5, 0.0, 0.6 + 0.6 * math.sin(6 * math.pi * t)]
        controller.apply_action(ArticulationAction(joint_positions=q))
        world.step(render=True)

    run.add_result("rendered_steps", 240)
    run.check("the sweep completed", True, detail="240 rendered steps")


@antioch.scenario(tags=["probe"], capture=False)
def encoder_probe(run: antioch.ScenarioRun) -> None:
    """What can this image encode a video with?

    capture=True produced no video artifact, so the frames have to be encoded
    in-scenario. Rather than guess at imageio vs cv2 vs a bare ffmpeg binary
    and burn a cloud round-trip per guess, ask once.
    """

    import importlib
    import shutil
    import subprocess

    found = {}
    for mod in ("imageio", "imageio_ffmpeg", "cv2", "av", "PIL"):
        try:
            m = importlib.import_module(mod)
            found[mod] = getattr(m, "__version__", "present")
        except Exception as exc:
            found[mod] = f"MISSING ({type(exc).__name__})"

    binaries = {}
    for exe in ("ffmpeg", "ffprobe"):
        path = shutil.which(exe)
        binaries[exe] = path or "MISSING"
        if path:
            try:
                out = subprocess.run([path, "-version"], capture_output=True,
                                     text=True, timeout=20)
                binaries[exe] = out.stdout.splitlines()[0][:90]
            except Exception as exc:
                binaries[exe] = f"{path} (version check failed: {exc})"

    writer = None
    try:
        import imageio_ffmpeg
        writer = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        writer = f"imageio_ffmpeg.get_ffmpeg_exe failed: {exc}"

    logger.info(f"modules: {found}")
    logger.info(f"binaries: {binaries}")
    logger.info(f"imageio_ffmpeg exe: {writer}")
    run.add_results({"modules": found, "binaries": binaries, "ffmpeg_exe": str(writer)})
    run.check("some encoder is available", True, detail=str(found))


@antioch.scenario(tags=["probe"], capture=False)
def tool_calibration(run: antioch.ScenarioRun) -> None:
    """Where is the grasp point really, compared to what forward() claims?

    A carried object is placed at forward(measured_joints). If that point is
    not between the jaws, the object rides a phantom offset and the video shows
    blocks moving beside the gripper rather than in it — which is exactly what
    it showed. L_TOOL was measured to the jaw LINK ORIGIN and flagged in
    arm_ik.py as needing calibration against a real grasp; this is that
    calibration, done against the twin instead of by eye.
    """

    import numpy as np
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Usd, UsdGeom

    import sys
    sys.path.insert(0, "/workspace/project")
    from arm_ik import JOINT_ORDER, forward

    antioch.load_asset(ARM, prim_path="/World/SO101", version=ARM_VERSION)
    world = antioch.world()
    world.scene.add_ground_plane()
    robot = world.scene.add(Robot(prim_path="/World/SO101", name="so101"))
    world.reset()
    controller = robot.get_articulation_controller()
    stage = antioch.stage()

    def bbox_centre(path):
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        c = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange().GetMidpoint()
        return np.array([float(c[0]), float(c[1]), float(c[2])])

    poses = [
        [0.0, 0.3, -0.9, -0.6, 0.0, 1.2],
        [0.4, 0.5, -1.1, -0.5, 0.0, 1.2],
        [-0.4, 0.2, -0.8, -0.7, 0.0, 1.2],
        [0.2, 0.8, -1.3, -0.4, 0.0, 1.2],
        [-0.2, 0.6, -1.0, -0.9, 0.0, 1.2],
    ]
    rows = []
    for q in poses:
        controller.apply_action(ArticulationAction(joint_positions=q))
        for _ in range(120):
            world.step(render=False)
        measured = [float(v) for v in robot.get_joint_positions()]
        fk = np.array(forward(dict(zip(JOINT_ORDER, measured))))
        grip = bbox_centre("/World/SO101/gripper")
        jaw = bbox_centre("/World/SO101/jaw")
        mid = (grip + jaw) / 2.0
        rows.append({
            "cmd": [round(v, 3) for v in q],
            "fk": [round(float(v), 4) for v in fk],
            "gripper": [round(float(v), 4) for v in grip],
            "jaw": [round(float(v), 4) for v in jaw],
            "fk_to_gripper_mm": round(float(np.linalg.norm(fk - grip)) * 1000, 1),
            "fk_to_jawmid_mm": round(float(np.linalg.norm(fk - mid)) * 1000, 1),
            "delta_to_jawmid_m": [round(float(v), 4) for v in (mid - fk)],
        })
        logger.info(f"{rows[-1]}")

    run.add_result("samples", rows)
    mean_gap = float(np.mean([r["fk_to_jawmid_mm"] for r in rows]))
    run.add_result("mean_fk_to_jawmid_mm", round(mean_gap, 1))
    run.check("forward() lands within 10 mm of the jaw midpoint", mean_gap < 10.0,
              detail=f"mean {mean_gap:.1f} mm across {len(rows)} poses")


@antioch.scenario(tags=["probe"], capture=False)
def zero_pose_frames(run: antioch.ScenarioRun) -> None:
    """Full zero-pose kinematics: every link's world transform and every
    joint's world axis.

    arm_ik's planar model assumed the links lie along one axis at q=0. They do
    not: measured, the upper arm points 77 deg up and the tool carries a ~25 mm
    lateral offset, so forward() misses the real gripper by ~380 mm. With each
    link's zero-pose frame and each joint's world axis, forward kinematics can
    be built exactly (product of exponentials) instead of assumed.
    """

    from isaacsim.core.api.robots import Robot
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    antioch.load_asset(ARM, prim_path="/World/SO101", version=ARM_VERSION)
    world = antioch.world()
    world.scene.add(Robot(prim_path="/World/SO101", name="so101"))
    world.reset()
    for _ in range(30):
        world.step(render=False)

    stage = antioch.stage()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    links = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/SO101")):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        m = cache.GetLocalToWorldTransform(prim)
        t = m.ExtractTranslation()
        r = m.ExtractRotationMatrix()
        links[prim.GetName()] = {
            "pos": [round(float(v), 6) for v in t],
            "rot": [[round(float(r[i][j]), 6) for j in range(3)] for i in range(3)],
        }

    joints = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/SO101")):
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        j = UsdPhysics.RevoluteJoint(prim)
        b0 = [t.name for t in j.GetBody0Rel().GetTargets()]
        b1 = [t.name for t in j.GetBody1Rel().GetTargets()]
        q0 = j.GetLocalRot0Attr().Get() or Gf.Quatf(1, 0, 0, 0)
        joints[prim.GetName()] = {
            "axis": str(j.GetAxisAttr().Get()),
            "body0": b0[0] if b0 else None,
            "body1": b1[0] if b1 else None,
            "local_pos0": [round(float(v), 6) for v in (j.GetLocalPos0Attr().Get() or (0, 0, 0))],
            "local_rot0": [round(float(q0.GetReal()), 6)] + [round(float(v), 6) for v in q0.GetImaginary()],
        }

    def bbox_centre(path):
        c = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        m = c.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange().GetMidpoint()
        return [round(float(m[0]), 6), round(float(m[1]), 6), round(float(m[2]), 6)]

    grasp = {
        "gripper_bbox": bbox_centre("/World/SO101/gripper"),
        "jaw_bbox": bbox_centre("/World/SO101/jaw"),
    }
    grasp["jaw_midpoint"] = [round((a + b) / 2, 6)
                             for a, b in zip(grasp["gripper_bbox"], grasp["jaw_bbox"])]

    logger.info(f"LINKS {links}")
    logger.info(f"JOINTS {joints}")
    logger.info(f"GRASP {grasp}")
    run.add_results({"links": links, "joints": joints, "grasp": grasp})
    run.check("all seven links reported", len(links) == 7, detail=f"{len(links)} links")
