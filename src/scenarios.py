"""
Example scenarios for Isaac Sim: one fast check, and one parameter sweep.

    antioch scenario run --scenario falling_cube        one run
    antioch suite run smoke                  the fast check
    antioch suite run sweep                  every case on one machine
    antioch suite run sweep --machines 4     opt into multi-machine fan-out

The sweep shows how one scenario declaration expands into independent cases.
Queue staging adds the submitted project source to the immutable sim image;
add a Dockerfile only when the project needs custom packages or another image
layer. Development watch rules are not part of queued runs.

`falling_cube` publishes pictures and 3D geometry; `cube_bounce` is
deliberately scalar-only, because six sweep cases want one comparable chart
rather than six sets of frames.
"""

from __future__ import annotations

import antioch

logger = antioch.Logger("cube")


# `capture=False` turns off the automatic platform viewport, which points
# wherever Kit last left it. The pictures below are authored instead: aimed at
# the drop, and published only when the frame is worth reviewing.
@antioch.scenario(tags=["smoke"], capture=False)
def falling_cube(
    run: antioch.ScenarioRun,
    drop_height: float = antioch.param(2.0, ge=0.5, le=10.0, description="Initial cube height in meters"),
    steps: int = antioch.param(180, ge=1, description="Physics steps to simulate"),
) -> None:
    """
    Drop a dynamic cube and verify that it settles on the ground.
    """

    import numpy as np
    import rerun as rr
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.viewports import set_camera_view

    world = antioch.world()
    world.scene.add_ground_plane(restitution=0.0)
    # A dome alone flattens the scene; the distant light casts the shadow that
    # makes height readable in a still frame
    create_prim("/World/dome_light", "DomeLight", attributes={"inputs:intensity": 200.0})
    create_prim("/World/key_light", "DistantLight", attributes={"inputs:intensity": 400.0})
    cube = world.scene.add(
        DynamicCuboid(prim_path="/World/cube", name="cube", position=np.array([0.0, 0.0, drop_height]), size=0.5, color=np.array([0.2, 0.4, 0.9]))
    )
    world.reset()
    # Frame the whole drop, not a fixed vantage. The distance has to grow with
    # the height or a tall drop starts above the frame and lands below it —
    # measured at drop_height 10, where a fixed camera published no usable
    # picture at all and the run still passed.
    span = max(drop_height, 1.0)
    set_camera_view(eye=[1.4 * span, 1.4 * span, 0.75 * span], target=[0.0, 0.0, 0.45 * span], camera_prim_path="/OmniverseKit_Persp")
    # Boxes draw geometry; a transform alone would position an empty 3D pane
    logger.value("scene/ground", rr.Boxes3D(centers=[[0.0, 0.0, -0.025]], sizes=[[4.0, 4.0, 0.05]], colors=[[110, 110, 110]]))

    # The first capture of a run hands back a frame from before the scene was
    # drawn — measured as the bare Kit grid with no subject in it. It follows
    # the first call, not any particular step, so no cadence avoids it, and it
    # sits well inside the exposure band, so no check below can tell it from a
    # real picture. Spend it here and the review set never contains it.
    antioch.capture_viewport()

    attempted = 0
    review_frames = 0
    for step in range(steps):
        # Rendering is what gives `capture_viewport` something to return
        world.step(render=step % 3 == 0)
        position = cube.get_world_pose()[0]
        logger.scalar("height", float(position[2]))
        logger.value("scene/body", rr.Boxes3D(centers=[position.tolist()], sizes=[[0.5, 0.5, 0.5]], colors=[[51, 102, 230]]))
        # A 2 m drop is over in about 38 steps at 60 Hz, so every sixtieth step
        # photographed the cube twice and both times after it had landed
        if step % 10 == 0:
            attempted += 1
            frame = antioch.capture_viewport()
            if frame is not None:
                rgb = np.asarray(frame)[:, :, :3]
                # A black or blown-out picture is worse evidence than none
                if 10.0 <= float(rgb.mean()) <= 220.0:
                    logger.image("camera/rgb", rgb)
                    review_frames += 1

    final_z = float(cube.get_world_pose()[0][2])
    run.add_result("final_z", round(final_z, 4))
    # Both counts, so a run that published nothing says so instead of looking
    # like a run that never tried
    run.add_result("review_frames", review_frames)
    run.add_result("review_frames_attempted", attempted)
    run.check("the cube came to rest on the ground", final_z < 0.4, detail=f"cube centre rested at {final_z:.3f} m")
    run.check("the review camera published usable frames", review_frames > 0, detail=f"{review_frames} of {attempted} attempts passed the exposure gate")


@antioch.scenario(
    tags=["sweep"], capture=False, cases=[antioch.case(grid={"drop_height": [0.5, 2.0, 6.0], "restitution": [0.0, 0.7]}, id="h{drop_height}-e{restitution}")]
)
def cube_bounce(
    run: antioch.ScenarioRun,
    drop_height: float = antioch.param(2.0, ge=0.1, le=10.0, description="Initial cube height in meters"),
    restitution: float = antioch.param(0.0, ge=0.0, le=1.0, description="How bouncy the ground is"),
    steps: int = antioch.param(240, ge=1, description="Physics steps to simulate"),
) -> None:
    """
    Drop a cube onto ground of varying bounciness and measure the rebound.

    Six children from one declaration: three heights against two materials.
    """

    import numpy as np
    from isaacsim.core.api.objects import DynamicCuboid

    world = antioch.world()
    world.scene.add_ground_plane(restitution=restitution)
    cube = world.scene.add(
        DynamicCuboid(prim_path="/World/cube", name="cube", position=np.array([0.0, 0.0, drop_height]), size=0.3, color=np.array([0.9, 0.4, 0.2]))
    )
    world.reset()

    # The rebound is the highest point AFTER the cube first reaches the floor,
    # so the drop itself is never mistaken for a bounce
    landed = False
    rebound = 0.0
    for _ in range(steps):
        world.step(render=False)
        height = float(cube.get_world_pose()[0][2])
        logger.scalar("height", height)
        landed = landed or height < 0.2
        if landed:
            rebound = max(rebound, height)

    run.add_result("rebound_height", round(rebound, 4))
    # Both criteria are recorded even when the first one fails, so a run that
    # never reached the floor still reports what its rebound measured
    run.check("the cube reached the floor", landed, detail=f"within {steps} steps from {drop_height:.2f} m")
    run.check("the cube rebounded no higher than it was dropped", rebound < drop_height, detail=f"rebounded to {rebound:.3f} m from {drop_height:.2f} m")
