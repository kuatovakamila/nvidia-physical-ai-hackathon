"""
Rain cubes onto a ground plane, so a livestream is visibly live.

    antioch run src/main.py                   run with the default livestream
    antioch run --no-stream src/main.py       run headlessly
    antioch run src/main.py -- --seconds 5    quick enough to iterate on
    antioch run src/main.py -- --seconds 600  long enough to walk away from

Every knob is an argument because the point of this file is to be dispatched
many ways, often at once: concurrent runs of different lengths are what
exercise a project's machines, the console's stream picker, and the
interactive quota.
"""

from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING, cast

import antioch

if TYPE_CHECKING:
    from isaacsim.core.api import World

CYCLE_SECONDS = 4.0


def main() -> None:
    """
    Drop a batch of cubes, let them settle, lift them, and do it again.

    The cubes are spawned once and re-dropped by resetting the world, because
    physics handles bind at reset — adding prims to a running scene is the one
    thing this loop must not do.

    Streaming is on by default.  Pass ``--no-stream`` for a headless run; this
    same file supports both launch modes.
    """

    parser = argparse.ArgumentParser(description="Rain cubes onto a ground plane")
    parser.add_argument("--seconds", type=float, default=30.0, help="How long to simulate before exiting")
    parser.add_argument("--cubes", type=int, default=12, help="How many cubes to drop each cycle")
    arguments = parser.parse_args()

    antioch.boot()

    import numpy as np
    from isaacsim.core.api import World as WorldSingleton
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.viewports import set_camera_view

    # The singleton __new__ is annotated with the base class upstream; the cast
    # restores the World type without changing runtime behavior
    world = cast("World", WorldSingleton())
    # Procedural geometry only: remote asset fetches make runs slow and
    # non-reproducible — prefer generated prims or `antioch.load_asset`.
    world.scene.add_ground_plane(restitution=0.4)
    create_prim("/World/dome_light", "DomeLight", attributes={"inputs:intensity": 200.0})
    create_prim("/World/key_light", "DistantLight", attributes={"inputs:intensity": 400.0})

    # A fixed seed, so two concurrent runs look alike and a rerun is comparable
    placement = np.random.default_rng(7)
    for index in range(arguments.cubes):
        world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/cube_{index}",
                name=f"cube_{index}",
                position=np.array([placement.uniform(-1.5, 1.5), placement.uniform(-1.5, 1.5), 2.0 + index * 0.35]),
                size=0.3,
                color=np.array([placement.uniform(0.2, 1.0), placement.uniform(0.2, 1.0), 0.9]),
            )
        )
    world.reset()
    set_camera_view(eye=[4.5, 4.5, 3.5], target=[0.0, 0.0, 1.2], camera_prim_path="/OmniverseKit_Persp")

    # Rendering every step is the whole point: a stream of a static frame says
    # nothing about whether streaming works.
    deadline = time.monotonic() + arguments.seconds
    next_drop = time.monotonic() + CYCLE_SECONDS
    while time.monotonic() < deadline:
        world.step(render=True)
        if time.monotonic() >= next_drop:
            world.reset()
            set_camera_view(eye=[4.5, 4.5, 3.5], target=[0.0, 0.0, 1.2], camera_prim_path="/OmniverseKit_Persp")
            next_drop = time.monotonic() + CYCLE_SECONDS
    print(f"dropped {arguments.cubes} cubes over {arguments.seconds:g}s to t={world.current_time:.2f}s")


if __name__ == "__main__":
    main()
