"""
kitchen_suite.py — eval + demo scenario scaffold for the Kitchen Assistant.

STATUS: SCAFFOLD. The @antioch.scenario / load_asset forms below are copied
from the event packet's own examples (Guide 2 / Guide 4); verify every
signature against the console docs after signup, and replace each TODO with
the real functions from the my-sim scripted expert. planner.py and monitor.py
are final and dependency-free — this file is the glue.

Owner: P1. Keep Isaac imports INSIDE the scenario function (packet rule:
the CLI imports this file on your laptop, where Isaac is not installed).

The two eval batches that produce the headline number:
  .venv/bin/antioch scenario run --scenario kitchen_suite -p supervisor_on=false
  .venv/bin/antioch scenario run --scenario kitchen_suite -p supervisor_on=true
"""

import json
import math
import random

import antioch  # provided by the my-sim project env

from planner import Planner, Obj
from monitor import monitor, placement_gate, PHASES

VIAL_R, CUP_R = 0.015, 0.040
TABLE = (0.10, 0.50, -0.25, 0.25)          # workspace bounds, metres
ZONE = (0.20, 0.40, 0.14, 0.24)            # accessibility zone — paint it!
HOME = (0.0, 0.0)                          # gripper home in table coords


@antioch.scenario(sim=antioch.BootProfile(physics_dt=0.005, render_dt=0.02))
def kitchen_suite(run,
                  episodes: int = 20,
                  supervisor_on: bool = True,
                  blocker_rate: float = 0.5,     # half the scenes are blocked
                  grasp_bias_mm: float = 15.0,   # manufactured failure, both modes
                  sabotage_at_s: float = -1.0,   # teleport vial at t; -1 = off
                  seed: int = 0):
    rng = random.Random(seed)

    # ---- scene ------------------------------------------------------------
    antioch.load_asset("so101_antioch", prim_path="/World/SO101",
                       version="1.3.2")                     # per Guide 2
    # TODO(P1): load vial ("medicine"), container ("water"), 1-2 cubes,
    #           and a bright zone marker. 10:30 recon decides exact asset
    #           names; fall back to labeled cubes if vial is missing.
    # TODO(P1): world = antioch.world(); world.reset(); wrist camera for
    #           perception.py (remember warm-render: first frame is black).

    results = []
    for ep in range(episodes):
        scene = _randomize(rng, blocked=rng.random() < blocker_rate)
        _teleport_all(scene)                                # TODO(P1)
        plan = Planner({o.name: o for o in scene}, TABLE, HOME,
                       naive=not supervisor_on)
        # Language layer replaces this line in the demo build:
        plan.request("medicine", _zone_center(), anchor="water")

        ep_log = {"ep": ep, "blocked": any(o.name == "cup" for o in scene),
                  "goals": []}
        while (g := plan.next_action()) is not None:
            bias = grasp_bias_mm / 1000.0 if g.attempts == 0 else 0.0
            ok = _execute(g, bias, sabotage_at_s, plan)     # phases + monitor
            plan.on_result(g, ok, new_pose=_true_xy(g.obj)) # TODO(P1): sim truth
            ep_log["goals"].append({"obj": g.obj, "reason": g.reason,
                                    "attempts": g.attempts, "ok": ok})
        ep_log["success"] = _final_gate("medicine")         # TODO(P1)
        ep_log["recovered"] = ep_log["success"] and any(
            gl["attempts"] > 0 or gl["reason"] != "user" for gl in ep_log["goals"])
        results.append(ep_log)
        run.log(json.dumps(ep_log))                         # TODO: real log API

    n = len(results)
    ok = sum(r["success"] for r in results)
    print(f"[suite] supervisor_on={supervisor_on}: {ok}/{n} "
          f"({100*ok/max(n,1):.0f}%), recovered={sum(r['recovered'] for r in results)}")


# ---- helpers (all TODO glue) ------------------------------------------------

def _randomize(rng, blocked: bool):
    """Vial + water always; a cup in the gripper->vial corridor when blocked."""
    vial = Obj("medicine", rng.uniform(0.36, 0.48), rng.uniform(-0.10, 0.10), VIAL_R)
    water = Obj("water", 0.30, 0.22, 0.045)
    scene = [vial, water]
    if blocked:
        t = rng.uniform(0.45, 0.75)          # partway along the corridor
        scene.append(Obj("cup", HOME[0] + t * (vial.x - HOME[0]),
                         HOME[1] + t * (vial.y - HOME[1]) + rng.uniform(-0.02, 0.02),
                         CUP_R))
    return scene


def _execute(goal, bias_m, sabotage_at_s, plan) -> bool:
    """Run approach/grasp/lift/carry/place via the my-sim expert, checking
    the monitor after each phase. Returns False on the first failure event."""
    # TODO(P1): wp = expert_waypoints(_true_pose(goal.obj), goal.to_xy, bias_m)
    for phase in PHASES:
        # TODO(P1): step the sim through this phase's targets;
        #           inside the loop: if sim_time crosses sabotage_at_s,
        #           teleport the vial 60 mm sideways (Guide 2 'r' mechanism)
        #           and plan.on_result-resync is NOT needed — the monitor
        #           will fail the phase and the retry replans from truth.
        ev = monitor(phase, _state_for(goal.obj))           # TODO(P1)
        if ev.failed:
            print(f"[monitor] ep goal={goal.obj} {ev.kind} @ {phase}: {ev.detail}")
            return False
    return True


def _zone_center():
    xmin, xmax, ymin, ymax = ZONE
    return ((xmin + xmax) / 2, (ymin + ymax) / 2)


def _teleport_all(scene):   raise NotImplementedError  # TODO(P1)
def _true_xy(name):         raise NotImplementedError  # TODO(P1)
def _state_for(name):       raise NotImplementedError  # TODO(P1)
def _final_gate(name):      raise NotImplementedError  # TODO(P1): placement_gate(...)
