"""
monitor.py — physics-state gates for the Kitchen Assistant.

Pure functions over poses; no sim imports. Feed them ground-truth state
inside the scenario after each phase. Philosophy per Guide 4: gate on
physics, not on video. Vision (HSV color id) lives in perception.py and
answers *which* object; this file answers *did the motion work*.

Owner: P2 (thresholds), P1 (wiring into run_phase).
Tune thresholds against the real vial asset once you have console access;
these defaults assume a ~15 mm-radius vial and the tabletop from Guide 2.
"""

from dataclasses import dataclass
import math

# ------------------------------------------------------------- thresholds --

GRASP_MISS_LIFT_M = 0.030    # after LIFT: object must be this high off table
DROP_DIST_M       = 0.080    # during CARRY: object-to-gripper distance limit
UPRIGHT_MAX_TILT_DEG = 15.0  # at PLACE: a vial must stand, not lie
SETTLE_SPEED_M_S  = 0.05     # at PLACE: object velocity must be below this
GRACE_TICKS       = 25       # ignore gates this many ticks after each phase
                             # start (0.5 s at 50 Hz) — Guide 4: judge the
                             # caught state, not the transient

PHASES = ("approach", "grasp", "lift", "carry", "place")


@dataclass
class Event:
    kind: str        # "GRASP_MISS" | "DROP" | "PLACE_FAIL" | "OK"
    phase: str
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.kind != "OK"


# ------------------------------------------------------------ phase checks --

def check_lift(obj_z_m: float, table_z_m: float = 0.0) -> Event:
    h = obj_z_m - table_z_m
    if h < GRASP_MISS_LIFT_M:
        return Event("GRASP_MISS", "lift", f"height {h*1000:.0f} mm")
    return Event("OK", "lift")


def check_carry(obj_xyz, gripper_xyz) -> Event:
    d = math.dist(obj_xyz, gripper_xyz)
    if d > DROP_DIST_M:
        return Event("DROP", "carry", f"obj-gripper {d*1000:.0f} mm")
    return Event("OK", "carry")


def tilt_deg_from_quat(qw, qx, qy, qz) -> float:
    """Angle between the object's local +Z and world +Z, in degrees."""
    # world-Z component of rotated local-Z axis:
    zz = 1.0 - 2.0 * (qx * qx + qy * qy)
    zz = max(-1.0, min(1.0, zz))
    return math.degrees(math.acos(zz))


def placement_gate(obj_xy, zone_rect, tilt_deg: float,
                   speed_m_s: float = 0.0) -> Event:
    """Success = inside the accessibility zone, upright, and settled.
    zone_rect = (xmin, xmax, ymin, ymax) — keep the zone axis-aligned and
    paint it bright in the scene so the gate is legible on the stream."""
    xmin, xmax, ymin, ymax = zone_rect
    x, y = obj_xy
    if not (xmin <= x <= xmax and ymin <= y <= ymax):
        return Event("PLACE_FAIL", "place", f"outside zone ({x:.2f},{y:.2f})")
    if tilt_deg > UPRIGHT_MAX_TILT_DEG:
        return Event("PLACE_FAIL", "place", f"tilt {tilt_deg:.0f} deg")
    if speed_m_s > SETTLE_SPEED_M_S:
        return Event("PLACE_FAIL", "place", f"still moving {speed_m_s:.2f} m/s")
    return Event("OK", "place")


def monitor(phase: str, state: dict) -> Event:
    """One entry point for run_phase(). `state` keys (fill from sim truth):
       obj_xyz, gripper_xyz, obj_quat (w,x,y,z), obj_speed, zone_rect,
       table_z, ticks_in_phase."""
    if state.get("ticks_in_phase", GRACE_TICKS) < GRACE_TICKS:
        return Event("OK", phase, "grace")
    if phase == "lift":
        return check_lift(state["obj_xyz"][2], state.get("table_z", 0.0))
    if phase == "carry":
        return check_carry(state["obj_xyz"], state["gripper_xyz"])
    if phase == "place":
        w, x, y, z = state["obj_quat"]
        return placement_gate(state["obj_xyz"][:2], state["zone_rect"],
                              tilt_deg_from_quat(w, x, y, z),
                              state.get("obj_speed", 0.0))
    return Event("OK", phase)


# --------------------------------------------------------------- self-test --

if __name__ == "__main__":
    assert check_lift(0.010).kind == "GRASP_MISS"
    assert check_lift(0.050).kind == "OK"
    assert check_carry((0.3, 0.0, 0.15), (0.3, 0.0, 0.16)).kind == "OK"
    assert check_carry((0.3, 0.0, 0.02), (0.3, 0.0, 0.16)).kind == "DROP"
    assert abs(tilt_deg_from_quat(1, 0, 0, 0)) < 1e-6                # upright
    assert abs(tilt_deg_from_quat(0.7071, 0.7071, 0, 0) - 90) < 0.1  # on side
    zone = (0.20, 0.40, 0.10, 0.30)
    assert placement_gate((0.30, 0.20), zone, 5.0).kind == "OK"
    assert placement_gate((0.30, 0.20), zone, 40.0).kind == "PLACE_FAIL"
    assert placement_gate((0.55, 0.20), zone, 5.0).kind == "PLACE_FAIL"
    assert monitor("lift", {"obj_xyz": (0, 0, 0.01), "ticks_in_phase": 5}).kind == "OK"
    print("monitor self-test OK")
