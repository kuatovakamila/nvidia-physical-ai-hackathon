"""
arm_ik.py — analytic inverse kinematics for the SO-101.

Pure Python, zero dependencies, like planner.py and monitor.py. Run
`python3 arm_ik.py` to see the self-test: FK(IK(p)) == p across the workspace,
plus the reach envelope that the scene layout has to respect.

Every constant here was MEASURED off the calibrated `so101_antioch` twin
(scenario `arm_kinematics`, run cc3bebba…), not read off a vendor drawing.
Link lengths were cross-checked two ways — joint anchor offsets and world link
origins — and agree to five decimals. Because the twin is of this exact arm,
the same numbers drive the servos.

Frame: base at the origin, +x forward (away from the operator), +y left,
+z up, metres, radians. Joint order matches the articulation exactly:

    Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw

Owner: P2 alongside planner.py. Consumed by both executors — the same six
numbers go to Isaac and to SO101Follower.send_action().
"""

from dataclasses import dataclass
import math

# ----------------------------------------------------------------- geometry --
# Measured: joints/Pitch anchor -> joints/Elbow anchor, etc.
L_UPPER = 0.1160      # Pitch axis  -> Elbow axis
L_LOWER = 0.1350      # Elbow axis  -> Wrist_Pitch axis
L_TOOL = 0.0944       # Wrist_Pitch axis -> jaw. Calibrate against a real grasp:
                      # the useful point is between the fingers, not the jaw
                      # link origin, and that offset is a few mm of taste.

# The Pitch axis in base coordinates. The yaw axis is vertical and passes a
# couple of centimetres off the base origin; at this scale that is not noise,
# so it is carried explicitly rather than assumed to be zero.
SHOULDER_X = 0.0519
SHOULDER_Y = 0.0026
SHOULDER_Z = 0.1194

# Measured limits, radians. Jaw is asymmetric: it closes slightly past zero.
LIMITS = {
    "Rotation": (-1.9199, 1.9199),
    "Pitch": (-1.7453, 1.7453),
    "Elbow": (-1.7453, 1.5708),
    "Wrist_Pitch": (-1.6581, 1.6581),
    "Wrist_Roll": (-2.7925, 2.7925),
    "Jaw": (-0.1745, 1.7453),
}
JOINT_ORDER = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")

JAW_OPEN = 1.2        # rad, comfortably clear of a 40 mm block
JAW_CLOSED = 0.0      # rad; squeeze past this only with a real object between


class Unreachable(ValueError):
    """Raised when a Cartesian goal has no valid joint solution."""


@dataclass
class Pose:
    """A grasp target: where the tool tip goes, and how the wrist is tilted.

    `pitch` is the tool's angle below horizontal. -pi/2 is straight down, which
    is what a top-down pick wants and what keeps a vial upright.
    """

    x: float
    y: float
    z: float
    pitch: float = -math.pi / 2


def _clamp(name: str, value: float) -> float:
    lo, hi = LIMITS[name]
    return max(lo, min(hi, value))


def forward(q: dict[str, float]) -> tuple[float, float, float]:
    """Tool-tip position for a joint dict. Exists so IK can be checked rather
    than believed — a closed-form solve that is never verified against the
    forward map is a guess with extra steps."""
    yaw = q["Rotation"]
    a1, a2, a3 = q["Pitch"], q["Elbow"], q["Wrist_Pitch"]

    # Planar chain in the vertical plane containing the yaw direction. Angles
    # accumulate: each link is placed relative to the previous one.
    t1 = a1
    t2 = a1 + a2
    t3 = a1 + a2 + a3
    r = L_UPPER * math.cos(t1) + L_LOWER * math.cos(t2) + L_TOOL * math.cos(t3)
    z = L_UPPER * math.sin(t1) + L_LOWER * math.sin(t2) + L_TOOL * math.sin(t3)

    return (
        SHOULDER_X + r * math.cos(yaw),
        SHOULDER_Y + r * math.sin(yaw),
        SHOULDER_Z + z,
    )


def inverse(pose: Pose, elbow_up: bool = True) -> dict[str, float]:
    """Closed-form IK. Raises Unreachable rather than returning a bent pose
    that silently misses — a planner that thinks it commanded a grasp and did
    not is the worst failure mode we can build."""
    dx = pose.x - SHOULDER_X
    dy = pose.y - SHOULDER_Y
    yaw = math.atan2(dy, dx)
    if not (LIMITS["Rotation"][0] <= yaw <= LIMITS["Rotation"][1]):
        raise Unreachable(f"yaw {math.degrees(yaw):.0f} deg is outside the base range")

    # Reduce to the planar problem in the (r, z) half-plane.
    r = math.hypot(dx, dy)
    z = pose.z - SHOULDER_Z

    # Back the tool off the target to find the wrist centre, so the last link
    # arrives at the commanded tilt instead of wherever the solve lands.
    rw = r - L_TOOL * math.cos(pose.pitch)
    zw = z - L_TOOL * math.sin(pose.pitch)

    d2 = rw * rw + zw * zw
    d = math.sqrt(d2)
    if d > L_UPPER + L_LOWER:
        raise Unreachable(
            f"wrist centre {d*1000:.0f} mm from shoulder, arm spans {(L_UPPER+L_LOWER)*1000:.0f} mm"
        )
    if d < abs(L_UPPER - L_LOWER):
        raise Unreachable(f"wrist centre {d*1000:.0f} mm from shoulder is inside the dead zone")

    # Two-link cosine solve.
    cos_elbow = (d2 - L_UPPER * L_UPPER - L_LOWER * L_LOWER) / (2 * L_UPPER * L_LOWER)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    elbow = math.acos(cos_elbow)
    if elbow_up:
        elbow = -elbow

    # Shoulder angle: direction to the wrist centre, plus the offset the elbow
    # bend introduces.
    pitch = math.atan2(zw, rw) - math.atan2(
        L_LOWER * math.sin(elbow), L_UPPER + L_LOWER * math.cos(elbow)
    )
    wrist_pitch = pose.pitch - pitch - elbow

    q = {
        "Rotation": yaw,
        "Pitch": pitch,
        "Elbow": elbow,
        "Wrist_Pitch": wrist_pitch,
        "Wrist_Roll": 0.0,
        "Jaw": JAW_OPEN,
    }
    for name in ("Pitch", "Elbow", "Wrist_Pitch"):
        lo, hi = LIMITS[name]
        if not (lo - 1e-9 <= q[name] <= hi + 1e-9):
            raise Unreachable(
                f"{name} needs {math.degrees(q[name]):.0f} deg, "
                f"limit is {math.degrees(lo):.0f}..{math.degrees(hi):.0f}"
            )
    return q


def reachable(x: float, y: float, z: float, pitch: float = -math.pi / 2) -> bool:
    """Cheap predicate for the planner's free-spot sampler: never propose a
    parking spot the arm cannot actually visit."""
    try:
        inverse(Pose(x, y, z, pitch))
        return True
    except Unreachable:
        return False


def to_vector(q: dict[str, float]) -> list[float]:
    """Joint dict -> the six-element vector both executors consume, clamped."""
    return [_clamp(name, q[name]) for name in JOINT_ORDER]


def workspace(z: float = 0.02, pitch: float = -math.pi / 2, step: float = 0.005):
    """Measured reach envelope at one height: (x_min, x_max, y_max, area_m2).

    The scene layout has to come from this, not from a guess. Sampling beats
    deriving it: the limits interact, and a closed form for the intersection of
    six ranges is a lot of algebra to get one rectangle.
    """
    xs, ys = [], []
    x = 0.0
    while x <= 0.50:
        y = -0.35
        while y <= 0.35:
            if reachable(x, y, z, pitch):
                xs.append(x)
                ys.append(y)
            y += step
        x += step
    if not xs:
        return None
    return (min(xs), max(xs), max(abs(v) for v in ys), len(xs) * step * step)


# --------------------------------------------------------------- self-test --

if __name__ == "__main__":
    print(f"link lengths: upper={L_UPPER*1000:.0f} lower={L_LOWER*1000:.0f} "
          f"tool={L_TOOL*1000:.0f} mm   shoulder at z={SHOULDER_Z*1000:.0f} mm")

    # FK(IK(p)) == p is the only check that matters; everything else is taste.
    tested = 0
    rejected = 0
    worst = 0.0
    for x in (0.14, 0.18, 0.22, 0.26, 0.30):
        for y in (-0.14, -0.07, 0.0, 0.07, 0.14):
            for z in (0.02, 0.06, 0.12):
                try:
                    q = inverse(Pose(x, y, z))
                except Unreachable:
                    rejected += 1
                    continue
                fx, fy, fz = forward(q)
                err = math.dist((x, y, z), (fx, fy, fz))
                worst = max(worst, err)
                assert err < 1e-6, f"IK/FK disagree at ({x},{y},{z}) by {err*1000:.3f} mm"
                tested += 1
    print(f"round-trip: {tested} solved, {rejected} out of reach, "
          f"worst error {worst*1e9:.1f} nm")
    assert tested > 30, "workspace sample too thin to trust"

    env = workspace(z=0.02)
    assert env is not None
    x_min, x_max, y_max, area = env
    print(f"reach at 20 mm above the table: x {x_min:.3f}..{x_max:.3f} m, "
          f"|y| <= {y_max:.3f} m, area {area*1e4:.0f} cm2")

    # The scaffold's TABLE = (0.10, 0.50, -0.25, 0.25) is mostly fiction.
    assert not reachable(0.48, 0.0, 0.02), "0.48 m should be well out of reach"
    assert reachable(0.22, 0.0, 0.02), "0.22 m straight ahead must be reachable"

    # Unreachable must raise, not silently return a bent pose.
    try:
        inverse(Pose(0.60, 0.0, 0.02))
    except Unreachable as exc:
        print(f"out-of-reach raises correctly: {exc}")
    else:
        raise AssertionError("0.60 m must not produce a solution")

    print("\narm_ik self-test OK")
