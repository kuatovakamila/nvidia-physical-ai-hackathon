"""
arm_ik.py — exact kinematics for the SO-101.

Pure Python, zero dependencies, like planner.py and monitor.py. Run
`python3 arm_ik.py` for the self-test.

Every constant here was MEASURED off the calibrated `so101_antioch` twin
(scenario `zero_pose_frames`), not assumed. That distinction is the whole
history of this file: the first version modelled the arm as a planar chain
whose links lie along one axis at q=0, solved beautifully, round-tripped
FK(IK(p)) == p to nanometres, and was wrong by 380 mm against the real
robot — because at q=0 the upper arm actually points 77 deg UP, the forearm
points 3 deg forward, and the tool carries a ~25 mm lateral offset that a
planar model cannot express at all. A self-consistent model is not a correct
one; `tool_calibration` is what caught it, by comparing forward() against the
simulator's own gripper.

So kinematics here are a product of exponentials: each joint is a rotation
about a measured world-frame axis line, composed distal-to-proximal. No
assumption about how the links are laid out survives into the code.

Frame: world metres, +x forward, +z up. Joint order matches the articulation:

    Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw

Owner: P2 alongside planner.py. Consumed by both executors — the same six
numbers go to Isaac and to SO101Follower.send_action().
"""

import math

# ------------------------------------------------------- measured geometry --
# Zero-pose axis lines: a point on the axis and its direction, both in world
# coordinates, read off the twin with every joint at zero.
AXES = {
    "Rotation":    ((0.023155, 0.020730, 0.064761), (-0.004807, 0.000046, -0.999988)),
    "Pitch":       ((0.053771, 0.002377, 0.118814), (0.002392, 0.999997, 0.000034)),
    "Elbow":       ((0.084650, 0.002299, 0.230628), (0.002392, 0.999997, 0.000034)),
    "Wrist_Pitch": ((0.219649, 0.001976, 0.230940), (0.002392, 0.999997, 0.000035)),
    "Wrist_Roll":  ((0.280743, 0.019930, 0.228481), (-0.999187, 0.002385, 0.040257)),
}

# Where the object sits when gripped, at zero pose: midway between the fixed
# finger and the moving jaw. Calibrate against a real grasp if the hardware
# disagrees by a few mm — but never against a drawing.
TOOL0 = (0.336584, 0.032893, 0.234457)
# A second point up the tool axis, so the approach direction can be computed
# without hard-coding which way "down the gripper" points.
WRIST0 = (0.219649, 0.001976, 0.230940)

LIMITS = {
    "Rotation": (-1.9199, 1.9199),
    "Pitch": (-1.7453, 1.7453),
    "Elbow": (-1.7453, 1.5708),
    "Wrist_Pitch": (-1.6581, 1.6581),
    "Wrist_Roll": (-2.7925, 2.7925),
    "Jaw": (-0.1745, 1.7453),
}
JOINT_ORDER = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
# The joints IK is allowed to move. Wrist_Roll only spins the gripper about its
# own axis and Jaw only opens it; neither helps reach a point.
IK_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch")

JAW_OPEN = 1.2
JAW_CLOSED = 0.0

REST = {"Rotation": 0.0, "Pitch": 0.6, "Elbow": -1.0, "Wrist_Pitch": -0.6,
        "Wrist_Roll": 0.0, "Jaw": JAW_OPEN}

# Radians of tilt error worth one metre of position error. Small enough that
# position always wins a genuine conflict, large enough that the wrist is
# actually driven to the requested angle rather than left where it landed.
TILT_WEIGHT = 0.08

# Damped least squares finds the nearest solution to wherever it started, so a
# single seed makes reachable points look unreachable whenever the straight
# line to them crosses a fold. These cover elbow-up, elbow-down and folded-in.
SEEDS = (
    {"Rotation": 0.0, "Pitch": 0.6, "Elbow": -1.0, "Wrist_Pitch": -0.6},
    {"Rotation": 0.0, "Pitch": 1.2, "Elbow": -1.5, "Wrist_Pitch": -1.0},
    {"Rotation": 0.0, "Pitch": 0.2, "Elbow": -0.5, "Wrist_Pitch": -0.2},
    {"Rotation": 0.0, "Pitch": 1.5, "Elbow": -0.8, "Wrist_Pitch": -1.4},
)


class Unreachable(ValueError):
    """Raised when a Cartesian goal has no valid joint solution."""


class Pose:
    """A grasp target. `pitch` is kept for call-compatibility and is used as a
    soft preference for how far the tool tilts off vertical, not a hard
    constraint — the arm has four useful joints and three position equations,
    so there is exactly one degree of freedom left to spend on it."""

    __slots__ = ("x", "y", "z", "pitch")

    def __init__(self, x, y, z, pitch=-math.pi / 2):
        self.x, self.y, self.z, self.pitch = x, y, z, pitch


def _clamp(name, value):
    lo, hi = LIMITS[name]
    return max(lo, min(hi, value))


def _rotate(point, origin, axis, theta):
    """Rodrigues rotation of `point` about the line (origin, axis)."""
    if theta == 0.0:
        return point
    ux, uy, uz = axis
    n = math.sqrt(ux * ux + uy * uy + uz * uz)
    ux, uy, uz = ux / n, uy / n, uz / n
    px, py, pz = point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]
    c, s = math.cos(theta), math.sin(theta)
    dot = ux * px + uy * py + uz * pz
    cx, cy, cz = uy * pz - uz * py, uz * px - ux * pz, ux * py - uy * px
    return (
        origin[0] + px * c + cx * s + ux * dot * (1 - c),
        origin[1] + py * c + cy * s + uy * dot * (1 - c),
        origin[2] + pz * c + cz * s + uz * dot * (1 - c),
    )


def _fk_point(p0, q):
    """Product of exponentials: apply each joint's zero-pose rotation to a
    zero-pose point, distal joint first."""
    p = p0
    for name in reversed(IK_JOINTS):
        origin, axis = AXES[name]
        p = _rotate(p, origin, axis, q.get(name, 0.0))
    return p


def forward(q):
    """World position of the grasp point for a joint dict."""
    return _fk_point(TOOL0, q)


def tool_axis(q):
    """Unit vector pointing from the wrist toward the grasp point."""
    a = _fk_point(WRIST0, q)
    b = _fk_point(TOOL0, q)
    d = [b[i] - a[i] for i in range(3)]
    n = math.sqrt(sum(v * v for v in d)) or 1.0
    return [v / n for v in d]


def _jacobian(q, names):
    """Numeric position Jacobian. Finite differences rather than closed form:
    six joints is small enough that the cost is irrelevant, and a hand-derived
    Jacobian is one more place for an assumption to hide."""
    h = 1e-6
    base = forward(q)
    cols = []
    for name in names:
        qh = dict(q)
        qh[name] = q[name] + h
        p = forward(qh)
        cols.append([(p[i] - base[i]) / h for i in range(3)])
    return base, cols


def _lin_solve(a, b):
    """Solve a square system by Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def _residual(q, target, want_tilt):
    """Position error in metres, plus tilt error scaled into metres.

    Tilt is a real constraint, not a preference. Solving position alone leaves
    the wrist wherever the arm happens to land — measured: -22 to +15 degrees
    off horizontal, i.e. the gripper reaching sideways at a block it is meant
    to pick up from above. Four joints and four equations is an even trade.
    TILT_WEIGHT converts radians into comparable metres so neither term
    swamps the other.
    """
    p = forward(q)
    got = math.asin(max(-1.0, min(1.0, tool_axis(q)[2])))
    return [target[0] - p[0], target[1] - p[1], target[2] - p[2],
            TILT_WEIGHT * (want_tilt - got)]


def inverse(pose, seed=None, iters=120, tol=5e-4):
    """Try each seed in turn and return the first solution that lands."""
    starts = [seed] if seed is not None else list(SEEDS)
    last = None
    for start in starts:
        try:
            return _inverse_from(pose, start, iters, tol)
        except Unreachable as exc:
            last = exc
    raise last if last else Unreachable("no seed attempted")


def _inverse_from(pose, seed, iters, tol):
    """Damped least squares IK on position, biased toward the requested tilt.

    Returns a full joint dict. Raises Unreachable rather than returning a pose
    that silently misses — a planner that thinks it commanded a grasp and did
    not is the worst failure mode we can build.
    """
    target = (pose.x, pose.y, pose.z)
    q = dict(REST if seed is None else seed)
    for name in JOINT_ORDER:
        q.setdefault(name, REST[name])

    lam = 0.03
    h = 1e-6
    for _ in range(iters):
        err = _residual(q, target, pose.pitch)
        if math.sqrt(sum(e * e for e in err[:3])) < tol and abs(err[3]) < 0.02:
            break
        # 4x4 Jacobian of [position, weighted tilt] against the four joints.
        cols = []
        for name in IK_JOINTS:
            qh = dict(q)
            qh[name] = q[name] + h
            eh = _residual(qh, target, pose.pitch)
            cols.append([(err[i] - eh[i]) / h for i in range(4)])
        jjt = [[sum(cols[k][i] * cols[k][j] for k in range(4))
                + (lam * lam if i == j else 0.0) for j in range(4)] for i in range(4)]
        w = _lin_solve(jjt, err)
        if w is None:
            break
        for k, name in enumerate(IK_JOINTS):
            step = sum(cols[k][i] * w[i] for i in range(4))
            q[name] = _clamp(name, q[name] + max(-0.3, min(0.3, step)))

    gap = math.dist(forward(q), target)
    if gap > 0.005:
        raise Unreachable(f"closest approach {gap*1000:.0f} mm at "
                          f"({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")
    q["Jaw"] = q.get("Jaw", JAW_OPEN)
    return q


def solve(x, y, z, prefer_pitch=-math.pi / 2, seed=None):
    """Joint targets for a grasp point. Kept as the executor's entry point."""
    return inverse(Pose(x, y, z, prefer_pitch), seed=seed)


def reachable(x, y, z, pitch=-math.pi / 2):
    try:
        solve(x, y, z, pitch)
        return True
    except Unreachable:
        return False


def to_vector(q):
    return [_clamp(name, q[name]) for name in JOINT_ORDER]


def workspace(z=0.02, pitch=-math.pi / 2, step=0.02):
    xs, ys = [], []
    x = 0.0
    while x <= 0.45:
        y = -0.30
        while y <= 0.30:
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
    zero = {n: 0.0 for n in JOINT_ORDER}
    p = forward(zero)
    print(f"zero pose grasp point: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")
    assert math.dist(p, TOOL0) < 1e-9, "FK must reproduce the measured zero pose"

    # IK/FK round trip. The bar is the tolerance IK solves to, not machine
    # epsilon: this is an iterative solver, and pretending otherwise is how the
    # last version looked perfect while being wrong.
    solved = unreachable = 0
    worst = 0.0
    for x in (0.16, 0.20, 0.24, 0.28):
        for y in (-0.12, -0.04, 0.04, 0.12):
            for z in (0.03, 0.08, 0.15):
                try:
                    q = solve(x, y, z)
                except Unreachable:
                    unreachable += 1
                    continue
                err = math.dist(forward(q), (x, y, z))
                worst = max(worst, err)
                assert err <= 0.005, f"IK returned a {err*1000:.1f} mm miss"
                solved += 1
    print(f"round-trip: {solved} solved, {unreachable} out of reach, "
          f"worst miss {worst*1000:.2f} mm")
    assert solved > 20, "workspace sample too thin to trust"

    env = workspace(z=0.03)
    assert env is not None
    print(f"reach at 30 mm: x {env[0]:.2f}..{env[1]:.2f} m, |y| <= {env[2]:.2f} m")

    try:
        solve(0.70, 0.0, 0.02)
    except Unreachable as exc:
        print(f"out-of-reach raises correctly: {exc}")
    else:
        raise AssertionError("0.70 m must not produce a solution")

    print("\narm_ik self-test OK")
