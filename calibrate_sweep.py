"""
calibrate_sweep.py — time-boxed calibration recorder for the SO-101.

lerobot's own record_ranges_of_motion() reads RAW encoder counts and tracks a
running min/max. When a joint's raw position sits near the 0/4095 boundary its
travel crosses it, and the recorded range comes back as exactly 0..4095 —
measured on this arm: shoulder_lift sits at raw 322 and elbow_flex at 904, so
both wrapped and both wrote garbage, twice, no matter how carefully the joints
were swept. Nothing about the operator's technique can fix that.

This records the same thing with two differences:

  * Positions are UNWRAPPED: a jump of more than half a turn between samples is
    read as crossing the boundary, not as a real move. Range then means travel,
    which is what calibration is asking about.
  * It is time-boxed rather than ENTER-gated, so it can be started for you and
    finishes on its own.

    python3 calibrate_sweep.py --seconds 45
    python3 calibrate_sweep.py --seconds 45 --id interx_follower --write

Torque is disabled while recording, so THE ARM WILL GO LIMP AND FALL if you let
go. Support its weight — and note that a collapse under gravity is itself a way
to record a bogus range.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import time

MOTORS = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper",
}
FULL_TURN = "wrist_roll"      # continuous joint: range is declared, not measured
RESOLUTION = 4096
HALF = RESOLUTION // 2


def _bus(port):
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    bus = FeetechMotorsBus(
        port=port,
        motors={n: Motor(i, "sts3215", MotorNormMode.DEGREES) for i, n in MOTORS.items()},
    )
    bus._connect(handshake=False)
    bus.set_baudrate(1_000_000)
    return bus


def record(port, seconds, hz=120.0):
    bus = _bus(port)
    try:
        bus.disable_torque()
    except Exception as exc:
        print(f"  (could not disable torque: {exc})")

    first = bus.sync_read("Present_Position", normalize=False)
    prev = dict(first)
    # Unwrapped position, in counts, relative to nothing in particular — only
    # differences matter.
    acc = {n: float(v) for n, v in first.items()}
    lo = dict(acc)
    hi = dict(acc)
    # A sample-to-sample step approaching half a turn is ambiguous: it could be
    # the counter wrapping or a genuinely fast joint, and the unwrapper has to
    # guess. Guessing wrong corrupts the accumulator for the rest of the run —
    # measured on wrist_flex, which is light enough to flick faster than the
    # loop samples. Count them so a bad run announces itself.
    suspicious = {n: 0 for n in first}

    print(f"recording {seconds:.0f}s — move every joint through its full travel now")
    end = time.time() + seconds
    last_print = 0.0
    while time.time() < end:
        try:
            raw = bus.sync_read("Present_Position", normalize=False)
        except Exception:
            continue
        for n, v in raw.items():
            d = v - prev[n]
            # A step bigger than half a turn is the counter wrapping, not the
            # joint teleporting.
            if abs(abs(d) - HALF) < HALF * 0.25:
                suspicious[n] += 1
            if d > HALF:
                d -= RESOLUTION
            elif d < -HALF:
                d += RESOLUTION
            acc[n] += d
            lo[n] = min(lo[n], acc[n])
            hi[n] = max(hi[n], acc[n])
            prev[n] = v
        now = time.time()
        if now - last_print > 3.0:
            last_print = now
            spans = "  ".join(f"{n[:9]}:{(hi[n]-lo[n])*360/RESOLUTION:5.0f}°" for n in MOTORS.values())
            print(f"  {end-now:4.0f}s left   {spans}")
        time.sleep(1.0 / hz)

    bus.port_handler.closePort()
    flagged = {n: c for n, c in suspicious.items() if c}
    if flagged:
        print(f"\n  WARNING ambiguous jumps (move that joint slower): {flagged}")
    return acc, lo, hi


PROGRESS = pathlib.Path(".calib_progress.json")


def merge(lo, hi):
    """Keep the best sweep of each joint across runs.

    One 45-second window is not enough to sweep six joints well, and missing a
    different joint each time is a loop with no exit — measured twice. Each
    joint's (lo, hi) pair is carried whole from whichever run swept it
    furthest: the pairs are in raw-equivalent counts, so they stay valid
    individually even though different joints come from different runs.
    """
    best = {}
    if PROGRESS.exists():
        best = json.loads(PROGRESS.read_text())
    for name in MOTORS.values():
        span = hi[name] - lo[name]
        prev = best.get(name)
        if prev is None or span > (prev["hi"] - prev["lo"]):
            best[name] = {"lo": lo[name], "hi": hi[name]}
    PROGRESS.write_text(json.dumps(best, indent=2))
    return ({n: v["lo"] for n, v in best.items()},
            {n: v["hi"] for n, v in best.items()})


def build(lo, hi):
    """Centre each joint's travel on the encoder's mid-scale.

    Putting the midpoint of measured travel at 2048 is what keeps the joint
    away from the wrap boundary in normal use — which is the whole failure this
    script exists to route around.
    """
    cal = {}
    for mid, name in MOTORS.items():
        span = hi[name] - lo[name]
        if name == FULL_TURN:
            cal[name] = {"id": mid, "drive_mode": 0, "homing_offset": 0,
                         "range_min": 0, "range_max": RESOLUTION - 1}
            continue
        # The accumulator is unwrapped, so its centre can land outside one
        # turn — measured: the elbow's came out at -876, giving a homing offset
        # of 2924 that the servo cannot store at all (Homing_Offset is sign
        # plus 11 bits, so +-2047). Fold the centre back into one turn, and
        # build the range symmetrically about mid-scale rather than by adding
        # the offset to raw endpoints.
        centre = ((hi[name] + lo[name]) / 2.0) % RESOLUTION
        homing = int(round(HALF - centre))
        if homing > HALF - 1:
            homing -= RESOLUTION
        elif homing < -(HALF - 1):
            homing += RESOLUTION
        rmin = int(round(HALF - span / 2.0))
        rmax = int(round(HALF + span / 2.0))
        cal[name] = {"id": mid, "drive_mode": 0, "homing_offset": homing,
                     "range_min": max(0, rmin), "range_max": min(RESOLUTION - 1, rmax)}
    return cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--id", default="interx_follower")
    ap.add_argument("--port", default=None)
    ap.add_argument("--write", action="store_true", help="save the calibration file")
    ap.add_argument("--identify", action="store_true",
                    help="name whichever joint you are moving; records nothing")
    a = ap.parse_args()

    port = a.port or sorted(glob.glob("/dev/tty.usbmodem*"))[0]
    print(f"port: {port}\n")
    if a.identify:
        cmd_identify(port, a.seconds)
        return 0
    acc, lo, hi = record(port, a.seconds)
    lo, hi = merge(lo, hi)

    print(f"\n{'joint':<16}{'travel':>9}   verdict   (best across runs)")
    cal = build(lo, hi)
    bad = []
    for name in MOTORS.values():
        deg = (hi[name] - lo[name]) * 360.0 / RESOLUTION
        if name == FULL_TURN:
            verdict = "continuous, declared"
        elif deg < 25:
            verdict = "NOT SWEPT"
            bad.append(name)
        else:
            verdict = "ok"
        print(f"{name:<16}{deg:>8.1f}°   {verdict}")

    if bad:
        print(f"\nnot usable yet: {bad} — re-run and sweep those")
        return 1
    if not a.write:
        print("\nlooks good. Re-run with --write to save it.")
        return 0

    path = (pathlib.Path.home() / ".cache/huggingface/lerobot/calibration/robots/"
            f"so_follower/{a.id}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal, indent=2))
    print(f"\nwritten: {path}")
    return 0


def cmd_identify(port, seconds=40.0):
    """Say which joint is moving, live.

    Motor names are opaque and two of them sit next to each other doing
    similar-looking things: wrist_flex nods the gripper, wrist_roll spins it.
    Naming them in prose invites exactly the mix-up that costs a whole sweep,
    so let the arm answer instead — wiggle one joint and read its name.
    """
    bus = _bus(port)
    try:
        bus.disable_torque()
    except Exception:
        pass
    prev = bus.sync_read("Present_Position", normalize=False)
    print("wiggle ONE joint at a time — its name appears below\n")
    end = time.time() + seconds
    while time.time() < end:
        time.sleep(0.15)
        try:
            raw = bus.sync_read("Present_Position", normalize=False)
        except Exception:
            continue
        moved = []
        for n, v in raw.items():
            d = v - prev[n]
            if d > HALF:
                d -= RESOLUTION
            elif d < -HALF:
                d += RESOLUTION
            if abs(d) > 8:
                moved.append((abs(d), n))
        prev = raw
        if moved:
            moved.sort(reverse=True)
            top = moved[0][1]
            bar = "#" * min(40, int(moved[0][0] / 4))
            print(f"  {top:<16} {bar}", flush=True)
    bus.port_handler.closePort()

if __name__ == "__main__":
    raise SystemExit(main())
