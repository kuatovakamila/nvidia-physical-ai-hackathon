"""
bus_monitor.py — live view of which servos are answering.

    ~/lerobot-env/bin/python bus_monitor.py

Reseat a connector and watch the IDs come back. Exists because every other
tool here reads all six motors at once and simply fails when the bus is
incomplete, which tells you nothing about WHERE it is incomplete.

Chain order on the SO-101, board outward:
    1 shoulder_pan  2 shoulder_lift  3 elbow_flex
    4 wrist_flex    5 wrist_roll     6 gripper
The servos share one half-duplex bus daisy-chained through each joint, so a
loose connector silences that joint AND everything past it. The lowest missing
ID is where to look first.
"""

import glob
import sys
import time

NAMES = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex",
         4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}


def main(seconds=120.0):
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    port = sorted(glob.glob("/dev/tty.usbmodem*"))
    if not port:
        print("no usbmodem port — the board itself is unplugged")
        return 1
    bus = FeetechMotorsBus(port=port[0],
                           motors={"p": Motor(1, "sts3215", MotorNormMode.DEGREES)})
    bus._connect(handshake=False)
    bus.set_baudrate(1_000_000)
    print(f"port: {port[0]}   watching — Ctrl-C to stop\n")
    end = time.time() + seconds
    prev = None
    try:
        while time.time() < end:
            alive = []
            for mid in NAMES:
                try:
                    if bus.ping(mid, num_retry=1) is not None:
                        alive.append(mid)
                except Exception:
                    pass
            row = "  ".join(f"{'OK' if i in alive else '--':>2} {NAMES[i][:11]}"
                            for i in NAMES)
            missing = [NAMES[i] for i in NAMES if i not in alive]
            tag = "ALL SIX" if not missing else f"missing: {', '.join(missing)}"
            if alive != prev:
                print(f"\n{row}\n   {len(alive)}/6 — {tag}", flush=True)
                prev = alive
            else:
                print(".", end="", flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            bus.port_handler.closePort()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main(float(sys.argv[1]) if len(sys.argv) > 1 else 120.0))
