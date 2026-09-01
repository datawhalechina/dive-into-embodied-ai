"""Virtual Xbox gamepad for the simulate bridge, served over a named pipe.

The bridge's Joystick class opens joystick_device and reads 8-byte Linux
js_event structs without any ioctls, so a FIFO fed by this script is
indistinguishable from a real gamepad. The scripted sequence walks the deploy
FSM (Passive -> FixStand -> Velocity via LT+up and RT+A), then holds forward,
lateral, and turn command segments on the stick axes; segment boundaries are
written as JSON for the tracking analysis.

Axis/button numbers follow the bridge's XBoxJoystick mapping; axis values are
scaled by 1 << (joystick_bits - 1) with joystick_bits = 16.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import time

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
AXIS_SCALE = 1 << 15

AX_LX = 0  # vy = -lx in Velocity mode
AX_LY = 1  # vx = ly; axis is inverted (ly = -axis / scale)
AX_LT = 2  # pressed when > 0
AX_RX = 3  # wz = -rx
AX_RT = 5  # pressed when > 0
AX_DPAD_Y = 7  # up pressed when < 0
BTN_A = 0
BTN_B = 1

SEGMENTS = [
  ("forward", 1.0, 0.0, 0.0, 8.0),
  ("halt", 0.0, 0.0, 0.0, 2.0),
  ("lateral", 0.0, 0.4, 0.0, 8.0),
  ("halt", 0.0, 0.0, 0.0, 2.0),
  ("turn", 0.0, 0.0, 0.8, 6.0),
  ("halt", 0.0, 0.0, 0.0, 2.0),
]


def _emit(fd: int, kind: int, number: int, value: int) -> None:
  os.write(fd, struct.pack("<IhBB", int(time.monotonic() * 1e3) & 0xFFFFFFFF,
                           value, kind, number))


def _axis_units(fraction: float) -> int:
  return max(-AXIS_SCALE, min(AXIS_SCALE - 1, round(fraction * AXIS_SCALE)))


def _set_command(fd: int, vx: float, vy: float, wz: float) -> None:
  _emit(fd, JS_EVENT_AXIS, AX_LY, _axis_units(-vx))
  _emit(fd, JS_EVENT_AXIS, AX_LX, _axis_units(-vy))
  _emit(fd, JS_EVENT_AXIS, AX_RX, _axis_units(-wz))


def _chord(fd: int, hold: tuple[int, int], tap: tuple[int, int, int]) -> None:
  """Hold a trigger axis, tap a second control, release both."""
  axis, value = hold
  kind, number, pressed = tap
  _emit(fd, JS_EVENT_AXIS, axis, value)
  time.sleep(0.3)
  _emit(fd, kind, number, pressed)
  time.sleep(0.5)
  _emit(fd, kind, number, 0)
  time.sleep(0.2)
  _emit(fd, JS_EVENT_AXIS, axis, 0)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", required=True, help="FIFO the bridge reads")
  parser.add_argument("--segments-out", required=True, help="segment JSON path")
  parser.add_argument("--stand-wait", type=float, default=5.0,
                      help="seconds to let FixStand finish")
  args = parser.parse_args()

  # O_RDWR so the open never blocks waiting for the reader.
  fd = os.open(args.device, os.O_RDWR)

  print("neutral")
  for axis in (AX_LX, AX_LY, AX_LT, AX_RX, AX_RT, AX_DPAD_Y):
    _emit(fd, JS_EVENT_AXIS, axis, 0)
  for button in (BTN_A, BTN_B):
    _emit(fd, JS_EVENT_BUTTON, button, 0)
  time.sleep(2.0)

  print("LT+up: Passive -> FixStand")
  _chord(fd, (AX_LT, AXIS_SCALE - 1), (JS_EVENT_AXIS, AX_DPAD_Y, -AXIS_SCALE))
  time.sleep(args.stand_wait)

  print("RT+A: FixStand -> Velocity")
  _chord(fd, (AX_RT, AXIS_SCALE - 1), (JS_EVENT_BUTTON, BTN_A, 1))
  time.sleep(2.0)

  records = []
  for name, vx, vy, wz, duration in SEGMENTS:
    print(f"segment {name}: vx={vx} vy={vy} wz={wz} for {duration}s")
    _set_command(fd, vx, vy, wz)
    start = time.time_ns()
    time.sleep(duration)
    records.append({"name": name, "vx": vx, "vy": vy, "wz": wz,
                    "start_ns": start, "end_ns": time.time_ns()})
  _set_command(fd, 0.0, 0.0, 0.0)
  time.sleep(1.0)

  print("LT+B: Velocity -> Passive")
  _chord(fd, (AX_LT, AXIS_SCALE - 1), (JS_EVENT_BUTTON, BTN_B, 1))
  os.close(fd)

  with open(args.segments_out, "w") as f:
    json.dump(records, f, indent=2)
  print(f"wrote {args.segments_out}")


if __name__ == "__main__":
  main()
