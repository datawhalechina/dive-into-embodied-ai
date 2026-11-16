"""VLA skill -> Petoi OpenCat firmware command bridge (real-robot handover prep).

The sim VLA outputs discrete skills (forward/steer_L/steer_R/stop). On the real
Quaddle the OpenCat firmware provides good gaits, so we map skills 1:1 to OpenCat
command tokens instead of re-running the sim CPG. No hardware required to import;
the serial sender is a stub until a real robot is attached (Dec delivery).

Mapping (per 《细化方案》: forward<->kwkF, spin/steer<->ktrL):
    forward   -> kwkF   (walk forward)
    steer_L   -> ktrL   (turn left)
    steer_R   -> ktrR   (turn right)
    stop      -> kbalance (hold stance)
"""
from __future__ import annotations

SKILL_TO_OPENCAT = {
    "forward": "kwkF",
    "steer_L": "ktrL",
    "steer_R": "ktrR",
    "stop": "kbalance",
}


def skill_to_opencat(skill: str) -> str:
    return SKILL_TO_OPENCAT[skill]


def quantize_to_skill(action2) -> str:
    """Same quantizer as eval_rollout_v5 (2D [vx_dir, wz_dir])."""
    vx, wz = float(action2[0]), float(action2[1])
    if vx < 0:
        vx = 0.0
    if abs(wz) > 0.5:
        return "steer_L" if wz > 0 else "steer_R"
    if max(abs(vx), abs(wz)) < 0.12:
        return "stop"
    if abs(wz) >= abs(vx) and abs(wz) > 0.15:
        return "steer_L" if wz > 0 else "steer_R"
    return "forward" if vx > 0 else "stop"


class OpenCatBridge:
    """Send OpenCat tokens over serial. Stub (dry-run) when no port/hardware."""

    def __init__(self, port: str | None = None, baud: int = 115200, dry_run: bool = True):
        self.dry_run = dry_run or port is None
        self._ser = None
        if not self.dry_run:
            import serial  # pyserial, only needed with real hardware
            self._ser = serial.Serial(port, baud, timeout=1)

    def send_skill(self, skill: str) -> str:
        token = skill_to_opencat(skill)
        if not self.dry_run and self._ser is not None:
            self._ser.write((token + "\n").encode())
        return token

    def send_action(self, action2) -> str:
        return self.send_skill(quantize_to_skill(action2))

    def close(self):
        if self._ser is not None:
            self._ser.close()


if __name__ == "__main__":
    b = OpenCatBridge(dry_run=True)
    for a in [(1, 0), (0, 1), (0, -1), (0, 0)]:
        print(a, "->", b.send_action(a))