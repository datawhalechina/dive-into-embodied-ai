"""Render throughput results as standalone SVG charts.

Emits inline SVG that inherits its colors from CSS custom properties, so the
charts follow the page's light/dark theme. Series are assigned palette slots in
the order given, never cycled.

    uv run python benchmarks/make_charts.py --out-dir charts/ \
        "R9700 (ROCm)=rocm.json" "A40=a40.json" "CPU=cpu.json"

The first series is the baseline the speedup chart compares against.
"""

import argparse
import itertools
import json
import math

W, H = 720, 380
PAD_L, PAD_R, PAD_T, PAD_B = 62, 150, 26, 52
LABEL_GAP = 32  # end labels carry a name + a value, so they need real room

INK = "var(--text-primary)"
MUTED = "var(--text-secondary)"
GRID = "var(--grid)"


def _color(i: int) -> str:
  return f"var(--series-{i + 1})"


def _fmt(n: float) -> str:
  return f"{n / 1000:.0f}k" if n >= 1000 else f"{n:.0f}"


def _log_scale(vmin, vmax, lo, hi):
  a, b = math.log10(vmin), math.log10(vmax)
  return lambda v: lo + (math.log10(v) - a) / (b - a) * (hi - lo)


def _header(title: str, subtitle: str, height: int = H) -> list[str]:
  return [
    (
      f'<svg viewBox="0 0 {W} {height}" width="100%" role="img" '
      f'aria-label="{title}" xmlns="http://www.w3.org/2000/svg" '
      'style="font-family:inherit;overflow:visible">'
    ),
    f'<text x="0" y="14" fill="{INK}" font-size="14" font-weight="600">{title}</text>',
    f'<text x="0" y="32" fill="{MUTED}" font-size="11.5">{subtitle}</text>',
  ]


def _spread(ys: list[float]) -> list[float]:
  """Nudge end labels apart so they never overlap, keeping their order."""
  order = sorted(range(len(ys)), key=lambda i: ys[i])
  out = list(ys)
  for prev, cur in itertools.pairwise(order):
    if out[cur] - out[prev] < LABEL_GAP:
      out[cur] = out[prev] + LABEL_GAP
  return out


def line_chart(series, key, title, subtitle, ylabel, fmt=_fmt, log_y=True) -> str:
  """Log y suits throughput (orders of magnitude); linear suits compile time."""
  # Union, not series[0]: a backend may stop short of the others (the R9700
  # cannot host 8192 envs), and a rung outside the scale lands in the margin.
  xs = sorted({r["num_envs"] for _, rows in series for r in rows})
  ys = [r[key] for _, rows in series for r in rows]
  x = _log_scale(min(xs), max(xs), PAD_L, W - PAD_R)
  ymin = min(v for v in ys if v > 0)

  if log_y:
    y = _log_scale(ymin * 0.75, max(ys) * 1.35, H - PAD_B, PAD_T + 24)
    d0 = math.floor(math.log10(ymin * 0.75))
    d1 = math.ceil(math.log10(max(ys) * 1.35))
    ticks = [10.0**e for e in range(d0, d1 + 1)]
  else:
    top = max(ys) * 1.15
    span = H - PAD_B - PAD_T - 24

    def y(v):
      return (H - PAD_B) - v / top * span

    step = 10 ** math.floor(math.log10(top))
    if top / step < 3:
      step /= 2
    ticks = [i * step for i in range(int(top / step) + 1)]

  o = _header(title, subtitle)
  for v in ticks:
    yy = y(v)
    if not (PAD_T + 20 <= yy <= H - PAD_B):
      continue
    o.append(
      f'<line x1="{PAD_L}" y1="{yy:.1f}" x2="{W - PAD_R}" y2="{yy:.1f}" '
      f'stroke="{GRID}" stroke-width="1"/>'
    )
    o.append(
      f'<text x="{PAD_L - 8}" y="{yy + 3.5:.1f}" fill="{MUTED}" font-size="10.5" '
      f'text-anchor="end">{fmt(v)}</text>'
    )
  for xv in xs:
    o.append(
      f'<text x="{x(xv):.1f}" y="{H - PAD_B + 18}" fill="{MUTED}" font-size="10.5" '
      f'text-anchor="middle">{xv}</text>'
    )
  o.append(
    f'<text x="{(PAD_L + W - PAD_R) / 2:.0f}" y="{H - 12}" fill="{MUTED}" '
    f'font-size="11" text-anchor="middle">并行环境数</text>'
  )
  o.append(
    f'<text transform="translate(14,{(PAD_T + H - PAD_B) / 2:.0f}) rotate(-90)" '
    f'fill="{MUTED}" font-size="11" text-anchor="middle">{ylabel}</text>'
  )

  ends = _spread([y(rows[-1][key]) for _, rows in series])
  for i, (label, rows) in enumerate(series):
    color = _color(i)
    pts = [(x(r["num_envs"]), y(r[key])) for r in rows]
    path = " ".join(
      f"{'M' if j == 0 else 'L'}{px:.1f},{py:.1f}" for j, (px, py) in enumerate(pts)
    )
    o.append(
      f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" '
      'stroke-linejoin="round" stroke-linecap="round"/>'
    )
    for px, py in pts:
      o.append(
        f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}" '
        f'stroke="var(--surface-1)" stroke-width="2"/>'
      )
    ex, ey = pts[-1][0], ends[i]
    value = f"{rows[-1][key]:,.1f}" if key == "compile_s" else fmt(rows[-1][key])
    o.append(f'<circle cx="{ex + 14:.1f}" cy="{ey:.1f}" r="4" fill="{color}"/>')
    o.append(
      f'<text x="{ex + 23:.1f}" y="{ey + 4:.1f}" fill="{INK}" font-size="11.5" '
      f'font-weight="600">{label}</text>'
    )
    o.append(
      f'<text x="{ex + 23:.1f}" y="{ey + 17:.1f}" fill="{MUTED}" font-size="10.5">'
      f"{value}</text>"
    )
  o.append("</svg>")
  return "\n".join(o)


def speedup_chart(baseline, target, title, subtitle) -> str:
  """Horizontal bars: how many times faster `target` is than `baseline`."""
  # Pair by env count, not by position: the ladders can differ in length, and
  # zipping them would compare different rungs to each other. A rung only one
  # card reached has no ratio and is left out.
  base_by_envs = {r["num_envs"]: r for r in baseline[1]}
  rows = [
    (t["num_envs"], t["steps_per_s"] / base_by_envs[t["num_envs"]]["steps_per_s"])
    for t in target[1]
    if t["num_envs"] in base_by_envs
  ]
  h, gap = 26, 12
  height = PAD_T + 40 + len(rows) * (h + gap) + 34
  o = _header(title, subtitle, height)
  x0 = PAD_L + 30
  xmax = W - PAD_R + 40
  vmax = max(v for _, v in rows) * 1.08

  def sx(v):
    return x0 + v / vmax * (xmax - x0)

  one = sx(1.0)
  o.append(
    f'<line x1="{one:.1f}" y1="{PAD_T + 34}" x2="{one:.1f}" '
    f'y2="{height - 30}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="3 3"/>'
  )
  o.append(
    f'<text x="{one:.1f}" y="{height - 16}" fill="{MUTED}" font-size="10.5" '
    f'text-anchor="middle">1× 打平</text>'
  )
  for i, (envs, v) in enumerate(rows):
    yy = PAD_T + 44 + i * (h + gap)
    o.append(
      f'<text x="{x0 - 12}" y="{yy + h / 2 + 4:.0f}" fill="{MUTED}" '
      f'font-size="11" text-anchor="end">{envs}</text>'
    )
    o.append(
      f'<rect x="{x0}" y="{yy}" width="{max(sx(v) - x0, 2):.1f}" height="{h}" '
      f'rx="4" fill="{_color(0)}"/>'
    )
    o.append(
      f'<text x="{sx(v) + 9:.1f}" y="{yy + h / 2 + 4:.0f}" fill="{INK}" '
      f'font-size="11.5" font-weight="600">{v:.2f}×</text>'
    )
  o.append(
    f'<text x="{x0 - 12}" y="{PAD_T + 34}" fill="{MUTED}" font-size="10.5" '
    f'text-anchor="end">并行环境数</text>'
  )
  o.append("</svg>")
  return "\n".join(o)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("series", nargs="+", help="LABEL=path.json, baseline first")
  p.add_argument("--out-dir", default=".")
  a = p.parse_args()

  loaded = []
  for spec in a.series:
    label, _, path = spec.partition("=")
    with open(path) as f:
      loaded.append((label, json.load(f)["results"]))

  gpus = [s for s in loaded if not s[0].upper().startswith("CPU")]
  charts = {
    "throughput.svg": line_chart(
      loaded,
      "steps_per_s",
      "吞吐量：每秒能算多少个物理步",
      "越高越好。两轴都是对数刻度，所以等距离代表等倍数。",
      "物理步 / 秒",
    ),
    "speedup.svg": speedup_chart(
      gpus[1],
      gpus[0],
      f"{gpus[0][0]} 相对 {gpus[1][0]} 的加速比",
      "同一份代码、同一个模型，只是换了显卡。1× 表示两者打平。",
    ),
    "compile.svg": line_chart(
      gpus,
      "compile_s",
      "首次编译耗时",
      "JAX 在第一次调用时编译内核，这部分开销只付一次，之后的每一步都不再编译。",
      "秒",
      fmt=lambda v: f"{v:.0f}",
      log_y=False,
    ),
  }
  for name, svg in charts.items():
    with open(f"{a.out_dir}/{name}", "w") as f:
      f.write(svg)
    print(f"wrote {a.out_dir}/{name}")


if __name__ == "__main__":
  main()
