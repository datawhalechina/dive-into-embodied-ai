"""Render per-seed reward curves with per-backend min/max bands as one SVG.

Same conventions as make_reward_chart.py: inline SVG, colors from CSS custom
properties, backends assigned palette slots in argument order. Each backend
argument bundles its seeds; the band is the pointwise min/max across them.

    uv run python benchmarks/make_seed_band_chart.py --out bands.svg \
        "R9700 (ROCm)=r0/metrics.jsonl,r1/metrics.jsonl,r2/metrics.jsonl" \
        "A40 (CUDA)=a0/metrics.jsonl,a1/metrics.jsonl,a2/metrics.jsonl"
"""

import argparse
import itertools
import json

W, H = 720, 380
PAD_L, PAD_R, PAD_T, PAD_B = 62, 150, 26, 52
LABEL_GAP = 32

INK = "var(--text-primary)"
MUTED = "var(--text-secondary)"
GRID = "var(--grid)"

REWARD_KEY = "eval/episode_reward"


def _color(i: int) -> str:
  return f"var(--series-{i + 1})"


def _spread(ys: list[float]) -> list[float]:
  """Nudge end labels apart so they never overlap, keeping their order."""
  order = sorted(range(len(ys)), key=lambda i: ys[i])
  out = list(ys)
  for prev, cur in itertools.pairwise(order):
    if out[cur] - out[prev] < LABEL_GAP:
      out[cur] = out[prev] + LABEL_GAP
  return out


def band_chart(backends, title, subtitle) -> str:
  xmax = max(r["step"] for _, seed_rows in backends for rows in seed_rows for r in rows)
  ys = [r[REWARD_KEY] for _, seed_rows in backends for rows in seed_rows for r in rows]
  ymin, ymax = min(0.0, min(ys)), max(ys) * 1.15
  span_x = W - PAD_R - PAD_L
  span_y = H - PAD_B - PAD_T - 24

  def x(v):
    return PAD_L + v / xmax * span_x

  def y(v):
    return (H - PAD_B) - (v - ymin) / (ymax - ymin) * span_y

  o = [
    (
      f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
      f'aria-label="{title}" xmlns="http://www.w3.org/2000/svg" '
      'style="font-family:inherit;overflow:visible">'
    ),
    f'<text x="0" y="14" fill="{INK}" font-size="14" font-weight="600">{title}</text>',
    f'<text x="0" y="32" fill="{MUTED}" font-size="11.5">{subtitle}</text>',
  ]
  step = 10 if ymax > 30 else 5
  ticks = [v for v in range(int(ymin), int(ymax) + 1) if v % step == 0]
  for v in ticks:
    yy = y(v)
    o.append(
      f'<line x1="{PAD_L}" y1="{yy:.1f}" x2="{W - PAD_R}" y2="{yy:.1f}" '
      f'stroke="{GRID}" stroke-width="1"/>'
    )
    o.append(
      f'<text x="{PAD_L - 8}" y="{yy + 3.5:.1f}" fill="{MUTED}" font-size="10.5" '
      f'text-anchor="end">{v}</text>'
    )
  for frac in (0, 0.25, 0.5, 0.75, 1.0):
    v = frac * xmax
    o.append(
      f'<text x="{x(v):.1f}" y="{H - PAD_B + 18}" fill="{MUTED}" font-size="10.5" '
      f'text-anchor="middle">{v / 1e6:.0f}M</text>'
    )
  o.append(
    f'<text x="{(PAD_L + W - PAD_R) / 2:.0f}" y="{H - 12}" fill="{MUTED}" '
    f'font-size="11" text-anchor="middle">训练步数</text>'
  )
  o.append(
    f'<text transform="translate(14,{(PAD_T + H - PAD_B) / 2:.0f}) rotate(-90)" '
    f'fill="{MUTED}" font-size="11" text-anchor="middle">单集回报</text>'
  )

  finals = []
  for _, seed_rows in backends:
    finals.append(sum(rows[-1][REWARD_KEY] for rows in seed_rows) / len(seed_rows))
  ends = _spread([y(v) for v in finals])

  for i, (label, seed_rows) in enumerate(backends):
    color = _color(i)
    steps = [r["step"] for r in seed_rows[0]]
    lo = [min(rows[j][REWARD_KEY] for rows in seed_rows) for j in range(len(steps))]
    hi = [max(rows[j][REWARD_KEY] for rows in seed_rows) for j in range(len(steps))]
    upper = " ".join(
      f"{'M' if j == 0 else 'L'}{x(s):.1f},{y(v):.1f}"
      for j, (s, v) in enumerate(zip(steps, hi))
    )
    lower = " ".join(
      f"L{x(s):.1f},{y(v):.1f}" for s, v in zip(reversed(steps), reversed(lo))
    )
    o.append(f'<path d="{upper} {lower} Z" fill="{color}" opacity="0.16"/>')
    for rows in seed_rows:
      pts = [(x(r["step"]), y(r[REWARD_KEY])) for r in rows]
      path = " ".join(
        f"{'M' if j == 0 else 'L'}{px:.1f},{py:.1f}" for j, (px, py) in enumerate(pts)
      )
      o.append(
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.4" '
        'opacity="0.85" stroke-linejoin="round" stroke-linecap="round"/>'
      )
    ex, ey = x(steps[-1]), ends[i]
    o.append(f'<circle cx="{ex + 14:.1f}" cy="{ey:.1f}" r="4" fill="{color}"/>')
    o.append(
      f'<text x="{ex + 23:.1f}" y="{ey + 4:.1f}" fill="{INK}" font-size="11.5" '
      f'font-weight="600">{label}</text>'
    )
    o.append(
      f'<text x="{ex + 23:.1f}" y="{ey + 17:.1f}" fill="{MUTED}" font-size="10.5">'
      f"{len(seed_rows)} 个种子，末值均 {finals[i]:.1f}</text>"
    )
  o.append("</svg>")
  return "\n".join(o)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("backends", nargs="+", help="LABEL=metrics.jsonl,metrics.jsonl,...")
  p.add_argument("--out", required=True)
  p.add_argument("--title", default="训练回报：逐种子曲线与后端包络带")
  p.add_argument("--subtitle", default="同一提交、同一配置、种子 0/1/2，只是换了显卡。")
  a = p.parse_args()

  loaded = []
  for spec in a.backends:
    label, _, paths = spec.partition("=")
    seed_rows = []
    for path in paths.split(","):
      with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
      seed_rows.append([r for r in rows if REWARD_KEY in r])
    loaded.append((label, seed_rows))

  svg = band_chart(loaded, a.title, a.subtitle)
  with open(a.out, "w") as f:
    f.write(svg)
  print(f"wrote {a.out}")


if __name__ == "__main__":
  main()
