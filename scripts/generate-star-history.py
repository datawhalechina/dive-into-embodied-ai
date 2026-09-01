#!/usr/bin/env python3
"""Generate a repository-owned SVG chart from GitHub stargazer timestamps."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


API_VERSION = "2022-11-28"
REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="GitHub repository in owner/name form (defaults to GITHUB_REPOSITORY)",
    )
    parser.add_argument(
        "--output",
        default="assets/star-history.svg",
        help="Output SVG path",
    )
    return parser.parse_args()


def github_request(path: str, token: str) -> tuple[object, dict[str, str]]:
    request = Request(
        f"https://api.github.com{path}",
        headers={
            "Accept": "application/vnd.github.star+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "dive-into-embodied-ai-star-history",
            "X-GitHub-Api-Version": API_VERSION,
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response), dict(response.headers.items())
    except HTTPError as error:
        body = error.read(500).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"GitHub API returned HTTP {error.code} for {path}: {body}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"Could not reach the GitHub API: {error.reason}") from error


def fetch_star_dates(repo: str, token: str) -> tuple[list[date], int]:
    encoded_repo = quote(repo, safe="/")
    metadata, _ = github_request(f"/repos/{encoded_repo}", token)
    if not isinstance(metadata, dict):
        raise RuntimeError("GitHub returned unexpected repository metadata")

    expected_count = int(metadata.get("stargazers_count", 0))
    star_dates: list[date] = []
    page = 1

    while True:
        payload, _ = github_request(
            f"/repos/{encoded_repo}/stargazers?per_page=100&page={page}", token
        )
        if not isinstance(payload, list):
            raise RuntimeError("GitHub returned unexpected stargazer data")
        for record in payload:
            if not isinstance(record, dict) or not record.get("starred_at"):
                raise RuntimeError(
                    "Stargazer timestamps are unavailable; check the token's repository permissions"
                )
            timestamp = datetime.fromisoformat(record["starred_at"].replace("Z", "+00:00"))
            star_dates.append(timestamp.astimezone(timezone.utc).date())
        if len(payload) < 100:
            break
        page += 1

    star_dates.sort()
    if len(star_dates) != expected_count:
        print(
            f"Warning: repository reports {expected_count} stars but the API returned "
            f"{len(star_dates)} timestamped records.",
            file=sys.stderr,
        )
    return star_dates, expected_count


def nice_ceiling(value: int) -> int:
    if value <= 5:
        return max(1, value)
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    for multiplier in (1, 2, 5, 10):
        if normalized <= multiplier:
            return multiplier * magnitude
    return 10 * magnitude


def evenly_spaced_dates(start: date, end: date, count: int = 4) -> list[date]:
    span = (end - start).days
    values = [start + timedelta(days=round(span * index / (count - 1))) for index in range(count)]
    return list(OrderedDict.fromkeys(values))


def render_svg(repo: str, star_dates: list[date], reported_count: int) -> str:
    width, height = 960, 560
    plot_left, plot_right = 88, 920
    plot_top, plot_bottom = 112, 456

    if star_dates:
        first_star = star_dates[0]
        last_star = star_dates[-1]
        start_date = first_star - timedelta(days=1)
        end_date = max(last_star, start_date + timedelta(days=1))
        cumulative: OrderedDict[date, int] = OrderedDict()
        cumulative[start_date] = 0
        for index, starred_on in enumerate(star_dates, start=1):
            cumulative[starred_on] = index
        current_count = len(star_dates)
        updated_label = last_star.isoformat()
    else:
        start_date = date.today() - timedelta(days=1)
        end_date = date.today()
        cumulative = OrderedDict(((start_date, 0), (end_date, 0)))
        current_count = reported_count
        updated_label = "waiting for the first star"

    if reported_count > current_count:
        cumulative[end_date] = reported_count
        current_count = reported_count

    x_span = max(1, (end_date - start_date).days)
    y_max = nice_ceiling(max(1, current_count))

    def x_position(value: date) -> float:
        return plot_left + ((value - start_date).days / x_span) * (plot_right - plot_left)

    def y_position(value: int) -> float:
        return plot_bottom - (value / y_max) * (plot_bottom - plot_top)

    points = [(x_position(day), y_position(count)) for day, count in cumulative.items()]
    line_path = " ".join(
        ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
        for index, (x, y) in enumerate(points)
    )
    area_path = (
        f"M {points[0][0]:.2f} {plot_bottom} "
        + " ".join(f"L {x:.2f} {y:.2f}" for x, y in points)
        + f" L {points[-1][0]:.2f} {plot_bottom} Z"
    )

    if y_max <= 5:
        y_ticks = list(range(0, y_max + 1))
    else:
        y_ticks = sorted({round(y_max * index / 4) for index in range(5)})
    x_ticks = evenly_spaced_dates(start_date, end_date)

    escaped_repo = html.escape(repo)
    subtitle = f"{current_count:,} stars · updated through {updated_label}"
    svg: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        f"  <title id=\"title\">{escaped_repo} Star History</title>",
        f"  <desc id=\"description\">Cumulative GitHub stars for {escaped_repo}: {current_count:,}</desc>",
        "  <defs>",
        '    <linearGradient id="area" x1="0" y1="0" x2="0" y2="1">',
        '      <stop offset="0%" stop-color="#58a6ff" stop-opacity="0.42" />',
        '      <stop offset="100%" stop-color="#58a6ff" stop-opacity="0.03" />',
        "    </linearGradient>",
        '    <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">',
        '      <feGaussianBlur stdDeviation="3" result="blur" />',
        '      <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>',
        "    </filter>",
        "  </defs>",
        f'  <rect width="{width}" height="{height}" rx="18" fill="#0d1117" />',
        f'  <rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="17.5" fill="none" stroke="#30363d" />',
        f'  <text x="48" y="50" fill="#f0f6fc" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif" font-size="25" font-weight="700">{escaped_repo} Star History</text>',
        f'  <text x="48" y="79" fill="#8b949e" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif" font-size="15">{html.escape(subtitle)}</text>',
    ]

    for tick in y_ticks:
        y = y_position(tick)
        svg.extend(
            [
                f'  <line x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" stroke="#21262d" stroke-width="1" />',
                f'  <text x="{plot_left - 14}" y="{y + 5:.2f}" text-anchor="end" fill="#8b949e" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif" font-size="13">{tick}</text>',
            ]
        )

    for index, tick in enumerate(x_ticks):
        x = x_position(tick)
        anchor = "start" if index == 0 else "end" if index == len(x_ticks) - 1 else "middle"
        svg.extend(
            [
                f'  <line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_bottom}" stroke="#21262d" stroke-width="1" stroke-dasharray="3 5" />',
                f'  <text x="{x:.2f}" y="{plot_bottom + 29}" text-anchor="{anchor}" fill="#8b949e" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif" font-size="13">{tick.isoformat()}</text>',
            ]
        )

    last_x, last_y = points[-1]
    svg.extend(
        [
            f'  <path d="{area_path}" fill="url(#area)" />',
            f'  <path d="{line_path}" fill="none" stroke="#58a6ff" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />',
            f'  <circle cx="{last_x:.2f}" cy="{last_y:.2f}" r="6" fill="#58a6ff" filter="url(#glow)" />',
            f'  <circle cx="{last_x:.2f}" cy="{last_y:.2f}" r="2.5" fill="#f0f6fc" />',
            f'  <text x="{plot_left}" y="{height - 28}" fill="#6e7681" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif" font-size="12">Generated from GitHub stargazer timestamps</text>',
            "</svg>",
            "",
        ]
    )
    return "\n".join(svg)


def main() -> int:
    args = parse_args()
    token = os.environ.get("GITHUB_TOKEN")
    if not args.repo or not REPO_PATTERN.fullmatch(args.repo):
        print("--repo must be provided in owner/name form", file=sys.stderr)
        return 2
    if not token:
        print("GITHUB_TOKEN is required", file=sys.stderr)
        return 2

    try:
        star_dates, reported_count = fetch_star_dates(args.repo, token)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            render_svg(args.repo, star_dates, reported_count),
            encoding="utf-8",
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Generated {args.output} from {len(star_dates)} stargazer timestamps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
