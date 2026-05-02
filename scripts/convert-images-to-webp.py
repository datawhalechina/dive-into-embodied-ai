#!/usr/bin/env python3
"""Convert repository raster images to WebP and rewrite local references."""

from __future__ import annotations

import argparse
import os
import posixpath
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RASTER_EXTS = {".png", ".jpg", ".jpeg"}
TEXT_EXTS = {
    ".md",
    ".mdx",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".css",
    ".json",
    ".yml",
    ".yaml",
}
SKIP_DIRS = {".git", "node_modules", "build", ".docusaurus"}
MAX_BYTES = 512_000
QUALITIES = [85, 78, 70, 62, 55, 48, 42, 36, 30, 24, 18, 12]


def is_skipped(path: Path) -> bool:
    rel_parts = path.relative_to(ROOT).parts
    return any(part in SKIP_DIRS for part in rel_parts)


def iter_raster_images() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not is_skipped(path)
        and path.suffix.lower() in RASTER_EXTS
    )


def iter_text_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not is_skipped(path)
        and path.suffix.lower() in TEXT_EXTS
    )


def run_cwebp(source: Path, output: Path, quality: int) -> None:
    subprocess.run(
        [
            "cwebp",
            "-quiet",
            "-m",
            "6",
            "-mt",
            "-q",
            str(quality),
            str(source),
            "-o",
            str(output),
        ],
        check=True,
    )


def convert_image(path: Path) -> tuple[Path, int, int, int]:
    target = path.with_suffix(".webp")
    original_size = path.stat().st_size

    best: tuple[int, int, Path] | None = None
    for quality in QUALITIES:
        tmp = target.with_name(f"{target.name}.q{quality}.tmp")
        tmp.unlink(missing_ok=True)
        run_cwebp(path, tmp, quality)
        size = tmp.stat().st_size
        if best is None or size < best[0]:
            if best is not None:
                best[2].unlink(missing_ok=True)
            best = (size, quality, tmp)
        else:
            tmp.unlink(missing_ok=True)
        if size <= MAX_BYTES:
            break

    if best is None:
        raise RuntimeError(f"failed to convert {path}")

    size, quality, tmp = best
    target.unlink(missing_ok=True)
    tmp.replace(target)
    path.unlink()
    return target, original_size, size, quality


def reference_variants(old_path: Path, new_path: Path, text_file: Path) -> dict[str, str]:
    old_repo = old_path.relative_to(ROOT).as_posix()
    new_repo = new_path.relative_to(ROOT).as_posix()
    old_rel = posixpath.relpath(old_path.as_posix(), text_file.parent.as_posix())
    new_rel = posixpath.relpath(new_path.as_posix(), text_file.parent.as_posix())

    variants = {
        old_repo: new_repo,
        old_rel: new_rel,
    }
    if not old_rel.startswith(".."):
        variants[f"./{old_rel}"] = f"./{new_rel}"
    return variants


def rewrite_references(converted: list[tuple[Path, Path]]) -> int:
    changed = 0
    for text_file in iter_text_files():
        content = text_file.read_text(errors="ignore")
        updated = content
        for old_path, new_path in converted:
            for old, new in sorted(
                reference_variants(old_path, new_path, text_file).items(),
                key=lambda item: len(item[0]),
                reverse=True,
            ):
                updated = updated.replace(old, new)
        if updated != content:
            text_file.write_text(updated)
            changed += 1
    return changed


def check_consistency() -> int:
    leftovers = iter_raster_images()
    if leftovers:
        print("ERROR: raster images must be WebP. Run `npm run assets:webp`.")
        for path in leftovers:
            print(f"  {path.relative_to(ROOT).as_posix()}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        return check_consistency()

    images = iter_raster_images()
    converted: list[tuple[Path, Path]] = []
    for image in images:
        target, old_size, new_size, quality = convert_image(image)
        converted.append((image, target))
        print(
            f"{image.relative_to(ROOT).as_posix()} -> "
            f"{target.relative_to(ROOT).as_posix()} "
            f"({old_size / 1024:.0f} KiB -> {new_size / 1024:.0f} KiB, q{quality})"
        )

    changed_files = rewrite_references(converted)
    print(f"Converted {len(converted)} image(s); updated {changed_files} text file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
