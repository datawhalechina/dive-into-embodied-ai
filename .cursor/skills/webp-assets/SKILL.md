---
name: webp-assets
description: Keep Docusaurus documentation image assets consistent by converting PNG/JPG/JPEG files to WebP, updating references, and checking pre-commit asset rules. Use when migrating docs, adding images, fixing oversized image files, or when the user mentions WebP, image consistency, asset conversion, or 500KB image limits.
---

# WebP Assets

## When To Use

Use this skill in `dw-dive-into-embodied-ai` when:

- Migrating docs or figures from another repo.
- Adding or replacing static documentation images.
- Fixing files blocked by the 500KB pre-commit limit.
- The user asks to keep image formats consistent.

## Rules

- Static raster images must be WebP: convert `png`, `jpg`, and `jpeg`.
- Keep SVG as SVG.
- Keep GIF/video files as-is; they are handled by existing LFS rules.
- Do not rewrite image names in code examples unless they are actual asset references.

## Commands

From the repo root:

```bash
npm run assets:webp
```

This converts repository raster images to WebP and rewrites local references in Markdown, MDX, TS, TSX, JS, JSX, CSS, JSON, YAML, and YML files.

Validate before finishing:

```bash
npm run assets:webp:check
npm run build
npm run typecheck
```

## Pre-Commit Behavior

The pre-commit hook runs:

```bash
python3 scripts/convert-images-to-webp.py --check
```

If a `png`, `jpg`, or `jpeg` file remains in the repo, the hook should fail and instruct the user to run `npm run assets:webp`.

## Notes

- The converter requires `cwebp` on the machine.
- If a specific WebP becomes visibly degraded, keep the file as WebP but regenerate it from a better source image or adjust the converter quality strategy in `scripts/convert-images-to-webp.py`.
