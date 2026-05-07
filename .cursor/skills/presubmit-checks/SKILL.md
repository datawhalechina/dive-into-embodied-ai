---
name: presubmit-checks
description: Run before committing docs or asset changes in dive-into-embodied-ai. Two commands catch the recurring regressions (PNG/JPG leak, MDX errors, broken links, sidebar id mismatch, mermaid SSR crash). Use when finishing any change under docs/, src/, sidebars*.ts, _category_.json, or after syncing from upstream dive-into-embodied-ai.
---

# Presubmit Checks

## Two commands, in order

```bash
npm run assets:webp:check
npm run build
```

Both must exit 0. If either fails, fix the regression below and re-run.

## Known regressions and fixes

| Symptom (first error line) | Cause | Fix |
|---|---|---|
| `ERROR: raster images must be WebP` | New PNG/JPG/JPEG leaked into the repo | `npm run assets:webp` (uses the `webp-assets` skill) |
| `Markdown image with URL X.png ... couldn't be resolved` | Source content references `.png` but target only has `.webp` | Replace `.png` / `.jpg` → `.webp` in the markdown |
| `Can't find any doc with ID X` | `_category_.json` `link.id` or `sidebars*.ts` references an old / wrong doc id | Update the id to the new Docusaurus path; do not keep source-side short ids like `cs123/intro` |
| `Docusaurus found broken links` referencing a sibling `./N.X.md` | Relative file links not shifted after a prefix-offset sync | Shift the `N.` prefix to match target's prefix |
| `ReactContextError ... useColorMode ... MermaidRenderer` (SSR) | `theme-mermaid` + `future.faster: true` is broken in this Docusaurus version | Replace ` ```mermaid ` fences with ` ```text ` until upstream fixes it; do **not** disable `faster` |
| `MDX compilation failed ... Unexpected VariableDeclaration` in `.mdx` | Top-level `const` instead of `export const` (MDX is strict ESM at top level) | Change `const X =` → `export const X =` |
| `Unable to resolve loader raw-loader` | Source uses `import x from '!!raw-loader!./x.py'` | Inline the file content as `export const x = "..."` (do not add raw-loader as dependency) |

## When syncing from upstream `dive-into-embodied-ai`

After running the two commands, also confirm:

- Source frontmatter `id:` and `slug:` are stripped.
- Path prefixes rewritten: `/docs/projects/` → `/docs/practices/...`, `/docs/tutorial/` → `/docs/foundations/...`.
- GitHub URL rewritten: upstream source repositories → `github.com/datawhalechina/dive-into-embodied-ai/...`.
- VLA chapters: source prefix N → target prefix N+1; cross-refs `第 N 讲` shifted by +1 inside the body.
- Skip non-reader assets like `.paper-work/`, LaTeX sources, `source.bundle`.

Detailed sync recipe lives in `docs/plans/d2eai-migration-plan-v3-mapping.md` §7.6.
