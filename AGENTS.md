# Agent Guide

This file helps AI agents work on this repository. See [README.md](README.md) for the learning content and [CONTRIBUTING.md](CONTRIBUTING.md) for contributor setup and asset rules.

## Project Overview

Dive into Embodied AI is a Chinese-first, open-source tutorial covering embodied AI foundations, simulation, and robotics projects. It combines a Docusaurus documentation website with standalone practice code. The website supports Chinese (`zh-Hans`), English (`en`), and Italian (`it`).

**Website stack:** Docusaurus 3, React, TypeScript, Markdown/MDX, MathJax, and Mermaid. Interactive demos also use Three.js and MuJoCo. Python robotics projects have their own dependencies and environments.

## Development Setup

Use the Node.js version in `.nvmrc` and the npm version in `package.json`. `.npmrc` enforces these versions and configures Node options for static builds.

```bash
# With nvm; another version manager can select the same version.
nvm install
nvm use

# Install Git LFS on your machine before these commands.
git lfs install
git lfs pull
npm ci
```

For code under `codes/`, read the relevant project's README and dependency files before installing or running anything. Do not assume the website's setup installs simulation, training, GPU, or hardware dependencies.

## Key Commands

Run website commands from the repository root:

```bash
npm run dev                          # Chinese development server
npm run start -- --locale en          # English development server
npm run start -- --locale it          # Italian development server
sh local-build.sh                    # Build all locales and serve locally
npm run build                        # Production build for all configured locales
npm run serve                        # Serve an existing build
npm run typecheck                    # TypeScript checks
npm run check:build-lock              # Concurrent builds and interrupted-process recovery
npm run assets:webp:check             # Raster asset policy
npm run check:i18n                    # After building: translations, routes, anchors
npm run check:page-markdown           # After building: page Markdown export
```

Before committing, run `npm run assets:webp:check` and `npm run build`, as required by `CONTRIBUTING.md`. Run `npm run typecheck` for TypeScript changes. The deployment workflow also runs `check:i18n` and `check:page-markdown` after building.

Use `npm run build` for builds and `npm run install:locked` for dependency reinstalls while working in a shared checkout. These commands share a process lock and reject overlapping work. Direct Docusaurus build and `npm ci` invocations bypass the lock. Do not delete `.docusaurus-build-lock.sqlite`; the OS releases the lock when the owning process exits.

Use the additional checks relevant to the changed area:

| Area | Command |
| --- | --- |
| Documentation images and references | `npm run check:doc-assets` |
| Mathematical cross-references | `npm run check:math-refs` |
| Documentation sidebars | `npm run check:doc-sidebars` |
| Practice navigation | `npm run check:practice-sidebars` |
| Interactive playgrounds | `npm run check:playgrounds` |
| Quaddle browser simulation | `npm run check:quaddle-web` |

For practice code, use the project's documented checks. Report which checks ran and any environment limitations; a website build does not validate robot hardware or training results.

## Repository Structure

- `docs/` — Chinese source tutorials: introduction, foundations, learning maps, and practice guides.
- `i18n/en/`, `i18n/it/` — Translated documentation and Docusaurus UI, navigation, footer, and sidebar catalogs.
- `src/pages/`, `src/components/` — Website pages, shared UI, and interactive demos.
- `src/theme/`, `src/css/` — Docusaurus theme customizations and site styles.
- `codes/practices/` — Standalone robotics and simulation projects with their own setup instructions.
- `static/`, `assets/`, and chapter `figs/` directories — Site assets and tutorial media.
- `scripts/` — Validation, asset generation, and local development helpers.
- `docusaurus.config.ts`, `sidebars.ts` — Website configuration, locales, plugins, and documentation navigation.
- `.github/workflows/deploy.yml` — Website build, validation, and GitHub Pages deployment.

## Content and Translations

- Chinese is the primary source of truth and the site's default language.
- When Chinese content or UI text changes, update the corresponding English version in the same task. Keep existing Italian translations aligned for affected content as well. Synchronize additions, deletions, links, and examples across the affected translations.
- Update the affected README, learning map, and Docusaurus translation files together. Translated chapters belong under `i18n/<locale>/docusaurus-plugin-content-docs/current/`; preserve source document IDs, slugs, and explicit section anchors.
- Unchanged Chinese-only tutorial chapters do not need a full translation. Keep fallback notices accurate; do not label Chinese fallback content as translated text.
- For multilingual website changes, build all configured locales and run `npm run check:i18n`. Verify language switching and shared section anchors when affected.
- Preserve the site's `/dive-into-embodied-ai/` base path and locale prefixes when changing links or routes. Check navigation and relevant overview pages when adding or moving chapters.

## Assets and Website Changes

- Use WebP for new raster images; `npm run assets:webp` converts supported images. Follow `.gitattributes` for Git LFS assets, and check that local media contains real data rather than LFS pointer text.
- Edit source files, not generated output in `build/`, `.docusaurus/`, or `static/quaddle/`. The Quaddle plugin generates browser assets from the pinned npm package and canonical model files.
- Preserve the existing Docusaurus architecture and browser conventions. For UI changes, check accessibility, typography, color contrast, responsive layout, and clear writing, including keyboard use and light/dark themes where affected.
- Keep personal agent configuration, installed skills, and local design notes in ignored directories such as `.agents/`. Shared contributor instructions must work without those private files.

## Git Workflow

- Inspect the working tree before editing and preserve unrelated changes.
- Commit and push only after the user explicitly approves those actions for the current work. Requests to implement, fix, or verify changes do not authorize committing or pushing.
- Keep approved commits focused and include corresponding translations with the source changes. Summarize the changes and validation results for review.
