---
name: commit
description: Stage changes and create a commit with emoji-prefixed conventional commit message
disable-model-invocation: true
argument-hint: "[optional message]"
allowed-tools: Bash, Read, Grep, Glob
---

# Emoji Commit

Stage relevant changes and create a commit following this project's emoji + conventional commit style.

## Steps

1. Run `git status` and `git diff` to understand current changes
2. Determine the commit type and choose the matching emoji
3. Write a concise Chinese commit message (1-2 sentences)
4. Stage the relevant files (prefer specific files over `git add -A`)
5. Commit with the format: `<emoji> <type>: <description>`

## Emoji mapping

| Emoji | Type       | When to use                    |
|-------|------------|--------------------------------|
| ✨    | feat       | New feature                    |
| 🐛    | fix        | Bug fix                        |
| 📝    | docs       | Documentation changes          |
| 💄    | style      | Formatting, UI, cosmetic       |
| ♻️    | refactor   | Code restructure, no behavior change |
| ⚡    | perf       | Performance improvement        |
| ✅    | test       | Add or update tests            |
| 🔧    | chore      | Build, config, tooling         |
| 🚀    | deploy     | Deployment related             |
| 🔥    | remove     | Remove code or files           |

## Commit message rules

- Format: `<emoji> <type>: <description>`
- Description in Chinese, concise
- Do NOT append any Co-Authored-By line

## User hint

$ARGUMENTS

If the user provided a message hint above, use it to guide the commit message content. Otherwise, infer from the diff.
