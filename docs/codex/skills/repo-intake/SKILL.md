---
name: repo-intake
description: Use at the start of work in an unfamiliar repository or subsystem. This skill gives a compact orientation workflow for reading docs, locating entry points, finding ownership boundaries, checking current git state, and narrowing the change surface before editing.
---

# Repo Intake

Use this skill before making changes in a codebase you do not already know well.

## Intake workflow

1. Read the top-level README and the main project config files.
2. Inspect the repository tree to find app code, libraries, scripts, and tests.
3. Check `git status` so you do not overwrite unrelated local work.
4. Trace the request to the smallest set of files that can satisfy it.
5. Identify the validation path before making edits.

## What to find quickly

- primary entry points
- test locations and commands
- build, lint, and typecheck commands
- central configuration files
- boundaries between modules or packages

## Search behavior

- Prefer fast codebase search over browsing file-by-file.
- Read only enough surrounding code to understand the local pattern.
- When a subsystem is unclear, summarize its shape before editing.

## Before editing

Write down or keep in mind:

- which files are in scope
- which files are adjacent but out of scope
- what command will verify the change
- whether delegation would help without creating overlap

## Delegation rule

Spawn an explorer only for a narrow question such as:

- where a feature starts
- which call sites use an API
- how tests are organized for one subsystem

Keep architectural decisions and cross-cutting changes in the main thread.
