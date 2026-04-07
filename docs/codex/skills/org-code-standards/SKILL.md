---
name: org-code-standards
description: Use when work should follow shared engineering conventions across repositories, especially for refactors, reviews, consistency fixes, and behavior-preserving cleanup. This skill defines how to prefer repo-local standards first, then apply org-wide defaults for naming, error handling, logging, tests, and review quality.
---

# Org Code Standards

Use this skill when the task is about consistency, cleanup, review quality, or aligning changes with team norms.

## Decision order

Apply standards in this order:

1. Existing code in the touched area
2. Repo configuration and tooling
3. Shared org conventions
4. Personal preference

If the repository already has a clear local pattern, follow it even if another repo does things differently.

## Working rules

- Keep diffs narrow and behavior-preserving unless the task asks for behavior change.
- Reuse existing abstractions before creating new helpers or wrappers.
- Match naming, logging, and error-handling patterns already present in the touched area.
- Avoid mixing style cleanup with unrelated logic changes unless it materially improves the requested change.
- Prefer removing duplication only when the extracted abstraction is already justified by nearby usage.

## Consistency defaults

Use these defaults only when the repo does not already define a stronger rule:

- Favor explicit names over shortened ones.
- Keep public APIs stable unless the task explicitly changes them.
- Keep error messages actionable and specific.
- Make logs structured and low-noise where the stack supports it.
- Add tests for behavior changes and regression risks.

## Before finishing

- Run the repo's formatter, linter, and relevant tests if available.
- Call out any place where the repository's current pattern conflicts with the requested direction.
- If a better standard should be enforced automatically, suggest moving it into tooling instead of relying on prompts.
