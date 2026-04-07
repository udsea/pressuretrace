---
name: change-validation
description: Use before calling a change complete. This skill standardizes how to choose validation commands, run the narrowest useful checks first, expand coverage when risk is higher, and report exactly what was and was not verified.
---

# Change Validation

Use this skill after edits and before presenting work as done.

## Validation order

Run checks from narrowest to broadest:

1. Targeted unit or file-specific tests
2. Relevant lint or formatter checks for touched files
3. Relevant type checks or build steps
4. Broader test suites only when risk justifies them

## Selection rules

- Prefer repository-native commands over ad hoc ones.
- Validate the behavior you changed, not the whole world by default.
- Broaden coverage when the change affects shared infrastructure, public APIs, schemas, or concurrency.
- If a command is slow or unavailable, say so explicitly rather than implying full verification.

## Reporting rules

Always report:

- commands run
- whether they passed or failed
- any checks you intentionally skipped
- any remaining risk that validation did not cover

## Failure handling

- If validation fails because of your change, fix it or explain the blocker.
- If validation fails for unrelated pre-existing reasons, do not hide it. Isolate the issue and state why it appears unrelated.
- If no automated check exists, describe the smallest manual verification that would increase confidence.
