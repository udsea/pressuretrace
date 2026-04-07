# Org Codex Workflow Template

Use this as the default operating model across repositories.

## 1. Working model

- Main agent owns planning, integration, and final review.
- Subagents are for bounded side work with clear ownership.
- Repo tooling is the source of truth for style and correctness.
- Skills capture repeated judgment and workflow, not rules that CI can enforce.

## 2. Default prompt shape

Use prompts with these fields:

```text
Goal: <what needs to change>
Scope: <files, packages, or layers in bounds>
Constraints: <performance, API, migration, style, safety>
Verification: <tests, lint, typecheck, manual checks>
Mode: <analyze only | implement>
Delegation: <allowed | avoid unless needed>
```

Example:

```text
Goal: fix retry handling for API timeouts
Scope: client transport and retry policy only
Constraints: no API shape changes, keep logs stable
Verification: targeted tests plus relevant lint/typecheck
Mode: implement
Delegation: one explorer allowed for call-site tracing
```

## 3. Consistency stack

Enforce consistency in this order:

1. Formatter and linter
2. Type checker and tests
3. Shared repo structure and naming conventions
4. Shared agent instructions
5. Skills for repeated judgment-heavy workflows

If a rule can be enforced in tooling, do not rely on a skill for it.

## 4. When to delegate

Delegate only when:

- the task is parallelizable
- ownership is clear
- file overlap is unlikely
- the main thread is not blocked immediately

Avoid delegation when:

- the change is small
- the task is underspecified
- multiple areas need to be designed together

## 5. Standard operating steps

For unfamiliar repositories:

1. Read top-level docs and config.
2. Inspect the tree and identify entry points.
3. Check git status before editing.
4. Find the narrowest files that can satisfy the request.
5. Verify with the smallest useful command set before broadening.

For code changes:

1. Match existing patterns before introducing new ones.
2. Keep diffs narrow and local.
3. Add or update tests when behavior changes.
4. Run the narrowest relevant validation first.
5. Report exactly what was verified and what was not.

## 6. Recommended base skills

- `org-code-standards`: consistency and review bar
- `repo-intake`: orientation in unfamiliar codebases
- `change-validation`: what must be checked before calling work done

## 7. What to standardize outside Codex

- formatter config
- lint rules
- import ordering
- typecheck settings
- test command names
- commit and PR conventions
- directory layout for apps, libraries, and tests
