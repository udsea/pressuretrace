# Subagent Decision Checklist

Use this before spawning a subagent.

## Spawn a subagent when all of these are true

- The task is concrete and bounded.
- The write scope is separate from other in-flight work, or the task is read-only.
- The main thread can keep moving without waiting for the answer immediately.
- The expected output is specific: files changed, findings, commands run, or a short decision.

## Keep the work in the main thread when any of these are true

- The next step is blocked on that result.
- The task is ambiguous or needs judgment across multiple parts of the codebase.
- Multiple agents would likely touch the same files.
- The task is small enough to finish faster than briefing a worker.

## Good delegation patterns

- Explorer: trace a code path, find entry points, summarize a subsystem, list affected files.
- Worker: own one package, one service, one test area, or one script directory.
- Validation worker: run a bounded verification pass while the main thread keeps implementing.

## Bad delegation patterns

- "Understand the repo."
- "Fix this however you think best."
- Two workers editing the same module tree.
- Handing off the core design decision and then waiting idle.

## Prompt template

Use this shape when spawning:

```text
Own <path or module>.

Goal: <one sentence>.
Scope: <what is in bounds>.
Out of scope: <what must not be touched>.
Deliverable: <diff / findings / failing tests / file list>.

You are not alone in the codebase. Do not revert unrelated changes. Adjust to existing edits if needed.
```

## Operating rule

The main agent should keep ownership of:

- task framing
- critical-path decisions
- integration across modules
- final review and user-facing explanation
