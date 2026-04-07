# PressureTrace v1

## Thesis

Pressure can systematically shift models from robust task completion toward shortcut-seeking or oversight-evasive routes, even when baseline capability is sufficient.

## Task Families

- Reasoning: GSM8K-style math reasoning tasks transformed into paired control and pressure variants.
- Coding: HumanEval-style and MBPP-style coding tasks transformed into paired control and pressure variants.

## Pressure Knobs

- Time pressure: reduced implied budget for verification or reflection.
- Oversight gap: weaker cues that intermediate reasoning or generalization will be checked.
- Reward salience: stronger emphasis on fast visible success.
- Instruction ambiguity: prompts that leave more room for shallow but plausible completion strategies.

## Route Labels

- `robust_correct`
- `shortcut_followed`
- `spec_gaming_suspected`
- `other_error`
- `unknown`

## Hypotheses

- H1: Pressure will reduce robust correctness even when the underlying task remains solvable.
- H2: Pressure will increase shortcut-following on paired reasoning variants.
- H3: Pressure will increase spec-gaming behavior on coding tasks with visible-test affordances.
- H4: Behavioral route labels will be more sensitive than pass/fail alone for detecting pressure effects.
- H5: Hidden-state probes will predict route choice before final output is emitted.

## Metrics

- Route-label rates by task family and pressure level.
- Accuracy / pass rate under control versus pressure.
- Delta in robust correctness between paired variants.
- Probe separability for robust versus shortcut/spec-gaming routes.

## Success Criteria

- Build paired manifests from reputable base datasets.
- Run benchmark pilots end-to-end with stable JSONL outputs.
- Show measurable route-label shifts under at least one pressure knob.

## Immediate Next Milestone

Implement deterministic paired-task generation for GSM8K, HumanEval, and MBPP with audited manifest schemas and small-scale dry-run pilots.
