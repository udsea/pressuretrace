# PressureTrace Frozen Reasoning Probe Artifacts

- Frozen root: `/Users/udbhav/Dev/tinkering/pressuretrace/pressuretrace-frozen/reasoning_v2_qwen3_14b_off`
- Source manifest: `/Users/udbhav/Dev/tinkering/pressuretrace/pressuretrace-frozen/reasoning_v2_qwen3_14b_off/data/manifests/reasoning_paper_slice_qwen-qwen3-14b_off.jsonl`
- Source results: `/Users/udbhav/Dev/tinkering/pressuretrace/pressuretrace-frozen/reasoning_v2_qwen3_14b_off/results/reasoning_paper_slice_qwen-qwen3-14b_off.jsonl`
- Hidden-state file: `/Users/udbhav/Dev/tinkering/pressuretrace/pressuretrace-frozen/reasoning_v2_qwen3_14b_off/results/reasoning_probe_hidden_states_qwen-qwen3-14b_off.jsonl`
- Probe dataset: `/Users/udbhav/Dev/tinkering/pressuretrace/pressuretrace-frozen/reasoning_v2_qwen3_14b_off/results/reasoning_probe_dataset_qwen-qwen3-14b_off.jsonl`
- Probe metrics JSONL: `/Users/udbhav/Dev/tinkering/pressuretrace/pressuretrace-frozen/reasoning_v2_qwen3_14b_off/results/reasoning_probe_metrics_qwen-qwen3-14b_off.jsonl`
- Probe summary TXT: `/Users/udbhav/Dev/tinkering/pressuretrace/pressuretrace-frozen/reasoning_v2_qwen3_14b_off/results/reasoning_probe_summary_qwen-qwen3-14b_off.txt`
- Metrics CSV: `/Users/udbhav/Dev/tinkering/pressuretrace/pressuretrace-frozen/reasoning_v2_qwen3_14b_off/results/reasoning_probe_metrics_qwen-qwen3-14b_off.csv`
- Paper table CSV: `/Users/udbhav/Dev/tinkering/pressuretrace/pressuretrace-frozen/reasoning_v2_qwen3_14b_off/results/reasoning_probe_table_qwen-qwen3-14b_off.csv`
- Paper table Markdown: `/Users/udbhav/Dev/tinkering/pressuretrace/pressuretrace-frozen/reasoning_v2_qwen3_14b_off/results/reasoning_probe_table_qwen-qwen3-14b_off.md`
- Patch pairs: `/Users/udbhav/Dev/tinkering/pressuretrace/pressuretrace-frozen/reasoning_v2_qwen3_14b_off/results/reasoning_patch_pairs_qwen-qwen3-14b_off.jsonl`

## Configuration

- Model name: `Qwen/Qwen3-14B`
- Thinking mode: `off`
- Pressure conditions used in probing: `neutral_wrong_answer_cue, teacher_anchor`
- Label definition: `1 = shortcut_followed`, `0 = robust_correct`
- Split rule: `base_task_id` train/test split, stratified when possible
- Layers used: `-10, -8, -6, -4, -2, -1`
- Representations used: `last_token, mean_pool`

## Result Snapshot

- Best hidden-state probe: layer `-10`, representation `last_token`, ROC AUC `0.8138`
- Prompt-length baseline ROC AUC: `0.6413`
- TF-IDF baseline ROC AUC: `0.6290`
