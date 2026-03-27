# End-to-End Evaluation Summary
## Experiment Configuration
- Mode: benchmark
- Seeds: 0, 1, 2
- Variants: baseline, guarded_no_declass, guarded_structured_declass
- Scenario count: 7

## Publication Table
| variant | attack_success_rate | unsafe_action_rate | security_failure_rate | blocked_malicious_action_rate | indirect_prompt_influence_rate | task_completion_rate_attacked | task_completion_rate_clean | task_token_match_rate_attacked | task_token_match_rate_clean | benign_action_preservation_rate | overblocking_rate | runtime_overhead_vs_baseline_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.048 ± 0.000 | 1.000 ± 0.000 | 0.286 ± 0.000 | 0.857 ± 0.000 | 0.619 ± 0.000 | 0.857 ± 0.000 | 0.143 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| guarded_no_declass | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.429 ± 0.000 | 0.774 ± 0.000 | 1.000 ± 0.000 | 0.286 ± 0.000 | 0.286 ± 0.000 | 0.619 ± 0.000 | 0.619 ± 0.000 | 0.857 ± 0.000 | 0.000 ± 0.000 | -5.137 ± 22.008 |
| guarded_structured_declass | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.857 ± 0.000 | 0.857 ± 0.000 | 0.857 ± 0.000 | 0.857 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 79.946 ± 45.207 |
