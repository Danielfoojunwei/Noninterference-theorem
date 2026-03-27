# End-to-End Empirical Evaluation

This directory contains the **executable empirical layer** for the repository. Its purpose is not to restate the formal theorem, but to test whether a contaminated-input agent runtime with explicit taint tracking, guard enforcement, and declassification can improve security without destroying task utility.

## Scope

The current evaluation package supports three distinct system variants.

| Variant | Description |
| --- | --- |
| `baseline` | An unguarded runtime that reads contaminated inputs and applies proposed actions without taint-aware restriction. |
| `guarded_no_declass` | A taint-aware guarded runtime that blocks or repairs control-relevant actions but does not recover trusted facts through declassification. |
| `guarded_structured_declass` | The guarded runtime plus a structured schema-based declassification strategy that promotes validated facts into verified state before later actions depend on them. |

The benchmark executes both **attacked** and **clean** versions of each scenario, so security and utility can be compared directly on the same tasks.

## What is implemented

| Component | File | Function |
| --- | --- | --- |
| Guarded runtime | `src/guarded_runtime.py` | Explicit contaminated-input state, provenance, taint flow, memory, plan state, and control-point enforcement. |
| Declassification | `src/declassification.py` | Structured extraction with typed schema validation, verified-fact promotion, and quarantine of unsafe spans. |
| End-to-end benchmark | `src/e2e_benchmark.py` | Scenario library, attacked/clean execution, and variant comparison. |
| Metrics | `src/e2e_metrics.py` | Security, utility, noninterference, provenance, and runtime-overhead metrics. |
| Reproducible runner | `src/run_e2e_eval.py` | Seeded experiment execution, config snapshots, raw artifact saving, aggregate summaries, and publication tables. |

## Threat coverage

The current benchmark covers seven scenario families saved in the canonical benchmark run configuration.

| Scenario ID | Threat pattern | Task family |
| --- | --- | --- |
| `email_order_lookup_markdown` | Markdown embedded instruction override | `email_summarization` |
| `hostile_web_search_html` | HTML comment plus paraphrased terminal instruction | `retrieval_browsing` |
| `memory_poisoning_followup` | Two-step persistence poisoning | `memory_persistence` |
| `tool_output_json_poisoning` | JSON field plus adversarial agent note | `tool_output_poisoning` |
| `declassification_bypass_attempt` | Field-level declassification bypass | `declassification_attack` |
| `markdown_codeblock_obfuscation` | Code-block embedded obfuscated override | `document_processing` |
| `authenticated_tool_paraphrase_poisoning` | Authenticated JSON plus paraphrased override | `authenticated_tool_poisoning` |

These scenarios are intentionally small and local. They are meant to provide **honest end-to-end coverage** of important agent failure modes, not to stand in for a production-scale evaluation corpus.

## Metrics

The package reports both security and utility outcomes.

| Metric family | Included measures |
| --- | --- |
| Security | Attack success rate, unsafe action rate, security failure rate, blocked malicious action rate, indirect prompt influence rate, plan corruption rate, persistence-poisoning success rate, unsafe final-answer rate |
| Utility | Attacked and clean task completion, attacked and clean task-token match, benign-action preservation rate, legitimate tool use, overblocking |
| Provenance | Verified-memory rate, verified-action usage rate |
| Operational | Runtime, attempted action count, executed tool count, runtime overhead versus baseline |
| Pairwise invariance | Tool, memory, answer, and task-outcome invariance between attacked and clean runs |

## Installation

Use the Python dependencies declared in `requirements.txt`.

```bash
cd noninterference_eval
pip3 install -r requirements.txt
```

The reproducibility targets in the repository root also check for core dependencies before running experiments.

## One-command workflows

From the repository root:

```bash
make check-eval-deps
make test-e2e
make e2e-smoke
make e2e-benchmark
```

To aggregate previously saved runs or regenerate compact tables:

```bash
make e2e-aggregate RUN_DIRS="noninterference_eval/results/e2e/benchmark_run_a noninterference_eval/results/e2e/benchmark_run_b"
make e2e-tables RUN_DIRS="noninterference_eval/results/e2e/benchmark_run_a noninterference_eval/results/e2e/benchmark_run_b"
```

The same workflows can be invoked directly.

```bash
cd noninterference_eval
python3.11 src/run_e2e_eval.py smoke --output-dir ./results/e2e
python3.11 src/run_e2e_eval.py benchmark --output-dir ./results/e2e
```

## Saved outputs

Each run directory contains a configuration snapshot, per-seed raw artifacts, metric bundles, and aggregate summaries.

| Artifact | Description |
| --- | --- |
| `config_snapshot.yaml` | Exact seeds, variants, scenario IDs, output directory, and timestamp |
| `seed_*/attacked_artifacts.json` | Raw attacked-run traces |
| `seed_*/clean_artifacts.json` | Raw clean-run traces |
| `seed_*/scenario_table.json` | Per-scenario metric rows |
| `seed_*/metrics_bundle.json` | Full metric bundle for that seed |
| `aggregate_seed_summary.json` | Mean and standard deviation across seeds |
| `publication_table.json` | Compact publication-facing table |
| `summary.md` | Human-readable summary |

## Canonical benchmark result in this repository

The benchmark run produced during the repository upgrade is stored at:

```text
results/e2e/benchmark_20260327T025110Z/
```

Its headline results are shown below.

| Variant | Attack Success Rate | Unsafe Action Rate | Security Failure Rate | Attacked Task Completion | Clean Task Completion |
| --- | --- | --- | --- | --- | --- |
| `baseline` | `1.000 ± 0.000` | `1.000 ± 0.000` | `1.000 ± 0.000` | `0.286 ± 0.000` | `0.857 ± 0.000` |
| `guarded_no_declass` | `0.000 ± 0.000` | `0.000 ± 0.000` | `0.429 ± 0.000` | `0.286 ± 0.000` | `0.286 ± 0.000` |
| `guarded_structured_declass` | `0.000 ± 0.000` | `0.000 ± 0.000` | `0.000 ± 0.000` | `0.857 ± 0.000` | `0.857 ± 0.000` |

The result pattern matters more than the absolute scale. The unguarded baseline is vulnerable throughout the benchmark. The guarded runtime without declassification removes malicious tool execution but leaves residual security failures and sharply reduced usefulness. The guarded runtime with structured declassification recovers the clean-task completion rate on this benchmark while preserving zero unsafe actions.

## Testing

The focused end-to-end and subsystem tests can be run with:

```bash
cd noninterference_eval
pytest -q tests/test_guarded_runtime.py \
          tests/test_declassification.py \
          tests/test_e2e_benchmark.py \
          tests/test_e2e_metrics.py
```

From the repository root, the equivalent command is:

```bash
make test-e2e
```

## Limitations

This package does not claim production-grade coverage. The scenario library is small, the tasks are local and hand-constructed, and the runtime is a benchmarked control architecture rather than a full deployment framework. Tool trust assignments remain assumption-sensitive, especially for authenticated channels. The overhead figures come from this lightweight harness and should not be interpreted as deployment latency guarantees.
