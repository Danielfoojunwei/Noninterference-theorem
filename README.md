# Noninterference Theorem for Indirect Prompt Injection in Agentic AI

> **A formal noninterference theorem paired with an executable end-to-end guarded-agent prototype and a saved empirical benchmark.**

---

## Overview

This repository studies **indirect prompt injection** as a control-plane security problem for agentic AI systems. The central formal claim is that, under explicit architectural assumptions, a system can enforce a **noninterference-style guarantee**: variations in untrusted content should not influence control-relevant behavior such as tool invocation, plan updates, persistent memory writes, or final answers [1] [2].

The repository now contains two distinct contributions that should be read separately. First, it contains a **formal specification and theorem** for taint-aware action selection. Second, it contains an **implemented empirical prototype** that evaluates a real contaminated-input execution path with explicit guard enforcement and a concrete declassification mechanism. The empirical system is intentionally modest in scale, but it is executable, reproducible, and materially stronger than the earlier shortcut-based prototype.

## Formal claim versus empirical claim

The most important distinction in this repository is between what is **proved** and what is **measured**.

| Layer | What is claimed | Current status |
| --- | --- | --- |
| Formal theorem | If tainted content is excluded from the control-relevant dependency set under the stated invariants, control outputs are noninterfering with respect to adversarial changes in untrusted input. | **Proved in the paper** under explicit assumptions. |
| Guarded runtime implementation | A contaminated-input runtime can carry provenance and taint labels through state and enforce policy at multiple control points. | **Implemented** in `noninterference_eval/src/guarded_runtime.py`. |
| Declassification | Untrusted content can be reduced to policy-checked structured facts before it influences actions. | **Implemented** in `noninterference_eval/src/declassification.py`. |
| End-to-end empirical evaluation | Baseline, guarded-without-declassification, and guarded-with-structured-declassification systems can be compared on both security and utility. | **Implemented and executed** in `noninterference_eval/src/e2e_benchmark.py` with saved artifacts under `noninterference_eval/results/e2e/benchmark_20260327T025110Z/`. |
| Real-world security guarantee | The implementation solves indirect prompt injection in arbitrary production agents. | **Not claimed.** |

## Current empirical status

The earlier version of this repository relied too heavily on shortcut-style evaluation, especially guarded runs that effectively depended on cleaned inputs outside the guarded execution path. That is no longer the headline evidence. The current empirical story is narrower but more honest.

The main benchmark now runs three variants across seven attacked scenarios and seven clean counterparts, with three saved seeds. The evaluated variants are `baseline`, `guarded_no_declass`, and `guarded_structured_declass`, and the scenario set covers markdown injection, hostile HTML retrieval poisoning, two-step memory poisoning, JSON tool-output poisoning, a declassification-bypass attempt, code-block obfuscation, and authenticated-tool paraphrase poisoning [3] [4].

The saved publication table shows a clear tradeoff. The unguarded baseline is fully vulnerable on this benchmark, with **1.000 ± 0.000 attack success rate**, **1.000 ± 0.000 unsafe action rate**, and only **0.286 ± 0.000 attacked task completion**. The guarded system without declassification eliminates malicious tool execution but still suffers **0.429 ± 0.000 security failure rate** and substantial utility loss, reaching only **0.286 ± 0.000 attacked task completion** and **0.286 ± 0.000 clean task completion**. The guarded system with structured declassification achieves **0.000 ± 0.000 attack success**, **0.000 ± 0.000 unsafe action rate**, **0.000 ± 0.000 security failure rate**, and **0.857 ± 0.000 attacked task completion**, matching its clean task completion rate on this benchmark [5].

| Variant | Attack Success Rate | Unsafe Action Rate | Security Failure Rate | Attacked Task Completion | Clean Task Completion | Benign-Action Preservation | Runtime Overhead vs Baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.286 ± 0.000 | 0.857 ± 0.000 | 0.143 ± 0.000 | 0.000 ± 0.000 |
| guarded_no_declass | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.429 ± 0.000 | 0.286 ± 0.000 | 0.286 ± 0.000 | 0.857 ± 0.000 | -5.137 ± 22.008 |
| guarded_structured_declass | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.857 ± 0.000 | 0.857 ± 0.000 | 1.000 ± 0.000 | 79.946 ± 45.207 |

These figures support a limited but concrete empirical claim: **in this implemented benchmark, structured declassification recovers task utility while preserving guarded control behavior** [5]. They do **not** justify a broad claim that arbitrary agent frameworks, arbitrary task distributions, or arbitrary authenticated tools are now solved.

## Implemented system architecture

The empirical prototype is built around an explicit stateful runtime rather than an oracle that substitutes clean inputs for guarded ones. Both trusted and untrusted information enter through explicit ingestion channels, and provenance is maintained at the node level throughout execution.

| Component | File | Implemented role |
| --- | --- | --- |
| Guarded runtime | `noninterference_eval/src/guarded_runtime.py` | Maintains state nodes, provenance, taint, persistent memory, and action logs; enforces policy at control points. |
| Declassification module | `noninterference_eval/src/declassification.py` | Implements a strategy interface and a structured schema extraction strategy that promotes only validated fields to verified facts. |
| End-to-end benchmark | `noninterference_eval/src/e2e_benchmark.py` | Defines realistic contaminated-input scenarios and runs baseline, guarded, and declassification-enabled variants. |
| Metrics and aggregation | `noninterference_eval/src/e2e_metrics.py` | Computes security, utility, provenance, noninterference, and runtime-overhead metrics. |
| Reproducible experiment runner | `noninterference_eval/src/run_e2e_eval.py` | Saves config snapshots, raw artifacts, per-seed metrics, aggregate summaries, and publication tables. |

The guarded runtime enforces policy at four key control points required by the audit: **tool calls or action execution, plan updates, memory writes, and final-answer emission**. Tainted evidence can remain in the system for inspection, logging, and candidate extraction, but it cannot directly drive control-relevant behavior unless it is explicitly promoted through the declassification path.

## Declassification design

The practical difficulty in agent security is not merely blocking malicious content, but allowing useful work on top of untrusted inputs. This repository now includes a concrete answer in code.

The implemented declassification mechanism uses **structured schema extraction**. Untrusted text is parsed into candidate fields, each candidate is validated against a typed specification and policy checks, and only accepted fields are promoted to `VerifiedFact` nodes. Unsafe spans are quarantined rather than allowed to shape control decisions. The current strategy supports typed extraction such as email addresses, ASINs, dates, prices, and URLs, and it rejects instruction-like strings such as embedded override directives, exfiltration prompts, or persistence commands [6].

| Declassification stage | Effect |
| --- | --- |
| Candidate extraction | Pulls bounded, typed values from untrusted text into candidate facts. |
| Validation and normalization | Applies type checks, length checks, allow/reject patterns, and instruction detection. |
| Verified promotion | Promotes only accepted values into verified nodes that may influence later control actions. |
| Quarantine | Retains unsafe spans for reference but prevents them from becoming action-safe facts. |

This mechanism is intentionally modular. The strategy interface was designed so later work can benchmark alternative declassification approaches without changing the rest of the runtime.

## Threat model

The implemented benchmark is broader than the earlier toy evaluation, but it remains a bounded experimental threat model rather than a universal security claim.

| Threat dimension | Covered in the benchmark | Notes |
| --- | --- | --- |
| Direct embedded override instructions | Yes | Example: markdown email injection. |
| Obfuscated or paraphrased instructions | Yes | Example: code-block and paraphrased override scenarios. |
| HTML or markdown channel attacks | Yes | Example: hostile HTML retrieval and markdown injection scenarios. |
| Multi-step persistence poisoning | Yes | Example: two-turn memory-poisoning follow-up. |
| Tool-output poisoning | Yes | Includes JSON tool-output poisoning and authenticated-tool paraphrase poisoning. |
| Attacks on declassification itself | Yes | Field-level declassification-bypass scenario. |
| Malicious user intent | No | Out of scope; the user is treated as an authorized principal. |
| Compromised system prompt | No | Out of scope for both theorem and implementation. |
| Arbitrary production tool ecosystems | No | The benchmark uses a small local scenario library rather than a production-scale agent stack. |

## Evaluation methodology

The repository’s headline empirical evidence is now the end-to-end benchmark under `noninterference_eval`. Each benchmark run evaluates attacked and clean versions of the same scenario across the three system variants. The runner saves per-seed attacked artifacts, clean artifacts, scenario tables, variant summaries, an aggregate mean-and-standard-deviation summary, and a publication-ready table [3] [5].

The reported metrics include both **security** and **usefulness**. Security metrics include attack success rate, unsafe action rate, security failure rate, blocked malicious action rate, indirect prompt influence rate, plan corruption, persistence-poisoning success, and unsafe final-answer rate. Utility metrics include attacked and clean task completion, task-token match, benign-action preservation, overblocking, legitimate tool use, verified-memory success, and runtime overhead.

## Reproducibility

The repository now includes one-command entry points for smoke runs, benchmark runs, aggregation, and table generation. Each major run saves its own configuration snapshot and raw outputs.

| Task | Command |
| --- | --- |
| Check core evaluation dependencies | `make check-eval-deps` |
| Run end-to-end test suite | `make test-e2e` |
| Run smoke benchmark | `make e2e-smoke` |
| Run full benchmark | `make e2e-benchmark` |
| Aggregate existing benchmark runs | `make e2e-aggregate RUN_DIRS="path/to/run1 path/to/run2"` |
| Regenerate tables from existing benchmark runs | `make e2e-tables RUN_DIRS="path/to/run1 path/to/run2"` |

A full benchmark can also be launched directly:

```bash
cd noninterference_eval
python3.11 src/run_e2e_eval.py benchmark --output-dir ./results/e2e
```

The canonical saved run produced during this repository upgrade is:

```text
noninterference_eval/results/e2e/benchmark_20260327T025110Z/
```

That directory contains:

| Artifact | Purpose |
| --- | --- |
| `config_snapshot.yaml` | Exact seeds, variants, scenario IDs, timestamp, and output directory. |
| `seed_*/attacked_artifacts.json` | Raw attacked-run traces per seed. |
| `seed_*/clean_artifacts.json` | Raw clean-run traces per seed. |
| `seed_*/scenario_table.json` | Per-scenario metric table for each seed. |
| `aggregate_seed_summary.json` | Mean and standard deviation across seeds for the headline metrics. |
| `publication_table.json` | Preformatted publication-facing table. |
| `summary.md` | Human-readable run summary. |

## Repository structure

| Path | Purpose |
| --- | --- |
| `paper/` | Formal theorem, definitions, discussion, and empirical write-up. |
| `noninterference_eval/src/` | Runtime, declassification, benchmark, metrics, and CLI code. |
| `noninterference_eval/tests/` | Unit and end-to-end tests for taint flow, guard logic, declassification, and benchmark metrics. |
| `noninterference_eval/results/` | Saved benchmark outputs and reports. |
| `EMPIRICAL_GAP_CLOSURE.md` | Audit-style summary of what was weak before, what changed, what is now supported, and what remains future work. |

## What is now supported

The repository now supports the following claims.

| Claim | Supported? | Evidence |
| --- | --- | --- |
| The theorem formalizes a noninterference-style architecture for taint-aware control decisions. | Yes | `paper/definitions.tex`, `paper/proof.tex`. |
| The repository implements a contaminated-input guarded runtime rather than evaluating guarded behavior by substituting clean inputs. | Yes | `noninterference_eval/src/guarded_runtime.py`. |
| The repository implements at least one concrete declassification mechanism. | Yes | `noninterference_eval/src/declassification.py`. |
| The repository empirically compares unguarded, guarded-without-declassification, and guarded-with-declassification systems on security and utility. | Yes | `noninterference_eval/src/e2e_benchmark.py`, saved results under `noninterference_eval/results/e2e/benchmark_20260327T025110Z/`. |
| Structured declassification improves utility relative to guarded-no-declassification on the saved benchmark while preserving zero unsafe actions. | Yes | `aggregate_seed_summary.json`, `publication_table.json`. |
| The implementation proves security for arbitrary production agents or arbitrary authenticated tools. | No | Outside the scope of the current implementation. |

## Limitations

This repository is stronger than the initial prototype, but it remains a research artifact with bounded evidence.

The end-to-end benchmark is intentionally small, with seven scenario families rather than thousands of real-world traces. The runtime is model-agnostic and locally simulated rather than integrated into a production agent framework. The overhead numbers are artifact-level timings from a lightweight benchmark harness and should not be interpreted as deployment-grade latency estimates. Authenticated tool outputs remain assumption-sensitive, because a compromised trusted channel can still defeat policy if the architecture mislabels it. Finally, the formal theorem still depends on the verifier and labeling invariants being implemented correctly; the code improves empirical credibility, but it does not remove the need for careful verification and future adversarial testing.

## References

[1]: https://ieeexplore.ieee.org/document/4568387 "Security Policies and Security Models"
[2]: https://arxiv.org/abs/2302.12173 "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection"
[3]: ./noninterference_eval/results/e2e/benchmark_20260327T025110Z/config_snapshot.yaml "Saved benchmark configuration snapshot"
[4]: ./noninterference_eval/results/e2e/benchmark_20260327T025110Z/seed_0/scenario_table.json "Saved per-scenario benchmark metrics"
[5]: ./noninterference_eval/results/e2e/benchmark_20260327T025110Z/aggregate_seed_summary.json "Saved aggregate benchmark summary"
[6]: ./noninterference_eval/src/declassification.py "Structured declassification implementation"
