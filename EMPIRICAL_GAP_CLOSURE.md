# Empirical Gap Closure

## Purpose

This document explains how the repository changed from a promising formal-security prototype into a more credible **empirical research artifact**. It records four things: what was previously weak or circular, what was fixed in code and evaluation, what claims are now genuinely supported by executable artifacts, and what still remains future work.

The guiding rule throughout this upgrade was simple:

> **Do not preserve claims that are not backed by executable experiments and saved artifacts.**

Accordingly, the repository’s current positioning is intentionally narrower than a broad production-security claim. The result is a smaller but much more defensible artifact.

## Executive Summary

Before this upgrade, the repository’s formal contribution was coherent, but its empirical layer was not yet strong enough to support publication-grade end-to-end security claims. The main problem was that the guarded system was often evaluated by effectively re-running clean inputs or by using stylized surrogate behavior instead of operating as a genuine guarded agent on contaminated context.

After this upgrade, the repository now contains a real contaminated-input guarded runtime, an explicit declassification mechanism, an end-to-end benchmark comparing baseline versus guarded variants on both security and utility, reproducible multi-seed experiment runners, saved result artifacts, and documentation aligned to those artifacts.

## What Was Previously Weak or Circular

The original empirical weaknesses fell into five categories.

| Gap | Previous problem | Why it mattered |
| --- | --- | --- |
| Shortcut-based guarded evaluation | Guarded runs were evaluated on taint-stripped or duplicated clean inputs rather than on the same contaminated context as the baseline. | This made the guarded noninterference result partly tautological rather than an end-to-end demonstration. |
| No real guarded agent loop | There was no implemented runtime that carried taint/provenance through explicit state and enforced policy at multiple control points. | The repository lacked a credible execution model for how guarding would work in an actual agent loop. |
| Declassification existed only in theory | The paper described verified-fact promotion, but the code did not implement a concrete declassification pipeline. | The hardest practical question—how useful work survives under guarding—was left unanswered in implementation. |
| Utility was asserted more than measured | Proxy metrics existed, but task completion, benign-action preservation, overblocking, and memory-poisoning outcomes were not properly evaluated end-to-end. | A safe but useless system would still look good under narrow security-only metrics. |
| Reproducibility was incomplete | Raw per-run artifacts, config snapshots, seed aggregation, and one-command benchmark entry points were missing or insufficient. | Even valid results were harder to inspect, reproduce, or cite. |

## What Was Implemented to Close the Gaps

The repository now includes concrete implementations for each of the critical missing pieces.

### 1. Real contaminated-input guarded execution

A new stateful runtime was added in `noninterference_eval/src/guarded_runtime.py`.

This runtime explicitly ingests trusted and untrusted inputs, tracks provenance and taint in node state, and enforces policy at the required control points:

| Control point | Current behavior |
| --- | --- |
| Tool call / action execution | Blocks tainted or instruction-bearing control actions unless policy and verified evidence permit them. |
| Plan update | Blocks tainted or instruction-bearing plan modifications. |
| Memory write | Quarantines tainted writes instead of promoting them into verified memory. |
| Final answer | Repairs tainted or instruction-bearing final output to a safe template when required. |

Every attempted action is logged with whether it was allowed, what evidence supported it, which evidence was tainted versus verified, and what blocking rules applied.

### 2. Concrete declassification mechanism

A real declassification module was added in `noninterference_eval/src/declassification.py`.

The first implemented strategy is **structured schema extraction**. It works by extracting typed candidate fields from untrusted text, validating them against explicit schema and policy constraints, rejecting instructional content, and promoting only accepted values into verified facts.

| Stage | Implemented effect |
| --- | --- |
| Candidate extraction | Pulls bounded typed values from untrusted content |
| Validation | Applies type checks, size limits, allow/reject rules, and instruction detection |
| Verified promotion | Converts accepted candidates into verified facts that later actions may depend on |
| Quarantine | Preserves unsafe spans for reference without allowing them to control behavior |

This closes the largest practical gap in the original repository: the code now contains an actual pathway by which useful information from contaminated inputs can be used safely.

### 3. End-to-end benchmark harness

A new end-to-end benchmark harness was added in `noninterference_eval/src/e2e_benchmark.py`.

The benchmark compares three system variants on the same attacked and clean tasks:

| Variant | Purpose |
| --- | --- |
| `baseline` | Measure the vulnerability of an unguarded contaminated-input agent |
| `guarded_no_declass` | Measure the security benefit and utility cost of strict guarding |
| `guarded_structured_declass` | Measure whether concrete declassification restores utility safely |

The benchmark uses realistic local task patterns rather than only abstract decision-point matching.

### 4. Expanded threat coverage

The new benchmark includes seven attacked scenario families and their clean counterparts:

| Scenario ID | Threat family |
| --- | --- |
| `email_order_lookup_markdown` | Markdown embedded instruction override |
| `hostile_web_search_html` | Hostile HTML retrieval poisoning |
| `memory_poisoning_followup` | Two-step memory poisoning |
| `tool_output_json_poisoning` | Tool-output poisoning via JSON plus adversarial note |
| `declassification_bypass_attempt` | Attack on the declassification pathway |
| `markdown_codeblock_obfuscation` | Obfuscated code-block embedded override |
| `authenticated_tool_paraphrase_poisoning` | Poisoning through an authenticated tool channel with paraphrased override |

This threat model is still bounded, but it is materially broader and more agent-realistic than the earlier shortcut-driven evaluation.

### 5. Proper security and utility metrics

A dedicated metrics module was added in `noninterference_eval/src/e2e_metrics.py`.

It reports both security and usefulness.

| Metric family | Included measures |
| --- | --- |
| Security | Attack success, unsafe action, security failure, blocked malicious action, indirect prompt influence, plan corruption, persistence poisoning, unsafe final answer |
| Utility | Attacked and clean task completion, task-token match, benign-action preservation, legitimate tool use, overblocking |
| Provenance | Verified-memory success, verified-action usage |
| Operational | Runtime, attempted actions, executed tools, overhead versus baseline |
| Invariance | Tool, memory, answer, and task-outcome invariance |

This is important because the repository can now describe the **security–utility tradeoff** directly rather than implying it through proxy scores.

### 6. Reproducibility and saved artifacts

A reproducible runner was added in `noninterference_eval/src/run_e2e_eval.py`, and the repository `Makefile` now includes one-command targets for smoke runs, full benchmarks, aggregation, tables, dependency checks, and testing.

Each major run now saves:

| Artifact | Purpose |
| --- | --- |
| `config_snapshot.yaml` | Exact seeds, variants, scenario set, timestamp, and output directory |
| `seed_*/attacked_artifacts.json` | Raw attacked-run traces |
| `seed_*/clean_artifacts.json` | Raw clean-run traces |
| `seed_*/scenario_table.json` | Per-scenario outcome table |
| `seed_*/metrics_bundle.json` | Structured metric bundle |
| `aggregate_seed_summary.json` | Mean and standard deviation across seeds |
| `publication_table.json` | Compact publication-ready table |
| `summary.md` | Human-readable run summary |

This closes the reproducibility gap that previously limited the repository’s credibility.

## What Was Actually Run

The canonical full run produced during this upgrade is stored at:

```text
noninterference_eval/results/e2e/benchmark_20260327T025110Z/
```

According to the saved configuration snapshot, the benchmark evaluated three variants across seven attacked scenarios and seven clean counterparts with seeds `0`, `1`, and `2`.

## What Is Now Empirically Supported

The repository can now support a much cleaner set of claims.

### Headline empirical findings

Using the saved aggregate results, the following statements are now backed by executable experiments and stored artifacts.

| Claim | Supported? | Evidence |
| --- | --- | --- |
| The unguarded baseline is vulnerable across the benchmarked contaminated-input scenarios. | Yes | `aggregate_seed_summary.json` shows `1.000 ± 0.000` attack success and unsafe action rate for `baseline`. |
| Strict guarding without declassification eliminates unsafe actions but sacrifices large amounts of useful behavior. | Yes | `guarded_no_declass` has `0.000 ± 0.000` unsafe action rate but only `0.286 ± 0.000` attacked and clean task completion. |
| Structured declassification restores major lost utility while preserving safe behavior on this benchmark. | Yes | `guarded_structured_declass` reaches `0.857 ± 0.000` attacked task completion with `0.000 ± 0.000` attack success and unsafe action rate. |
| The new benchmark measures both security and utility rather than relying only on proxy influence metrics. | Yes | The metrics and saved artifacts include attack success, unsafe action, task completion, token match, preservation, overblocking, and overhead. |
| The repository now includes a real guarded runtime operating on contaminated inputs. | Yes | Implemented in `guarded_runtime.py` and exercised through the new end-to-end harness. |
| The repository now includes a real declassification mechanism used in the main benchmark. | Yes | Implemented in `declassification.py` and used by the `guarded_structured_declass` benchmark arm. |

### Core numbers from the saved benchmark

| Variant | Attack Success Rate | Unsafe Action Rate | Security Failure Rate | Attacked Task Completion | Clean Task Completion | Benign-Action Preservation |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | `1.000 ± 0.000` | `1.000 ± 0.000` | `1.000 ± 0.000` | `0.286 ± 0.000` | `0.857 ± 0.000` | `0.143 ± 0.000` |
| guarded_no_declass | `0.000 ± 0.000` | `0.000 ± 0.000` | `0.429 ± 0.000` | `0.286 ± 0.000` | `0.286 ± 0.000` | `0.857 ± 0.000` |
| guarded_structured_declass | `0.000 ± 0.000` | `0.000 ± 0.000` | `0.000 ± 0.000` | `0.857 ± 0.000` | `0.857 ± 0.000` | `1.000 ± 0.000` |

These numbers support the central practical conclusion of the upgraded repository:

> **On the implemented benchmark, structured declassification closes the utility gap that remains under strict guarding, without reintroducing unsafe actions.**

That is a real and useful result, but it is intentionally scoped to the implemented benchmark.

## What Claims Were Downgraded or Clarified

The documentation was rewritten to make the supported scope explicit.

| Documentation area | Change made |
| --- | --- |
| `README.md` | Now clearly separates the formal theorem from the empirical prototype, includes a current empirical status section, and removes solved-in-general framing. |
| `noninterference_eval/README.md` | Now describes the implemented runtime, benchmark variants, metrics, saved artifacts, and limits instead of overstating external-benchmark validation. |
| `paper/main.tex` | Now distinguishes formal proof from empirical evidence, reports the saved end-to-end benchmark rather than earlier shortcut-style surrogate claims, and frames the result as bounded rather than universal. |

This was an essential part of the gap closure. An honest narrower claim is better than an ambitious unsupported one.

## What Still Remains Future Work

The repository is now materially stronger, but it is not complete in the absolute sense.

| Remaining limitation | Why it still matters |
| --- | --- |
| Small scenario library | Seven scenario families are useful for credibility, but they are not a substitute for large-scale heterogeneous evaluation. |
| Local benchmark runtime | The implementation is a reusable research runtime, not yet an integration into major real-world agent frameworks. |
| Limited declassification family | Only one concrete declassification strategy is implemented so far. More strategies should be benchmarked and stress-tested. |
| Authenticated-channel assumptions | Trust labeling for tool outputs remains security-critical and can fail if the architecture misclassifies channels. |
| Lightweight runtime overhead numbers | Current timing results are valid for the harness but should not be treated as production-latency guarantees. |
| No claim of universal noninterference in production | The theorem is conditional, and the implementation demonstrates a bounded prototype rather than a solved deployment story. |

## Final Supported-Claims Statement

At the end of this upgrade, the repository now truly supports the following statement:

> This repository provides a formal noninterference theorem for taint-aware agent control, an implemented contaminated-input guarded runtime, a concrete structured declassification mechanism, and a reproducible end-to-end benchmark showing that, on the saved seven-scenario benchmark, structured declassification can preserve safety while recovering task utility relative to strict guarding alone.

It does **not** support the stronger statement that indirect prompt injection is fully solved for arbitrary agents, arbitrary tools, or arbitrary real-world deployments.

## Reproduction Commands

From the repository root, the main commands are:

```bash
make check-eval-deps
make test-e2e
make e2e-smoke
make e2e-benchmark
```

The canonical benchmark artifacts cited in the documentation are located under:

```text
noninterference_eval/results/e2e/benchmark_20260327T025110Z/
```
