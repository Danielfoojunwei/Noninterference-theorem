# Internal Audit and Implementation Plan

## Purpose

This note records the initial repository audit requested before major edits. It distinguishes what is currently **formally specified**, what is **actually implemented**, and what is **empirically supported by saved artifacts**.

The central audit conclusion is that the repository already contains a coherent **formal theorem and system specification**, but its current empirical pipeline is **not yet a credible end-to-end guarded-agent evaluation**. The present code mainly demonstrates that if tainted content is removed before the model sees it, then deterministic outputs remain unchanged. That finding is valid as a theorem-alignment check, but it is not yet sufficient evidence for practical agentic-AI security claims.

## Repository Areas Audited

The initial audit covered the following files in detail:

| Area | Files examined | Main audit purpose |
|---|---|---|
| Top-level documentation | `README.md`, `Makefile` | Check headline claims, reproducibility, and publication readiness |
| Paper | `paper/main.tex`, `paper/definitions.tex` | Separate formal claims from empirical claims; inspect declassification status |
| Evaluation docs | `noninterference_eval/README.md` | Check whether the empirical pipeline description matches the real code |
| Core implementation | `noninterference_eval/src/agent.py`, `noninterference_eval/src/llm_agent.py` | Inspect guarded execution semantics and whether contaminated execution is real |
| Evaluation harness | `noninterference_eval/src/evaluator.py`, `noninterference_eval/src/run_llm_eval.py` | Identify circular evaluation paths, metric weaknesses, and clean-input shortcuts |
| Saved results | `noninterference_eval/results/llm_report_20260211_000906.txt` | Verify what was actually executed and what conclusions are justified |

## Executive Audit Summary

The repository currently supports a strong statement of the following form:

> If a deterministic enforcement architecture excludes tainted content from the action-selection input, then action outputs are noninterfering with respect to variations in untrusted input under the theorem’s assumptions.

However, the repository does **not yet** support the stronger empirical claim that a **real guarded agent operating on contaminated inputs** can safely use untrusted content, preserve utility, and resist indirect prompt injection end-to-end.

The main reasons are summarized below.

| Audit question | Current status | Main issue |
|---|---|---|
| Is guarded execution evaluated on the same contaminated context as baseline? | No | Guarded evaluation re-runs on taint-stripped clean input |
| Is there a real guard around multiple control points? | No | Current implementation is mostly tool-selection filtering only |
| Is declassification implemented in code? | No | `VerifiedFact` is specified in the paper, not implemented as an executable pathway |
| Is utility measured end-to-end? | No | Current metrics are proxy metrics only, not task completion under defense |
| Are saved artifacts sufficient for publication-grade reproducibility? | No | Results are limited; raw per-run artifacts, configs, seeds, and one-command pipelines are missing |

## Detailed Audit Findings

## 1. Claims that currently depend on shortcuts, oracles, or unrealistic assumptions

The repository contains several empirical claims that rely on assumptions stronger than a realistic guarded-agent deployment.

### 1.1 Guarded noninterference is measured by duplicated clean-input inference

In `noninterference_eval/src/run_llm_eval.py`, the guarded InjecAgent path runs `g_clean` on `case.tool_response_clean` and then runs `g_inj` on `case.tool_response_clean` again, with the comment that this represents taint-stripped input. The BIPIA guarded path does the same pattern by passing `case.context` to both guarded runs.

This means the reported guarded 0% influence and 100% noninterference are obtained by comparing **clean input against the same clean input again**, rather than by executing a guard inside a contaminated run. The result is therefore a theorem-consistency check, not an end-to-end empirical demonstration.

### 1.2 The LLM guarded path removes untrusted content before model exposure

In `noninterference_eval/src/llm_agent.py`, the `LLMGuardedAgent` receives `tool_response_clean` or `context_clean` directly. This architecture does not model a real agent that must ingest contaminated content, label it, and then decide what parts may influence behavior. Instead, the evaluation harness externally supplies already-clean input.

That is acceptable for illustrating the theorem’s architectural idea, but it is not a realistic guarded-agent execution path.

### 1.3 The rule-based evaluation also contains the same shortcut

In `noninterference_eval/src/evaluator.py`, guarded InjecAgent execution is run on injected content, but noninterference is still checked by separately running a second clean guarded pass and comparing outputs. The BIPIA rule-based path is even more stylized: the baseline deliberately parrots the injection while the guarded variant simply ignores it.

This makes the non-LLM path useful as a unit-level sanity check, but too stylized to serve as headline empirical evidence.

### 1.4 Saved report confirms the guarded result is expected by construction

The saved artifact `noninterference_eval/results/llm_report_20260211_000906.txt` explicitly states that 0% guarded influence is expected because the guarded agent receives identical input regardless of injection under deterministic decoding. This is an honest admission in the report, but it also confirms that the current empirical result is not testing a realistic contaminated execution path.

### 1.5 Utility claims rely on weak proxy metrics

The README and report discuss utility using guarded tool accuracy and token-overlap relevance. These are limited proxies. The report itself shows very low usefulness in practice, including **12.0% guarded tool accuracy** on InjecAgent and **0.083 guarded relevance** on BIPIA. That means the current implementation does not yet show that the defense preserves meaningful task utility.

### 1.6 Contradictory security framing remains in the current report artifacts

The saved report shows **guarded ASR = 82.0%** on InjecAgent while also claiming 0% guarded influence and 100% noninterference. This is possible only because the current metrics mix incompatible notions of success and because the model output quality is poor. Publication-facing documentation should not present such figures without strong clarification.

## 2. Places where the guarded model is effectively re-run on clean inputs instead of actually being guarded in contaminated context

The following locations are the clearest instances of the forbidden shortcut.

| File | Location | Audit finding |
|---|---|---|
| `noninterference_eval/src/run_llm_eval.py` | InjecAgent evaluation | `g_inj` is run on `case.tool_response_clean`, not contaminated input |
| `noninterference_eval/src/run_llm_eval.py` | BIPIA evaluation | `g_inj` is run on `case.context`, same as `g_clean` |
| `noninterference_eval/src/llm_agent.py` | Guarded agent API | Guarded methods accept pre-cleaned inputs instead of contaminated input plus labels/guard logic |
| `noninterference_eval/src/evaluator.py` | InjecAgent evaluator | Noninterference is checked by comparing injected guarded output against a separate clean guarded run |
| `noninterference_eval/src/evaluator.py` | BIPIA evaluator | Guarded path is a simplified ignore-the-injection behavior, not realistic guarded execution |

These are the core places that must be redesigned before new empirical claims are made.

## 3. Places where declassification is mentioned but not implemented

The paper specifies a meaningful declassification concept, but the repository does not yet implement it as an executable, benchmarked subsystem.

### 3.1 Declassification exists in the paper specification only

In `paper/definitions.tex`, the repository defines `VerifiedFact` promotion, including cross-reference checking, schema validation, and allowlist matching. The same section explicitly remarks that the declassification pathway is **not empirically evaluated**.

### 3.2 Code contains node types but no real `VerifiedFact` promotion pipeline

In `noninterference_eval/src/agent.py`, the node types include `CandidateFact` and `VerifiedFact`, but the agent logic does not implement a declassification module that parses tainted content, validates fields, promotes safe fields, and then routes only promoted content into downstream control decisions.

### 3.3 No benchmark compares multiple declassification strategies

There is currently no declassification interface, no plug-in strategy abstraction, no tests for false accepts or false rejects, and no evaluation comparing guarded-without-declassification versus guarded-with-declassification.

## 4. Places where task utility is asserted but not measured sufficiently

Several parts of the repository correctly acknowledge the utility gap, but the documentation still overstates what is known empirically.

### 4.1 README contribution framing is too strong relative to the evidence

The README currently lists a “Utility evaluation” contribution and states that the defense does not degrade utility relative to the undefended clean baseline. While technically true for one narrow proxy metric on a weak surrogate, this wording risks implying practical usefulness that the saved artifacts do not support.

### 4.2 The paper explicitly acknowledges the gap, but the evaluation still lacks end-to-end utility measurement

`paper/main.tex` already admits that task completion under defense and declassification evaluation remain open. That honesty should be preserved. However, the empirical section still does not measure:

| Missing utility measure | Why it matters |
|---|---|
| Task completion rate | Determines whether the defended system can still do useful work |
| Benign-action preservation rate | Measures overblocking of legitimate actions |
| Overblocking rate | Quantifies how often safety blocks benign behavior |
| Memory usefulness under defense | Needed for multi-step agent workflows |
| Latency and overhead by control point | Necessary for realistic deployability assessment |

### 4.3 Current proxy metrics are insufficient for the new target claim

Tool accuracy and Jaccard relevance are not enough to support a publication-grade claim that guarded agents remain useful while safe. End-to-end tasks with judged success criteria are required.

## 5. Documentation claims that should be downgraded, clarified, or removed

The following claim categories should be revised before publication-quality positioning.

### 5.1 README headline language

The README currently presents the system as a “system-level security guarantee with empirical validation on canonical benchmarks.” That should be split into two claims:

1. a **formal theorem** about an enforcement architecture under explicit assumptions; and
2. a **limited surrogate empirical check** showing that a taint-stripped decision-point implementation behaves consistently with the theorem.

It should not read like a solved end-to-end agent security result.

### 5.2 README abstract result language

The abstract currently reports 0% guarded influence and 100% noninterference, then discusses equal tool accuracy relative to the clean baseline. This needs clarification that:

- the guarded result is expected by construction under duplicated clean inputs;
- the guarded pathway currently strips untrusted content rather than safely using it;
- utility remains poor overall;
- no real declassification mechanism is yet evaluated.

### 5.3 `noninterference_eval/README.md`

This file currently says the framework uses “industry-standard models to validate the theorem’s claims.” That wording should be softened because the implemented evaluation is a mix of toy agents and a surrogate FLAN-T5 decision-point model, not a realistic tool-calling agent benchmark.

### 5.4 Paper empirical framing

The paper should keep the theorem and the scope discussion, but the empirical section should explicitly state that the existing pipeline is a **surrogate theorem-alignment study** and not the main evidence for real guarded-agent robustness. Once the new end-to-end harness exists, the paper can elevate stronger empirical claims.

## 6. Missing implementation capabilities relative to the target artifact

The repository does not yet implement the following capabilities requested for a credible empirical security artifact.

| Capability | Current status | Required change |
|---|---|---|
| Contaminated-input guarded agent loop | Missing | Agent must ingest the same contaminated input as baseline and guard decisions in-process |
| Multiple control points | Missing | Add final-answer, tool-call, memory-write, and plan-update guards |
| Explicit audit log | Missing | Log allowed/blocked actions with reasons and provenance |
| Real declassification module | Missing | Add typed extraction / policy-checked summarization / verifier-based promotion |
| Guarded-with-declassification benchmark arm | Missing | Compare utility/security tradeoff against guarded-without-declassification |
| Multi-step memory-poisoning scenarios | Missing | Add persistent-state tasks and two-step attacks |
| Reproducibility runner | Weak | Add config-driven smoke/full/aggregate/table commands |
| Statistical rigor | Missing | Add seeds, aggregation, raw logs, and mean/std reporting |

## 7. Concrete implementation plan

The implementation work should proceed in the following order.

### 7.1 Workstream A — Replace the clean-input shortcut with real guarded execution

Build a stateful agent loop that accepts the same contaminated context as the baseline. The loop should preserve labeled state rather than passing pre-cleaned strings into a guarded model API.

Key changes:

| Task | Intended result |
|---|---|
| Introduce explicit input channels | Trusted user/system input and untrusted web/tool/document content enter through separate labeled records |
| Add provenance-bearing memory/state objects | Taint and provenance flow through state rather than disappearing at prompt-construction time |
| Add action-time guard | Tool calls, final answers, memory writes, and plan updates must pass policy checks |
| Add structured action log | Every allowed or blocked action is saved with reason, source labels, and policy rule |

### 7.2 Workstream B — Implement real declassification

Implement at least one concrete declassification strategy that lets the agent use untrusted content without giving it free control over behavior.

Initial strategy to implement first:

> **Structured extraction into a typed schema with field-level policy checks.**

Example behavior:

- untrusted email/web text is parsed into typed candidate fields;
- only allowed fields such as `product_name`, `date`, `price`, `sender_address`, or `summary_facts` may flow forward;
- instruction-like fields are either dropped or retained as quarantined reference text;
- only validated fields may influence downstream control decisions.

Recommended abstraction:

| Interface | Purpose |
|---|---|
| `DeclassificationStrategy` | Common base interface for future strategies |
| `SchemaExtractionStrategy` | First concrete implementation |
| `PolicyCheckedSummaryStrategy` | Second optional implementation for benchmarking |
| `VerificationResult` | Standard record with accepted fields, rejected spans, and reasons |

### 7.3 Workstream C — Build a real evaluation harness

Create an end-to-end evaluation with three main systems:

1. **Unguarded baseline agent**
2. **Guarded agent without declassification**
3. **Guarded agent with declassification**

Recommended benchmark task families:

| Task family | Why it matters |
|---|---|
| Email summarization/reply planning | Uses untrusted content in a realistic workflow |
| Retrieval/browsing with hostile snippets | Captures common indirect injection channels |
| Tool-use planning | Tests whether injections can redirect actions |
| Memory persistence tasks | Tests durable poisoning across turns |

### 7.4 Workstream D — Add metrics that jointly measure security and utility

Security metrics should include attack success rate, unsafe action rate, blocked malicious action rate, indirect prompt influence rate, and persistence-poisoning success rate for multi-step tasks.

Utility metrics should include task completion rate, exact-match or judged correctness, benign-action preservation rate, overblocking rate, and latency overhead.

All major runs should save both per-case raw traces and aggregate metrics.

### 7.5 Workstream E — Improve reproducibility and reporting

The repository should expose one-command flows for smoke tests, full runs, aggregation, and table generation. Each run should snapshot:

- configuration;
- seed;
- selected cases;
- raw traces;
- aggregated metrics;
- generated tables.

The current `Makefile` is paper-only and must be expanded.

## 8. Proposed immediate coding order

The next implementation steps should be executed in this order:

| Step | Immediate next action |
|---|---|
| 1 | Refactor agent abstractions so contaminated input enters baseline and guarded systems identically |
| 2 | Add provenance-aware state, guard decisions, and action logging |
| 3 | Implement one concrete declassification strategy with tests |
| 4 | Add realistic end-to-end tasks and multi-step attack scenarios |
| 5 | Add metrics, aggregation, and reproducibility runners |
| 6 | Run smoke experiments first, then fuller evaluation |
| 7 | Rewrite README, paper, and produce `EMPIRICAL_GAP_CLOSURE.md` from actual artifacts |

## 9. Current truthful claim boundary

At this point in the audit, the strongest claims that appear supportable are the following.

> The repository contains a formal noninterference theorem for a taint-filtering enforcement architecture and a limited empirical surrogate evaluation showing that when tainted content is removed before deterministic model inference, guarded outputs remain invariant across untrusted-input variations.

The following stronger claims are **not yet supported** and should not remain in final publication-facing text unless new experiments are added:

- that a real tool-calling guarded agent has been evaluated end-to-end;
- that declassification has been implemented and validated;
- that task utility is preserved in realistic workflows;
- that the system delivers practical real-world noninterference beyond the theorem’s assumptions.

## 10. Next development objective

The immediate implementation objective is therefore:

> Replace the current clean-input guarded shortcut with a genuine contaminated-input guarded agent loop, then add explicit declassification and end-to-end security/utility evaluation before rewriting the paper and README.

This note should be updated as implementation proceeds and converted later into the user-facing `EMPIRICAL_GAP_CLOSURE.md` deliverable.
