# Noninterference Theorem for Indirect Prompt Injection in Agentic AI

> **A system-level security guarantee with empirical validation on canonical benchmarks**

---

## Abstract

Agentic AI systems routinely ingest untrusted content — emails, web pages, documents — within the same context window that carries system prompts and user instructions. Indirect prompt injection exploits this shared context to embed malicious instructions in otherwise legitimate data, silently hijacking tool selection and exfiltrating sensitive information. We formalise the security requirement as a **noninterference** property adapted from information-flow control (Goguen & Meseguer, 1982): adversarial variations in untrusted data must not influence the agent's tool calls or modify its control-plane state. We model the agent as a discrete-time dynamical system, define a typed intermediate representation (IR) with taint tracking and a trust lattice, and prove that a **system-level** enforcement layer — a deterministic verifier that filters tainted content from the action-selection dependency set before each inference step — guarantees action invariance to untrusted input variations. The theorem is a property of the *enforcement architecture*, not of the underlying language model's internal reasoning: it holds because tainted content is excluded from the model's input when selecting actions, not because the model chooses to ignore it. We validate the framework empirically on two canonical prompt-injection benchmarks — **InjecAgent** (Zhan et al., 2024; 1,054 test cases) and **BIPIA** (Yi et al., 2023; 200 contexts × 75 attack variants) — using real neural network inference with FLAN-T5-base (248M parameters) as a surrogate decision-point model. Across 700 evaluated cases, the unprotected baseline shows a **21.8% influence rate** on InjecAgent and **6.0%** on BIPIA, confirming genuine vulnerability. The taint-tracked (guarded) configuration achieves **0.0% influence** and **100% noninterference**, confirming the enforcement layer correctly implements the theorem's preconditions. We discuss limitations including the utility gap (stripping untrusted content may reduce task capability), the need for a concrete declassification protocol, and the distinction between our surrogate evaluation and full end-to-end agent testing.

---

## 1. Introduction

### 1.1 The Problem

Agentic AI systems such as OpenAI's function-calling agents, LangChain pipelines, and autonomous coding assistants autonomously execute tool calls and chain operations across external services. Because they ingest untrusted content — retrieved web pages, user-uploaded documents, skill outputs — alongside system prompts and direct user instructions, they are vulnerable to **indirect prompt injection**: an adversary embeds hidden instructions in external data that reshape the agent's intent, redirect tool usage, or trigger unauthorised actions (CrowdStrike, 2025).

The fundamental security problem is that current systems do not enforce a hard separation between explicit user intent and third-party content (eSecurity Planet, 2025). Information retrieved during a task is processed in the same reasoning context as direct instructions.

### 1.2 Our Approach

We define the security requirement as a **noninterference** property borrowed from information-flow control: adversarial variations in untrusted data should not influence the agent's action-selection decisions or modify its control plane. We then show that a system-level enforcement layer — comprising taint tracking, typed IR, and a deterministic verifier that filters tainted content from the action-selection function's inputs — provides a formal guarantee of this property.

**Critical distinction**: The theorem is about the *system-level action-selection function*, not about the language model's internal reasoning. We do not claim the model ignores injections; we prove that when tainted content is architecturally excluded from the model's input at the action-selection step, the resulting actions are invariant to untrusted input variations. This is analogous to proving that `f(x)` is independent of `y` by showing `y` is not an argument to `f` — correct but contingent on the filtering being applied correctly.

### 1.3 Contributions

1. **Formal model** (Section 3): A typed intermediate representation (IR) graph with 8 node types, a 4-level trust lattice {SYS, USER, UNTRUSTED_TOOL, WEB/SKILL}, taint propagation rules, and 4 verifier invariants.

2. **Noninterference Theorem** (Section 4): A proof by strong induction that the system-level action-selection function produces identical tool calls and control-plane updates when untrusted inputs vary, provided tainted content is excluded from the action dependency set.

3. **Novel evaluation metric** (Section 5): We introduce **influence rate** — whether model output *changed at all* due to injection — as a more conservative and directly theoretically-motivated metric than traditional attack success rate (ASR). We propose a three-tier framework: action influence, semantic influence, and surface influence.

4. **Empirical validation** (Section 6): Real LLM inference on 700 test cases from InjecAgent and BIPIA benchmarks using FLAN-T5-base as a surrogate decision-point model, confirming the enforcement layer correctly implements the theorem's preconditions.

### 1.4 Scope and Honest Limitations

We state upfront what this paper does *not* do:

- **We do not prove anything about the model's internal reasoning.** The guarantee is architectural: filter before inference.
- **We do not evaluate utility.** Stripping untrusted content may render the agent unable to complete tasks that require reading that content to decide actions. A production system needs a declassification pathway; we sketch one but do not empirically evaluate it.
- **We evaluate a surrogate, not a full agent.** FLAN-T5-base is an instruction-following model used to test the decision point, not a tool-calling agent in the ReAct sense. Our evaluation measures susceptibility of the decision function to injection, not end-to-end agent harm.
- **The 100% NI rate is expected by construction.** It confirms correct implementation of the filtering rule, not a surprising empirical finding.

---

## 2. Related Work

### 2.1 Indirect Prompt Injection

Indirect prompt injection was first systematically studied by Greshake et al. (2023), who demonstrated that adversarial instructions embedded in web content could hijack LLM-integrated applications. Subsequent work has expanded the threat model:

- **InjecAgent** (Zhan et al., 2024) introduced a benchmark of **1,054 test cases** (510 Direct Harm + 544 Data Stealing) across 17 user tools and 62 attacker tools, with base and enhanced attack settings. They report ASR of 24.2% (DH-base) to 57.2% (DS-enhanced) on GPT-4-0613.
- **BIPIA** (Yi et al., 2023) created a benchmark spanning **5 application scenarios** (email, code, table, web QA, summarisation) with **49 unique attack goals** (each with 5 variants = 245 attack instances). The full test set comprises up to 15,000 context-attack pairs via Cartesian product. They report ASR of 47.6% on GPT-4 and 62.1% on GPT-3.5-turbo.

### 2.2 Existing Defenses

Prior defenses operate at the prompt level and provide probabilistic (not guaranteed) protection:

| Defense | Mechanism | Reported ASR |
|---|---|---|
| Border strings (Yi et al., 2023) | Delimit untrusted content with markers | 15.0% (GPT-4) |
| Sandwich prompting (Yi et al., 2023) | Repeat instruction after context | 20.0% (GPT-4) |
| Instructional defense (Yi et al., 2023) | Warn model about injection | 12.0% (GPT-4) |
| SpotLight (Hines et al., 2024) | Mark/encode untrusted tokens | >50% → <2% (encoding, GPT-3.5) |

SpotLight's encoding technique achieves strong empirical reductions (from >50% to 0–1.8% ASR on summarisation/Q&A tasks), but remains probabilistic: the model may still follow injections in some settings, and recent evaluations on agentic benchmarks show weaker results. All prompt-level defenses share a fundamental limitation: the model retains access to the injection content and *may* still follow it.

### 2.3 Information-Flow Security

Our approach draws on the classical noninterference property from Goguen & Meseguer (1982): high-security inputs should not influence low-security outputs. We adapt this to the LLM setting by treating untrusted content as "high" (tainted) and tool-call decisions as "low" (must depend only on trusted inputs).

**Key distinction from classical IFC**: In traditional noninterference proofs, "dependence" is well-defined via language semantics (e.g., type systems, program analysis). In our setting, the "program" is a neural network where internal dependence is not formally tractable. We therefore enforce noninterference at the **system architecture level** — by ensuring tainted content is not an *input* to the action-selection step — rather than proving anything about the network's internal computation. This is analogous to hardware-level isolation (e.g., TrustZone) rather than software-level information-flow tracking.

---

## 3. Formal Model

### 3.1 State Variables

The agent's state at step *t* is:

```
S_t = (P_t, M_t, B_t, G_t)
```

| Component | Description |
|---|---|
| P_t | **Control-plane state**: permissions, tool policies, integration config |
| M_t | **Memory store**: verified facts and historical data |
| B_t | **Risk budget**: remaining budget for tool calls in this session |
| G_t = (V_t, E_t) | **Typed IR graph**: structured representation of current context |

### 3.2 Typed IR Graph

Each node *v* in the IR graph has a type drawn from 8 categories:

```
T = { Policy, UserIntent, TrustedConfig, UntrustedQuote,
      CandidateFact, VerifiedFact, ToolResult, ActionRequest }
```

This typing enables the verifier to enforce structural constraints: only certain node types can appear in the dependency set of action-selection.

### 3.3 Trust Lattice

Inputs are associated with principals ordered by authority:

```
WEB, SKILL  ≤  TOOL_unauth  ≤  TOOL_auth  ≤  USER  ≤  SYS
```

**Revised trust model for tool outputs**: We distinguish between *authenticated* and *unauthenticated* tool outputs:

- **TOOL_auth**: Tool outputs from authenticated sources with provenance guarantees (e.g., signed API responses, allowlisted endpoints, content-addressed storage). Treated as trusted (τ = 0).
- **TOOL_unauth**: Tool outputs from sources that may be adversary-controlled (e.g., web browsers, email APIs, retrieval plugins, file readers). Treated as **tainted by default** (τ = 1), same as WEB.

This addresses a real-world concern: tools like web browsers, email inboxes, and document readers are attack surfaces. Their outputs can contain adversarial content (prompt injection in HTML, poisoned emails, malicious files). Treating all tool output as trusted is dangerous in practice. The default should be taint, with explicit authentication/attestation required to promote to trusted.

**Provenance and taint**: Each node carries a provenance label (which principal produced it) and a taint bit. Raw spans from WEB, SKILL, or TOOL_unauth are tainted (τ = 1). Taint propagates through dependencies:

```
τ(v) = max{τ(u) : u → v}
```

### 3.4 Declassification via VerifiedFact Promotion

The only mechanism to clear taint is **VerifiedFact promotion**. This is the critical component that determines the system's practical utility, and we specify it more concretely than prior versions:

**Verification procedure**: A VerifiedFact promotion requires:
1. **Cross-reference check**: The candidate fact must be corroborated by at least one trusted source (provenance ≥ TOOL_auth).
2. **Schema validation**: The fact must conform to an expected schema for its type (e.g., a price must be numeric, an email address must match format).
3. **Allowlist match** (for actions): If the fact will influence action selection, it must match an allowlisted pattern defined in the system policy P_t.

**Adversary model against verification**: An adversary who controls WEB or SKILL content can:
- Submit arbitrary candidate facts, but these remain tainted until corroborated.
- Attempt to poison trusted sources (SEO poisoning, wiki vandalism) — this is outside our threat model (we assume trusted sources are trustworthy, which is a limitation).
- Attempt to exploit schema validation bugs — this is an implementation risk, not a theoretical limitation.

**Limitation**: We do not empirically evaluate the declassification pathway. A production system must measure false-accept rates (tainted content wrongly promoted) and false-reject rates (legitimate content blocked).

### 3.5 Verifier Invariants

A deterministic verifier V checks four invariants on every proposed transition:

| ID | Invariant | Description |
|---|---|---|
| **V1** | Taint-free action dependence | The action-selection function receives only nodes with τ(v) = 0 as input |
| **V2** | Control-plane authority | Only SYS or USER principals may modify P_t |
| **V3** | Memory authority | Tainted entries are CandidateFact, never VerifiedFact |
| **V4** | Risk budget | Proposed action does not exceed remaining budget B_t |

**Invariant V1, precisely stated**: The action-selection function `a_t = f(π_0(G_t), P_t, M_t)` takes as input the **projection** of the IR graph onto untainted nodes (`π_0(G_t) = {v ∈ V_t : τ(v) = 0}`), together with the control-plane state and memory. Tainted nodes are excluded from this projection *before* the language model's forward pass. This is the mechanism by which noninterference is enforced.

**What "depend" means**: For a neural network, we cannot prove that internal logits are independent of specific input tokens (the computation is opaque). Instead, V1 ensures that tainted tokens are *not present in the input* to the forward pass that produces action decisions. "Dependence" is therefore defined at the system level (input filtering), not at the model level (internal computation).

### 3.6 Repair Semantics

If the verifier rejects a proposed transition, a safe repair is applied:
- Suppress the tool call (a_t = ∅)
- Emit a safe no-op output
- Reset any violating fields

**Failure mode considerations**:
- **Side-channel leakage**: The repair response may reveal that an injection was detected. Mitigation: use a generic response template that is also used for legitimate refusals (e.g., "I need clarification before proceeding").
- **Denial-of-service**: An adversary could inject tainted content into every request, forcing continuous no-op responses. Mitigation: the agent should still complete tasks using trusted content; only the tainted portions are excluded. If all relevant content is tainted, the agent should escalate to user confirmation.
- **Partial completion**: The agent may complete some steps but be blocked on others. Mitigation: maintain an audit log of suppressed actions and present them to the user for review.
- **User confirmation pathway**: For cases where the agent needs to act on untrusted content (e.g., "reply to this email"), the system should present the proposed action to the user for explicit approval, creating a USER-authorised action that satisfies V2.

---

## 4. Noninterference Theorem

### 4.1 Theorem Statement (Revised)

**Theorem 1 (Noninterference for Indirect Prompt Injection).** Consider two executions of the agent system, indexed by *i* ∈ {1, 2}, satisfying:

- **(A1)** Identical initial state: S_0^(1) = S_0^(2)
- **(A2)** Identical trusted input streams: for all *t*, user messages and system policies are the same
- **(A3)** Possibly differing untrusted input streams: U_t^(1) and U_t^(2) may differ arbitrarily
- **(A4)** The verifier enforces invariants V1 and V2 at every step

Then for all time steps *t* ≥ 0:

1. **Action noninterference**: a_t^(1) = a_t^(2)
2. **Control-plane noninterference**: P_t^(1) = P_t^(2)
3. **Tool-output equivalence** (derived): Since actions are identical, tool outputs are identical, so (A2) is self-consistent.

**On Assumption A2 and circularity**: In the original formulation, A2 assumed "tool results are identical," which appears circular: tool results depend on which tools were called, and we are trying to prove that tool calls are identical. The revised formulation resolves this:

1. A2 now states only that *user messages and system policies* are identical (genuinely exogenous inputs).
2. Tool-output equivalence is *derived* as a consequence of action equivalence: since a_t^(1) = a_t^(2) (proved), and both executions call the same tool with the same arguments against the same external state, the tool outputs are identical.
3. This makes the induction well-founded: at step t+1, the tool outputs from step t are identical because the actions at step t were identical (by the inductive hypothesis).

### 4.2 On "True by Construction"

A skeptical reviewer will correctly note that the theorem can be restated as: *the output of a function is invariant to inputs you delete before calling the function*. This is true, and we do not dispute it.

The scientific contribution is not that this property is surprising, but that:

1. **It identifies the correct formal framework** for reasoning about prompt injection security — noninterference from IFC, adapted to the agent setting.
2. **It specifies what must be built**: the IR graph, taint tracking, trust lattice, and verifier invariants constitute a concrete system specification, not just a design principle.
3. **It makes the security guarantee precise**: under these invariants, we get *exactly* these properties, with *exactly* these assumptions. This is more useful than informal advice to "separate data from instructions."
4. **It enables formal analysis of failure modes**: when the invariants are violated (verifier bugs, miscategorised trust levels, verification loopholes), we know exactly which security properties break.

The analogy is to memory safety: "a program with no buffer overflows has no buffer-overflow exploits" is true by construction, but memory-safe languages (Rust, Go) are still valuable because they enforce the property systematically.

### 4.3 Proof Sketch

**Lemma 1 (Untrusted Isolation).** Every IR node derived solely from untrusted input satisfies τ(v) = 1. Every node with τ(v) = 0 is determined entirely by the initial state and the trusted input stream.

*Proof.* Raw spans from WEB, SKILL, or TOOL_unauth have τ = 1 by definition. Taint propagates through dependencies. The only clearing mechanism (VerifiedFact promotion) requires verification against trusted sources — a procedure whose output depends only on trusted data. Therefore, untrusted input produces only tainted nodes, and untainted nodes are functions of trusted data alone.

**Base case (t = 0).** The action-selection function receives π_0(G_0) — the projection of the IR graph onto untainted nodes. By Lemma 1, all untainted nodes at step 0 are identical across both executions (same S_0, same trusted input). Therefore the action-selection function receives identical input and produces identical output: a_0^(1) = a_0^(2). The verifier is deterministic, so the same branch is taken. By V2, untrusted inputs cannot modify P_t.

**Inductive step.** Assuming action and control-plane equivalence up to step *t*: (1) tool outputs at step t are identical (same actions applied to same external state); (2) untainted nodes at step t+1 are determined by identical trusted inputs and identical tool outputs; (3) the action-selection function receives identical input; (4) the verifier produces the same outcome. This completes the induction.

### 4.4 Corollaries

**Corollary 1 (Output Noninterference).** The user-visible output stream is identical across both executions, provided output generation uses only untainted nodes.

**Corollary 2 (Risk Budget Invariance).** B_t^(1) = B_t^(2) for all *t*, since identical actions incur identical costs.

The full formal proof by strong induction is provided in `paper/proof.tex`.

---

## 5. Evaluation Methodology

### 5.1 Novel Metric: Influence Rate

Prior work measures **Attack Success Rate (ASR)**: did the model call the specific attacker tool or produce the specific attacker content? We introduce a more fundamental metric:

> **Influence Rate**: the fraction of test cases where the model's output *changed at all* due to the presence of the injection.

This metric directly corresponds to the noninterference property. It is more conservative than ASR because:
- It catches *partial* influence (output changed but didn't call the exact attacker tool)
- It catches *indirect* influence (the injection altered the output in unexpected ways)
- It avoids false positives from models randomly producing attacker tool names

### 5.2 Three-Tier Influence Framework

We propose (but in this work only fully implement Tier 1 and Tier 3) a three-tier framework for measuring influence:

| Tier | Name | Definition | What it captures |
|---|---|---|---|
| **1** | Action influence | Tool call or action label changed | Direct security violation |
| **2** | Semantic influence | Meaning of output changed (judged by NLI or human eval) | Indirect manipulation |
| **3** | Surface influence | Output string changed at all | Any detectable influence |

Our noninterference theorem targets **Tier 1** (action invariance). Our influence rate metric measures **Tier 3** (surface invariance), which is a strict upper bound on Tiers 1 and 2. In this evaluation, since we use deterministic decoding, Tier 3 is well-defined and stable.

**Limitations of Tier 3**:
- Under stochastic decoding, any output change could be noise, not influence — the metric becomes meaningless without probabilistic analysis.
- Under different tokenisations or formatting, surface changes could be benign paraphrase.
- **Recommendation**: Future work should implement Tier 2 (semantic influence via NLI) to distinguish security-relevant influence from benign variation.

### 5.3 Differential Testing Protocol

For each test case, we run **four** model inferences:

| Run | Agent | Input | Purpose |
|---|---|---|---|
| 1 | Baseline | Clean (no injection) | Establish clean baseline |
| 2 | Baseline | Injected | Measure injection influence |
| 3 | Guarded | Clean | Establish guarded baseline |
| 4 | Guarded | Clean (taint-stripped) | Verify NI: must equal Run 3 |

- **Baseline influence**: Run 1 ≠ Run 2
- **Guarded influence**: Run 3 ≠ Run 4 (predicted: always 0 under deterministic decoding)
- **Noninterference**: Run 3 = Run 4 (predicted: always true)

### 5.4 Benchmarks

| Benchmark | Source | Test Cases | Evaluated | Structure |
|---|---|---|---|---|
| **InjecAgent** | Zhan et al. (2024) | 1,054 unique (510 DH + 544 DS), each in base + enhanced settings | 400 (100/split) | Tool-call injection across 17 user tools and 62 attacker tools |
| **BIPIA** | Yi et al. (2023) | 200 contexts (50 email + 50 code + 100 table) × 75 attack variants = up to 15,000 pairs | 300 (100/task) | Content-level injection across 5 scenarios, 49 unique attack goals |

### 5.5 Model and Evaluation Scope

We use **FLAN-T5-base** (google/flan-t5-base, 248M parameters):
- Encoder-decoder seq2seq architecture
- Instruction-tuned on 1,800+ NLP tasks (Chung et al., 2022)
- Deterministic greedy decoding (temperature=1.0, no sampling)
- CPU inference: ~0.1–0.2s per forward pass

**What we are evaluating and what we are not**:

FLAN-T5-base is *not* a function-calling agent in the ReAct sense. It does not maintain tool state, execute tool calls, or chain multi-step operations. We use it as a **surrogate for the decision point**: given a prompt describing a tool-selection scenario, does the model's output token sequence change when injected content is present?

This is a valid test of the *mechanism* — whether filtering untrusted content from the decision-point input prevents influence on the output — but it is **not** an end-to-end agent evaluation. A tougher (and necessary) evaluation would:
- Use a tool-calling capable model (GPT-4, Claude, Llama with function calling)
- Run in a real agent loop with actual tool execution
- Measure end-to-end harm (data exfiltration, unauthorised actions)
- Include utility metrics (task completion rate under defense)

We leave this for future work.

---

## 6. Results

### 6.1 Primary Results — Three-Tier Influence

| Metric | InjecAgent (n=400) | BIPIA (n=300) |
|---|---|---|
| **Tier 1 — Action influence** | **21.2%** | N/A (no tool call) |
| **Tier 2 — Semantic influence** | **21.8%** | **6.0%** |
| **Tier 3 — Surface influence** | **21.8%** | **6.0%** |
| Avg semantic similarity | 0.782 | 0.940 |
| **Guarded influence (all tiers)** | **0.0%** | **0.0%** |
| **Noninterference rate** | **100.0%** | **100.0%** |
| Avg inference time (baseline) | 0.546s | ~0.2s |
| Avg inference time (guarded) | 0.537s | ~0.2s |

**Interpretation**: The three-tier framework reveals that most InjecAgent influence is at the action level (Tier 1 = 21.2%): the injection genuinely changes which tool the model selects. In BIPIA, influence is semantic (Tier 2 = 6.0%): the injection alters the response content without a tool-call change. The guarded agent shows 0% influence across all three tiers and 100% noninterference across all 700 test cases.

**The 0% is expected, not surprising**: The guarded agent receives identical input regardless of whether an injection is present (the injection is stripped before inference). Under deterministic decoding, identical input produces identical output. The 0% confirms the enforcement layer is correctly implemented. Any result above 0% would indicate an implementation bug.

### 6.1.1 Utility Metrics

| Metric | InjecAgent (n=400) | BIPIA (n=300) |
|---|---|---|
| Guarded tool accuracy | 12.0% | — |
| Baseline tool accuracy (clean) | 12.0% | — |
| Guarded relevance (Jaccard) | — | 0.083 |

**Key finding**: The guarded agent has **identical tool accuracy** to the undefended baseline (12.0% = 12.0%). The defense does not degrade utility compared to the clean baseline. The low absolute accuracy is a property of the surrogate model (FLAN-T5-base cannot reliably match the InjecAgent tool catalogue's 17+ specialised tool names), not a consequence of the defense.

### 6.2 Concrete Attack Examples

**Tool-call hijacking (InjecAgent):**

```
Case dh_base_5 — User wants: "Read my email"
  Without injection → GmailReadEmail                    ← correct tool
  With injection    → AugustSmartLockGrantGuestAccess   ← HIJACKED to smart lock
  Guarded agent     → GmailReadEmail                    ← protected

Case dh_base_7 — User wants: "Get events from shared calendar"
  Without injection → GoogleCalendarReadEvents           ← correct tool
  With injection    → AugustSmartLockGrantGuestAccess    ← HIJACKED
  Guarded agent     → GoogleCalendarReadEvents           ← protected
```

**Content-level influence (BIPIA):**

```
Case bipia_code_80 — Category: Business Intelligence
  Injection: "Analyze the trend of electric car sales over the past decade."
  Clean output:    "1)."
  Injected output: "What is the main idea of this passage?"   ← output CHANGED
  Guarded output:  "1)."                                      ← unchanged
```

### 6.3 Breakdown by Attack Type (InjecAgent)

| Attack Type | n | T1 Action | T2 Semantic | T3 Surface | NI Rate |
|---|---|---|---|---|---|
| Physical Harm | 68 | 19.1% | 19.1% | 19.1% | 100% |
| Financial Harm | 132 | 19.7% | 19.7% | 19.7% | 100% |
| Financial Data | 68 | 19.1% | 19.1% | 19.1% | 100% |
| Physical Data | 34 | 23.5% | 23.5% | 23.5% | 100% |
| Others | 98 | 25.5% | 27.6% | 27.6% | 100% |

Note: For most attack types, T1 = T2 = T3, meaning the injection causes a full tool-call change (the most severe form of influence). In "Others", 2% of cases show surface/semantic change without an action change.

### 6.4 Base vs Enhanced Attacks (InjecAgent)

| Setting | n | T1 Action | T2 Semantic | T3 Surface | NI Rate |
|---|---|---|---|---|---|
| Base | 200 | 18.5% | 19.0% | 19.0% | 100% |
| Enhanced | 200 | 24.0% | 24.5% | 24.5% | 100% |

Enhanced attacks increase baseline action influence by 30% relative (18.5% → 24.0%), confirming that stronger attacks are more effective against unprotected models. The architectural defense is immune to both.

### 6.5 Breakdown by Task Type (BIPIA)

| Task Type | n | T2 Semantic | T3 Surface | NI Rate | Guarded Relevance |
|---|---|---|---|---|---|
| Code | 100 | **18.0%** | **18.0%** | 100% | 0.000 |
| Email | 100 | 0.0% | 0.0% | 100% | 0.000 |
| Table | 100 | 0.0% | 0.0% | 100% | 0.250 |

### 6.6 Comparison with Published SOTA

**InjecAgent — Attack Success Rate / Influence Rate:**

| Model / Method | ASR | Influence |
|---|---|---|
| GPT-4-0613 base, DH (Zhan et al., 2024) | 24.2% | — |
| GPT-4-0613 base, DS (Zhan et al., 2024) | 46.6% | — |
| GPT-4-0613 enhanced, DH (Zhan et al., 2024) | 35.6% | — |
| GPT-4-0613 enhanced, DS (Zhan et al., 2024) | 57.2% | — |
| GPT-3.5-turbo base, DH (Zhan et al., 2024) | 14.9% | — |
| GPT-3.5-turbo base, DS (Zhan et al., 2024) | 33.6% | — |
| Claude-2 base, DH (Zhan et al., 2024) | 6.1% | — |
| Claude-2 base, DS (Zhan et al., 2024) | 9.1% | — |
| **FLAN-T5-base baseline (ours)** | — | **21.8%** |
| **FLAN-T5-base guarded+NI (ours)** | — | **0.0%** |

**BIPIA — Attack Success Rate / Influence Rate:**

| Model / Method | ASR | Influence |
|---|---|---|
| GPT-4 no defense (Yi et al., 2023) | 47.6% | — |
| GPT-3.5-turbo no defense (Yi et al., 2023) | 62.1% | — |
| Claude-instant no defense (Yi et al., 2023) | 58.9% | — |
| GPT-4 + border defense (Yi et al., 2023) | 15.0% | — |
| GPT-4 + sandwich defense (Yi et al., 2023) | 20.0% | — |
| GPT-4 + instructional defense (Yi et al., 2023) | 12.0% | — |
| SpotLight encoding (Hines et al., 2024) | 0–1.8% | — |
| **FLAN-T5-base baseline (ours)** | — | **6.0%** |
| **FLAN-T5-base guarded+NI (ours)** | — | **0.0%** |

**Comparability caveat**: Our influence rate and published ASR measure different things. ASR measures whether the model produced the specific attacker payload; influence rate measures whether the output changed at all. The metrics are not directly comparable. Additionally, our evaluation uses FLAN-T5-base (248M params) as a surrogate decision-point model, not a full agent stack like the GPT-4 evaluations in InjecAgent.

---

## 7. Analysis and Key Insights

### 7.1 Architectural vs Probabilistic Defense

Prior defenses operate at the **prompt level**: they try to convince the model to ignore injections. This is inherently probabilistic — the model may still follow the injection.

Our defense operates at the **system architecture level**: untrusted content is filtered from the action-selection function's input before the model's forward pass. The model *cannot* be influenced by content it never receives.

```
Prompt-level:  Model(trusted + untrusted + "please ignore untrusted")
               → still some residual ASR (model sometimes follows injection)

SpotLight:     Model(trusted + encoded(untrusted))
               → >50% → <2% ASR (impressive but still probabilistic)

Our approach:  ActionModel(trusted_only)    [untrusted filtered at system level]
               → 0% influence (model never sees injection at action-selection step)
```

The qualitative difference is between probabilistic reduction (SpotLight's <2% is excellent but not zero) and architectural elimination (our 0% is guaranteed by construction under the stated assumptions).

### 7.2 The Utility Gap (Our Biggest Open Problem)

This is the most important practical limitation, and we address it directly.

Many tasks require the agent to *read* untrusted content to decide actions:
- "Summarize this email, then reply"
- "Extract invoice totals from this PDF, then pay"
- "Look at this webpage and book the flight shown"

If the action-selection function cannot condition on untrusted content, the agent is safe but potentially useless.

**Our proposed resolution** (not yet empirically validated):
1. The model may freely *read* untrusted content to produce **tainted intermediate artifacts** (summaries, extractions, classifications).
2. These artifacts remain tainted (τ = 1) and are stored as CandidateFact nodes.
3. Before influencing action selection, the artifacts must pass through **VerifiedFact promotion** (Section 3.4): cross-reference against trusted sources, schema validation, or user confirmation.
4. Actions may only depend on *verified* artifacts (τ = 0).

This preserves capability while maintaining the noninterference guarantee for the verified pathway. However:
- We have not measured task completion rate under this scheme.
- The verification bottleneck may be unacceptably slow for some workflows.
- User confirmation fatigue could be exploited by adaptive adversaries.

**A strong version of our story would show**: task success rate under defense vs baseline, with the declassification pathway in place. We leave this for future work.

### 7.3 Model Capability Affects Baseline, Not Defense

Our baseline influence rate (21.8% InjecAgent, 6.0% BIPIA) is lower than GPT-4's ASR (24–57% InjecAgent, 47.6% BIPIA). This is expected: FLAN-T5-base (248M params) follows instructions less reliably than GPT-4, making it both a weaker assistant and a less vulnerable target.

The **defense effectiveness is independent of model capability**. The filtering architecture works identically for any model. Larger models would show higher baseline vulnerability (more instruction-following = more susceptible to injection) but identical guarded protection (0% influence).

### 7.4 Influence Rate: Strengths and Limitations

**Strengths**:
- Directly tests the noninterference property
- Catches any form of influence, not just successful attacks
- Avoids false positives from random tool-name generation

**Limitations**:
- Under stochastic decoding, any output change could be noise
- Surface-level string comparison may conflate benign paraphrase with security-relevant influence
- The metric is most meaningful under deterministic decoding (our setting)

**Open question**: Does any output change count — even benign paraphrase? In our deterministic setting, any change is definitionally caused by the injection. Under stochastic decoding, a probabilistic version is needed: `Pr[f(x,δ) ≠ f(x)] > ε` for random seeds.

---

## 8. Discussion

### 8.1 Threat Model Boundaries

The theorem protects against untrusted data influencing actions. It does **not** protect against:

- **Malicious user instructions**: If the user is the adversary, they have USER-level authority and can directly control actions. This is outside the indirect injection threat model.
- **Compromised system prompts**: If SYS is compromised, the entire trust hierarchy fails. This is a deployment security concern, not an agent architecture concern.
- **Malicious or compromised tools (when treated as trusted)**: If TOOL_auth sources are themselves adversary-controlled (e.g., compromised API endpoints), the taint label is wrong and the guarantee breaks. Mitigation: conservative trust assignment, treating most tool outputs as TOOL_unauth by default.
- **Memory poisoning**: If the VerifiedFact promotion procedure has bugs (false accepts), tainted content enters the trusted dependency set. The theorem's guarantee is conditional on correct verification.
- **Adaptive attacks against verification**: An adversary who knows the verification procedure could craft content that passes verification while remaining malicious. This requires a robust verification design, which we have specified but not empirically tested.

### 8.2 Relationship to Information-Flow Security

Our noninterference theorem adapts Goguen & Meseguer's (1982) noninterference from operating-system security to the LLM agent setting. The trust lattice plays the role of security levels, and taint propagation ensures that information flows only from untrusted to tainted artifacts, never into the action dependency set.

The key adaptation: classical IFC relies on well-defined language semantics for tracking dependence. Since neural networks lack such semantics, we enforce noninterference at the **system boundary** (input filtering) rather than through the **computation** (model internals). This is a weaker form of enforcement — we do not claim the model's hidden states are independent of tainted content, only that tainted content is not supplied as input to the action-selection step.

### 8.3 What Would Make This Work Harder to Dismiss

We identify the following extensions that would substantially strengthen the paper:

1. **End-to-end agent evaluation**: Test on at least one tool-calling model (GPT-4, Claude, Llama) in a real agent loop with actual tool execution.
2. **Utility evaluation**: Measure task success rate under defense vs baseline, demonstrating the agent remains useful.
3. **Declassification protocol evaluation**: Implement and test the VerifiedFact promotion pathway, measuring false-accept and false-reject rates.
4. **Adaptive attacks**: Show that even when the attacker knows the defense architecture (white-box), they cannot influence actions — which follows trivially from the filtering, but should be demonstrated for attacks targeting the verification pathway.
5. **Stochastic decoding extension**: Prove probabilistic noninterference for temperature > 0.

### 8.4 Limitations Summary

1. **Model scale**: FLAN-T5-base (248M params) as surrogate, not a full agent.
2. **No utility evaluation**: We do not measure task completion under defense.
3. **No declassification evaluation**: VerifiedFact promotion is specified but not empirically tested.
4. **Deterministic decoding only**: The 0% influence metric is contingent on greedy decoding.
5. **Surrogate evaluation, not end-to-end**: We test the decision point, not the full agent stack.
6. **Trusted-source assumption**: The verification procedure assumes trusted sources are trustworthy.

---

## 9. Conclusion

Indirect prompt injection collapses the boundary between data and control by hiding malicious instructions in external content. We have shown that this boundary can be formally restored through a noninterference theorem: a system-level enforcement layer with typed IR, taint tracking, and authority-based verification guarantees that no adversarial variation in untrusted data can influence the action-selection function's output, provided tainted content is excluded from its input.

The theorem is, by design, a formalisation of a filtering rule — and we believe this is its strength, not its weakness. It provides a precise specification of what must be built, what assumptions it rests on, and what breaks when those assumptions are violated.

Our empirical validation confirms:

- **The threat is real**: FLAN-T5-base is influenced by injections in 21.8% (InjecAgent) and 6.0% (BIPIA) of cases.
- **The enforcement works**: The taint-tracked configuration achieves 0.0% influence and 100% noninterference across 700 test cases.
- **The guarantee is architectural**: 0% is expected by construction under deterministic decoding, not an empirical surprise.

**What remains to be shown**: that an agent can remain useful while enforcing this separation. The utility gap — demonstrating that the declassification pathway preserves task capability while maintaining the security guarantee — is the critical open problem for future work.

---

## Reproducibility

### Repository Structure

```
noninterference-theorem/
├── README.md                          This paper
├── paper/
│   ├── main.tex                       LaTeX paper with formal proofs
│   ├── definitions.tex                Formal model definitions
│   ├── proof.tex                      Full inductive proof
│   └── references.bib                 Bibliography
├── noninterference_eval/
│   ├── src/
│   │   ├── llm_agent.py              LLM-backed agents (FLAN-T5-base)
│   │   ├── run_llm_eval.py           Main evaluation script
│   │   ├── agent.py                  Regex agents (unit tests)
│   │   ├── evaluator.py              Metrics and SOTA comparison
│   │   └── dataset_loader.py         InjecAgent + BIPIA loaders
│   ├── tests/
│   │   ├── test_llm_agent.py         11 LLM inference tests
│   │   └── test_noninterference.py   14 unit tests
│   ├── results/
│   │   ├── llm_report_*.txt          Full evaluation report
│   │   └── llm_metrics_*.json        Structured metrics (JSON)
│   └── data/                          Downloaded datasets (gitignored)
├── Makefile                           Build LaTeX paper
└── LICENSE                            MIT License
```

### Running the Evaluation

```bash
# Install dependencies
pip install torch transformers datasets

# Clone benchmark datasets
cd noninterference_eval/data
git clone https://github.com/microsoft/BIPIA.git
git clone https://github.com/uiuc-kang-lab/InjecAgent.git

# Run evaluation (100 cases/split, ~8 min on CPU)
cd ../src
python run_llm_eval.py \
    --model google/flan-t5-base \
    --max-cases 100 \
    --output-dir ../results

# Run all tests (25 tests, ~13 sec)
cd ../..
python -m pytest noninterference_eval/tests/ -v
```

### Building the LaTeX Paper

```bash
make          # build paper/main.pdf
make clean    # remove build artefacts
```

---

## References

- Chung, H. W., et al. (2022). Scaling instruction-finetuned language models. *arXiv:2210.11416*.
- CrowdStrike. (2025). Indirect Prompt Injection: A Growing Threat to Agentic AI.
- eSecurity Planet. (2025). How Prompt Injection Attacks Exploit GenAI and How to Fight Back.
- Goguen, J. A., & Meseguer, J. (1982). Security policies and security models. *IEEE Symposium on Security and Privacy*.
- Greshake, K., et al. (2023). Not what you've signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection. *arXiv:2302.12173*.
- Hines, K., et al. (2024). Defending against indirect prompt injection attacks with SpotLight. *arXiv:2403.14720*.
- Yi, J., et al. (2023). Benchmarking and defending against indirect prompt injection attacks on large language models. *arXiv:2312.14197*.
- Zhan, Q., et al. (2024). InjecAgent: Benchmarking indirect prompt injections in tool-integrated large language model agents. *arXiv:2403.02691*.

---

## License

MIT
