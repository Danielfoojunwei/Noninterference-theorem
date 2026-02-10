# Noninterference Theorem for Indirect Prompt Injection in Agentic AI

A formal treatment of the noninterference property for agentic AI systems that
ingest untrusted content. The theorem proves that, under a recursive language
model (RLM) equipped with a typed intermediate representation, taint tracking,
and authority-based update rules, adversarial variations in untrusted data
cannot influence tool selection or modify the control plane.

## Overview

Indirect prompt injection allows adversaries to embed malicious instructions in
otherwise legitimate data (emails, web pages, documents) that is processed by
an agentic AI system. Because the untrusted content shares the same context
window as system prompts and user instructions, these hidden instructions can
silently influence the agent's behaviour.

This repository formalises the **noninterference** security property: the
agent's decisions and control-plane state must be invariant to arbitrary
adversarial variations in untrusted input. We provide:

- **Formal model definitions** — state variables, trust lattice, taint labels,
  and a typed IR graph.
- **The Noninterference Theorem** — a precise statement and proof that tool
  calls and control-plane state depend only on trusted inputs.
- **An evaluation plan** — differential testing, side-effect monitoring, and
  performance overhead measurement.

## Repository structure

```
├── README.md                   This file
├── LICENSE                     MIT License
├── paper/
│   ├── main.tex                Main LaTeX document
│   ├── definitions.tex         Formal model definitions
│   ├── proof.tex               Full proof of the noninterference theorem
│   └── references.bib          Bibliography
└── Makefile                    Build the PDF with latexmk
```

## Building the paper

```bash
make          # build paper/main.pdf
make clean    # remove build artefacts
```

Requires a TeX distribution with `latexmk`, `pdflatex`, `amsmath`, `amssymb`,
`amsthm`, `mathtools`, `hyperref`, and `natbib`.

## Key result

**Theorem (Noninterference for Indirect Prompt Injection).** Consider two
executions of the agent with identical initial state and identical trusted input
streams but differing untrusted input streams. If the verifier enforces that
candidate tool calls depend only on untainted IR nodes and that control-plane
updates originate only from `SYS` or `USER` principals, then for all time
steps *t*:

1. The control-plane states are equal: *P_t^(1) = P_t^(2)*.
2. The tool calls are equal: *a_t^(1) = a_t^(2)*.

## License

MIT