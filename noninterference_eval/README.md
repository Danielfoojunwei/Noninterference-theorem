# Noninterference Theorem Evaluation Framework

Empirical evaluation framework for the **Noninterference Theorem for Indirect
Prompt Injection in Agentic AI**.  Uses real benchmarks, canonical datasets and
industry-standard models to validate the theorem's claims.

## Overview

The framework implements:

- **Recursive Language Model (RLM)** with typed intermediate representation (IR)
- **Taint tracking** for untrusted inputs
- **Trust lattice** with principal-based access control
- **Verifier** for control-plane invariants
- **Differential testing** to verify noninterference properties

## Architecture

```
noninterference_eval/
├── src/
│   ├── agent.py           # Agent with RLM and taint tracking
│   ├── dataset_loader.py  # Loaders for BIPIA, InjecAgent, synthetic data
│   ├── evaluator.py       # Differential testing and metrics
│   └── main.py            # Main evaluation script
├── data/                  # Dataset storage (populated by user)
├── results/               # Evaluation results and reports
├── tests/                 # Unit tests
└── requirements.txt       # Python dependencies
```

## Installation

```bash
cd noninterference_eval
pip install -r requirements.txt
```

Optionally set API keys for LLM models:

```bash
export OPENAI_API_KEY="your-api-key-here"
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Datasets

### Synthetic (built-in)

```bash
python src/main.py --dataset synthetic --num-cases 20
```

### BIPIA

Source: [microsoft/BIPIA](https://github.com/microsoft/BIPIA)

```bash
git clone https://github.com/microsoft/BIPIA.git data/BIPIA
python src/main.py --dataset bipia --bipia-dir data/BIPIA/benchmark --max-per-dataset 50
```

### InjecAgent

Source: [uiuc-kang-lab/InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent)

```bash
git clone https://github.com/uiuc-kang-lab/InjecAgent.git data/InjecAgent
python src/main.py --dataset injecagent --injecagent-dir data/InjecAgent/data --max-per-dataset 50
```

### All datasets

```bash
python src/main.py --dataset all \
    --bipia-dir data/BIPIA/benchmark \
    --injecagent-dir data/InjecAgent/data \
    --max-per-dataset 50
```

## Command-line options

| Option              | Description                                           | Default        |
|---------------------|-------------------------------------------------------|----------------|
| `--dataset`         | `synthetic`, `bipia`, `injecagent`, or `all`          | `synthetic`    |
| `--bipia-dir`       | Path to BIPIA benchmark directory                     | —              |
| `--injecagent-dir`  | Path to InjecAgent data directory                     | —              |
| `--num-cases`       | Number of synthetic test cases                        | `10`           |
| `--max-per-dataset` | Max cases per external dataset                        | `50`           |
| `--model`           | Model name for the agent                              | `gpt-3.5-turbo`|
| `--output-dir`      | Directory to save results                             | `./results`    |

## Evaluation metrics

| Metric                  | Description                                                       |
|-------------------------|-------------------------------------------------------------------|
| Noninterference Rate    | % of test cases where behaviour is identical across runs          |
| Attack Success Rate     | % of attacks that successfully manipulate the agent               |
| Control-Plane Match     | Whether control-plane states remain identical                     |
| Tool Calls Match        | Whether tool call sequences remain identical                      |
| Execution Time          | Latency per test case                                             |

All metrics are also reported per attack type.

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

## References

1. Yi, J., et al. (2025). *Benchmarking and defending against indirect prompt
   injection attacks on large language models*. ACM CCS.
2. Kang, D., et al. (2024). *InjecAgent: Benchmarking Indirect Prompt
   Injections in Tool-Integrated Large Language Model Agents*.
   arXiv:2403.02691.
3. Liu, Y., et al. (2024). *Formalizing and Benchmarking Prompt Injection
   Attacks and Defenses*. USENIX Security.

## License

MIT
