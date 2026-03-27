"""Reproducible CLI for the end-to-end guarded-agent benchmark.

This entry point adds the publication-facing execution layer requested in the
repository audit:

- explicit experiment configuration snapshots;
- repeated seeded runs with mean/std reporting;
- saved raw artifacts for attacked and clean executions;
- aggregate metric bundles and compact publication tables;
- one-command smoke and benchmark modes.

The benchmark itself is deterministic today, but the runner still records and
replays seeds so future stochastic controllers can be evaluated without changing
artifact formats.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised in real CLI usage
    raise SystemExit(
        "Missing dependency 'pyyaml'. Install requirements from "
        "noninterference_eval/requirements.txt before running this CLI."
    ) from exc

from e2e_benchmark import (
    DEFAULT_VARIANTS,
    SystemVariant,
    build_reference_scenarios,
    run_benchmark_suite,
)
from e2e_metrics import build_metrics_bundle


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "e2e"
SUMMARY_METRICS = [
    "attack_success_rate",
    "unsafe_action_rate",
    "security_failure_rate",
    "blocked_malicious_action_rate",
    "indirect_prompt_influence_rate",
    "task_completion_rate_attacked",
    "task_completion_rate_clean",
    "task_token_match_rate_attacked",
    "task_token_match_rate_clean",
    "benign_action_preservation_rate",
    "overblocking_rate",
    "runtime_overhead_vs_baseline_pct",
]


@dataclass
class ExperimentConfig:
    mode: str
    seeds: List[int]
    variants: List[str]
    output_dir: str
    timestamp_utc: str
    scenario_ids: List[str]
    scenario_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible end-to-end noninterference experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ["smoke", "benchmark"]:
        sub = subparsers.add_parser(command, help=f"Run the {command} experiment suite")
        sub.add_argument(
            "--output-dir",
            type=str,
            default=str(DEFAULT_RESULTS_DIR),
            help="Directory where run artifacts will be saved.",
        )
        sub.add_argument(
            "--seeds",
            type=int,
            nargs="+",
            default=[0] if command == "smoke" else [0, 1, 2],
            help="Seeds to execute. The harness is deterministic today, but seeds are recorded for future stochastic variants.",
        )
        sub.add_argument(
            "--variants",
            nargs="+",
            default=[variant.value for variant in DEFAULT_VARIANTS],
            choices=[variant.value for variant in SystemVariant],
            help="System variants to evaluate.",
        )

    for command in ["aggregate", "tables"]:
        sub = subparsers.add_parser(command, help=f"Build {command} outputs from existing run directories")
        sub.add_argument(
            "run_dirs",
            nargs="+",
            help="One or more run directories previously produced by the smoke/benchmark commands.",
        )
        sub.add_argument(
            "--output-dir",
            type=str,
            default=str(DEFAULT_RESULTS_DIR),
            help="Directory where aggregate outputs will be written.",
        )

    return parser.parse_args()


def _variant_objects(variant_names: Sequence[str]) -> List[SystemVariant]:
    return [SystemVariant(name) for name in variant_names]


def _set_seed(seed: int) -> None:
    random.seed(seed)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _yaml_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _artifact_list_to_json(artifacts: Iterable[Any]) -> List[Dict[str, Any]]:
    return [artifact.to_dict() for artifact in artifacts]


def _run_single_seed(seed: int, variants: Sequence[SystemVariant]) -> Dict[str, Any]:
    _set_seed(seed)
    scenarios = build_reference_scenarios()
    attacked_artifacts = run_benchmark_suite(attacked=True, variants=variants)
    clean_artifacts = run_benchmark_suite(attacked=False, variants=variants)
    metrics_bundle = build_metrics_bundle(attacked_artifacts, clean_artifacts, scenarios)
    return {
        "seed": seed,
        "attacked_artifacts": _artifact_list_to_json(attacked_artifacts),
        "clean_artifacts": _artifact_list_to_json(clean_artifacts),
        "metrics_bundle": metrics_bundle,
    }


def _summary_frame(metrics_bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for variant, summary in metrics_bundle["variant_summary"].items():
        row = {"variant": variant}
        for key in SUMMARY_METRICS:
            row[key] = summary.get(key, 0.0)
        row["security_failure_scenarios"] = summary.get("security_failure_scenarios", [])
        rows.append(row)
    return rows


def _seed_metric_rows(seed_runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in seed_runs:
        metrics_bundle = run["metrics_bundle"]
        for variant, summary in metrics_bundle["variant_summary"].items():
            row = {"seed": run["seed"], "variant": variant}
            for key in SUMMARY_METRICS:
                row[key] = summary.get(key, 0.0)
            rows.append(row)
    return rows


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    numeric = [float(v) for v in values]
    if not numeric:
        return {"mean": 0.0, "std": 0.0}
    if len(numeric) == 1:
        return {"mean": numeric[0], "std": 0.0}
    return {"mean": statistics.mean(numeric), "std": statistics.stdev(numeric)}


def _aggregate_seed_runs(seed_runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_variant: Dict[str, Dict[str, List[float]]] = {}
    for run in seed_runs:
        for variant, summary in run["metrics_bundle"]["variant_summary"].items():
            bucket = by_variant.setdefault(variant, {metric: [] for metric in SUMMARY_METRICS})
            for metric in SUMMARY_METRICS:
                bucket[metric].append(float(summary.get(metric, 0.0)))

    aggregated_variant_summary: Dict[str, Dict[str, float]] = {}
    for variant, metric_lists in by_variant.items():
        aggregated_variant_summary[variant] = {
            metric: _mean_std(values)
            for metric, values in metric_lists.items()
        }

    return {
        "n_seed_runs": len(seed_runs),
        "per_seed_rows": _seed_metric_rows(seed_runs),
        "aggregated_variant_summary": aggregated_variant_summary,
    }


def _publication_table(seed_runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    aggregate = _aggregate_seed_runs(seed_runs)
    rows: List[Dict[str, Any]] = []
    for variant, summary in aggregate["aggregated_variant_summary"].items():
        row = {"variant": variant}
        for metric in SUMMARY_METRICS:
            stats = summary[metric]
            row[metric] = f"{stats['mean']:.3f} ± {stats['std']:.3f}"
        rows.append(row)
    return rows


def _markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, separator] + body) + "\n"


def _write_run_outputs(root: Path, config: ExperimentConfig, seed_runs: Sequence[Dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _yaml_dump(root / "config_snapshot.yaml", config.to_dict())

    for run in seed_runs:
        seed_dir = root / f"seed_{run['seed']}"
        _json_dump(seed_dir / "attacked_artifacts.json", run["attacked_artifacts"])
        _json_dump(seed_dir / "clean_artifacts.json", run["clean_artifacts"])
        _json_dump(seed_dir / "metrics_bundle.json", run["metrics_bundle"])
        _json_dump(seed_dir / "variant_summary.json", run["metrics_bundle"]["variant_summary"])
        _json_dump(seed_dir / "scenario_table.json", run["metrics_bundle"]["scenario_table"])

    aggregate = _aggregate_seed_runs(seed_runs)
    _json_dump(root / "aggregate_seed_summary.json", aggregate)
    table_rows = _publication_table(seed_runs)
    _json_dump(root / "publication_table.json", table_rows)
    _json_dump(root / "latest_variant_summary.json", seed_runs[-1]["metrics_bundle"]["variant_summary"])

    md = [
        "# End-to-End Evaluation Summary\n",
        "## Experiment Configuration\n",
        f"- Mode: {config.mode}\n",
        f"- Seeds: {', '.join(str(seed) for seed in config.seeds)}\n",
        f"- Variants: {', '.join(config.variants)}\n",
        f"- Scenario count: {config.scenario_count}\n",
        "\n## Publication Table\n",
        _markdown_table(
            table_rows,
            ["variant"] + list(SUMMARY_METRICS),
        ),
    ]
    (root / "summary.md").write_text("".join(md))


def _load_seed_run(run_dir: Path) -> Dict[str, Any]:
    config_path = run_dir / "config_snapshot.yaml"
    if not config_path.exists():
        raise SystemExit(f"Missing config snapshot in {run_dir}")

    seed_dirs = sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("seed_"))
    if not seed_dirs:
        raise SystemExit(f"No per-seed outputs found in {run_dir}")

    seed_runs: List[Dict[str, Any]] = []
    for seed_dir in seed_dirs:
        metrics_path = seed_dir / "metrics_bundle.json"
        if not metrics_path.exists():
            raise SystemExit(f"Missing metrics bundle: {metrics_path}")
        seed = int(seed_dir.name.split("_", 1)[1])
        seed_runs.append({
            "seed": seed,
            "metrics_bundle": json.loads(metrics_path.read_text()),
        })
    return {
        "run_dir": str(run_dir),
        "config": yaml.safe_load(config_path.read_text()),
        "seed_runs": seed_runs,
    }


def _combine_existing_runs(run_dirs: Sequence[Path]) -> Dict[str, Any]:
    combined_seed_runs: List[Dict[str, Any]] = []
    configs: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        loaded = _load_seed_run(run_dir)
        configs.append(loaded["config"])
        combined_seed_runs.extend(loaded["seed_runs"])
    return {
        "source_configs": configs,
        "seed_runs": combined_seed_runs,
    }


def _write_cross_run_outputs(root: Path, combined: Dict[str, Any], label: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _yaml_dump(root / "combined_source_configs.yaml", {"runs": combined["source_configs"]})
    aggregate = _aggregate_seed_runs(combined["seed_runs"])
    table_rows = _publication_table(combined["seed_runs"])
    _json_dump(root / f"{label}_aggregate_seed_summary.json", aggregate)
    _json_dump(root / f"{label}_publication_table.json", table_rows)
    (root / f"{label}_summary.md").write_text(
        "# Aggregated End-to-End Evaluation Table\n\n"
        + _markdown_table(table_rows, ["variant"] + list(SUMMARY_METRICS))
    )


def _run_command(command: str, output_dir: Path, seeds: Sequence[int], variant_names: Sequence[str]) -> Path:
    variants = _variant_objects(variant_names)
    scenarios = build_reference_scenarios()
    timestamp = _utc_timestamp()
    run_root = output_dir / f"{command}_{timestamp}"
    config = ExperimentConfig(
        mode=command,
        seeds=list(seeds),
        variants=[variant.value for variant in variants],
        output_dir=str(run_root),
        timestamp_utc=timestamp,
        scenario_ids=[scenario.scenario_id for scenario in scenarios],
        scenario_count=len(scenarios),
    )
    seed_runs = [_run_single_seed(seed, variants) for seed in seeds]
    _write_run_outputs(run_root, config, seed_runs)
    return run_root


def main() -> None:
    args = _parse_args()

    if args.command in {"smoke", "benchmark"}:
        run_root = _run_command(
            command=args.command,
            output_dir=Path(args.output_dir),
            seeds=args.seeds,
            variant_names=args.variants,
        )
        print(f"Saved run outputs to {run_root}")
        return

    run_dirs = [Path(path).resolve() for path in args.run_dirs]
    combined = _combine_existing_runs(run_dirs)
    output_root = Path(args.output_dir).resolve() / f"{args.command}_{_utc_timestamp()}"
    _write_cross_run_outputs(output_root, combined, label=args.command)
    print(f"Saved {args.command} outputs to {output_root}")


if __name__ == "__main__":
    main()
