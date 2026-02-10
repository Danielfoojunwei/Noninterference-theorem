#!/usr/bin/env python3
"""
Full evaluation of the noninterference theorem on canonical benchmarks.

Runs:
  1. InjecAgent (Zhan et al., 2024) – all 2118 test cases
     (dh_base, dh_enhanced, ds_base, ds_enhanced)
  2. BIPIA (Yi et al., 2023) – email, table, code tasks x text attacks

For each benchmark, both a *baseline* agent (no taint tracking, vulnerable)
and a *guarded* agent (taint-tracked, noninterference enforced) are evaluated.

Results are compared against published SOTA numbers.

Usage:
    python main.py \\
        --injecagent-dir ../data/InjecAgent/data \\
        --bipia-dir ../data/BIPIA/benchmark \\
        --output-dir ../results
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dataset_loader import InjecAgentLoader, BIPIALoader
from evaluator import (
    InjecAgentEvaluator, BIPIAEvaluator,
    compute_injecagent_metrics, compute_bipia_metrics,
    generate_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Empirical evaluation of the noninterference theorem",
    )
    parser.add_argument(
        "--injecagent-dir", type=str,
        default="../data/InjecAgent/data",
        help="Path to InjecAgent data/ directory",
    )
    parser.add_argument(
        "--bipia-dir", type=str,
        default="../data/BIPIA/benchmark",
        help="Path to BIPIA benchmark/ directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="../results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--max-cases", type=int, default=None,
        help="Cap per dataset split (None = all)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    logger.info("=" * 90)
    logger.info("NONINTERFERENCE THEOREM — FULL EMPIRICAL EVALUATION")
    logger.info("=" * 90)
    t_start = time.time()

    # ── InjecAgent ───────────────────────────────────────────────────
    injecagent_results = []
    ia_loader = InjecAgentLoader(args.injecagent_dir)
    ia_eval = InjecAgentEvaluator()

    for attack_type in ["dh", "ds"]:
        for setting in ["base", "enhanced"]:
            cases = ia_loader.load(attack_type, setting, args.max_cases)
            if cases:
                logger.info(
                    "Running InjecAgent %s/%s  (%d cases)",
                    attack_type, setting, len(cases),
                )
                results = ia_eval.run(cases)
                injecagent_results.extend(results)
                logger.info(
                    "  baseline_attacked=%d  guarded_attacked=%d  NI_held=%d",
                    sum(r.baseline_attacked for r in results),
                    sum(r.guarded_attacked for r in results),
                    sum(r.noninterference_held for r in results),
                )

    injecagent_metrics = compute_injecagent_metrics(injecagent_results)

    # ── BIPIA ────────────────────────────────────────────────────────
    bipia_results = []
    bp_loader = BIPIALoader(args.bipia_dir)
    bp_eval = BIPIAEvaluator()

    for task in ["email", "table", "code"]:
        cases = bp_loader.load(task, "text", args.max_cases)
        if cases:
            logger.info(
                "Running BIPIA %s  (%d cases)", task, len(cases),
            )
            results = bp_eval.run(cases)
            bipia_results.extend(results)
            logger.info(
                "  baseline_injected=%d  guarded_injected=%d  NI_held=%d",
                sum(r.baseline_injected for r in results),
                sum(r.guarded_injected for r in results),
                sum(r.noninterference_held for r in results),
            )

    bipia_metrics = compute_bipia_metrics(bipia_results)

    # ── Report ───────────────────────────────────────────────────────
    report = generate_report(injecagent_metrics, bipia_metrics, output_dir)

    elapsed = time.time() - t_start
    logger.info("")
    logger.info("Total wall-clock time: %.2fs", elapsed)
    logger.info("")
    print(report)


if __name__ == "__main__":
    main()
