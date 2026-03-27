import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from e2e_benchmark import SystemVariant, build_reference_scenarios, run_benchmark_suite
from e2e_metrics import (
    aggregate_by_dimension,
    aggregate_variant_metrics,
    build_metrics_bundle,
    compute_metrics_table,
    compute_noninterference_metrics,
)


class TestE2EMetrics:
    def test_variant_summary_captures_security_utility_tradeoff(self):
        scenarios = build_reference_scenarios()
        attacked = run_benchmark_suite(
            attacked=True,
            variants=[
                SystemVariant.BASELINE,
                SystemVariant.GUARDED_NO_DECLASS,
                SystemVariant.GUARDED_STRUCTURED_DECLASS,
            ],
        )
        clean = run_benchmark_suite(
            attacked=False,
            variants=[
                SystemVariant.BASELINE,
                SystemVariant.GUARDED_NO_DECLASS,
                SystemVariant.GUARDED_STRUCTURED_DECLASS,
            ],
        )

        summary = aggregate_variant_metrics(attacked + clean, scenarios)

        assert summary["baseline"]["attack_success_rate"] > 0.0
        assert summary["baseline"]["security_failure_rate"] > 0.0
        assert summary["guarded_no_declass"]["attack_success_rate"] == 0.0
        assert summary["guarded_no_declass"]["task_completion_rate_attacked"] < 1.0
        assert summary["guarded_structured_declass"]["attack_success_rate"] == 0.0
        assert summary["guarded_structured_declass"]["task_completion_rate_attacked"] > summary["guarded_no_declass"]["task_completion_rate_attacked"]
        assert summary["guarded_structured_declass"]["verified_action_usage_rate"] > summary["guarded_no_declass"]["verified_action_usage_rate"]

    def test_metrics_table_distinguishes_legitimate_and_malicious_same_tool_use(self):
        scenarios = build_reference_scenarios()
        attacked = run_benchmark_suite(
            attacked=True,
            variants=[SystemVariant.BASELINE, SystemVariant.GUARDED_STRUCTURED_DECLASS],
        )
        rows = compute_metrics_table(attacked, scenarios)

        baseline_memory = [
            row for row in rows
            if row["variant"] == "baseline" and row["scenario_id"] == "memory_poisoning_followup"
        ][0]
        guarded_memory = [
            row for row in rows
            if row["variant"] == "guarded_structured_declass" and row["scenario_id"] == "memory_poisoning_followup"
        ][0]

        assert baseline_memory["malicious_tool_executed"] is True
        assert baseline_memory["legitimate_tool_executed"] is False
        assert guarded_memory["malicious_tool_executed"] is False
        assert guarded_memory["legitimate_tool_executed"] is True

    def test_noninterference_summary_reflects_variant_behavior(self):
        scenarios = build_reference_scenarios()
        attacked = run_benchmark_suite(
            attacked=True,
            variants=[
                SystemVariant.BASELINE,
                SystemVariant.GUARDED_NO_DECLASS,
                SystemVariant.GUARDED_STRUCTURED_DECLASS,
            ],
        )
        clean = run_benchmark_suite(
            attacked=False,
            variants=[
                SystemVariant.BASELINE,
                SystemVariant.GUARDED_NO_DECLASS,
                SystemVariant.GUARDED_STRUCTURED_DECLASS,
            ],
        )

        ni = compute_noninterference_metrics(attacked, clean, scenarios)

        assert ni["baseline"]["tool_invariance_rate"] < 1.0
        assert ni["guarded_structured_declass"]["tool_invariance_rate"] == 1.0
        assert ni["guarded_structured_declass"]["task_outcome_invariance_rate"] == 1.0

    def test_aggregation_by_dimension_and_bundle_shape(self):
        scenarios = build_reference_scenarios()
        attacked = run_benchmark_suite(
            attacked=True,
            variants=[SystemVariant.BASELINE, SystemVariant.GUARDED_STRUCTURED_DECLASS],
        )
        clean = run_benchmark_suite(
            attacked=False,
            variants=[SystemVariant.BASELINE, SystemVariant.GUARDED_STRUCTURED_DECLASS],
        )

        by_family = aggregate_by_dimension(attacked + clean, scenarios, dimension="task_family")
        bundle = build_metrics_bundle(attacked, clean, scenarios)

        assert "baseline" in by_family
        assert "guarded_structured_declass" in by_family
        assert "email_summarization" in by_family["baseline"]
        assert "variant_summary" in bundle
        assert "noninterference" in bundle
        assert len(bundle["scenario_table"]) == len(attacked) + len(clean)
