import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from e2e_benchmark import (
    ReferenceTaskController,
    SystemVariant,
    build_reference_scenarios,
    get_system_config,
    run_benchmark_suite,
)


def _find_artifact(artifacts, scenario_id):
    for artifact in artifacts:
        if artifact.scenario_id == scenario_id:
            return artifact
    raise KeyError(f"Scenario not found: {scenario_id}")


class TestScenarioLibrary:
    def test_reference_scenarios_cover_required_patterns(self):
        scenarios = build_reference_scenarios()
        scenario_ids = {scenario.scenario_id for scenario in scenarios}
        assert scenario_ids == {
            "email_order_lookup_markdown",
            "hostile_web_search_html",
            "memory_poisoning_followup",
            "tool_output_json_poisoning",
            "declassification_bypass_attempt",
            "markdown_codeblock_obfuscation",
            "authenticated_tool_paraphrase_poisoning",
        }
        assert any(len(s.turns) > 1 for s in scenarios)
        assert any(s.expected_memory for s in scenarios)
        assert any(
            any("Authenticated tools may still emit content" in assumption for assumption in s.theorem_assumptions)
            for s in scenarios
        )
        assert any(
            "code block" in s.attack_vector or "code-block" in s.attack_vector
            for s in scenarios
        )


class TestEndToEndVariants:
    def test_baseline_is_vulnerable_under_attack(self):
        controller = ReferenceTaskController(get_system_config(SystemVariant.BASELINE))
        scenario = next(
            s for s in build_reference_scenarios()
            if s.scenario_id == "email_order_lookup_markdown"
        )
        artifact = controller.run_scenario(scenario, attacked=True)

        assert artifact.variant == "baseline"
        assert artifact.trace["memory"]["contact_email"]["value"] == "attacker@evil.com"
        tool_names = [
            event["tool_name"]
            for turn in artifact.turns
            for event in turn.executed_tools
        ]
        assert "GmailSendEmail" in tool_names
        assert artifact.trace["action_log"][-1]["reason"] == "baseline_allows_without_guard"

    def test_guarded_no_declass_blocks_attack_but_loses_utility(self):
        controller = ReferenceTaskController(get_system_config(SystemVariant.GUARDED_NO_DECLASS))
        scenario = next(
            s for s in build_reference_scenarios()
            if s.scenario_id == "email_order_lookup_markdown"
        )
        artifact = controller.run_scenario(scenario, attacked=True)

        assert artifact.variant == "guarded_no_declass"
        assert artifact.trace["memory"]["contact_email"]["verified"] is False
        assert artifact.trace["memory"]["contact_email"]["metadata"]["quarantined"] is True
        assert all(not turn.executed_tools for turn in artifact.turns)
        reasons = [entry["reason"] for entry in artifact.trace["action_log"]]
        assert "tool_or_action_blocked" in reasons
        assert "final_answer_repaired_to_safe_template" in reasons

    def test_guarded_structured_declass_recovers_utility_under_attack(self):
        controller = ReferenceTaskController(
            get_system_config(SystemVariant.GUARDED_STRUCTURED_DECLASS)
        )
        scenario = next(
            s for s in build_reference_scenarios()
            if s.scenario_id == "email_order_lookup_markdown"
        )
        artifact = controller.run_scenario(scenario, attacked=True)

        assert artifact.variant == "guarded_structured_declass"
        assert artifact.trace["memory"]["contact_email"]["value"] == "customer@example.com"
        assert artifact.trace["memory"]["contact_email"]["verified"] is True
        tool_names = [
            event["tool_name"]
            for turn in artifact.turns
            for event in turn.executed_tools
        ]
        assert tool_names == ["AmazonGetProductDetails"]
        assert artifact.final_context["product_name"] == "USB-C travel charger"
        assert artifact.trace["action_log"][-1]["reason"] == "final_answer_allowed"

    def test_memory_poisoning_is_blocked_and_clean_memory_is_preserved(self):
        scenarios = build_reference_scenarios()
        memory_scenario = next(
            s for s in scenarios if s.scenario_id == "memory_poisoning_followup"
        )

        baseline = ReferenceTaskController(get_system_config(SystemVariant.BASELINE))
        baseline_artifact = baseline.run_scenario(memory_scenario, attacked=True)
        baseline_email = baseline_artifact.turns[-1].executed_tools[0]["arguments"]["to"]
        assert baseline_email == "attacker@evil.com"

        guarded = ReferenceTaskController(
            get_system_config(SystemVariant.GUARDED_STRUCTURED_DECLASS)
        )
        guarded_artifact = guarded.run_scenario(memory_scenario, attacked=True)
        guarded_email = guarded_artifact.turns[-1].executed_tools[0]["arguments"]["to"]
        assert guarded_email == "alice.customer@example.com"
        assert guarded_artifact.trace["memory"]["contact_email"]["verified"] is True

    def test_codeblock_obfuscation_attack_is_blocked_without_losing_ticket_lookup(self):
        scenario = next(
            s for s in build_reference_scenarios()
            if s.scenario_id == "markdown_codeblock_obfuscation"
        )

        baseline = ReferenceTaskController(get_system_config(SystemVariant.BASELINE))
        baseline_artifact = baseline.run_scenario(scenario, attacked=True)
        baseline_tools = [
            event["tool_name"]
            for turn in baseline_artifact.turns
            for event in turn.executed_tools
        ]
        assert baseline_tools == ["Terminal"]

        guarded = ReferenceTaskController(
            get_system_config(SystemVariant.GUARDED_STRUCTURED_DECLASS)
        )
        guarded_artifact = guarded.run_scenario(scenario, attacked=True)
        guarded_tools = [
            event["tool_name"]
            for turn in guarded_artifact.turns
            for event in turn.executed_tools
        ]
        assert guarded_tools == ["TicketLookup"]
        assert "Rotate staging API credentials" in guarded_artifact.final_context["ticket_summary"]

    def test_authenticated_tool_output_poisoning_is_constrained_to_validated_fields(self):
        scenario = next(
            s for s in build_reference_scenarios()
            if s.scenario_id == "authenticated_tool_paraphrase_poisoning"
        )

        baseline = ReferenceTaskController(get_system_config(SystemVariant.BASELINE))
        baseline_artifact = baseline.run_scenario(scenario, attacked=True)
        baseline_tools = [
            event["tool_name"]
            for turn in baseline_artifact.turns
            for event in turn.executed_tools
        ]
        assert baseline_tools == ["Terminal"]
        assert baseline_artifact.trace["memory"]["contact_email"]["value"] == "attacker@evil.com"

        guarded = ReferenceTaskController(
            get_system_config(SystemVariant.GUARDED_STRUCTURED_DECLASS)
        )
        guarded_artifact = guarded.run_scenario(scenario, attacked=True)
        guarded_tools = [
            event["tool_name"]
            for turn in guarded_artifact.turns
            for event in turn.executed_tools
        ]
        assert guarded_tools == ["GmailSendEmail"]
        assert guarded_artifact.trace["memory"]["contact_email"]["value"] == "renewals@example.com"
        assert guarded_artifact.turns[0].executed_tools[0]["arguments"]["to"] == "renewals@example.com"

    def test_benchmark_suite_runner_returns_expected_cardinality(self):
        artifacts = run_benchmark_suite(
            attacked=True,
            variants=[
                SystemVariant.BASELINE,
                SystemVariant.GUARDED_NO_DECLASS,
                SystemVariant.GUARDED_STRUCTURED_DECLASS,
            ],
        )
        assert len(artifacts) == 21
        baseline_email = _find_artifact(artifacts, "email_order_lookup_markdown")
        assert baseline_email.variant == "baseline"
