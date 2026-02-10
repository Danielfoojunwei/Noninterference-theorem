"""
Dataset loaders for BIPIA and InjecAgent benchmarks.

Loads the *actual* data files from cloned repositories and produces
structured TestCase objects for evaluation.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from itertools import product as cartesian

logger = logging.getLogger(__name__)


@dataclass
class InjecAgentCase:
    """One InjecAgent test case (tool-call attack)."""
    id: str
    user_instruction: str
    user_tool: str
    attacker_tools: List[str]
    attacker_instruction: str
    attack_type: str
    tool_response_clean: str   # tool response WITHOUT injection
    tool_response_injected: str  # tool response WITH injection
    expected_achievements: str
    setting: str  # "base" or "enhanced"


@dataclass
class BIPIACase:
    """One BIPIA test case (content-level attack)."""
    id: str
    task_type: str  # email, table, code
    question: str
    context: str  # benign context
    injection: str  # the injected instruction
    attack_category: str  # e.g. "Scams & Fraud"
    ideal: str  # expected answer


# ── InjecAgent loader ────────────────────────────────────────────────────

class InjecAgentLoader:
    """Load real InjecAgent data."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(
        self,
        attack_type: str = "dh",
        setting: str = "base",
        max_cases: Optional[int] = None,
    ) -> List[InjecAgentCase]:
        """Load test cases.

        Args:
            attack_type: "dh" (direct harm) or "ds" (data stealing)
            setting: "base" or "enhanced"
        """
        path = self.data_dir / (
            "test_cases_%s_%s.json" % (attack_type, setting)
        )
        if not path.exists():
            logger.warning("File not found: %s", path)
            return []

        with open(path) as f:
            raw = json.load(f)

        cases = []
        for i, item in enumerate(raw):
            if max_cases and i >= max_cases:
                break

            # Build clean tool response (without injection)
            template = item.get("Tool Response Template", "")
            clean_response = template.replace(
                "<Attacker Instruction>", "",
            )

            cases.append(InjecAgentCase(
                id="injecagent_%s_%s_%d" % (attack_type, setting, i),
                user_instruction=item["User Instruction"],
                user_tool=item["User Tool"],
                attacker_tools=item["Attacker Tools"],
                attacker_instruction=item["Attacker Instruction"],
                attack_type=item["Attack Type"],
                tool_response_clean=clean_response,
                tool_response_injected=item["Tool Response"],
                expected_achievements=item["Expected Achievements"],
                setting=setting,
            ))

        logger.info(
            "Loaded %d InjecAgent cases (%s/%s)", len(cases),
            attack_type, setting,
        )
        return cases


# ── BIPIA loader ─────────────────────────────────────────────────────────

class BIPIALoader:
    """Load real BIPIA benchmark data."""

    def __init__(self, benchmark_dir: str):
        self.benchmark_dir = Path(benchmark_dir)

    def _load_task_data(
        self, task: str, split: str = "test",
    ) -> List[Dict[str, Any]]:
        path = self.benchmark_dir / task / ("%s.jsonl" % split)
        if not path.exists():
            logger.warning("Task file not found: %s", path)
            return []
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def _load_attacks(
        self, attack_type: str = "text", split: str = "test",
    ) -> Dict[str, List[str]]:
        path = self.benchmark_dir / (
            "%s_attack_%s.json" % (attack_type, split)
        )
        if not path.exists():
            logger.warning("Attack file not found: %s", path)
            return {}
        with open(path) as f:
            return json.load(f)

    def load(
        self,
        task: str = "email",
        attack_type: str = "text",
        max_cases: Optional[int] = None,
    ) -> List[BIPIACase]:
        """Load BIPIA cases by pairing task data with attack instructions.

        Each task sample is paired with every attack instruction,
        producing |tasks| x |attacks| test cases (capped by max_cases).
        """
        task_data = self._load_task_data(task)
        attacks = self._load_attacks(attack_type)

        if not task_data or not attacks:
            return []

        # Flatten attacks into (category, instruction) pairs
        flat_attacks = []
        for cat, instructions in attacks.items():
            for instr in instructions:
                flat_attacks.append((cat, instr))

        cases = []
        count = 0
        for item in task_data:
            for cat, injection in flat_attacks:
                if max_cases and count >= max_cases:
                    break
                context = item.get("context", item.get("question", ""))
                question = item.get("question", "")
                ideal = item.get("ideal", "")
                if isinstance(ideal, list):
                    ideal = "\n".join(str(x) for x in ideal)
                if isinstance(context, list):
                    context = "\n".join(str(x) for x in context)

                cases.append(BIPIACase(
                    id="bipia_%s_%d" % (task, count),
                    task_type=task,
                    question=question,
                    context=context,
                    injection=injection,
                    attack_category=cat,
                    ideal=ideal,
                ))
                count += 1
            if max_cases and count >= max_cases:
                break

        logger.info(
            "Loaded %d BIPIA cases (%s, %s attacks)",
            len(cases), task, attack_type,
        )
        return cases
