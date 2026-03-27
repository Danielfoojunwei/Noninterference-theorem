"""Guarded agent runtime with explicit contaminated-input execution.

This module replaces the core empirical shortcut identified in the audit:
it is designed so that guarded execution operates on the *same contaminated
state* as the baseline, rather than receiving pre-cleaned inputs from the
outside.  Trusted and untrusted content enter through explicit channels,
provenance and taint are carried in state, and guard decisions are enforced at
multiple control points.

The runtime is intentionally model-agnostic.  A planner, LLM wrapper, or any
other controller may propose actions, but every proposal must pass through the
guard before it can affect the control plane, persistent memory, or the final
user-visible answer.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class Principal(Enum):
    """Origin principal for a piece of information."""

    SYS = "SYS"
    USER = "USER"
    TOOL_AUTH = "TOOL_AUTH"
    TOOL_UNAUTH = "TOOL_UNAUTH"
    WEB = "WEB"
    SKILL = "SKILL"
    MEMORY = "MEMORY"

    @property
    def is_tainted_by_default(self) -> bool:
        return self in {Principal.TOOL_UNAUTH, Principal.WEB, Principal.SKILL}


class TrustLevel(Enum):
    """Trust state of a node after propagation / promotion."""

    TRUSTED = "trusted"
    TAINTED = "tainted"
    VERIFIED = "verified"

    @property
    def is_action_safe(self) -> bool:
        return self in {TrustLevel.TRUSTED, TrustLevel.VERIFIED}


class NodeType(Enum):
    """Structured state node categories."""

    SYSTEM_POLICY = "SystemPolicy"
    USER_INTENT = "UserIntent"
    TRUSTED_CONFIG = "TrustedConfig"
    UNTRUSTED_CONTENT = "UntrustedContent"
    TOOL_OUTPUT = "ToolOutput"
    CANDIDATE_FACT = "CandidateFact"
    VERIFIED_FACT = "VerifiedFact"
    PLAN_STATE = "PlanState"
    MEMORY_ENTRY = "MemoryEntry"
    ACTION_PROPOSAL = "ActionProposal"
    FINAL_ANSWER = "FinalAnswer"
    ACTION_LOG = "ActionLog"


class ControlPoint(Enum):
    """Locations where the guard must enforce policy."""

    FINAL_ANSWER = "final_answer"
    TOOL_CALL = "tool_call"
    MEMORY_WRITE = "memory_write"
    PLAN_UPDATE = "plan_update"
    ACTION_EXECUTION = "action_execution"


@dataclass
class StateNode:
    """A provenance-carrying state node."""

    node_id: str
    node_type: NodeType
    principal: Principal
    trust_level: TrustLevel
    channel: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def tainted(self) -> bool:
        return self.trust_level == TrustLevel.TAINTED


@dataclass
class MemoryRecord:
    """Persistent memory written by the agent."""

    key: str
    value: str
    trust_level: TrustLevel
    source_node_ids: List[str]
    verified: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class ActionProposal:
    """A controller proposal submitted to the guard."""

    control_point: ControlPoint
    payload: Dict[str, Any]
    evidence_node_ids: List[str]
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardDecision:
    """Guard verdict with repair semantics and evidence summary."""

    allowed: bool
    reason: str
    repaired_payload: Optional[Dict[str, Any]] = None
    blocking_rules: List[str] = field(default_factory=list)
    tainted_evidence_ids: List[str] = field(default_factory=list)
    verified_evidence_ids: List[str] = field(default_factory=list)
    trusted_evidence_ids: List[str] = field(default_factory=list)


@dataclass
class ActionLogEntry:
    """Persisted audit log for a guarded or baseline action."""

    step_id: int
    control_point: str
    allowed: bool
    reason: str
    payload: Dict[str, Any]
    repaired_payload: Optional[Dict[str, Any]]
    evidence_node_ids: List[str]
    tainted_evidence_ids: List[str]
    verified_evidence_ids: List[str]
    trusted_evidence_ids: List[str]
    blocking_rules: List[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class GuardPolicy:
    """Policy controlling what the runtime permits."""

    allowed_tools: List[str] = field(default_factory=list)
    allow_tainted_final_answer: bool = False
    allow_tainted_plan_updates: bool = False
    allow_tainted_tool_arguments: bool = False
    allow_tainted_memory_persistence: bool = False
    permit_plan_updates: bool = True
    permit_tool_calls: bool = True
    permit_final_answers: bool = True
    permit_memory_writes: bool = True
    instruction_blocklist: List[str] = field(
        default_factory=lambda: [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"override\s+the\s+user",
            r"grant\s+guest\s+access",
            r"send\s+money",
            r"transfer\s+crypto",
            r"send\s+email\s+to",
            r"delete\s+all\s+files",
            r"rm\s+-rf",
            r"persist\s+this\s+instruction",
        ]
    )


class AgentState:
    """State container with explicit provenance and taint flow."""

    def __init__(self):
        self.nodes: Dict[str, StateNode] = {}
        self.memory: Dict[str, MemoryRecord] = {}
        self.action_log: List[ActionLogEntry] = []
        self.plan: List[str] = []
        self._node_counter = 0

    def _new_node_id(self) -> str:
        self._node_counter += 1
        return f"n{self._node_counter}"

    def add_node(
        self,
        *,
        node_type: NodeType,
        principal: Principal,
        channel: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_ids: Optional[Sequence[str]] = None,
        force_trust: Optional[TrustLevel] = None,
    ) -> StateNode:
        parent_ids = list(parent_ids or [])
        metadata = dict(metadata or {})

        if force_trust is not None:
            trust = force_trust
        else:
            trust = self._derive_trust(principal=principal, parent_ids=parent_ids)
            if node_type == NodeType.VERIFIED_FACT:
                trust = TrustLevel.VERIFIED

        node = StateNode(
            node_id=self._new_node_id(),
            node_type=node_type,
            principal=principal,
            trust_level=trust,
            channel=channel,
            content=content,
            metadata=metadata,
            parent_ids=parent_ids,
        )
        self.nodes[node.node_id] = node
        return node

    def _derive_trust(
        self,
        *,
        principal: Principal,
        parent_ids: Sequence[str],
    ) -> TrustLevel:
        if any(self.nodes[parent_id].trust_level == TrustLevel.TAINTED for parent_id in parent_ids):
            return TrustLevel.TAINTED
        if principal.is_tainted_by_default:
            return TrustLevel.TAINTED
        return TrustLevel.TRUSTED

    def get_nodes(self, node_ids: Iterable[str]) -> List[StateNode]:
        return [self.nodes[node_id] for node_id in node_ids if node_id in self.nodes]

    def evidence_summary(self, node_ids: Sequence[str]) -> Dict[str, List[str]]:
        trusted_ids: List[str] = []
        tainted_ids: List[str] = []
        verified_ids: List[str] = []
        for node in self.get_nodes(node_ids):
            if node.trust_level == TrustLevel.TAINTED:
                tainted_ids.append(node.node_id)
            elif node.trust_level == TrustLevel.VERIFIED:
                verified_ids.append(node.node_id)
            else:
                trusted_ids.append(node.node_id)
        return {
            "trusted": trusted_ids,
            "tainted": tainted_ids,
            "verified": verified_ids,
        }

    def serialise(self) -> Dict[str, Any]:
        return {
            "nodes": [asdict(node) for node in self.nodes.values()],
            "memory": {
                key: asdict(value) for key, value in self.memory.items()
            },
            "action_log": [asdict(entry) for entry in self.action_log],
            "plan": list(self.plan),
        }


class BaseAgentRuntime:
    """Common runtime for baseline and guarded execution."""

    def __init__(self, policy: Optional[GuardPolicy] = None):
        self.policy = policy or GuardPolicy()
        self.state = AgentState()
        self._step_id = 0

    # ------------------------------------------------------------------
    # Input ingestion
    # ------------------------------------------------------------------
    def ingest_trusted_input(
        self,
        content: str,
        *,
        channel: str = "trusted_input",
        principal: Principal = Principal.USER,
        metadata: Optional[Dict[str, Any]] = None,
        node_type: NodeType = NodeType.USER_INTENT,
    ) -> StateNode:
        return self.state.add_node(
            node_type=node_type,
            principal=principal,
            channel=channel,
            content=content,
            metadata=metadata,
        )

    def ingest_untrusted_input(
        self,
        content: str,
        *,
        channel: str,
        principal: Principal,
        metadata: Optional[Dict[str, Any]] = None,
        node_type: NodeType = NodeType.UNTRUSTED_CONTENT,
    ) -> StateNode:
        return self.state.add_node(
            node_type=node_type,
            principal=principal,
            channel=channel,
            content=content,
            metadata=metadata,
        )

    def ingest_tool_output(
        self,
        content: str,
        *,
        authenticated: bool,
        channel: str = "tool_output",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StateNode:
        principal = Principal.TOOL_AUTH if authenticated else Principal.TOOL_UNAUTH
        return self.state.add_node(
            node_type=NodeType.TOOL_OUTPUT,
            principal=principal,
            channel=channel,
            content=content,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Proposal application
    # ------------------------------------------------------------------
    def apply_proposal(self, proposal: ActionProposal) -> GuardDecision:
        raise NotImplementedError

    def _log_decision(
        self,
        proposal: ActionProposal,
        decision: GuardDecision,
    ) -> ActionLogEntry:
        self._step_id += 1
        entry = ActionLogEntry(
            step_id=self._step_id,
            control_point=proposal.control_point.value,
            allowed=decision.allowed,
            reason=decision.reason,
            payload=dict(proposal.payload),
            repaired_payload=(dict(decision.repaired_payload) if decision.repaired_payload else None),
            evidence_node_ids=list(proposal.evidence_node_ids),
            tainted_evidence_ids=list(decision.tainted_evidence_ids),
            verified_evidence_ids=list(decision.verified_evidence_ids),
            trusted_evidence_ids=list(decision.trusted_evidence_ids),
            blocking_rules=list(decision.blocking_rules),
        )
        self.state.action_log.append(entry)
        self.state.add_node(
            node_type=NodeType.ACTION_LOG,
            principal=Principal.SYS,
            channel="audit_log",
            content=json.dumps(asdict(entry), sort_keys=True),
            metadata={"step_id": entry.step_id},
        )
        return entry

    def export_trace(self) -> Dict[str, Any]:
        return self.state.serialise()


class BaselineAgentRuntime(BaseAgentRuntime):
    """Naive runtime with no taint-aware restrictions.

    This runtime is intentionally permissive and represents an agent that reads
    contaminated inputs and executes what it proposes without a meaningful
    control-plane guard.
    """

    def apply_proposal(self, proposal: ActionProposal) -> GuardDecision:
        evidence = self.state.evidence_summary(proposal.evidence_node_ids)
        decision = GuardDecision(
            allowed=True,
            reason="baseline_allows_without_guard",
            repaired_payload=dict(proposal.payload),
            tainted_evidence_ids=evidence["tainted"],
            verified_evidence_ids=evidence["verified"],
            trusted_evidence_ids=evidence["trusted"],
        )
        self._apply_effects(proposal.control_point, proposal.payload, evidence_ids=proposal.evidence_node_ids)
        self._log_decision(proposal, decision)
        return decision

    def _apply_effects(
        self,
        control_point: ControlPoint,
        payload: Dict[str, Any],
        *,
        evidence_ids: Sequence[str],
    ) -> None:
        if control_point == ControlPoint.PLAN_UPDATE:
            self.state.plan.append(payload.get("plan_update", ""))
            self.state.add_node(
                node_type=NodeType.PLAN_STATE,
                principal=Principal.MEMORY,
                channel="plan",
                content=payload.get("plan_update", ""),
                metadata={"guarded": False},
                parent_ids=list(evidence_ids),
            )
        elif control_point == ControlPoint.MEMORY_WRITE:
            key = payload.get("key", "memory")
            value = payload.get("value", "")
            self.state.memory[key] = MemoryRecord(
                key=key,
                value=value,
                trust_level=TrustLevel.TRUSTED,
                source_node_ids=list(evidence_ids),
                verified=True,
                metadata={"guarded": False},
            )
            self.state.add_node(
                node_type=NodeType.MEMORY_ENTRY,
                principal=Principal.MEMORY,
                channel="memory",
                content=value,
                metadata={"key": key, "verified": True, "guarded": False},
                parent_ids=list(evidence_ids),
                force_trust=TrustLevel.TRUSTED,
            )
        elif control_point == ControlPoint.FINAL_ANSWER:
            self.state.add_node(
                node_type=NodeType.FINAL_ANSWER,
                principal=Principal.MEMORY,
                channel="final_answer",
                content=payload.get("text", ""),
                metadata={"guarded": False},
                parent_ids=list(evidence_ids),
            )
        elif control_point in {ControlPoint.TOOL_CALL, ControlPoint.ACTION_EXECUTION}:
            self.state.add_node(
                node_type=NodeType.ACTION_PROPOSAL,
                principal=Principal.MEMORY,
                channel="tool_call",
                content=json.dumps(payload, sort_keys=True),
                metadata={"guarded": False, "executed": True},
                parent_ids=list(evidence_ids),
            )


class GuardedAgentRuntime(BaseAgentRuntime):
    """Taint-aware runtime that enforces multi-point control restrictions.

    The key property is that contaminated inputs remain present in state and may
    still be inspected, logged, or transformed into candidate artifacts, but
    tainted evidence cannot directly drive control-relevant actions unless it
    has been explicitly promoted to verified data.
    """

    def apply_proposal(self, proposal: ActionProposal) -> GuardDecision:
        decision = self._evaluate(proposal)
        payload_to_apply = (
            dict(decision.repaired_payload)
            if decision.repaired_payload is not None
            else dict(proposal.payload)
        )
        if decision.allowed:
            self._apply_guarded_effects(
                proposal.control_point,
                payload_to_apply,
                evidence_ids=proposal.evidence_node_ids,
            )
        self._log_decision(proposal, decision)
        return decision

    def promote_verified_fact(
        self,
        *,
        candidate_node_id: str,
        verified_value: str,
        verifier_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StateNode:
        """Promote a candidate fact to a verified fact.

        This method is the explicit declassification hook used by later phases.
        It is deliberately narrow: callers must name the candidate source node
        and the verifier responsible for promotion.
        """
        if candidate_node_id not in self.state.nodes:
            raise KeyError(f"Unknown candidate node: {candidate_node_id}")
        source = self.state.nodes[candidate_node_id]
        return self.state.add_node(
            node_type=NodeType.VERIFIED_FACT,
            principal=Principal.SYS,
            channel="verified_fact",
            content=verified_value,
            metadata={"verifier": verifier_name, **(metadata or {})},
            parent_ids=[candidate_node_id],
            force_trust=TrustLevel.VERIFIED,
        )

    def _evaluate(self, proposal: ActionProposal) -> GuardDecision:
        evidence = self.state.evidence_summary(proposal.evidence_node_ids)
        tainted = evidence["tainted"]
        verified = evidence["verified"]
        trusted = evidence["trusted"]

        # Disabled control points
        if proposal.control_point == ControlPoint.PLAN_UPDATE and not self.policy.permit_plan_updates:
            return GuardDecision(
                allowed=False,
                reason="plan_updates_disabled_by_policy",
                blocking_rules=["permit_plan_updates"],
                tainted_evidence_ids=tainted,
                verified_evidence_ids=verified,
                trusted_evidence_ids=trusted,
            )
        if proposal.control_point in {ControlPoint.TOOL_CALL, ControlPoint.ACTION_EXECUTION} and not self.policy.permit_tool_calls:
            return GuardDecision(
                allowed=False,
                reason="tool_calls_disabled_by_policy",
                blocking_rules=["permit_tool_calls"],
                tainted_evidence_ids=tainted,
                verified_evidence_ids=verified,
                trusted_evidence_ids=trusted,
            )
        if proposal.control_point == ControlPoint.MEMORY_WRITE and not self.policy.permit_memory_writes:
            return GuardDecision(
                allowed=False,
                reason="memory_writes_disabled_by_policy",
                blocking_rules=["permit_memory_writes"],
                tainted_evidence_ids=tainted,
                verified_evidence_ids=verified,
                trusted_evidence_ids=trusted,
            )
        if proposal.control_point == ControlPoint.FINAL_ANSWER and not self.policy.permit_final_answers:
            return GuardDecision(
                allowed=False,
                reason="final_answers_disabled_by_policy",
                blocking_rules=["permit_final_answers"],
                tainted_evidence_ids=tainted,
                verified_evidence_ids=verified,
                trusted_evidence_ids=trusted,
            )

        # Control-point-specific checks
        if proposal.control_point in {ControlPoint.TOOL_CALL, ControlPoint.ACTION_EXECUTION}:
            return self._evaluate_tool_or_action(proposal, tainted, verified, trusted)
        if proposal.control_point == ControlPoint.MEMORY_WRITE:
            return self._evaluate_memory_write(proposal, tainted, verified, trusted)
        if proposal.control_point == ControlPoint.PLAN_UPDATE:
            return self._evaluate_plan_update(proposal, tainted, verified, trusted)
        if proposal.control_point == ControlPoint.FINAL_ANSWER:
            return self._evaluate_final_answer(proposal, tainted, verified, trusted)

        return GuardDecision(
            allowed=False,
            reason="unknown_control_point",
            blocking_rules=["unknown_control_point"],
            tainted_evidence_ids=tainted,
            verified_evidence_ids=verified,
            trusted_evidence_ids=trusted,
        )

    def _evaluate_tool_or_action(
        self,
        proposal: ActionProposal,
        tainted: List[str],
        verified: List[str],
        trusted: List[str],
    ) -> GuardDecision:
        tool_name = str(proposal.payload.get("tool_name", ""))
        arguments = proposal.payload.get("arguments", {})

        rules: List[str] = []
        if tool_name not in self.policy.allowed_tools:
            rules.append("tool_not_allowlisted")
        if tainted:
            rules.append("tainted_evidence_for_control_action")
        if self._payload_contains_blocklisted_instruction(proposal.payload):
            rules.append("instructional_payload_detected")
        if tainted and not self.policy.allow_tainted_tool_arguments:
            rules.append("tainted_tool_arguments_disallowed")

        if rules:
            return GuardDecision(
                allowed=False,
                reason="tool_or_action_blocked",
                repaired_payload={
                    "tool_name": None,
                    "arguments": {},
                    "blocked_tool_name": tool_name,
                    "blocked_arguments_hash": self._stable_hash(arguments),
                },
                blocking_rules=rules,
                tainted_evidence_ids=tainted,
                verified_evidence_ids=verified,
                trusted_evidence_ids=trusted,
            )

        return GuardDecision(
            allowed=True,
            reason="tool_or_action_allowed",
            repaired_payload=dict(proposal.payload),
            tainted_evidence_ids=tainted,
            verified_evidence_ids=verified,
            trusted_evidence_ids=trusted,
        )

    def _evaluate_memory_write(
        self,
        proposal: ActionProposal,
        tainted: List[str],
        verified: List[str],
        trusted: List[str],
    ) -> GuardDecision:
        if tainted and not self.policy.allow_tainted_memory_persistence:
            repaired = dict(proposal.payload)
            repaired["verified"] = False
            repaired["quarantined"] = True
            repaired["storage_class"] = "candidate_fact"
            return GuardDecision(
                allowed=True,
                reason="memory_write_quarantined_due_to_taint",
                repaired_payload=repaired,
                blocking_rules=["tainted_memory_quarantined"],
                tainted_evidence_ids=tainted,
                verified_evidence_ids=verified,
                trusted_evidence_ids=trusted,
            )

        return GuardDecision(
            allowed=True,
            reason="memory_write_allowed",
            repaired_payload={**proposal.payload, "verified": True},
            tainted_evidence_ids=tainted,
            verified_evidence_ids=verified,
            trusted_evidence_ids=trusted,
        )

    def _evaluate_plan_update(
        self,
        proposal: ActionProposal,
        tainted: List[str],
        verified: List[str],
        trusted: List[str],
    ) -> GuardDecision:
        rules: List[str] = []
        if tainted and not self.policy.allow_tainted_plan_updates:
            rules.append("tainted_plan_update_disallowed")
        if self._payload_contains_blocklisted_instruction(proposal.payload):
            rules.append("instructional_payload_detected")
        if rules:
            return GuardDecision(
                allowed=False,
                reason="plan_update_blocked",
                repaired_payload={"plan_update": None, "blocked": True},
                blocking_rules=rules,
                tainted_evidence_ids=tainted,
                verified_evidence_ids=verified,
                trusted_evidence_ids=trusted,
            )
        return GuardDecision(
            allowed=True,
            reason="plan_update_allowed",
            repaired_payload=dict(proposal.payload),
            tainted_evidence_ids=tainted,
            verified_evidence_ids=verified,
            trusted_evidence_ids=trusted,
        )

    def _evaluate_final_answer(
        self,
        proposal: ActionProposal,
        tainted: List[str],
        verified: List[str],
        trusted: List[str],
    ) -> GuardDecision:
        rules: List[str] = []
        if tainted and not self.policy.allow_tainted_final_answer:
            rules.append("tainted_final_answer_disallowed")
        if self._payload_contains_blocklisted_instruction(proposal.payload):
            rules.append("instructional_payload_detected")
        if rules:
            safe_text = (
                "The requested content was reviewed, but the current policy does "
                "not allow directly emitting action-relevant output derived from "
                "untrusted content without verification."
            )
            return GuardDecision(
                allowed=True,
                reason="final_answer_repaired_to_safe_template",
                repaired_payload={"text": safe_text, "safe_template": True},
                blocking_rules=rules,
                tainted_evidence_ids=tainted,
                verified_evidence_ids=verified,
                trusted_evidence_ids=trusted,
            )
        return GuardDecision(
            allowed=True,
            reason="final_answer_allowed",
            repaired_payload=dict(proposal.payload),
            tainted_evidence_ids=tainted,
            verified_evidence_ids=verified,
            trusted_evidence_ids=trusted,
        )

    def _apply_guarded_effects(
        self,
        control_point: ControlPoint,
        payload: Dict[str, Any],
        *,
        evidence_ids: Sequence[str],
    ) -> None:
        if control_point == ControlPoint.PLAN_UPDATE:
            plan_update = payload.get("plan_update")
            if plan_update:
                self.state.plan.append(plan_update)
                self.state.add_node(
                    node_type=NodeType.PLAN_STATE,
                    principal=Principal.MEMORY,
                    channel="plan",
                    content=plan_update,
                    metadata={"guarded": True},
                    parent_ids=list(evidence_ids),
                )
        elif control_point == ControlPoint.MEMORY_WRITE:
            key = payload.get("key", "memory")
            value = payload.get("value", "")
            verified_flag = bool(payload.get("verified", False)) and not payload.get("quarantined", False)
            trust = TrustLevel.VERIFIED if verified_flag else TrustLevel.TAINTED
            self.state.memory[key] = MemoryRecord(
                key=key,
                value=value,
                trust_level=trust,
                source_node_ids=list(evidence_ids),
                verified=verified_flag,
                metadata={
                    "guarded": True,
                    "storage_class": payload.get("storage_class", "verified_memory" if verified_flag else "candidate_fact"),
                    "quarantined": bool(payload.get("quarantined", False)),
                },
            )
            self.state.add_node(
                node_type=NodeType.MEMORY_ENTRY,
                principal=Principal.MEMORY,
                channel="memory",
                content=value,
                metadata={
                    "key": key,
                    "verified": verified_flag,
                    "guarded": True,
                    "quarantined": bool(payload.get("quarantined", False)),
                },
                parent_ids=list(evidence_ids),
                force_trust=trust,
            )
        elif control_point == ControlPoint.FINAL_ANSWER:
            self.state.add_node(
                node_type=NodeType.FINAL_ANSWER,
                principal=Principal.MEMORY,
                channel="final_answer",
                content=payload.get("text", ""),
                metadata={"guarded": True, "safe_template": payload.get("safe_template", False)},
                parent_ids=list(evidence_ids),
                force_trust=TrustLevel.TRUSTED,
            )
        elif control_point in {ControlPoint.TOOL_CALL, ControlPoint.ACTION_EXECUTION}:
            self.state.add_node(
                node_type=NodeType.ACTION_PROPOSAL,
                principal=Principal.MEMORY,
                channel="tool_call",
                content=json.dumps(payload, sort_keys=True),
                metadata={"guarded": True, "executed": True},
                parent_ids=list(evidence_ids),
                force_trust=TrustLevel.TRUSTED,
            )

    def _payload_contains_blocklisted_instruction(self, payload: Dict[str, Any]) -> bool:
        text = json.dumps(payload, sort_keys=True).lower()
        return any(re.search(pattern, text) for pattern in self.policy.instruction_blocklist)

    @staticmethod
    def _stable_hash(value: Any) -> str:
        encoded = json.dumps(value, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def summarise_trace(runtime: BaseAgentRuntime) -> Dict[str, Any]:
    """Compact summary used by tests and experiment logs."""

    return {
        "plan": list(runtime.state.plan),
        "memory": {
            key: {
                "value": record.value,
                "trust_level": record.trust_level.value,
                "verified": record.verified,
                "source_node_ids": list(record.source_node_ids),
                "metadata": dict(record.metadata),
            }
            for key, record in runtime.state.memory.items()
        },
        "action_log": [asdict(entry) for entry in runtime.state.action_log],
        "node_count": len(runtime.state.nodes),
    }


def find_tainted_nodes_by_channel(
    runtime: BaseAgentRuntime,
    channel: str,
) -> List[StateNode]:
    """Helper for experiments that need to inspect contaminated state."""

    return [
        node
        for node in runtime.state.nodes.values()
        if node.channel == channel and node.trust_level == TrustLevel.TAINTED
    ]
