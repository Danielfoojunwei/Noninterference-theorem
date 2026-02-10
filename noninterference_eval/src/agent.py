"""
Agent Under Test (AUT) Implementation
Implements the Noninterference Theorem for Indirect Prompt Injection

This module implements a recursive language model (RLM) agent with:
- Typed intermediate representation (IR) graph
- Taint tracking for untrusted inputs
- Trust lattice with principal-based access control
- Verifier for control-plane invariants
"""

import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types for IR graph nodes"""
    POLICY = "Policy"
    USER_INTENT = "UserIntent"
    TRUSTED_CONFIG = "TrustedConfig"
    UNTRUSTED_QUOTE = "UntrustedQuote"
    CANDIDATE_FACT = "CandidateFact"
    VERIFIED_FACT = "VerifiedFact"
    TOOL_RESULT = "ToolResult"
    ACTION_REQUEST = "ActionRequest"


class Principal(Enum):
    """Principals in the trust lattice"""
    SYS = "SYS"
    USER = "USER"
    TOOL = "TOOL"
    WEB = "WEB"
    SKILL = "SKILL"

    def __le__(self, other):
        """Define the trust ordering: WEB, SKILL <= TOOL <= USER <= SYS"""
        order = {
            Principal.WEB: 0,
            Principal.SKILL: 0,
            Principal.TOOL: 1,
            Principal.USER: 2,
            Principal.SYS: 3,
        }
        return order[self] <= order[other]


@dataclass
class IRNode:
    """Node in the intermediate representation graph"""
    id: str
    node_type: NodeType
    content: str
    provenance: Principal
    tainted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)


@dataclass
class IRGraph:
    """Typed intermediate representation graph"""
    nodes: Dict[str, IRNode] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)

    def add_node(self, node: IRNode):
        """Add a node to the graph"""
        self.nodes[node.id] = node
        logger.debug(
            "Added node %s (type=%s, tainted=%s)",
            node.id, node.node_type.value, node.tainted,
        )

    def add_edge(self, from_id: str, to_id: str):
        """Add an edge to the graph"""
        self.edges.append((from_id, to_id))

    def get_untainted_nodes(self) -> List[IRNode]:
        """Get all untainted nodes"""
        return [node for node in self.nodes.values() if not node.tainted]

    def get_nodes_by_type(self, node_type: NodeType) -> List[IRNode]:
        """Get all nodes of a specific type"""
        return [node for node in self.nodes.values()
                if node.node_type == node_type]


@dataclass
class ControlPlaneState:
    """Control-plane state P_t"""
    permissions: Dict[str, bool] = field(default_factory=dict)
    tool_policies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    integration_config: Dict[str, Any] = field(default_factory=dict)

    def __eq__(self, other):
        """Check equality for noninterference verification"""
        if not isinstance(other, ControlPlaneState):
            return False
        return (
            self.permissions == other.permissions
            and self.tool_policies == other.tool_policies
            and self.integration_config == other.integration_config
        )

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "permissions": self.permissions,
            "tool_policies": self.tool_policies,
            "integration_config": self.integration_config,
        }


@dataclass
class MemoryStore:
    """Memory store M_t for verified facts"""
    facts: List[Dict[str, Any]] = field(default_factory=list)

    def add_fact(self, fact: Dict[str, Any], verified: bool = False):
        """Add a fact to memory"""
        fact["verified"] = verified
        self.facts.append(fact)


@dataclass
class RiskBudget:
    """Risk budget B_t"""
    total: float = 100.0
    spent: float = 0.0

    def can_spend(self, amount: float) -> bool:
        """Check if we can spend the given amount"""
        return (self.spent + amount) <= self.total

    def spend(self, amount: float):
        """Spend risk budget"""
        if self.can_spend(amount):
            self.spent += amount
            return True
        return False


@dataclass
class AgentState:
    """Complete agent state S_t = (P_t, M_t, B_t, G_t)"""
    control_plane: ControlPlaneState
    memory: MemoryStore
    risk_budget: RiskBudget
    ir_graph: IRGraph
    step: int = 0


@dataclass
class ToolCall:
    """Represents a tool call action"""
    tool_name: str
    arguments: Dict[str, Any]
    risk_score: float = 0.0

    def __eq__(self, other):
        """Check equality for noninterference verification"""
        if not isinstance(other, ToolCall):
            return False
        return (
            self.tool_name == other.tool_name
            and self.arguments == other.arguments
        )

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "risk_score": self.risk_score,
        }


class Verifier:
    """Deterministic verifier V that checks invariants"""

    @staticmethod
    def verify_tool_call(
        tool_call: ToolCall,
        ir_graph: IRGraph,
        control_plane: ControlPlaneState,
    ) -> bool:
        """
        Verify that a tool call only depends on untainted nodes
        and respects control-plane policies
        """
        # Check 1: Tool must be in allowed list
        if tool_call.tool_name not in control_plane.tool_policies:
            logger.warning("Tool %s not in allowed policies", tool_call.tool_name)
            return False

        # Check 2: Tool call must only depend on untainted nodes
        action_nodes = ir_graph.get_nodes_by_type(NodeType.ACTION_REQUEST)
        for node in action_nodes:
            if node.tainted:
                logger.warning(
                    "Tool call depends on tainted node %s", node.id,
                )
                return False

        return True

    @staticmethod
    def verify_control_plane_update(
        old_state: ControlPlaneState,
        new_state: ControlPlaneState,
        principal: Principal,
    ) -> bool:
        """
        Verify that control-plane updates only come from SYS or USER principals
        """
        if old_state == new_state:
            return True

        # Only SYS and USER can modify control plane
        if principal not in [Principal.SYS, Principal.USER]:
            logger.warning(
                "Principal %s attempted to modify control plane",
                principal.value,
            )
            return False

        return True


class RecursiveLanguageModel:
    """
    Recursive Language Model (RLM) controller T_theta

    This is a simplified implementation that demonstrates the core concepts.
    In a real implementation, this would use an actual LLM (GPT-4, Claude, etc.)
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.verifier = Verifier()

    def propose_update(
        self,
        state: AgentState,
        trusted_input: Dict[str, Any],
        untrusted_input: Optional[Dict[str, Any]] = None,
    ) -> Tuple[AgentState, str, Optional[ToolCall]]:
        """
        Propose an updated state and candidate tool call
        Returns: (new_state, response_text, tool_call)
        """
        # Create a new state
        new_state = AgentState(
            control_plane=ControlPlaneState(
                permissions=state.control_plane.permissions.copy(),
                tool_policies=state.control_plane.tool_policies.copy(),
                integration_config=state.control_plane.integration_config.copy(),
            ),
            memory=MemoryStore(facts=state.memory.facts.copy()),
            risk_budget=RiskBudget(
                total=state.risk_budget.total,
                spent=state.risk_budget.spent,
            ),
            ir_graph=IRGraph(
                nodes=state.ir_graph.nodes.copy(),
                edges=state.ir_graph.edges.copy(),
            ),
            step=state.step + 1,
        )

        # Process trusted input
        if trusted_input:
            self._process_trusted_input(new_state, trusted_input)

        # Process untrusted input with taint tracking
        if untrusted_input:
            self._process_untrusted_input(new_state, untrusted_input)

        # Choose action based ONLY on untainted nodes
        tool_call = self._choose_action(new_state)

        # Generate response text
        response_text = self._generate_response(new_state, tool_call)

        return new_state, response_text, tool_call

    def _process_trusted_input(
        self, state: AgentState, trusted_input: Dict[str, Any],
    ):
        """Process trusted input and add to IR graph"""
        principal = Principal[trusted_input.get("principal", "USER")]
        content = trusted_input.get("content", "")
        input_type = trusted_input.get("type", "USER_INTENT")

        node_id = self._generate_node_id(content)
        node = IRNode(
            id=node_id,
            node_type=NodeType[input_type],
            content=content,
            provenance=principal,
            tainted=False,  # Trusted input is never tainted
        )
        state.ir_graph.add_node(node)

    def _process_untrusted_input(
        self, state: AgentState, untrusted_input: Dict[str, Any],
    ):
        """Process untrusted input with taint tracking"""
        principal = Principal[untrusted_input.get("principal", "WEB")]
        content = untrusted_input.get("content", "")

        node_id = self._generate_node_id(content)
        node = IRNode(
            id=node_id,
            node_type=NodeType.UNTRUSTED_QUOTE,
            content=content,
            provenance=principal,
            tainted=True,  # Untrusted input is always tainted
        )
        state.ir_graph.add_node(node)

        # Extract candidate facts (these remain tainted until verified)
        facts = self._extract_facts(content)
        for fact in facts:
            fact_id = self._generate_node_id(fact)
            fact_node = IRNode(
                id=fact_id,
                node_type=NodeType.CANDIDATE_FACT,
                content=fact,
                provenance=principal,
                tainted=True,  # Candidate facts from untrusted sources are tainted
            )
            state.ir_graph.add_node(fact_node)
            state.ir_graph.add_edge(node_id, fact_id)

    def _choose_action(self, state: AgentState) -> Optional[ToolCall]:
        """
        Choose an action based ONLY on untainted IR nodes.
        This is the key enforcement point for noninterference.
        """
        # Get only untainted nodes
        untainted_nodes = state.ir_graph.get_untainted_nodes()

        # Look for USER_INTENT nodes to determine what tool to call
        intent_nodes = [
            n for n in untainted_nodes
            if n.node_type == NodeType.USER_INTENT
        ]

        if not intent_nodes:
            return None

        # Simple keyword matching to determine tool
        intent_text = " ".join(n.content for n in intent_nodes).lower()

        tool_call = None
        if "search" in intent_text or "find" in intent_text:
            tool_call = ToolCall(
                tool_name="web_search",
                arguments={"query": intent_text},
                risk_score=5.0,
            )
        elif "email" in intent_text or "send" in intent_text:
            tool_call = ToolCall(
                tool_name="send_email",
                arguments={"subject": "Response", "body": intent_text},
                risk_score=10.0,
            )
        elif "read" in intent_text or "file" in intent_text:
            tool_call = ToolCall(
                tool_name="read_file",
                arguments={"path": "/tmp/data.txt"},
                risk_score=3.0,
            )

        # Verify the tool call
        if tool_call:
            if not self.verifier.verify_tool_call(
                tool_call, state.ir_graph, state.control_plane,
            ):
                logger.warning("Tool call failed verification")
                return None

            # Check risk budget
            if not state.risk_budget.can_spend(tool_call.risk_score):
                logger.warning("Insufficient risk budget")
                return None

        return tool_call

    def _generate_response(
        self, state: AgentState, tool_call: Optional[ToolCall],
    ) -> str:
        """Generate a response text"""
        if tool_call:
            return (
                "Executing %s with arguments %s"
                % (tool_call.tool_name, tool_call.arguments)
            )
        return "No action taken"

    def _extract_facts(self, content: str) -> List[str]:
        """Extract facts from content (simplified)"""
        return [s.strip() for s in content.split(".") if s.strip()]

    def _generate_node_id(self, content: str) -> str:
        """Generate a unique node ID based on content"""
        return hashlib.md5(content.encode()).hexdigest()[:16]


class AgentUnderTest:
    """
    Main agent class that implements the noninterference theorem
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model = RecursiveLanguageModel(model_name)
        self.verifier = Verifier()
        self.execution_trace = []

    def initialize(
        self,
        initial_permissions: Dict[str, bool],
        tool_policies: Dict[str, Dict[str, Any]],
    ):
        """Initialize the agent with control-plane configuration"""
        self.state = AgentState(
            control_plane=ControlPlaneState(
                permissions=initial_permissions,
                tool_policies=tool_policies,
                integration_config={},
            ),
            memory=MemoryStore(),
            risk_budget=RiskBudget(total=100.0, spent=0.0),
            ir_graph=IRGraph(),
            step=0,
        )
        logger.info("Agent initialized")

    def run_task(
        self,
        task_description: str,
        trusted_inputs: List[Dict[str, Any]],
        untrusted_inputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run a task with the given inputs.
        Returns execution trace for analysis.
        """
        logger.info("Starting task: %s", task_description)

        # Add task description as a trusted input
        task_input = {
            "principal": "USER",
            "type": "USER_INTENT",
            "content": task_description,
        }

        # Process all inputs
        all_trusted = [task_input] + trusted_inputs

        for i, trusted in enumerate(all_trusted):
            untrusted = (
                untrusted_inputs[i]
                if i < len(untrusted_inputs)
                else None
            )
            logger.info("Step %d", i + 1)

            # Propose update
            new_state, response, tool_call = self.model.propose_update(
                self.state, trusted, untrusted,
            )

            # Verify control-plane update
            principal = Principal[trusted.get("principal", "USER")]
            if not self.verifier.verify_control_plane_update(
                self.state.control_plane,
                new_state.control_plane,
                principal,
            ):
                logger.error("Control-plane update failed verification")
                # Apply safe repair
                tool_call = None
                response = "Action blocked by verifier"
            else:
                # Update state
                self.state = new_state

            # Record trace
            trace_entry = {
                "step": i + 1,
                "control_plane": self.state.control_plane.to_dict(),
                "tool_call": tool_call.to_dict() if tool_call else None,
                "response": response,
                "risk_spent": self.state.risk_budget.spent,
            }
            self.execution_trace.append(trace_entry)

            logger.info("Response: %s", response)
            if tool_call:
                logger.info("Tool call: %s", tool_call.tool_name)
                # Spend risk budget
                self.state.risk_budget.spend(tool_call.risk_score)

        return {
            "task": task_description,
            "trace": self.execution_trace,
            "final_state": {
                "control_plane": self.state.control_plane.to_dict(),
                "risk_spent": self.state.risk_budget.spent,
            },
        }

    def get_execution_trace(self) -> List[Dict[str, Any]]:
        """Get the execution trace"""
        return self.execution_trace

    def reset(self):
        """Reset the agent state"""
        self.execution_trace = []
