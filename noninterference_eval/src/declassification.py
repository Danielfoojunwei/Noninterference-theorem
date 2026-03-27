"""Concrete declassification mechanisms for guarded agent execution.

The repository's formal model already describes declassification via verified-fact
promotion.  This module turns that idea into executable code that can be used in
real guarded runs and benchmarked as an explicit subsystem.

The initial implementation focuses on a structured extraction strategy:

1. untrusted text is parsed into candidate fields;
2. each candidate is validated against a typed schema and policy checks;
3. only accepted fields are promoted to verified facts;
4. unsafe spans are quarantined for reference rather than allowed to influence
   control-relevant behavior.

The strategy interface is intentionally extensible so future benchmark arms can
compare multiple declassification mechanisms without changing the guarded-agent
runtime.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from guarded_runtime import (
    GuardedAgentRuntime,
    NodeType,
    Principal,
)


@dataclass
class FieldSpec:
    """Typed field schema for structured declassification."""

    name: str
    field_type: str
    description: str = ""
    required: bool = False
    max_length: int = 256
    allow_patterns: List[str] = field(default_factory=list)
    reject_patterns: List[str] = field(default_factory=list)
    action_influencing: bool = False


@dataclass
class DeclassifiedField:
    """Outcome for a single extracted field."""

    name: str
    field_type: str
    raw_value: Optional[str]
    normalized_value: Optional[str]
    accepted: bool
    reason: str
    candidate_node_id: Optional[str] = None
    verified_node_id: Optional[str] = None


@dataclass
class DeclassificationResult:
    """Full result of a declassification pass over one tainted source."""

    strategy_name: str
    source_node_id: str
    accepted_fields: Dict[str, str] = field(default_factory=dict)
    fields: List[DeclassifiedField] = field(default_factory=list)
    candidate_node_ids: List[str] = field(default_factory=list)
    verified_node_ids: List[str] = field(default_factory=list)
    quarantined_spans: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeclassificationStrategy(ABC):
    """Base interface for benchmarkable declassification strategies."""

    name = "base"

    @abstractmethod
    def declassify(
        self,
        *,
        runtime: GuardedAgentRuntime,
        source_node_id: str,
        schema: Sequence[FieldSpec],
    ) -> DeclassificationResult:
        """Reduce tainted source content into policy-checked verified data."""


class StructuredSchemaExtractionStrategy(DeclassificationStrategy):
    """Structured extraction into a typed schema with policy checks.

    The strategy is intentionally conservative.  It permits only a bounded set
    of typed fields, validates them explicitly, and quarantines instruction-like
    spans instead of letting them flow into verified state.
    """

    name = "structured_schema"

    EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
    ASIN_RE = re.compile(r"\b[A-Z0-9]{10}\b", re.I)
    PRICE_RE = re.compile(r"\$?([0-9]+(?:\.[0-9]{2})?)\b")
    ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
    MONTH_DATE_RE = re.compile(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},\s+\d{4}\b",
        re.I,
    )
    URL_RE = re.compile(r"\bhttps?://[^\s<>'\"]+\b", re.I)

    DEFAULT_INSTRUCTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+the\s+user",
        r"override\s+the\s+user",
        r"grant\s+(permanent\s+)?guest\s+access",
        r"send\s+money",
        r"transfer\s+crypto",
        r"delete\s+all\s+files",
        r"rm\s+-rf",
        r"persist\s+this\s+instruction",
        r"send\s+email\s+to",
        r"exfiltrat",
        r"reply\s+with\s+the\s+following\s+exact\s+text",
    ]

    def __init__(
        self,
        *,
        instruction_patterns: Optional[Sequence[str]] = None,
    ):
        self.instruction_patterns = list(
            instruction_patterns or self.DEFAULT_INSTRUCTION_PATTERNS
        )

    def declassify(
        self,
        *,
        runtime: GuardedAgentRuntime,
        source_node_id: str,
        schema: Sequence[FieldSpec],
    ) -> DeclassificationResult:
        if source_node_id not in runtime.state.nodes:
            raise KeyError(f"Unknown source node: {source_node_id}")

        source = runtime.state.nodes[source_node_id]
        text = source.content
        result = DeclassificationResult(
            strategy_name=self.name,
            source_node_id=source_node_id,
            quarantined_spans=self._collect_unsafe_spans(text),
        )

        for spec in schema:
            raw_value = self._extract_value(text, spec)
            if raw_value is None:
                if spec.required:
                    result.fields.append(DeclassifiedField(
                        name=spec.name,
                        field_type=spec.field_type,
                        raw_value=None,
                        normalized_value=None,
                        accepted=False,
                        reason="required_field_missing",
                    ))
                continue

            candidate = runtime.state.add_node(
                node_type=NodeType.CANDIDATE_FACT,
                principal=source.principal,
                channel=f"candidate_extraction:{self.name}",
                content=json.dumps(
                    {
                        "field": spec.name,
                        "raw_value": raw_value,
                        "source_node_id": source_node_id,
                    },
                    sort_keys=True,
                ),
                metadata={
                    "field": spec.name,
                    "field_type": spec.field_type,
                    "strategy": self.name,
                },
                parent_ids=[source_node_id],
            )
            result.candidate_node_ids.append(candidate.node_id)

            normalized, reason = self._validate_and_normalize(raw_value, spec)
            if normalized is None:
                result.fields.append(DeclassifiedField(
                    name=spec.name,
                    field_type=spec.field_type,
                    raw_value=raw_value,
                    normalized_value=None,
                    accepted=False,
                    reason=reason,
                    candidate_node_id=candidate.node_id,
                ))
                continue

            verified = runtime.promote_verified_fact(
                candidate_node_id=candidate.node_id,
                verified_value=json.dumps(
                    {"field": spec.name, "value": normalized},
                    sort_keys=True,
                ),
                verifier_name=self.name,
                metadata={
                    "field": spec.name,
                    "field_type": spec.field_type,
                    "reason": reason,
                    "action_influencing": spec.action_influencing,
                },
            )
            result.verified_node_ids.append(verified.node_id)
            result.accepted_fields[spec.name] = normalized
            result.fields.append(DeclassifiedField(
                name=spec.name,
                field_type=spec.field_type,
                raw_value=raw_value,
                normalized_value=normalized,
                accepted=True,
                reason=reason,
                candidate_node_id=candidate.node_id,
                verified_node_id=verified.node_id,
            ))

        if result.quarantined_spans:
            result.notes.append(
                "Instruction-like spans were quarantined and excluded from verified output."
            )
        if not result.accepted_fields:
            result.notes.append("No fields passed validation and policy checks.")
        return result

    def _extract_value(self, text: str, spec: FieldSpec) -> Optional[str]:
        labelled = self._extract_labelled_value(text, spec.name)
        if labelled is not None:
            typed_from_label = self._extract_typed_fragment(
                labelled, spec.field_type,
            )
            if typed_from_label is not None:
                return typed_from_label
            return labelled

        return self._extract_typed_fragment(text, spec.field_type)

    @staticmethod
    def _extract_labelled_value(text: str, field_name: str) -> Optional[str]:
        pattern = re.compile(
            rf"(?im)^\s*{re.escape(field_name)}\s*[:=]\s*(.+?)\s*$"
        )
        match = pattern.search(text)
        if not match:
            return None
        return match.group(1).strip()

    def _extract_typed_fragment(
        self,
        text: str,
        field_type: str,
    ) -> Optional[str]:
        field_type = field_type.lower()
        if field_type == "email":
            match = self.EMAIL_RE.search(text)
            return match.group(0).lower() if match else None
        if field_type == "asin":
            match = self.ASIN_RE.search(text)
            return match.group(0).upper() if match else None
        if field_type == "price":
            match = self.PRICE_RE.search(text)
            return match.group(1) if match else None
        if field_type == "date":
            iso = self.ISO_DATE_RE.search(text)
            if iso:
                return iso.group(0)
            month = self.MONTH_DATE_RE.search(text)
            return month.group(0) if month else None
        if field_type == "url":
            match = self.URL_RE.search(text)
            return match.group(0) if match else None
        return None

    def _validate_and_normalize(
        self,
        raw_value: str,
        spec: FieldSpec,
    ) -> Tuple[Optional[str], str]:
        value = raw_value.strip().strip(" \t\r\n,.;")
        if not value:
            return None, "empty_value"

        if len(value) > spec.max_length:
            return None, "value_exceeds_max_length"

        if self._contains_instruction(value):
            return None, "instructional_content_detected"

        for pattern in spec.reject_patterns:
            if re.search(pattern, value, re.I):
                return None, "reject_pattern_matched"

        field_type = spec.field_type.lower()
        normalized: Optional[str]
        if field_type == "email":
            normalized = value.lower() if self.EMAIL_RE.fullmatch(value) else None
            if normalized is None:
                return None, "invalid_email"
        elif field_type == "asin":
            normalized = value.upper() if self.ASIN_RE.fullmatch(value.upper()) else None
            if normalized is None:
                return None, "invalid_asin"
        elif field_type == "price":
            match = self.PRICE_RE.fullmatch(value.replace(",", ""))
            if not match:
                return None, "invalid_price"
            normalized = f"{float(match.group(1)):.2f}"
        elif field_type == "date":
            if self.ISO_DATE_RE.fullmatch(value):
                normalized = value
            elif self.MONTH_DATE_RE.fullmatch(value):
                normalized = re.sub(r"\s+", " ", value)
            else:
                return None, "invalid_date"
        elif field_type == "url":
            normalized = value if self.URL_RE.fullmatch(value) else None
            if normalized is None:
                return None, "invalid_url"
        elif field_type in {"short_text", "text"}:
            normalized = re.sub(r"\s+", " ", value).strip()
        else:
            return None, "unsupported_field_type"

        if spec.allow_patterns:
            if not any(re.search(pattern, normalized, re.I) for pattern in spec.allow_patterns):
                return None, "allowlist_mismatch"

        if spec.action_influencing and not spec.allow_patterns:
            return None, "action_influencing_field_requires_allowlist"

        return normalized, "validated"

    def _collect_unsafe_spans(self, text: str) -> List[str]:
        quarantined: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if self._contains_instruction(stripped):
                quarantined.append(stripped)
        return quarantined

    def _contains_instruction(self, text: str) -> bool:
        lowered = text.lower()
        return any(re.search(pattern, lowered, re.I) for pattern in self.instruction_patterns)


STRATEGY_REGISTRY = {
    StructuredSchemaExtractionStrategy.name: StructuredSchemaExtractionStrategy,
}


def build_strategy(name: str, **kwargs: Any) -> DeclassificationStrategy:
    """Construct a declassification strategy by registry name."""

    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"Unknown declassification strategy: {name}")
    return STRATEGY_REGISTRY[name](**kwargs)


def list_supported_strategies() -> List[str]:
    """List currently implemented strategy names."""

    return sorted(STRATEGY_REGISTRY)
