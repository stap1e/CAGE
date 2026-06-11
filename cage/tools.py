from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .graph import ClaimModality, ClaimNode, EvidenceNode, RelationType


@dataclass
class ToolResult:
    """Standard output contract for all CAGE tools."""

    evidence: EvidenceNode
    relation_type: RelationType
    confidence: float = 1.0
    raw_output: Optional[Any] = None


class BaseTool(ABC):
    """Abstract interface for CAGE evidence-gathering tools."""

    name: str = "base_tool"

    @abstractmethod
    def can_handle(self, claim: ClaimNode) -> bool:
        """Return whether this tool is appropriate for the selected claim."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        claim: ClaimNode,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ToolResult]:
        """Run the tool and return evidence candidates."""
        raise NotImplementedError


class ToolRouter:
    """Maps selected claims to appropriate evidence tools."""

    def __init__(self, tools: Optional[List[BaseTool]] = None) -> None:
        self.tools = tools or []

    def register_tool(self, tool: BaseTool) -> None:
        self.tools.append(tool)

    def route(self, claim: ClaimNode) -> List[BaseTool]:
        return [tool for tool in self.tools if tool.can_handle(claim)]

    def gather_evidence(
        self,
        claim: ClaimNode,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ToolResult]:
        results: List[ToolResult] = []
        for tool in self.route(claim):
            results.extend(tool.run(claim, context=context))
        return results


class WikipediaToolStub(BaseTool):
    """Stub for textual factual retrieval."""

    name = "Wikipedia"

    def can_handle(self, claim: ClaimNode) -> bool:
        return claim.modality in {ClaimModality.TEXT, ClaimModality.CROSS_MODAL}

    def run(self, claim: ClaimNode, context: Optional[Dict[str, Any]] = None) -> List[ToolResult]:
        evidence = EvidenceNode(
            content=f"[Wikipedia stub evidence] {claim.claim_text}",
            source=self.name,
            credibility_score=0.75,
            metadata={"stub": True},
        )
        return [ToolResult(evidence=evidence, relation_type=RelationType.NEUTRAL, confidence=0.5)]


class LLavaVQAToolStub(BaseTool):
    """Stub for visual question answering."""

    name = "LLava-VQA"

    def can_handle(self, claim: ClaimNode) -> bool:
        return claim.modality in {ClaimModality.VISUAL, ClaimModality.CROSS_MODAL}

    def run(self, claim: ClaimNode, context: Optional[Dict[str, Any]] = None) -> List[ToolResult]:
        evidence = EvidenceNode(
            content=f"[LLava VQA stub answer] {claim.claim_text}",
            source=self.name,
            credibility_score=0.65,
            metadata={"stub": True},
        )
        return [ToolResult(evidence=evidence, relation_type=RelationType.NEUTRAL, confidence=0.5)]


class ForensicsToolStub(BaseTool):
    """Stub for image manipulation / forgery detection."""

    name = "PSCC-Net"

    def can_handle(self, claim: ClaimNode) -> bool:
        return claim.modality in {ClaimModality.VISUAL, ClaimModality.CROSS_MODAL}

    def run(self, claim: ClaimNode, context: Optional[Dict[str, Any]] = None) -> List[ToolResult]:
        evidence = EvidenceNode(
            content="[PSCC-Net stub] No strong manipulation signal detected.",
            source=self.name,
            credibility_score=0.7,
            metadata={"stub": True, "manipulation_score": 0.25},
        )
        return [ToolResult(evidence=evidence, relation_type=RelationType.NEUTRAL, confidence=0.6)]
