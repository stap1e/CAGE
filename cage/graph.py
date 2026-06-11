from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx


class RelationType(str, Enum):
    """Typed relations between evidence and claims."""

    SUPPORT = "support"
    REFUTE = "refute"
    CONFLICT = "conflict"
    NEUTRAL = "neutral"


class ClaimModality(str, Enum):
    """Modality associated with an atomic claim."""

    TEXT = "text"
    VISUAL = "visual"
    CROSS_MODAL = "cross-modal"


@dataclass
class ClaimNode:
    """Atomic claim extracted from multimodal input."""

    claim_text: str
    uncertainty_score: float
    modality: Union[str, ClaimModality]
    node_id: str = field(default_factory=lambda: f"claim:{uuid.uuid4()}")
    verified: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.modality, str):
            self.modality = ClaimModality(self.modality)
        self.uncertainty_score = min(max(float(self.uncertainty_score), 0.0), 1.0)


@dataclass
class EvidenceNode:
    """Evidence retrieved by a tool."""

    content: str
    source: str
    credibility_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    node_id: str = field(default_factory=lambda: f"evidence:{uuid.uuid4()}")

    def __post_init__(self) -> None:
        self.credibility_score = min(max(float(self.credibility_score), 0.0), 1.0)


GraphNode = Union[ClaimNode, EvidenceNode]


class EvidenceGraph:
    """Dynamic directed graph state used by CAGE MCTS.

    NetworkX node keys are stable node IDs. The original dataclass object is
    stored under ``graph.nodes[node_id]["data"]``.
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def copy(self) -> "EvidenceGraph":
        """Deep-copy the graph for isolated MCTS expansion/rollout."""
        return copy.deepcopy(self)

    def add_claim(self, claim: ClaimNode) -> str:
        self.graph.add_node(
            claim.node_id,
            data=claim,
            node_type="claim",
            label=claim.claim_text,
            modality=claim.modality.value,
            uncertainty_score=claim.uncertainty_score,
            verified=claim.verified,
        )
        return claim.node_id

    def add_evidence(self, evidence: EvidenceNode) -> str:
        self.graph.add_node(
            evidence.node_id,
            data=evidence,
            node_type="evidence",
            label=evidence.content,
            source=evidence.source,
            credibility_score=evidence.credibility_score,
            metadata=evidence.metadata,
        )
        return evidence.node_id

    def add_relation(
        self,
        source_node: Union[str, GraphNode],
        target_node: Union[str, GraphNode],
        relation_type: RelationType,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        source_id = self._resolve_node_id(source_node)
        target_id = self._resolve_node_id(target_node)
        if source_id not in self.graph:
            raise KeyError(f"Source node does not exist: {source_id}")
        if target_id not in self.graph:
            raise KeyError(f"Target node does not exist: {target_id}")
        self.graph.add_edge(
            source_id,
            target_id,
            relation_type=relation_type,
            weight=float(weight),
            metadata=metadata or {},
        )

    def get_node(self, node_id: str) -> GraphNode:
        return self.graph.nodes[node_id]["data"]

    def get_claim_nodes(self) -> List[ClaimNode]:
        return [
            attr["data"]
            for _, attr in self.graph.nodes(data=True)
            if attr.get("node_type") == "claim"
        ]

    def get_evidence_nodes(self) -> List[EvidenceNode]:
        return [
            attr["data"]
            for _, attr in self.graph.nodes(data=True)
            if attr.get("node_type") == "evidence"
        ]

    def get_unverified_claims(self) -> List[ClaimNode]:
        return [claim for claim in self.get_claim_nodes() if not claim.verified]

    def mark_claim_verified(self, claim: Union[str, ClaimNode]) -> None:
        claim_id = self._resolve_node_id(claim)
        node = self.get_node(claim_id)
        if not isinstance(node, ClaimNode):
            raise TypeError(f"Node is not a ClaimNode: {claim_id}")
        node.verified = True
        self.graph.nodes[claim_id]["verified"] = True

    def incoming_evidence_edges(
        self,
        claim: Union[str, ClaimNode],
        relation_types: Optional[Sequence[RelationType]] = None,
    ) -> List[Tuple[EvidenceNode, Dict[str, Any]]]:
        claim_id = self._resolve_node_id(claim)
        results: List[Tuple[EvidenceNode, Dict[str, Any]]] = []
        for source_id, _, edge_attr in self.graph.in_edges(claim_id, data=True):
            source_data = self.get_node(source_id)
            if not isinstance(source_data, EvidenceNode):
                continue
            if relation_types is not None and edge_attr.get("relation_type") not in relation_types:
                continue
            results.append((source_data, edge_attr))
        return results

    def claim_has_unresolved_conflict(self, claim: Union[str, ClaimNode]) -> bool:
        support_edges = self.incoming_evidence_edges(claim, [RelationType.SUPPORT])
        refute_edges = self.incoming_evidence_edges(claim, [RelationType.REFUTE])
        conflict_edges = self.incoming_evidence_edges(claim, [RelationType.CONFLICT])
        credible_support = [e for e, _ in support_edges if e.credibility_score >= 0.6]
        credible_refute = [e for e, _ in refute_edges if e.credibility_score >= 0.6]
        credible_conflict = [e for e, _ in conflict_edges if e.credibility_score >= 0.6]
        return bool(credible_conflict or (credible_support and credible_refute))

    def num_claims(self) -> int:
        return len(self.get_claim_nodes())

    def num_evidence(self) -> int:
        return len(self.get_evidence_nodes())

    def _resolve_node_id(self, node: Union[str, GraphNode]) -> str:
        return node if isinstance(node, str) else node.node_id

    def __len__(self) -> int:
        return self.graph.number_of_nodes()

    def __repr__(self) -> str:
        return (
            f"EvidenceGraph(claims={self.num_claims()}, "
            f"evidence={self.num_evidence()}, edges={self.graph.number_of_edges()})"
        )
