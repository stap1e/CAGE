from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import torch

from .graph import ClaimNode, EvidenceGraph, EvidenceNode, RelationType


@dataclass
class FinalDecision:
    prediction: str
    confidence: float
    explanation: str
    evidence_paths: List[List[str]]
    claim_scores: Dict[str, float]
    graph_reward: float


class GraphEvaluator:
    """Conflict-aware structural evaluator for CAGE evidence graphs."""

    def __init__(
        self,
        support_weight: float = 1.0,
        refute_weight: float = 1.2,
        conflict_penalty: float = 1.5,
        unresolved_claim_penalty: float = 0.3,
        source_diversity_weight: float = 0.2,
    ) -> None:
        self.support_weight = support_weight
        self.refute_weight = refute_weight
        self.conflict_penalty = conflict_penalty
        self.unresolved_claim_penalty = unresolved_claim_penalty
        self.source_diversity_weight = source_diversity_weight

    def evaluate_trajectory(self, graph: EvidenceGraph) -> float:
        reward = 0.0
        for claim in graph.get_claim_nodes():
            reward += self.score_claim(graph, claim)
            if not claim.verified:
                reward -= self.unresolved_claim_penalty
            if graph.claim_has_unresolved_conflict(claim):
                reward -= self.conflict_penalty
        reward += self._source_diversity_bonus(graph)
        return reward / max(graph.num_claims(), 1)

    def score_claim(self, graph: EvidenceGraph, claim: ClaimNode) -> float:
        support_score = 0.0
        refute_score = 0.0
        neutral_score = 0.0
        conflict_score = 0.0

        for evidence, edge_attr in graph.incoming_evidence_edges(claim):
            relation = edge_attr.get("relation_type")
            contribution = float(edge_attr.get("weight", 1.0)) * evidence.credibility_score
            if relation == RelationType.SUPPORT:
                support_score += contribution
            elif relation == RelationType.REFUTE:
                refute_score += contribution
            elif relation == RelationType.CONFLICT:
                conflict_score += contribution
            elif relation == RelationType.NEUTRAL:
                neutral_score += 0.1 * contribution

        return (
            self.support_weight * support_score
            - self.refute_weight * refute_score
            - self.conflict_penalty * conflict_score
            + neutral_score
            - 0.2 * claim.uncertainty_score
        )

    def make_final_decision(self, graph: EvidenceGraph) -> FinalDecision:
        claim_scores = {claim.node_id: self.score_claim(graph, claim) for claim in graph.get_claim_nodes()}
        graph_reward = self.evaluate_trajectory(graph)
        if not claim_scores:
            return FinalDecision("Unknown", 0.0, "No claims available for verification.", [], {}, graph_reward)

        mean_claim_score = sum(claim_scores.values()) / len(claim_scores)
        if mean_claim_score >= 0.25 and graph_reward >= 0.0:
            prediction = "True"
        elif mean_claim_score <= -0.25 or graph_reward < -0.25:
            prediction = "Fake"
        else:
            prediction = "Uncertain"

        confidence = torch.sigmoid(torch.tensor(abs(mean_claim_score))).item()
        evidence_paths = self._extract_explainable_paths(graph)
        explanation = self._build_explanation(graph, prediction, confidence, claim_scores, evidence_paths, graph_reward)
        return FinalDecision(prediction, confidence, explanation, evidence_paths, claim_scores, graph_reward)

    def build_llm_graph_of_thoughts_prompt(self, graph: EvidenceGraph) -> str:
        lines = [
            "You are a multimodal misinformation verification judge.",
            "Reason over support, refutation, conflicts, source credibility, and unresolved claims.",
            "",
            "CLAIMS:",
        ]
        for claim in graph.get_claim_nodes():
            lines.append(
                f"- Claim ID: {claim.node_id}\n"
                f"  Text: {claim.claim_text}\n"
                f"  Modality: {claim.modality.value}\n"
                f"  Uncertainty: {claim.uncertainty_score}\n"
                f"  Verified: {claim.verified}"
            )
        lines.append("\nEVIDENCE EDGES:")
        for source_id, target_id, edge_attr in graph.graph.edges(data=True):
            source = graph.get_node(source_id)
            target = graph.get_node(target_id)
            source_text = source.content if isinstance(source, EvidenceNode) else source.claim_text
            target_text = target.content if isinstance(target, EvidenceNode) else target.claim_text
            lines.append(
                f"- {source_id} -> {target_id}\n"
                f"  Relation: {edge_attr.get('relation_type')}\n"
                f"  Source text: {source_text}\n"
                f"  Target text: {target_text}\n"
                f"  Edge weight: {edge_attr.get('weight', 1.0)}"
            )
        lines.extend([
            "",
            "TASK:",
            "1. Identify supported, refuted, and unresolved claims.",
            "2. Identify unresolved conflicts.",
            "3. Decide whether the multimodal post is True, Fake, or Uncertain.",
            "4. Return an explainable evidence path for the decision.",
        ])
        return "\n".join(lines)

    def _source_diversity_bonus(self, graph: EvidenceGraph) -> float:
        sources = {evidence.source for evidence in graph.get_evidence_nodes()}
        return self.source_diversity_weight * math.log1p(len(sources))

    def _extract_explainable_paths(self, graph: EvidenceGraph) -> List[List[str]]:
        paths: List[List[str]] = []
        for claim in graph.get_claim_nodes():
            for evidence, edge_attr in graph.incoming_evidence_edges(claim):
                paths.append([evidence.node_id, str(edge_attr.get("relation_type")), claim.node_id])
        return paths

    def _build_explanation(
        self,
        graph: EvidenceGraph,
        prediction: str,
        confidence: float,
        claim_scores: Dict[str, float],
        evidence_paths: List[List[str]],
        graph_reward: float,
    ) -> str:
        lines = [f"Final prediction: {prediction}", f"Confidence: {confidence:.3f}", f"Graph reward: {graph_reward:.3f}", "", "Claim scores:"]
        for claim in graph.get_claim_nodes():
            lines.append(
                f"- {claim.claim_text}\n"
                f"  ID: {claim.node_id}\n"
                f"  Score: {claim_scores.get(claim.node_id, 0.0):.3f}\n"
                f"  Verified: {claim.verified}\n"
                f"  Unresolved conflict: {graph.claim_has_unresolved_conflict(claim)}"
            )
        lines.append("\nEvidence paths:")
        for path in evidence_paths:
            lines.append(f"- {' -> '.join(path)}")
        return "\n".join(lines)
