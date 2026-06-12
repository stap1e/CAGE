from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch

from cage.gcn import PolicyValueGCN
from cage.graph import DynamicEvidenceGraph, RelationType


TRAJECTORY_EVAL_PROMPT = """
You are a strict evaluator for a multimodal misinformation detection agent.
Given the full Thought/Action/Observation trajectory, evaluate whether the trajectory is logically correct.
Focus on action relevance, evidence use, contradiction handling, and whether the reasoning path actually helps solve the query.

Input:
- history: a list of Thought/Action/Observation dictionaries.

Output requirements:
- Provide a concise rationale.
- The final line must be exactly: Thus the correctness score is s
- s must be an integer from 1 to 10.

History:
{history}
""".strip()


CONFIDENCE_EVAL_PROMPT = """
You are a strict evaluator for evidence reliability in multimodal misinformation detection.
Given the user query and the leaf-node evidence, evaluate whether the evidence is reliable and sufficient to answer.
Consider source relevance, redundancy, visual/text consistency, entity grounding, and unresolved conflicts.

Input:
- query: the original user query.
- evidence: a list of leaf-node observations.

Output requirements:
- Provide a concise rationale.
- The final line must be exactly: Thus the reliability score is s
- s must be an integer from 1 to 10.

Query:
{query}

Evidence:
{evidence}
""".strip()


@dataclass
class DecisionCandidate:
    """A single candidate reasoning path used in final decision making."""

    history: Sequence[Mapping[str, Any]]
    evidence: Sequence[str]
    graph_state: Optional[DynamicEvidenceGraph]
    llm_confidence: float
    path_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FinalDecision:
    """Final graph-aware misinformation decision."""

    verdict: str
    fake_probability: float
    confidence: float
    rationale: str
    path_breakdown: list[Dict[str, float]]


class Evaluator:
    """Dual-value evaluator with graph-aware conflict fusion."""

    def __init__(
        self,
        alpha: float = 0.30,
        beta: float = 0.35,
        gamma: float = 0.35,
        llm_score_fn: Optional[Callable[[str], Tuple[float, str]]] = None,
        graph_model: Optional[PolicyValueGCN] = None,
    ) -> None:
        if min(alpha, beta, gamma) < 0.0:
            raise ValueError("alpha, beta, gamma must be non-negative")
        if alpha + beta + gamma <= 0:
            raise ValueError("alpha + beta + gamma must be > 0")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.llm_score_fn = llm_score_fn
        self.graph_model = graph_model

    def evaluate_trajectory(self, history: Sequence[Mapping[str, Any]]) -> float:
        prompt = self.build_trajectory_prompt(history)
        if self.llm_score_fn is not None:
            raw_score, _ = self.llm_score_fn(prompt)
            return self._score_1_to_10_to_unit(raw_score)
        return self._heuristic_trajectory_score(history)

    def evaluate_confidence(self, evidence: Sequence[str], query: str) -> float:
        prompt = self.build_confidence_prompt(evidence, query)
        if self.llm_score_fn is not None:
            raw_score, _ = self.llm_score_fn(prompt)
            return self._score_1_to_10_to_unit(raw_score)
        return self._heuristic_confidence_score(evidence)

    def infer_graph_conflict(self, graph_state: Optional[DynamicEvidenceGraph]) -> float:
        if graph_state is None:
            return 0.5

        if self.graph_model is not None:
            try:
                with torch.no_grad():
                    return float(self.graph_model.conflict_score(graph_state))
            except Exception:
                pass

        total_support = 0
        total_refute = 0
        total_conflict = 0

        for _, _, _, edge_attr in graph_state.graph.edges(keys=True, data=True):
            relation = edge_attr.get("relation_type")
            if relation == RelationType.SUPPORT:
                total_support += 1
            elif relation == RelationType.REFUTE:
                total_refute += 1
            elif relation == RelationType.CONFLICT:
                total_conflict += 1

        contradiction_mass = min(total_support, total_refute)
        raw_conflict = total_conflict + contradiction_mass
        denom = max(total_support + total_refute + total_conflict, 1)
        return self._clip(raw_conflict / denom)

    def combine_scores(
        self,
        trajectory_score: float,
        confidence_score: float,
        graph_conflict_score: float,
    ) -> float:
        graph_consistency = 1.0 - graph_conflict_score
        weighted = (
            self.alpha * trajectory_score
            + self.beta * confidence_score
            + self.gamma * graph_consistency
        )
        return self._clip(weighted / (self.alpha + self.beta + self.gamma))

    def evaluate(
        self,
        history: Sequence[Mapping[str, Any]],
        evidence: Sequence[str],
        query: str,
        graph_state: Optional[DynamicEvidenceGraph] = None,
        llm_confidence: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        trajectory_score = self.evaluate_trajectory(history)
        confidence_score = llm_confidence if llm_confidence is not None else self.evaluate_confidence(evidence, query)
        confidence_score = self._clip(confidence_score)
        graph_conflict_score = self.infer_graph_conflict(graph_state)
        value = self.combine_scores(trajectory_score, confidence_score, graph_conflict_score)
        return trajectory_score, confidence_score, value

    def make_final_decision(
        self,
        candidates: Sequence[DecisionCandidate],
        query: str,
    ) -> FinalDecision:
        if not candidates:
            return FinalDecision(
                verdict="Uncertain",
                fake_probability=0.5,
                confidence=0.0,
                rationale="No valid reasoning path was produced by MCTS.",
                path_breakdown=[],
            )

        path_breakdown: list[Dict[str, float]] = []
        fake_scores: list[float] = []
        conflict_scores: list[float] = []
        weights: list[float] = []

        for idx, candidate in enumerate(candidates):
            llm_conf = self._clip(candidate.llm_confidence)
            graph_conflict = self.infer_graph_conflict(candidate.graph_state)
            path_weight = max(candidate.path_score, 1e-4)
            fake_score = self._clip(0.65 * graph_conflict + 0.35 * (1.0 - llm_conf))

            path_breakdown.append(
                {
                    "path_index": float(idx),
                    "path_weight": float(path_weight),
                    "llm_confidence": float(llm_conf),
                    "graph_conflict": float(graph_conflict),
                    "fake_score": float(fake_score),
                }
            )
            fake_scores.append(fake_score)
            conflict_scores.append(graph_conflict)
            weights.append(path_weight)

        norm_weights = [w / sum(weights) for w in weights]
        weighted_fake = sum(w * s for w, s in zip(norm_weights, fake_scores))
        avg_conflict = sum(w * s for w, s in zip(norm_weights, conflict_scores))
        disagreement = statistics.pstdev(fake_scores) if len(fake_scores) > 1 else 0.0

        graph_dominance = self._clip(0.35 + 0.35 * avg_conflict + 0.30 * disagreement)
        llm_dominance = 1.0 - graph_dominance
        weighted_llm_fake = sum(w * (1.0 - p["llm_confidence"]) for w, p in zip(norm_weights, path_breakdown))
        final_fake_probability = self._clip(
            graph_dominance * weighted_fake + llm_dominance * weighted_llm_fake
        )

        if final_fake_probability >= 0.58:
            verdict = "Fake"
        elif final_fake_probability <= 0.42:
            verdict = "True"
        else:
            verdict = "Uncertain"

        confidence = self._clip(2.0 * abs(final_fake_probability - 0.5))
        rationale = (
            f"Graph-aware decision for query='{query}': "
            f"avg_conflict={avg_conflict:.3f}, path_disagreement={disagreement:.3f}, "
            f"graph_weight={graph_dominance:.3f}, final_fake_probability={final_fake_probability:.3f}. "
            f"High graph conflict or strong path disagreement causes the decision logic "
            f"to trust graph-level structure more than a single optimistic LLM path."
        )

        return FinalDecision(
            verdict=verdict,
            fake_probability=final_fake_probability,
            confidence=confidence,
            rationale=rationale,
            path_breakdown=path_breakdown,
        )

    def build_trajectory_prompt(self, history: Sequence[Mapping[str, Any]]) -> str:
        return TRAJECTORY_EVAL_PROMPT.format(history=list(history))

    def build_confidence_prompt(self, evidence: Sequence[str], query: str) -> str:
        return CONFIDENCE_EVAL_PROMPT.format(query=query, evidence=list(evidence))

    def parse_score_from_text(self, text: str, kind: str) -> float:
        if kind == "trajectory":
            pattern = r"Thus the correctness score is\s+([1-9]|10)"
        elif kind == "confidence":
            pattern = r"Thus the reliability score is\s+([1-9]|10)"
        else:
            raise ValueError(f"Unknown score kind: {kind}")
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Could not parse {kind} score from LLM output")
        return self._score_1_to_10_to_unit(float(match.group(1)))

    def _heuristic_trajectory_score(self, history: Sequence[Mapping[str, Any]]) -> float:
        if not history:
            return 0.0

        valid_steps = 0
        contradiction_penalty = 0.0
        for step in history:
            thought = str(step.get("thought", "")).strip()
            action = str(step.get("action", "")).strip()
            observation = str(step.get("observation", "")).strip().lower()

            if thought and (action or observation):
                valid_steps += 1
            if "error" in observation or "failed" in observation:
                contradiction_penalty += 0.1
            if "contradiction" in observation or "inconsistent" in observation:
                contradiction_penalty += 0.08

        base = valid_steps / max(len(history), 1)
        return self._clip(base - min(0.35, contradiction_penalty))

    def _heuristic_confidence_score(self, evidence: Sequence[str]) -> float:
        if not evidence:
            return 0.0

        joined = "\n".join(str(item) for item in evidence).lower()
        supportive_markers = [
            "retrieved",
            "entity",
            "matched",
            "support",
            "refute",
            "forgery",
            "manipulation",
            "source",
            "image",
            "claim",
        ]
        hits = sum(marker in joined for marker in supportive_markers)
        diversity_bonus = min(0.25, 0.05 * len(evidence))
        marker_bonus = min(0.40, 0.06 * hits)
        length_bonus = min(0.20, len(joined) / 1500.0)
        return self._clip(0.15 + diversity_bonus + marker_bonus + length_bonus)

    def _score_1_to_10_to_unit(self, score: float) -> float:
        return self._clip((float(score) - 1.0) / 9.0)

    def _clip(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class GraphEvaluator(Evaluator):
    """Compatibility alias for earlier graph-oriented CAGE experiments."""

    pass
