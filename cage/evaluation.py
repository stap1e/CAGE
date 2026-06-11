from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple


class Evaluator:
    """Dual-value evaluator for LLM-driven MCTS trajectories.

    The default implementation is a deterministic heuristic so the framework is
    runnable without an LLM. Production systems should inject an `llm_score_fn`
    that calls a large model with the prompts built below and parses structured
    scores from its response.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        llm_score_fn: Optional[Callable[[str], Tuple[float, str]]] = None,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self.llm_score_fn = llm_score_fn

    def evaluate_trajectory(self, history: list) -> float:
        """Compute S^T: reasoning trajectory coherence score in [0, 1]."""
        prompt = self.build_trajectory_prompt(history)
        if self.llm_score_fn is not None:
            score, _ = self.llm_score_fn(prompt)
            return self._clip(score)

        if not history:
            return 0.0
        valid_steps = 0
        for step in history:
            thought = str(step.get("thought", "")).strip()
            action = str(step.get("action", "")).strip()
            observation = str(step.get("observation", "")).strip()
            if thought and (action or observation):
                valid_steps += 1
        base = valid_steps / max(len(history), 1)
        failed = sum("error" in str(step.get("observation", "")).lower() for step in history)
        penalty = min(0.3, 0.1 * failed)
        return self._clip(base - penalty)

    def evaluate_confidence(self, evidence: list, query: str) -> float:
        """Compute S^C: evidence sufficiency confidence score in [0, 1]."""
        prompt = self.build_confidence_prompt(evidence, query)
        if self.llm_score_fn is not None:
            score, _ = self.llm_score_fn(prompt)
            return self._clip(score)

        if not evidence:
            return 0.0
        joined = "\n".join(str(item) for item in evidence).lower()
        decisive_markers = ["support", "refute", "true", "fake", "forgery", "manipulation", "retrieved"]
        marker_hits = sum(marker in joined for marker in decisive_markers)
        diversity_bonus = min(0.25, 0.05 * len(evidence))
        length_bonus = min(0.25, len(joined) / 1200.0)
        marker_bonus = min(0.5, 0.1 * marker_hits)
        return self._clip(0.15 + diversity_bonus + length_bonus + marker_bonus)

    def combine_scores(self, trajectory_score: float, confidence_score: float) -> float:
        """Combine V(s) = alpha * S^T + (1 - alpha) * S^C."""
        return self._clip(self.alpha * trajectory_score + (1.0 - self.alpha) * confidence_score)

    def evaluate(self, history: list, evidence: list, query: str) -> Tuple[float, float, float]:
        """Return (S^T, S^C, V)."""
        trajectory_score = self.evaluate_trajectory(history)
        confidence_score = self.evaluate_confidence(evidence, query)
        value = self.combine_scores(trajectory_score, confidence_score)
        return trajectory_score, confidence_score, value

    def build_trajectory_prompt(self, history: list) -> str:
        """Prompt placeholder for LLM-based trajectory scoring.

        Expected LLM input:
            - Full Thought/Action/Observation trajectory.
        Expected LLM output:
            JSON object: {"score": float in [0,1], "explanation": string}
        """
        return f"""
You are an expert evaluator for tool-augmented reasoning agents.

Evaluate the logical coherence of the following Thought-Action-Observation trajectory.
Focus on whether each action follows from the thought, whether each observation is used correctly,
and whether the chain avoids unsupported jumps or contradictions.

Trajectory:
{history}

Return strict JSON:
{{
  "score": <float between 0 and 1>,
  "explanation": "<short explanation>"
}}
""".strip()

    def build_confidence_prompt(self, evidence: list, query: str) -> str:
        """Prompt placeholder for LLM-based evidence confidence scoring.

        Expected LLM input:
            - User query.
            - Leaf-node evidence/observations.
        Expected LLM output:
            JSON object: {"score": float in [0,1], "is_decisive": bool, "explanation": string}
        """
        return f"""
You are an expert evaluator for evidence sufficiency.

User query:
{query}

Collected evidence:
{evidence}

Evaluate whether the evidence is sufficient to make a reliable final decision.
Consider evidence relevance, credibility, redundancy, conflicts, and missing information.

Return strict JSON:
{{
  "score": <float between 0 and 1>,
  "is_decisive": <true or false>,
  "explanation": "<short explanation>"
}}
""".strip()

    def _clip(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


# Backward-compatible lightweight graph evaluator placeholder for earlier CAGE code.
class GraphEvaluator(Evaluator):
    """Compatibility alias for earlier graph-oriented CAGE experiments."""

    pass
