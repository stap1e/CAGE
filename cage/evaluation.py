from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple


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


class Evaluator:
    """Dual-value evaluator for T2/CAGE MCTS.

    `llm_score_fn` is an optional provider hook. It receives a prompt and should
    return `(score_1_to_10, explanation)`. If absent, deterministic heuristics
    keep the system runnable on a bare Ubuntu server.
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
        """Compute normalized S^T in [0, 1]."""
        prompt = self.build_trajectory_prompt(history)
        if self.llm_score_fn is not None:
            raw_score, _ = self.llm_score_fn(prompt)
            return self._score_1_to_10_to_unit(raw_score)
        return self._heuristic_trajectory_score(history)

    def evaluate_confidence(self, evidence: list, query: str) -> float:
        """Compute normalized S^C in [0, 1]."""
        prompt = self.build_confidence_prompt(evidence, query)
        if self.llm_score_fn is not None:
            raw_score, _ = self.llm_score_fn(prompt)
            return self._score_1_to_10_to_unit(raw_score)
        return self._heuristic_confidence_score(evidence)

    def combine_scores(self, trajectory_score: float, confidence_score: float) -> float:
        """V(s) = alpha * S^T + (1 - alpha) * S^C."""
        return self._clip(self.alpha * trajectory_score + (1.0 - self.alpha) * confidence_score)

    def evaluate(self, history: list, evidence: list, query: str) -> Tuple[float, float, float]:
        trajectory_score = self.evaluate_trajectory(history)
        confidence_score = self.evaluate_confidence(evidence, query)
        return trajectory_score, confidence_score, self.combine_scores(trajectory_score, confidence_score)

    def build_trajectory_prompt(self, history: list) -> str:
        return TRAJECTORY_EVAL_PROMPT.format(history=history)

    def build_confidence_prompt(self, evidence: list, query: str) -> str:
        return CONFIDENCE_EVAL_PROMPT.format(query=query, evidence=evidence)

    def parse_score_from_text(self, text: str, kind: str) -> float:
        """Parse required score sentence from LLM output and return [0, 1]."""
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

    def _heuristic_trajectory_score(self, history: list) -> float:
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
        return self._clip(base - min(0.3, 0.1 * failed))

    def _heuristic_confidence_score(self, evidence: list) -> float:
        if not evidence:
            return 0.0
        joined = "\n".join(str(item) for item in evidence).lower()
        decisive_markers = ["support", "refute", "true", "fake", "forgery", "manipulation", "retrieved", "entity"]
        marker_hits = sum(marker in joined for marker in decisive_markers)
        diversity_bonus = min(0.25, 0.05 * len(evidence))
        length_bonus = min(0.25, len(joined) / 1200.0)
        marker_bonus = min(0.5, 0.1 * marker_hits)
        return self._clip(0.15 + diversity_bonus + length_bonus + marker_bonus)

    def _score_1_to_10_to_unit(self, score: float) -> float:
        return self._clip((float(score) - 1.0) / 9.0)

    def _clip(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class GraphEvaluator(Evaluator):
    """Compatibility alias for earlier graph-oriented CAGE experiments."""

    pass
