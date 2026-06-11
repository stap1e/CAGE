from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from cage.evaluation import Evaluator
from cage.mcts import MCTSConfig, MCTSNode, MCTSSearcher
from cage.tools import ToolRegistry, register_default_tools


class LLMController:
    """LLM controller for Thought/Action generation.

    This default implementation is runnable and deterministic. Replace
    `complete()` with a real LLM backend when deploying. Keep the public methods
    stable so MCTS remains independent from provider SDKs and model versions.
    """

    def __init__(self, tool_registry: ToolRegistry, system_prompt: Optional[str] = None) -> None:
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt or self.default_system_prompt()

    def propose_next(self, query: str, history: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """Generate next Thought, Action name, and Action input."""
        prompt = self.build_action_prompt(query, history)
        raw = self.complete(prompt)
        return self.parse_action_response(raw, query, history)

    def execute_action(self, action_name: str, action_input: str) -> str:
        if not action_name:
            return ""
        try:
            return self.tool_registry.execute(action_name, action_input)
        except Exception as exc:
            return f"[ToolError] action={action_name}, error={exc}"

    def complete(self, prompt: str) -> str:
        """Provider hook.

        Replace this method with a call to your preferred LLM/LVLM runtime.

        Expected prompt input:
            - system instruction
            - user query
            - current Thought/Action/Observation history
            - registered tool descriptions

        Expected model output JSON:
            {
              "thought": "short reasoning step",
              "action": "registered_tool_name or empty string",
              "action_input": "string payload for the tool"
            }
        """
        # Deterministic fallback policy for local smoke tests.
        lower_prompt = prompt.lower()
        if "forgery" in lower_prompt or "image" in lower_prompt or "visual" in lower_prompt:
            action = "forgery_detection"
        elif "step_count: 0" in lower_prompt:
            action = "web_search"
        elif "step_count: 1" in lower_prompt:
            action = "text_verifier"
        else:
            action = ""
        return json.dumps(
            {
                "thought": "Plan the next evidence-gathering step based on the current trajectory.",
                "action": action,
                "action_input": self._extract_query_from_prompt(prompt),
            },
            ensure_ascii=False,
        )

    def build_action_prompt(self, query: str, history: List[Dict[str, Any]]) -> str:
        """Build the LLM prompt for Thought/Action generation."""
        return f"""
{self.system_prompt}

You are controlling one expansion step in an MCTS tree for tool-augmented reasoning.

User query:
{query}

Current history as JSON:
{json.dumps(history, ensure_ascii=False, indent=2)}

STEP_COUNT: {len(history)}

Available tools:
{self.tool_registry.describe_tools()}

Decide the next Thought and Action.

Output strict JSON only:
{{
  "thought": "<what to verify or reason about next>",
  "action": "<one registered tool name, or empty string if no tool is needed>",
  "action_input": "<string input to pass to the selected tool>"
}}
""".strip()

    def parse_action_response(
        self,
        raw_response: str,
        query: str,
        history: List[Dict[str, Any]],
    ) -> Tuple[str, str, str]:
        try:
            data = json.loads(raw_response)
            thought = str(data.get("thought", "")).strip()
            action = str(data.get("action", "")).strip()
            action_input = str(data.get("action_input", query)).strip()
        except json.JSONDecodeError:
            thought = raw_response.strip() or "Continue reasoning."
            action = "web_search" if not history else ""
            action_input = query

        if action and action not in self.tool_registry.list_tool_names():
            action = "web_search" if "web_search" in self.tool_registry.list_tool_names() else ""
        return thought, action, action_input

    def default_system_prompt(self) -> str:
        return (
            "You are a rigorous multimodal tool-augmented reasoning controller. "
            "You choose tools only when they can improve evidence quality. "
            "Prefer concise, inspectable Thought/Action decisions."
        )

    def _extract_query_from_prompt(self, prompt: str) -> str:
        match = re.search(r"User query:\n(?P<query>.*?)\n\nCurrent history", prompt, re.S)
        return match.group("query").strip() if match else prompt[:512]


@dataclass
class AgentDecision:
    query: str
    answer: str
    confidence: float
    value: float
    trajectory: List[Dict[str, Any]]
    evidence: List[str]
    best_node: MCTSNode


class T2Agent:
    """Top-level LLM + Tools + MCTS agent wrapper.

    T2Agent initializes the singleton ToolRegistry, the dual Evaluator, and the
    MCTSSearcher. `run(query)` is the main external entry point.
    """

    def __init__(
        self,
        controller: Optional[LLMController] = None,
        registry: Optional[ToolRegistry] = None,
        evaluator: Optional[Evaluator] = None,
        mcts_config: Optional[MCTSConfig] = None,
        register_defaults: bool = True,
    ) -> None:
        self.registry = registry or ToolRegistry()
        if register_defaults:
            register_default_tools(self.registry, overwrite=True)
        self.controller = controller or LLMController(self.registry)
        self.evaluator = evaluator or Evaluator(alpha=0.5)
        self.searcher = MCTSSearcher(
            controller=self.controller,
            evaluator=self.evaluator,
            config=mcts_config or MCTSConfig(),
        )

    def run(self, query: str) -> AgentDecision:
        """Run MCTS search and return a final fused decision."""
        best_node = self.searcher.search(query)
        trajectory = best_node.trajectory()
        evidence = best_node.evidence()
        trajectory_score, confidence_score, value = self.evaluator.evaluate(trajectory, evidence, query)
        answer = self.make_final_decision(query, trajectory, evidence, confidence_score, value)
        return AgentDecision(
            query=query,
            answer=answer,
            confidence=confidence_score,
            value=value,
            trajectory=trajectory,
            evidence=evidence,
            best_node=best_node,
        )

    def make_final_decision(
        self,
        query: str,
        trajectory: List[Dict[str, Any]],
        evidence: List[str],
        confidence: float,
        value: float,
    ) -> str:
        """Heuristic probability-fusion decision making.

        Replace this with calibrated Bayesian fusion, learned reward models, or
        an LLM final-answer call once real tools are connected.
        """
        if confidence >= 0.85 and value >= 0.65:
            verdict = "SUPPORTED / HIGH_CONFIDENCE"
        elif confidence <= 0.35 or value <= 0.35:
            verdict = "INSUFFICIENT_EVIDENCE / LOW_CONFIDENCE"
        else:
            verdict = "UNCERTAIN / NEED_MORE_EVIDENCE"

        evidence_preview = "\n".join(f"- {item}" for item in evidence[:5]) or "- <no evidence>"
        return (
            f"Decision: {verdict}\n"
            f"Query: {query}\n"
            f"Fused value: {value:.3f}\n"
            f"Confidence: {confidence:.3f}\n"
            f"Evidence used:\n{evidence_preview}\n"
            f"Trajectory length: {len(trajectory)}"
        )


__all__ = ["LLMController", "T2Agent", "AgentDecision"]
