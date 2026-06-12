from __future__ import annotations

import base64
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cage.evaluation import Evaluator
from cage.mcts import MCTSConfig, MCTSNode, MCTSSearcher
from cage.tools import ToolRegistry, register_default_tools


INITIALIZATION_PROMPT = """
You are an expert multimodal misinformation detection controller.
Analyze the input text and optional image. Estimate three fake-probability dimensions:
1. text_authenticity_fake_prob: probability that the text claim is false or misleading.
2. image_authenticity_fake_prob: probability that the image is manipulated, synthetic, or visually misleading.
3. cross_modal_inconsistency_prob: probability that text and image are inconsistent.

Input text:
{text}

Image payload:
{image_payload_description}

Output strict JSON only:
{{
  "text_authenticity_fake_prob": <float 0.0-1.0>,
  "image_authenticity_fake_prob": <float 0.0-1.0>,
  "cross_modal_inconsistency_prob": <float 0.0-1.0>,
  "rationale": "<brief explanation>"
}}
""".strip()


class LLMController:
    """LLM/LVLM controller for Thought and Action generation.

    The class provides stable prompt construction and multimodal payload helpers.
    `complete()` is intentionally a provider hook: replace it with Claude,
    OpenAI, vLLM, Qwen-VL, LLaVA, or another runtime without changing MCTS.
    """

    def __init__(self, tool_registry: ToolRegistry, system_prompt: Optional[str] = None) -> None:
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt or self.default_system_prompt()

    def image_to_base64(self, image_path: str) -> str:
        """Read a local image and return base64 string for LVLM payloads."""
        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        with path.open("rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def build_multimodal_payload(self, text: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        """Build a provider-neutral multimodal payload description.

        Concrete providers will need this translated into their own schema.
        """
        payload: Dict[str, Any] = {"text": text, "image": None}
        if image_path:
            media_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
            payload["image"] = {
                "path": image_path,
                "media_type": media_type,
                "base64": self.image_to_base64(image_path),
            }
        return payload

    def initialize_multimodal_state(self, text: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        """Estimate initial fake probabilities for text/image/cross-modal dimensions."""
        image_desc = "<no image>"
        if image_path:
            image_desc = f"local image path={image_path}; base64 is available in runtime payload"
        prompt = INITIALIZATION_PROMPT.format(text=text, image_payload_description=image_desc)
        raw = self.complete(prompt, multimodal_payload=self.build_multimodal_payload(text, image_path))
        try:
            parsed = json.loads(raw)
            return {
                "text_authenticity_fake_prob": self._clip(parsed.get("text_authenticity_fake_prob", 0.5)),
                "image_authenticity_fake_prob": self._clip(parsed.get("image_authenticity_fake_prob", 0.5)),
                "cross_modal_inconsistency_prob": self._clip(parsed.get("cross_modal_inconsistency_prob", 0.5)),
                "rationale": str(parsed.get("rationale", "")),
            }
        except json.JSONDecodeError:
            return {
                "text_authenticity_fake_prob": 0.5,
                "image_authenticity_fake_prob": 0.5 if image_path else 0.0,
                "cross_modal_inconsistency_prob": 0.5 if image_path else 0.0,
                "rationale": raw,
            }

    def propose_next(self, query: str, history: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        prompt = self.build_action_prompt(query, history)
        raw = self.complete(prompt)
        return self.parse_action_response(raw, query, history)

    def execute_action(self, action_name: str, action_input: str) -> str:
        if not action_name:
            return ""
        try:
            return self.tool_registry.execute(action_name, action_input)
        except Exception as exc:  # noqa: BLE001
            return f"[ToolError] action={action_name}, error={exc}"

    def complete(self, prompt: str, multimodal_payload: Optional[Dict[str, Any]] = None) -> str:
        """Provider hook for LLM/LVLM inference.

        Replace this method with a concrete model call. The prompt already
        specifies the expected output format. If `multimodal_payload` is not
        None, it contains text plus image base64 data under payload["image"].
        """
        _ = multimodal_payload
        if "text_authenticity_fake_prob" in prompt:
            return json.dumps(
                {
                    "text_authenticity_fake_prob": 0.5,
                    "image_authenticity_fake_prob": 0.5 if "<no image>" not in prompt else 0.0,
                    "cross_modal_inconsistency_prob": 0.5 if "<no image>" not in prompt else 0.0,
                    "rationale": "Mock initialization; connect a real LVLM for calibrated estimates.",
                },
                ensure_ascii=False,
            )

        lower_prompt = prompt.lower()
        if "step_count: 0" in lower_prompt:
            action = "web_search"
        elif "image_path" in lower_prompt or "visual" in lower_prompt or "image" in lower_prompt:
            action = "vqa"
        elif "step_count: 1" in lower_prompt:
            action = "entity_recognition"
        else:
            action = "text_verifier"
        return json.dumps(
            {
                "thought": "Select the next tool that can most improve evidence quality for this branch.",
                "action": action,
                "action_input": self._extract_query_from_prompt(prompt),
            },
            ensure_ascii=False,
        )

    def build_action_prompt(self, query: str, history: List[Dict[str, Any]]) -> str:
        return f"""
{self.system_prompt}

You are expanding one node in an MCTS tree for multimodal misinformation detection.
Choose the next Thought and Action based on the current trajectory and available tools.

User query:
{query}

Current history as JSON:
{json.dumps(history, ensure_ascii=False, indent=2)}

STEP_COUNT: {len(history)}

Available tools:
{self.tool_registry.describe_tools()}

Output strict JSON only:
{{
  "thought": "<what to verify or reason about next>",
  "action": "<one registered tool name, or empty string if no tool is needed>",
  "action_input": "<string input to pass to the selected tool; can be JSON string for multimodal tools>"
}}
""".strip()

    def parse_action_response(self, raw_response: str, query: str, history: List[Dict[str, Any]]) -> Tuple[str, str, str]:
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
            "You are a rigorous T2-style multimodal misinformation detection controller. "
            "Plan evidence-gathering actions over text authenticity, image authenticity, "
            "and cross-modal consistency."
        )

    def _extract_query_from_prompt(self, prompt: str) -> str:
        match = re.search(r"User query:\n(?P<query>.*?)\n\nCurrent history", prompt, re.S)
        return match.group("query").strip() if match else prompt[:512]

    def _clip(self, value: Any) -> float:
        return max(0.0, min(1.0, float(value)))


@dataclass
class AgentDecision:
    query: str
    answer: str
    confidence: float
    value: float
    trajectory: List[Dict[str, Any]]
    evidence: List[str]
    best_node: MCTSNode
    initialization: Dict[str, Any]


class T2Agent:
    """Top-level multimodal T2 Agent wrapper."""

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
        self.searcher = MCTSSearcher(self.controller, self.evaluator, mcts_config or MCTSConfig())

    def run(self, query: str, image_path: Optional[str] = None) -> AgentDecision:
        initialization = self.controller.initialize_multimodal_state(query, image_path)
        enriched_query = self._build_enriched_query(query, image_path, initialization)
        best_node = self.searcher.search(enriched_query)
        trajectory = best_node.trajectory()
        evidence = best_node.evidence()
        trajectory_score, confidence_score, value = self.evaluator.evaluate(trajectory, evidence, enriched_query)
        answer = self.make_final_decision(query, image_path, initialization, trajectory, evidence, confidence_score, value)
        return AgentDecision(query, answer, confidence_score, value, trajectory, evidence, best_node, initialization)

    def make_final_decision(
        self,
        query: str,
        image_path: Optional[str],
        initialization: Dict[str, Any],
        trajectory: List[Dict[str, Any]],
        evidence: List[str],
        confidence: float,
        value: float,
    ) -> str:
        fake_prior = max(
            float(initialization.get("text_authenticity_fake_prob", 0.0)),
            float(initialization.get("image_authenticity_fake_prob", 0.0)),
            float(initialization.get("cross_modal_inconsistency_prob", 0.0)),
        )
        fused_fake_probability = 0.4 * fake_prior + 0.6 * (1.0 - value)
        if confidence >= 0.75 and fused_fake_probability >= 0.6:
            verdict = "FAKE_OR_MISLEADING"
        elif confidence >= 0.75 and fused_fake_probability <= 0.4:
            verdict = "LIKELY_TRUE"
        else:
            verdict = "UNCERTAIN"
        evidence_preview = "\n".join(f"- {item}" for item in evidence[:5]) or "- <no evidence>"
        return (
            f"Decision: {verdict}\n"
            f"Query: {query}\n"
            f"Image: {image_path or '<none>'}\n"
            f"Fused fake probability: {fused_fake_probability:.3f}\n"
            f"Evidence reliability: {confidence:.3f}\n"
            f"MCTS value: {value:.3f}\n"
            f"Initialization: {json.dumps(initialization, ensure_ascii=False)}\n"
            f"Evidence used:\n{evidence_preview}\n"
            f"Trajectory length: {len(trajectory)}"
        )

    def _build_enriched_query(self, query: str, image_path: Optional[str], initialization: Dict[str, Any]) -> str:
        return json.dumps(
            {
                "query": query,
                "image_path": image_path,
                "initialization": initialization,
            },
            ensure_ascii=False,
        )


__all__ = ["INITIALIZATION_PROMPT", "LLMController", "T2Agent", "AgentDecision"]
