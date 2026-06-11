from __future__ import annotations

import json
import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AgentToolInput:
    """Structured input passed into an agent tool."""

    payload: Dict[str, Any]


@dataclass
class AgentToolOutput:
    """Standardized output returned by an agent tool."""

    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class AgentBaseTool(ABC):
    """Abstract base class for tools used by the generic MCTS agent.

    A concrete tool may wrap a search API, database retriever, VQA model,
    code executor, or any other external capability.
    """

    name: str
    description: str

    @abstractmethod
    def execute(self, tool_input: AgentToolInput) -> AgentToolOutput:
        raise NotImplementedError

    def schema(self) -> Dict[str, Any]:
        """Return a JSON-schema-like description exposed to the LLM controller."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            },
        }


class ToolRegistry:
    """Dynamic registry and execution center for agent tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, AgentBaseTool] = {}

    def register(self, tool: AgentBaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def unregister(self, tool_name: str) -> None:
        self._tools.pop(tool_name, None)

    def get(self, tool_name: str) -> AgentBaseTool:
        if tool_name not in self._tools:
            raise KeyError(f"Tool not found: {tool_name}")
        return self._tools[tool_name]

    def list_tools(self) -> List[AgentBaseTool]:
        return list(self._tools.values())

    def tool_schemas(self) -> List[Dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, tool_name: str, tool_input: AgentToolInput) -> AgentToolOutput:
        return self.get(tool_name).execute(tool_input)


@dataclass
class AgentAction:
    """A tool action proposed by the LLM controller."""

    tool_name: str
    tool_input: AgentToolInput
    reasoning: str = ""


@dataclass
class ReasoningStep:
    """One Thought -> Action -> Observation step."""

    thought: str
    action: Optional[AgentAction]
    observation: Optional[AgentToolOutput]


@dataclass
class EvaluationResult:
    """Dual value evaluation for a trajectory."""

    trajectory_score: float
    confidence_score: float
    value: float
    is_decisive: bool = False
    explanation: str = ""


@dataclass
class AgentSearchResult:
    """Final search result returned by MCTSAgent.search()."""

    best_node: "AgentMCTSNode"
    best_trajectory: List[ReasoningStep]
    final_answer: Optional[str]
    root: "AgentMCTSNode"


class LLMController(ABC):
    """Provider-neutral LLM controller interface.

    Concrete implementations can wrap Claude, OpenAI, Gemini, local vLLM,
    Qwen, Llama, or any custom model server. This core skeleton intentionally
    contains no provider-specific SDK calls.
    """

    @abstractmethod
    def generate_thought_and_action(
        self,
        system_prompt: str,
        task: str,
        trajectory: List[ReasoningStep],
        available_tools: List[Dict[str, Any]],
    ) -> Tuple[str, Optional[AgentAction]]:
        raise NotImplementedError

    @abstractmethod
    def evaluate_trajectory(
        self,
        system_prompt: str,
        task: str,
        trajectory: List[ReasoningStep],
    ) -> Tuple[float, str]:
        """Return S^T, the trajectory coherence score in [0, 1]."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_confidence(
        self,
        system_prompt: str,
        task: str,
        trajectory: List[ReasoningStep],
    ) -> Tuple[float, str]:
        """Return S^C, the evidence sufficiency score in [0, 1]."""
        raise NotImplementedError

    @abstractmethod
    def generate_final_answer(
        self,
        system_prompt: str,
        task: str,
        trajectory: List[ReasoningStep],
    ) -> str:
        raise NotImplementedError


class PromptBuilder:
    """Prompt construction points for real LLM integration."""

    def build_action_prompt(
        self,
        task: str,
        trajectory: List[ReasoningStep],
        available_tools: List[Dict[str, Any]],
    ) -> str:
        """Build the prompt used to request the next Thought and Action.

        A concrete LLMController should call an LLM here and request structured
        output such as {"thought": ..., "action": {"tool_name": ..., ...}}.
        """
        trajectory_text = self.format_trajectory(trajectory)
        tools_text = json.dumps(available_tools, ensure_ascii=False, indent=2)
        return f"""
You are an LLM controller inside a tool-augmented MCTS reasoning agent.

Task:
{task}

Current trajectory:
{trajectory_text}

Available tools:
{tools_text}

Generate the next reasoning step:
1. Thought: what should be verified or explored next?
2. Action: which tool should be called, with what input?

Return a structured action. If no tool is needed, return action=null.
""".strip()

    def build_trajectory_evaluation_prompt(self, task: str, trajectory: List[ReasoningStep]) -> str:
        """Build the prompt for S^T trajectory scoring."""
        return f"""
Evaluate the logical coherence of the following reasoning trajectory.

Task:
{task}

Trajectory:
{self.format_trajectory(trajectory)}

Score the trajectory from 0 to 1.

Consider:
- Are the thoughts coherent?
- Are the actions relevant?
- Are observations used correctly?
- Does the trajectory avoid unsupported jumps?

Return:
{{
  "score": float,
  "explanation": str
}}
""".strip()

    def build_confidence_evaluation_prompt(self, task: str, trajectory: List[ReasoningStep]) -> str:
        """Build the prompt for S^C confidence scoring."""
        return f"""
Evaluate whether the collected observations are sufficient to answer the task.

Task:
{task}

Trajectory:
{self.format_trajectory(trajectory)}

Score confidence from 0 to 1.

Consider:
- Is the evidence enough?
- Are there unresolved ambiguities?
- Are there conflicting observations?
- Can a final conclusion be made safely?

Return:
{{
  "score": float,
  "explanation": str,
  "is_decisive": bool
}}
""".strip()

    def build_final_answer_prompt(self, task: str, trajectory: List[ReasoningStep]) -> str:
        return f"""
Use the following tool-augmented reasoning trajectory to answer the task.

Task:
{task}

Trajectory:
{self.format_trajectory(trajectory)}

Provide:
1. Final answer
2. Key evidence
3. Remaining uncertainty, if any
""".strip()

    def format_trajectory(self, trajectory: List[ReasoningStep]) -> str:
        if not trajectory:
            return "<empty trajectory>"

        lines: List[str] = []
        for i, step in enumerate(trajectory, start=1):
            lines.append(f"Step {i}:")
            lines.append(f"Thought: {step.thought}")
            if step.action is None:
                lines.append("Action: None")
            else:
                lines.append(f"Action Tool: {step.action.tool_name}")
                lines.append(f"Action Input: {step.action.tool_input.payload}")
                lines.append(f"Action Reasoning: {step.action.reasoning}")
            if step.observation is None:
                lines.append("Observation: None")
            else:
                lines.append(f"Observation Success: {step.observation.success}")
                lines.append(f"Observation Content: {step.observation.content}")
                if step.observation.error:
                    lines.append(f"Observation Error: {step.observation.error}")
                if step.observation.metadata:
                    lines.append(f"Observation Metadata: {step.observation.metadata}")
            lines.append("")
        return "\n".join(lines)


class AgentMCTSNode:
    """MCTS tree node storing a Thought-Action-Observation trajectory prefix."""

    def __init__(
        self,
        parent: Optional["AgentMCTSNode"] = None,
        step: Optional[ReasoningStep] = None,
        prior: float = 1.0,
        depth: int = 0,
    ) -> None:
        self.node_id = str(uuid.uuid4())
        self.parent = parent
        self.children: List[AgentMCTSNode] = []
        self.step = step
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.depth = depth
        self.is_terminal = False
        self.is_pruned = False
        self.evaluation: Optional[EvaluationResult] = None

    @property
    def mean_value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def add_child(self, child: "AgentMCTSNode") -> None:
        self.children.append(child)

    def trajectory(self) -> List[ReasoningStep]:
        steps: List[ReasoningStep] = []
        node: Optional[AgentMCTSNode] = self
        while node is not None:
            if node.step is not None:
                steps.append(node.step)
            node = node.parent
        return list(reversed(steps))

    def uct_score(self, exploration_constant: float) -> float:
        """Improved UCT with prior initialization for low-visit nodes."""
        parent_visits = 1 if self.parent is None else self.parent.visit_count
        exploitation = self.value_sum / (self.visit_count + 1)
        exploration = exploration_constant * math.sqrt(
            math.log(parent_visits + 1) / (self.visit_count + 1)
        )
        prior_bonus = self.prior / (self.visit_count + 1)
        return exploitation + exploration + prior_bonus

    def best_child(self, exploration_constant: float) -> "AgentMCTSNode":
        candidates = [child for child in self.children if not child.is_pruned]
        if not candidates:
            raise ValueError("No available non-pruned children.")
        return max(candidates, key=lambda child: child.uct_score(exploration_constant))

    def mark_pruned(self) -> None:
        self.is_pruned = True
        for child in self.children:
            child.mark_pruned()

    def __repr__(self) -> str:
        return (
            f"AgentMCTSNode(id={self.node_id[:8]}, depth={self.depth}, "
            f"N={self.visit_count}, V={self.value_sum:.3f}, mean={self.mean_value:.3f}, "
            f"children={len(self.children)}, terminal={self.is_terminal}, pruned={self.is_pruned})"
        )


@dataclass
class MCTSAgentConfig:
    max_iterations: int = 32
    max_depth: int = 5
    exploration_constant: float = 1.414
    alpha: float = 0.5
    decisive_confidence_threshold: float = 0.9
    expansion_width: int = 1
    prune_siblings_on_decisive: bool = True


class MCTSAgent:
    """LLM-controlled, tool-augmented MCTS agent.

    Main lifecycle:
        select -> expand -> evaluate -> backpropagate -> prune -> answer
    """

    def __init__(
        self,
        llm_controller: LLMController,
        tool_registry: ToolRegistry,
        prompt_builder: Optional[PromptBuilder] = None,
        config: Optional[MCTSAgentConfig] = None,
        system_prompt: str = "",
    ) -> None:
        self.llm = llm_controller
        self.tools = tool_registry
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.config = config or MCTSAgentConfig()
        self.system_prompt = system_prompt

    def search(self, task: str) -> AgentSearchResult:
        """Run the planning -> action -> observation -> evaluation -> decision loop."""
        root = AgentMCTSNode(parent=None, step=None, prior=1.0, depth=0)

        for _ in range(self.config.max_iterations):
            selected = self.select(root)
            if selected.is_pruned:
                continue

            expanded = self.expand(selected, task)
            target = expanded if expanded is not None else selected

            evaluation = self.evaluate(target, task)
            target.evaluation = evaluation

            self.backpropagate(target, evaluation.value)

            if evaluation.is_decisive:
                self.prune(target)

            if self.has_decisive_child(root):
                break

        best_node = self.choose_best_final_node(root)
        best_trajectory = best_node.trajectory()
        final_answer = self.llm.generate_final_answer(
            system_prompt=self.system_prompt,
            task=task,
            trajectory=best_trajectory,
        )
        return AgentSearchResult(best_node, best_trajectory, final_answer, root)

    def select(self, root: AgentMCTSNode) -> AgentMCTSNode:
        node = root
        while True:
            if node.is_terminal or node.depth >= self.config.max_depth:
                return node
            available_children = [child for child in node.children if not child.is_pruned]
            if not available_children:
                return node
            node = node.best_child(self.config.exploration_constant)

    def expand(self, node: AgentMCTSNode, task: str) -> Optional[AgentMCTSNode]:
        if node.depth >= self.config.max_depth:
            node.is_terminal = True
            return None

        trajectory = node.trajectory()
        thought, action = self.llm.generate_thought_and_action(
            system_prompt=self.system_prompt,
            task=task,
            trajectory=trajectory,
            available_tools=self.tools.tool_schemas(),
        )

        observation = self.execute_action(action) if action is not None else None
        step = ReasoningStep(thought=thought, action=action, observation=observation)
        child = AgentMCTSNode(
            parent=node,
            step=step,
            prior=self.estimate_prior(thought, action, observation),
            depth=node.depth + 1,
        )
        node.add_child(child)
        if child.depth >= self.config.max_depth:
            child.is_terminal = True
        return child

    def execute_action(self, action: AgentAction) -> AgentToolOutput:
        try:
            return self.tools.execute(action.tool_name, action.tool_input)
        except Exception as exc:
            return AgentToolOutput(
                content=None,
                success=False,
                error=str(exc),
                metadata={"tool_name": action.tool_name, "tool_input": action.tool_input.payload},
            )

    def estimate_prior(
        self,
        thought: str,
        action: Optional[AgentAction],
        observation: Optional[AgentToolOutput],
    ) -> float:
        if action is None:
            return 0.5
        if observation is not None and observation.success:
            return 1.0
        return 0.7

    def evaluate(self, node: AgentMCTSNode, task: str) -> EvaluationResult:
        trajectory = node.trajectory()
        trajectory_score, trajectory_explanation = self.llm.evaluate_trajectory(
            self.system_prompt,
            task,
            trajectory,
        )
        confidence_score, confidence_explanation = self.llm.evaluate_confidence(
            self.system_prompt,
            task,
            trajectory,
        )
        value = self.config.alpha * trajectory_score + (1.0 - self.config.alpha) * confidence_score
        is_decisive = confidence_score >= self.config.decisive_confidence_threshold
        explanation = (
            f"Trajectory evaluation: {trajectory_explanation}\n"
            f"Confidence evaluation: {confidence_explanation}"
        )
        return EvaluationResult(trajectory_score, confidence_score, value, is_decisive, explanation)

    def backpropagate(self, node: AgentMCTSNode, value: float) -> None:
        current: Optional[AgentMCTSNode] = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            current = current.parent

    def prune(self, decisive_node: AgentMCTSNode) -> None:
        decisive_node.is_terminal = True
        if not self.config.prune_siblings_on_decisive or decisive_node.parent is None:
            return
        for sibling in decisive_node.parent.children:
            if sibling is not decisive_node:
                sibling.mark_pruned()

    def has_decisive_child(self, root: AgentMCTSNode) -> bool:
        return any(
            node.evaluation is not None and node.evaluation.is_decisive
            for node in self.collect_nodes(root)
        )

    def choose_best_final_node(self, root: AgentMCTSNode) -> AgentMCTSNode:
        candidates = [node for node in self.collect_nodes(root) if node is not root and not node.is_pruned]
        if not candidates:
            return root
        decisive = [node for node in candidates if node.evaluation is not None and node.evaluation.is_decisive]
        if decisive:
            return max(decisive, key=lambda node: node.mean_value)
        return max(candidates, key=lambda node: node.mean_value)

    def collect_nodes(self, root: AgentMCTSNode) -> List[AgentMCTSNode]:
        nodes: List[AgentMCTSNode] = []
        stack = [root]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children)
        return nodes


class SearchTool(AgentBaseTool):
    """Placeholder search tool for local skeleton tests."""

    name = "search"
    description = "Search external knowledge sources for evidence relevant to a query."

    def execute(self, tool_input: AgentToolInput) -> AgentToolOutput:
        query = tool_input.payload.get("query", "")
        return AgentToolOutput(
            content=f"Mock search result for query: {query}",
            metadata={"query": query, "source": "mock_search"},
            success=True,
        )

    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        }


class CalculatorTool(AgentBaseTool):
    """Placeholder deterministic calculator tool.

    Do not use eval in production; replace it with a safe parser.
    """

    name = "calculator"
    description = "Evaluate a simple mathematical expression."

    def execute(self, tool_input: AgentToolInput) -> AgentToolOutput:
        expression = tool_input.payload.get("expression", "")
        try:
            result = eval(expression, {"__builtins__": {}})
            return AgentToolOutput(content=result, metadata={"expression": expression}, success=True)
        except Exception as exc:
            return AgentToolOutput(
                content=None,
                metadata={"expression": expression},
                success=False,
                error=str(exc),
            )

    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate.",
                    },
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        }


class MockLLMController(LLMController):
    """Mock controller for testing without a real LLM provider."""

    def generate_thought_and_action(
        self,
        system_prompt: str,
        task: str,
        trajectory: List[ReasoningStep],
        available_tools: List[Dict[str, Any]],
    ) -> Tuple[str, Optional[AgentAction]]:
        if len(trajectory) == 0:
            return (
                "I should search for relevant information first.",
                AgentAction(
                    tool_name="search",
                    tool_input=AgentToolInput(payload={"query": task}),
                    reasoning="Search can provide external evidence.",
                ),
            )
        if len(trajectory) == 1:
            return "I should check whether the gathered evidence is sufficient.", None
        return "No further tool use is needed.", None

    def evaluate_trajectory(
        self,
        system_prompt: str,
        task: str,
        trajectory: List[ReasoningStep],
    ) -> Tuple[float, str]:
        if not trajectory:
            return 0.1, "Empty trajectory."
        failed_steps = [
            step for step in trajectory
            if step.observation is not None and not step.observation.success
        ]
        if failed_steps:
            return 0.4, "Some tool calls failed."
        return 0.8, "Trajectory is logically coherent."

    def evaluate_confidence(
        self,
        system_prompt: str,
        task: str,
        trajectory: List[ReasoningStep],
    ) -> Tuple[float, str]:
        has_observation = any(
            step.observation is not None and step.observation.success
            for step in trajectory
        )
        if has_observation:
            return 0.92, "Evidence appears sufficient for a preliminary answer."
        return 0.2, "No evidence has been collected."

    def generate_final_answer(
        self,
        system_prompt: str,
        task: str,
        trajectory: List[ReasoningStep],
    ) -> str:
        return (
            "Final answer generated from the best MCTS trajectory. "
            "Replace MockLLMController with a real LLM implementation."
        )
