from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from cage.evaluation import Evaluator


class ControllerProtocol(Protocol):
    """Minimal controller interface required by MCTSSearcher."""

    def propose_next(self, query: str, history: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """Return (thought, action_name, action_input)."""
        ...

    def execute_action(self, action_name: str, action_input: str) -> str:
        """Execute an action and return an observation."""
        ...


@dataclass
class MCTSNode:
    """A node in the tool-augmented reasoning tree."""

    thought: str = ""
    action: str = ""
    observation: str = ""
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    prior: float = 1.0
    depth: int = 0
    terminal: bool = False
    pruned: bool = False
    confidence: float = 0.0

    def add_child(self, child: "MCTSNode") -> None:
        self.children.append(child)

    def trajectory(self) -> List[Dict[str, Any]]:
        path: List[MCTSNode] = []
        node: Optional[MCTSNode] = self
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        return [
            {"thought": n.thought, "action": n.action, "observation": n.observation}
            for n in path
            if n.parent is not None
        ]

    def evidence(self) -> List[str]:
        return [step["observation"] for step in self.trajectory() if step.get("observation")]

    def mean_value(self) -> float:
        return self.value / max(self.visits, 1)

    def uct_score(self, exploration_constant: float) -> float:
        """Biased UCT score.

        UCT(s_t) = V(s_t)/(N(s_t)+1)
                 + C * sqrt(ln(N(parent)+1)/(N(s_t)+1))

        `prior` is a lightweight bias for never/rarely visited nodes.
        """
        parent_visits = self.parent.visits if self.parent is not None else 1
        exploitation = self.value / (self.visits + 1)
        exploration = exploration_constant * math.sqrt(
            math.log(parent_visits + 1) / (self.visits + 1)
        )
        prior_bias = self.prior / (self.visits + 1)
        return exploitation + exploration + prior_bias

    def mark_pruned(self) -> None:
        self.pruned = True
        for child in self.children:
            child.mark_pruned()


@dataclass
class MCTSConfig:
    max_iterations: int = 32
    max_depth: int = 5
    exploration_constant: float = 1.414
    decisive_confidence_threshold: float = 0.92
    prune_siblings_on_decisive: bool = True


class MCTSSearcher:
    """Monte Carlo tree search over LLM-generated tool-use trajectories."""

    def __init__(
        self,
        controller: ControllerProtocol,
        evaluator: Evaluator,
        config: Optional[MCTSConfig] = None,
    ) -> None:
        self.controller = controller
        self.evaluator = evaluator
        self.config = config or MCTSConfig()

    def search(self, query: str) -> MCTSNode:
        root = MCTSNode(depth=0, prior=1.0)

        for _ in range(self.config.max_iterations):
            leaf = self.select(root)
            if leaf.pruned:
                continue

            expanded = self.expand(leaf, query)
            target = expanded if expanded is not None else leaf

            trajectory_score, confidence_score, value = self.evaluator.evaluate(
                history=target.trajectory(),
                evidence=target.evidence(),
                query=query,
            )
            target.confidence = confidence_score
            target.value += value

            self.backpropagate(target, value)

            if confidence_score >= self.config.decisive_confidence_threshold:
                target.terminal = True
                self.prune(target)
                break

        return self.best_node(root)

    def select(self, root: MCTSNode) -> MCTSNode:
        node = root
        while True:
            if node.terminal or node.depth >= self.config.max_depth:
                return node
            candidates = [child for child in node.children if not child.pruned]
            if not candidates:
                return node
            node = max(candidates, key=lambda child: child.uct_score(self.config.exploration_constant))

    def expand(self, node: MCTSNode, query: str) -> Optional[MCTSNode]:
        if node.depth >= self.config.max_depth:
            node.terminal = True
            return None

        history = node.trajectory()
        thought, action_name, action_input = self.controller.propose_next(query, history)
        observation = self.controller.execute_action(action_name, action_input) if action_name else ""

        child = MCTSNode(
            thought=thought,
            action=action_name,
            observation=observation,
            parent=node,
            prior=self.estimate_prior(thought, action_name, observation),
            depth=node.depth + 1,
        )
        node.add_child(child)
        if child.depth >= self.config.max_depth:
            child.terminal = True
        return child

    def backpropagate(self, node: MCTSNode, value: float) -> None:
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visits += 1
            if current is not node:
                current.value += value
            current = current.parent

    def prune(self, decisive_node: MCTSNode) -> None:
        if not self.config.prune_siblings_on_decisive or decisive_node.parent is None:
            return
        for sibling in decisive_node.parent.children:
            if sibling is not decisive_node:
                sibling.mark_pruned()

    def best_node(self, root: MCTSNode) -> MCTSNode:
        nodes = self.collect_nodes(root)
        candidates = [node for node in nodes if node is not root and not node.pruned]
        if not candidates:
            return root
        decisive = [node for node in candidates if node.confidence >= self.config.decisive_confidence_threshold]
        if decisive:
            return max(decisive, key=lambda node: node.mean_value())
        return max(candidates, key=lambda node: node.mean_value())

    def collect_nodes(self, root: MCTSNode) -> List[MCTSNode]:
        nodes: List[MCTSNode] = []
        stack = [root]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children)
        return nodes

    def estimate_prior(self, thought: str, action_name: str, observation: str) -> float:
        if not action_name:
            return 0.4
        if observation and "error" not in observation.lower():
            return 1.0
        return 0.6
