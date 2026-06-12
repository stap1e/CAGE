from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple

from cage.evaluation import DecisionCandidate, Evaluator, FinalDecision
from cage.gcn import PolicyValueGCN
from cage.graph import ClaimModality, ClaimNode, DynamicEvidenceGraph, EvidenceNode, RelationType


class ControllerProtocol(Protocol):
    """Controller interface used by the graph-guided MCTS searcher."""

    def propose_next(self, query: str, history: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        ...

    def execute_action(self, action_name: str, action_input: str) -> Any:
        ...


@dataclass
class MCTSNode:
    """A node in the graph-guided reasoning tree."""

    thought: str = ""
    action: str = ""
    action_input: str = ""
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
    predicted_conflict: float = 0.5
    graph_state: DynamicEvidenceGraph = field(default_factory=DynamicEvidenceGraph)
    step_node_ids: Dict[str, str] = field(default_factory=dict)
    expanded_actions: Set[str] = field(default_factory=set)

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
            {
                "thought": n.thought,
                "action": n.action,
                "action_input": n.action_input,
                "observation": n.observation,
            }
            for n in path
            if n.parent is not None
        ]

    def evidence(self) -> List[str]:
        return [step["observation"] for step in self.trajectory() if step.get("observation")]

    def mean_value(self) -> float:
        return self.value / max(self.visits, 1)

    def uct_score(self, exploration_constant: float) -> float:
        """Graph-guided adaptive UCT."""
        parent_visits = self.parent.visits if self.parent is not None else 1
        exploitation = self.value / (self.visits + 1)
        exploration = self.prior * exploration_constant * math.sqrt(
            math.log(parent_visits + 1) / (self.visits + 1)
        )
        return exploitation + exploration

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
    decisive_conflict_threshold: float = 0.20
    prune_siblings_on_decisive: bool = True
    default_tool_names: Tuple[str, ...] = (
        "web_search",
        "text_verifier",
        "forgery_detection",
        "vqa",
        "counterfactual",
        "entity_recognition",
    )


class MCTSSearcher:
    """GCN-guided Monte Carlo Tree Search over multimodal tool-use trajectories."""

    def __init__(
        self,
        controller: ControllerProtocol,
        evaluator: Evaluator,
        policy_value_net: Optional[PolicyValueGCN] = None,
        config: Optional[MCTSConfig] = None,
        tool_vocab: Optional[Sequence[str]] = None,
    ) -> None:
        self.controller = controller
        self.evaluator = evaluator
        self.policy_value_net = policy_value_net
        self.config = config or MCTSConfig()
        self.tool_vocab = list(tool_vocab or getattr(policy_value_net, "tool_vocab", []) or self.config.default_tool_names)
        self.last_root: Optional[MCTSNode] = None

    def search(
        self,
        query: str,
        root_graph: Optional[DynamicEvidenceGraph] = None,
    ) -> MCTSNode:
        graph = root_graph or DynamicEvidenceGraph(tool_vocab=self.tool_vocab)
        self._ensure_root_claim(graph, query)

        root = MCTSNode(
            thought="",
            action="",
            action_input="",
            observation="",
            parent=None,
            prior=1.0,
            depth=0,
            graph_state=graph,
        )
        self.last_root = root

        for _ in range(self.config.max_iterations):
            leaf = self.select(root, query)
            if leaf.pruned:
                continue

            expanded = self.expand(leaf, query)
            target = expanded if expanded is not None else leaf

            _, confidence_score, value = self.evaluator.evaluate(
                history=target.trajectory(),
                evidence=target.evidence(),
                query=query,
                graph_state=target.graph_state,
            )
            target.confidence = confidence_score
            target.predicted_conflict = self._predict_conflict(target.graph_state)
            self.backpropagate(target, value)

            if (
                confidence_score >= self.config.decisive_confidence_threshold
                and target.predicted_conflict <= self.config.decisive_conflict_threshold
            ):
                target.terminal = True
                self.prune(target)
                break

        return self.best_node(root)

    def final_decision(self, root: MCTSNode, query: str) -> FinalDecision:
        leaves = [
            node
            for node in self.collect_nodes(root)
            if node is not root and not node.pruned and not node.children
        ]
        candidates = [
            DecisionCandidate(
                history=node.trajectory(),
                evidence=node.evidence(),
                graph_state=node.graph_state,
                llm_confidence=node.confidence,
                path_score=node.mean_value(),
                metadata={"predicted_conflict": node.predicted_conflict},
            )
            for node in leaves
        ]
        return self.evaluator.make_final_decision(candidates, query=query)

    def select(self, root: MCTSNode, query: str) -> MCTSNode:
        node = root
        while True:
            if node.terminal or node.depth >= self.config.max_depth:
                return node

            available_actions = self._available_actions(query, node)
            unexpanded = [action for action in available_actions if action not in node.expanded_actions]
            if unexpanded:
                return node

            candidates = [child for child in node.children if not child.pruned]
            if not candidates:
                return node

            priors = self._predict_policy(node.graph_state)
            for child in candidates:
                child.prior = priors.get(child.action, child.prior)

            node = max(candidates, key=lambda child: child.uct_score(self.config.exploration_constant))

    def expand(self, node: MCTSNode, query: str) -> Optional[MCTSNode]:
        if node.depth >= self.config.max_depth:
            node.terminal = True
            return None

        available_actions = self._available_actions(query, node)
        unexpanded = [action for action in available_actions if action not in node.expanded_actions]
        if not unexpanded:
            return None

        priors = self._predict_policy(node.graph_state)
        selected_action = max(unexpanded, key=lambda action: priors.get(action, 0.0))

        thought, action_name, action_input = self._propose_step(query, node, selected_action)
        if not action_name:
            action_name = selected_action
        if not action_input:
            action_input = query

        observation_result = self.controller.execute_action(action_name, action_input)
        observation_text, observation_visual_feature, observation_metadata = self._normalize_observation(observation_result)

        child_graph = node.graph_state.copy()
        step_node_ids = child_graph.add_step(
            thought=thought,
            action=action_name,
            observation=observation_text,
            parent_node_id=child_graph.current_node_id,
            action_input=action_input,
            observation_visual_feature=observation_visual_feature,
            action_metadata={"selected_by_gcn": True, "prior": priors.get(action_name, 0.0)},
            observation_metadata=observation_metadata,
            extracted_entities=observation_metadata.get("entities"),
            aligned_pairs=observation_metadata.get("aligned_pairs"),
        )

        evidence_node = EvidenceNode(
            content=observation_text,
            source=action_name,
            credibility_score=float(observation_metadata.get("credibility_score", 0.5)),
            metadata=dict(observation_metadata),
        )
        evidence_id = child_graph.add_evidence(
            evidence_node,
            parent_node_id=step_node_ids["observation"],
            relation_type=RelationType.OBSERVES,
        )

        relation = self._infer_relation_from_observation(observation_text)
        for claim in child_graph.get_claim_nodes():
            child_graph.connect_claim_evidence(
                claim,
                evidence_id,
                relation_type=relation,
                weight=float(observation_metadata.get("relation_weight", 1.0)),
                metadata={"action": action_name},
            )

        child = MCTSNode(
            thought=thought,
            action=action_name,
            action_input=action_input,
            observation=observation_text,
            parent=node,
            prior=priors.get(action_name, 1.0 / max(len(self.tool_vocab), 1)),
            depth=node.depth + 1,
            graph_state=child_graph,
            step_node_ids=step_node_ids,
            predicted_conflict=self._predict_conflict(child_graph),
        )
        node.add_child(child)
        node.expanded_actions.add(action_name)

        if child.depth >= self.config.max_depth:
            child.terminal = True
        return child

    def backpropagate(self, node: MCTSNode, value: float) -> None:
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visits += 1
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

        decisive = [
            node
            for node in candidates
            if node.confidence >= self.config.decisive_confidence_threshold
            and node.predicted_conflict <= self.config.decisive_conflict_threshold
        ]
        if decisive:
            return max(decisive, key=lambda node: node.mean_value())

        return max(
            candidates,
            key=lambda node: node.mean_value() + 0.15 * node.confidence - 0.15 * node.predicted_conflict,
        )

    def collect_nodes(self, root: MCTSNode) -> List[MCTSNode]:
        nodes: List[MCTSNode] = []
        stack = [root]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children)
        return nodes

    def _available_actions(self, query: str, node: MCTSNode) -> List[str]:
        if hasattr(self.controller, "available_actions"):
            actions = getattr(self.controller, "available_actions")(query, node.trajectory(), node.graph_state)
            if actions:
                return list(actions)
        return list(self.tool_vocab)

    def _propose_step(self, query: str, node: MCTSNode, preferred_action: str) -> Tuple[str, str, str]:
        history = node.trajectory()

        if hasattr(self.controller, "propose_step"):
            thought, action_name, action_input = getattr(self.controller, "propose_step")(
                query,
                history,
                preferred_action,
                node.graph_state,
            )
            return thought, action_name or preferred_action, action_input

        thought, action_name, action_input = self.controller.propose_next(query, history)
        if not action_name:
            action_name = preferred_action
        elif preferred_action in self.tool_vocab:
            action_name = preferred_action
        return thought, action_name, action_input

    def _predict_policy(self, graph_state: DynamicEvidenceGraph) -> Dict[str, float]:
        if self.policy_value_net is None:
            uniform = 1.0 / max(len(self.tool_vocab), 1)
            return {tool: uniform for tool in self.tool_vocab}
        return self.policy_value_net.policy_dict(graph_state, current_node_id=graph_state.current_node_id)

    def _predict_conflict(self, graph_state: DynamicEvidenceGraph) -> float:
        if self.policy_value_net is None:
            return self.evaluator.infer_graph_conflict(graph_state)
        return self.policy_value_net.conflict_score(graph_state, current_node_id=graph_state.current_node_id)

    def _normalize_observation(self, result: Any) -> Tuple[str, Optional[Any], Dict[str, Any]]:
        if isinstance(result, dict):
            observation_text = str(result.get("text", ""))
            visual_feature = result.get("visual_feature")
            metadata = dict(result)
            metadata.pop("text", None)
            metadata.pop("visual_feature", None)
            return observation_text, visual_feature, metadata
        return str(result), None, {}

    def _infer_relation_from_observation(self, observation: str) -> RelationType:
        text = observation.lower()
        if any(token in text for token in ["forgery", "manipulated", "fake", "tampered", "inconsistent"]):
            return RelationType.REFUTE
        if any(token in text for token in ["support", "confirmed", "verified", "matched", "consistent"]):
            return RelationType.SUPPORT
        if any(token in text for token in ["conflict", "contradiction", "mismatch"]):
            return RelationType.CONFLICT
        return RelationType.NEUTRAL

    def _ensure_root_claim(self, graph: DynamicEvidenceGraph, query: str) -> None:
        if graph.num_claims() > 0:
            return
        root_claim = ClaimNode(
            claim_text=query,
            uncertainty_score=1.0,
            modality=ClaimModality.CROSS_MODAL,
            metadata={"seed_claim": True},
        )
        graph.add_claim(root_claim, parent_node_id=graph.root_node_id, relation_type=RelationType.ROOT)
