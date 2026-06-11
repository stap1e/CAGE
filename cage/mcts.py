from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from .evaluation import GraphEvaluator
from .graph import ClaimNode, EvidenceGraph
from .tools import ToolRouter


@dataclass(frozen=True)
class CAGEAction:
    """MCTS action: select one unverified atomic claim for verification."""

    claim_id: str


class ActionSpace:
    """Valid actions from an EvidenceGraph state."""

    def get_actions(self, graph: EvidenceGraph) -> List[CAGEAction]:
        return [CAGEAction(claim.node_id) for claim in graph.get_unverified_claims()]


class CAGEMCTSNode:
    """MCTS node whose state is an EvidenceGraph."""

    def __init__(
        self,
        state: EvidenceGraph,
        parent: Optional["CAGEMCTSNode"] = None,
        action: Optional[CAGEAction] = None,
        action_space: Optional[ActionSpace] = None,
    ) -> None:
        self.state = state
        self.parent = parent
        self.children: List["CAGEMCTSNode"] = []
        self.visit_count = 0
        self.value = 0.0
        self.action = action
        self.action_space = action_space or ActionSpace()
        self.untried_actions = self.action_space.get_actions(state)

    @property
    def mean_value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value / self.visit_count

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_terminal(self, max_depth: Optional[int] = None) -> bool:
        no_actions = len(self.action_space.get_actions(self.state)) == 0
        return no_actions if max_depth is None else no_actions or self.depth() >= max_depth

    def depth(self) -> int:
        depth = 0
        node = self.parent
        while node is not None:
            depth += 1
            node = node.parent
        return depth

    def best_child(self, exploration_weight: float = 1.414) -> "CAGEMCTSNode":
        if not self.children:
            raise ValueError("Cannot select best child from a leaf node.")

        def ucb_score(child: CAGEMCTSNode) -> float:
            if child.visit_count == 0:
                return float("inf")
            exploitation = child.mean_value
            exploration = exploration_weight * math.sqrt(math.log(max(self.visit_count, 1)) / child.visit_count)
            return exploitation + exploration

        return max(self.children, key=ucb_score)

    def add_child(self, child_state: EvidenceGraph, action: CAGEAction) -> "CAGEMCTSNode":
        child = CAGEMCTSNode(child_state, parent=self, action=action, action_space=self.action_space)
        self.children.append(child)
        return child

    def __repr__(self) -> str:
        return (
            f"CAGEMCTSNode(visits={self.visit_count}, value={self.value:.3f}, "
            f"mean={self.mean_value:.3f}, children={len(self.children)}, "
            f"untried={len(self.untried_actions)}, state={self.state})"
        )


class CAGEMCTS:
    """Claim-driven, graph-state MCTS for CAGE."""

    def __init__(
        self,
        tool_router: ToolRouter,
        evaluator: GraphEvaluator,
        action_space: Optional[ActionSpace] = None,
        exploration_weight: float = 1.414,
        max_depth: int = 5,
        rollout_depth: int = 3,
    ) -> None:
        self.tool_router = tool_router
        self.evaluator = evaluator
        self.action_space = action_space or ActionSpace()
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth
        self.rollout_depth = rollout_depth

    def search(self, initial_graph: EvidenceGraph, num_iterations: int = 50) -> CAGEMCTSNode:
        root = CAGEMCTSNode(initial_graph.copy(), action_space=self.action_space)
        for _ in range(num_iterations):
            leaf = self.select(root)
            expanded = self.expand(leaf)
            reward = self.simulate(expanded)
            self.backpropagate(expanded, reward)
        return root if not root.children else max(root.children, key=lambda child: child.mean_value)

    def select(self, node: CAGEMCTSNode) -> CAGEMCTSNode:
        current = node
        while (
            not current.is_terminal(max_depth=self.max_depth)
            and current.is_fully_expanded()
            and current.children
        ):
            current = current.best_child(self.exploration_weight)
        return current

    def expand(self, node: CAGEMCTSNode) -> CAGEMCTSNode:
        if node.is_terminal(max_depth=self.max_depth) or not node.untried_actions:
            return node

        action = node.untried_actions.pop(0)

        # MCTS expansion must never mutate the parent state.
        child_graph = node.state.copy()
        selected_claim = child_graph.get_node(action.claim_id)
        if not isinstance(selected_claim, ClaimNode):
            raise TypeError(f"Action does not point to a ClaimNode: {action.claim_id}")

        for result in self.tool_router.gather_evidence(selected_claim, context={"graph": child_graph}):
            evidence_id = child_graph.add_evidence(result.evidence)
            child_graph.add_relation(
                source_node=evidence_id,
                target_node=selected_claim.node_id,
                relation_type=result.relation_type,
                weight=result.confidence,
                metadata={"tool_raw_output": result.raw_output},
            )
        child_graph.mark_claim_verified(selected_claim)
        return node.add_child(child_graph, action)

    def simulate(self, node: CAGEMCTSNode) -> float:
        # Rollout must modify only this deep-copied trajectory graph.
        rollout_graph = node.state.copy()
        for _ in range(self.rollout_depth):
            actions = self.action_space.get_actions(rollout_graph)
            if not actions:
                break
            selected_action = self._rollout_policy(rollout_graph, actions)
            selected_claim = rollout_graph.get_node(selected_action.claim_id)
            if not isinstance(selected_claim, ClaimNode):
                continue
            for result in self.tool_router.gather_evidence(
                selected_claim,
                context={"graph": rollout_graph, "phase": "simulation"},
            ):
                evidence_id = rollout_graph.add_evidence(result.evidence)
                rollout_graph.add_relation(
                    source_node=evidence_id,
                    target_node=selected_claim.node_id,
                    relation_type=result.relation_type,
                    weight=result.confidence,
                    metadata={"phase": "simulation", "tool_raw_output": result.raw_output},
                )
            rollout_graph.mark_claim_verified(selected_claim)
        return self.evaluator.evaluate_trajectory(rollout_graph)

    def backpropagate(self, node: CAGEMCTSNode, reward: float) -> None:
        current: Optional[CAGEMCTSNode] = node
        while current is not None:
            current.visit_count += 1
            current.value += reward
            current = current.parent

    def _rollout_policy(self, graph: EvidenceGraph, actions: List[CAGEAction]) -> CAGEAction:
        def uncertainty(action: CAGEAction) -> float:
            node = graph.get_node(action.claim_id)
            return node.uncertainty_score if isinstance(node, ClaimNode) else 0.0

        return max(actions, key=uncertainty)
