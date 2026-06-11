"""CAGE: Claim-driven Active Graph Exploration."""

from .agent import AgentDecision, LLMController, T2Agent
from .evaluation import Evaluator, GraphEvaluator
from .gcn import CAGEGraphScorerGCN, GraphTensorizer
from .graph import ClaimModality, ClaimNode, EvidenceGraph, EvidenceNode, RelationType
from .mcts import MCTSConfig, MCTSNode, MCTSSearcher
from .tools import (
    BaseTool,
    ForgeryDetectionTool,
    TextVerifierTool,
    ToolRegistry,
    VQATool,
    WebSearchTool,
    register_default_tools,
)

__all__ = [
    "AgentDecision",
    "LLMController",
    "T2Agent",
    "Evaluator",
    "GraphEvaluator",
    "CAGEGraphScorerGCN",
    "GraphTensorizer",
    "ClaimModality",
    "ClaimNode",
    "EvidenceGraph",
    "EvidenceNode",
    "RelationType",
    "MCTSConfig",
    "MCTSNode",
    "MCTSSearcher",
    "BaseTool",
    "ForgeryDetectionTool",
    "TextVerifierTool",
    "ToolRegistry",
    "VQATool",
    "WebSearchTool",
    "register_default_tools",
]
