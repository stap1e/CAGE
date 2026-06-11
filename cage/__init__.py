"""
CAGE: Claim-driven Active Graph Exploration.

Core framework skeleton for claim-driven, conflict-aware graph reasoning in
multimodal misinformation detection.
"""

from .graph import (
    ClaimModality,
    ClaimNode,
    EvidenceGraph,
    EvidenceNode,
    RelationType,
)
from .tools import (
    BaseTool,
    ForensicsToolStub,
    LLavaVQAToolStub,
    ToolResult,
    ToolRouter,
    WikipediaToolStub,
)
from .mcts import ActionSpace, CAGEAction, CAGEMCTS, CAGEMCTSNode
from .agent import (
    AgentAction,
    AgentBaseTool,
    AgentMCTSNode,
    AgentSearchResult,
    AgentToolInput,
    AgentToolOutput,
    CalculatorTool,
    LLMController,
    MCTSAgent,
    MCTSAgentConfig,
    MockLLMController,
    PromptBuilder,
    ReasoningStep,
    SearchTool,
    ToolRegistry,
)
from .evaluation import FinalDecision, GraphEvaluator
from .gcn import GraphTensorizer, CAGEGraphScorerGCN

__all__ = [
    "ClaimModality",
    "ClaimNode",
    "EvidenceGraph",
    "EvidenceNode",
    "RelationType",
    "BaseTool",
    "ForensicsToolStub",
    "LLavaVQAToolStub",
    "ToolResult",
    "ToolRouter",
    "WikipediaToolStub",
    "ActionSpace",
    "CAGEAction",
    "CAGEMCTS",
    "CAGEMCTSNode",
    "AgentAction",
    "AgentBaseTool",
    "AgentMCTSNode",
    "AgentSearchResult",
    "AgentToolInput",
    "AgentToolOutput",
    "CalculatorTool",
    "LLMController",
    "MCTSAgent",
    "MCTSAgentConfig",
    "MockLLMController",
    "PromptBuilder",
    "ReasoningStep",
    "SearchTool",
    "ToolRegistry",
    "FinalDecision",
    "GraphEvaluator",
    "GraphTensorizer",
    "CAGEGraphScorerGCN",
]
