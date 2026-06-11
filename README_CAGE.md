# CAGE

CAGE (**Claim-driven Active Graph Exploration**) is a skeleton framework for claim-driven, conflict-aware graph reasoning in multimodal misinformation detection.

## Core idea

Traditional tool-driven MCTS agents often model state as a procedural trace:

```text
S_t = (Thought, Action, Observation)
```

CAGE instead models the reasoning state as an evidence graph:

```text
G_t = (C, E, R)
```

where:

- `C` is the set of atomic `ClaimNode`s extracted from multimodal input.
- `E` is the set of retrieved `EvidenceNode`s.
- `R` is the set of typed relations: `SUPPORT`, `REFUTE`, `CONFLICT`, and `NEUTRAL`.

This makes the MCTS action space semantic rather than procedural:

- **State:** dynamic `EvidenceGraph` backed by `networkx.DiGraph`
- **Action:** selecting an unverified atomic `ClaimNode`
- **Routing:** `ToolRouter` maps a selected claim to suitable evidence tools
- **Reward:** `GraphEvaluator` computes conflict-aware structural reward over claims, evidence, and relations

## Files

- `cage/graph.py` — `ClaimNode`, `EvidenceNode`, `EvidenceGraph`, `RelationType`
- `cage/tools.py` — `BaseTool` ABC, `ToolRouter`, tool stubs
- `cage/mcts.py` — strict CAGE evidence-graph MCTS implementation: select, expand, simulate, backpropagate
- `cage/agent.py` — generic LLM-controller + tool-registry MCTS agent skeleton
- `cage/evaluation.py` — conflict-aware reward and final decision logic
- `cage/gcn.py` — PyTorch GCN scorer stub and tensorizer
- `cage/data/` — PyTorch dataset wrappers for MMFakeBench and Hints-of-Truth
- `examples_cage_run.py` — minimal runnable example

## Run

```bash
pip install -r requirements.txt
python examples_cage_run.py
python examples_agent_mcts_run.py
```

## Generic LLM tool-augmented MCTS agent

`cage/agent.py` provides a provider-neutral skeleton for a large-model controller that plans with MCTS and calls external tools:

- `AgentBaseTool` and `ToolRegistry` define dynamic tool registration/execution.
- `LLMController` is the abstract interface for Thought/Action generation, dual evaluation, and final answer generation.
- `AgentMCTSNode` stores the tree state and Thought -> Action -> Observation trajectory.
- `MCTSAgent` implements select, expand, dual evaluate, backpropagate, and high-confidence pruning.

The framework intentionally contains no provider-specific SDK calls. Plug in a real model by subclassing `LLMController`:

```python
from cage import LLMController


class MyLLMController(LLMController):
    ...
```

Run the local mock example:

```bash
python examples_agent_mcts_run.py
```

## Dataset usage

CAGE includes normalized PyTorch `Dataset` wrappers for the two local multimodal datasets:

```python
from cage.data import HintsOfTruthDataset, MMFakeBenchDataset, cage_collate_fn

mm_val = MMFakeBenchDataset(
    root="/data/lhy_data/MMFakeBench",
    split="val",
    load_image=True,
)

hot_train = HintsOfTruthDataset(
    root="/data/lhy_data/hints_of_truth",
    split="dev1",
    load_image=True,
)

sample = mm_val[0]
print(sample.text, sample.label_name, type(sample.image))
```

The two datasets have different tasks and should not be merged under one label space:

- **MMFakeBench**: misinformation truthfulness, `0=true`, `1=fake`; use `val` for development and `test` for final evaluation.
- **Hints-of-Truth**: check-worthiness, `0=not_check_worthy`, `1=check_worthy`; use `dev1` for training, `dev2` for validation, and `test` for final evaluation.

MMFakeBench images are loaded lazily from zip members without extracting the large archives. Hints-of-Truth images are decoded lazily from parquet image bytes in `__getitem__`.

For `DataLoader` usage, use the provided collate function so PIL images remain as lists unless your transform converts them to same-shaped tensors:

```python
from torch.utils.data import DataLoader

loader = DataLoader(mm_val, batch_size=4, collate_fn=cage_collate_fn)
batch = next(iter(loader))
```

## Conflict edge generation

The skeleton keeps conflict insertion modular. A tool or relation classifier can return `ToolResult(..., relation_type=RelationType.CONFLICT)` when it detects disagreement. Future conflict-edge generators can include:

- LLM/NLI zero-shot relation judgment
- Cross-source disagreement detection
- VLM/text mismatch detection
- External prior alignment, e.g. timestamp, location, and entity consistency checks

## Extension points

Implement your own tools by subclassing `BaseTool`:

```python
from cage import BaseTool, ClaimModality, ClaimNode, EvidenceNode, RelationType, ToolResult


class GoogleSearchTool(BaseTool):
    name = "Google"

    def can_handle(self, claim: ClaimNode) -> bool:
        return claim.modality in {ClaimModality.TEXT, ClaimModality.CROSS_MODAL}

    def run(self, claim: ClaimNode, context=None) -> list[ToolResult]:
        evidence = EvidenceNode(
            content="retrieved snippet",
            source=self.name,
            credibility_score=0.8,
        )
        return [ToolResult(evidence=evidence, relation_type=RelationType.SUPPORT)]
```

Replace the heuristic `GraphEvaluator` with a trained GNN, an LLM Graph-of-Thoughts judge, or a hybrid calibrated verifier.
