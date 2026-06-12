import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_fake_torch_geometric() -> None:
    """Install a tiny torch_geometric stub for environments without PyG."""
    if "torch_geometric" in sys.modules:
        return

    tg_module = types.ModuleType("torch_geometric")
    data_module = types.ModuleType("torch_geometric.data")
    nn_module = types.ModuleType("torch_geometric.nn")

    class Data:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def to(self, device: torch.device):
            for key, value in list(self.__dict__.items()):
                if isinstance(value, torch.Tensor):
                    setattr(self, key, value.to(device))
            return self

    class _BaseConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, **_: object) -> None:
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            if edge_index.numel() == 0:
                return self.linear(x)

            source, target = edge_index
            aggregated = torch.zeros_like(x)
            aggregated.index_add_(0, target, x[source])

            degree = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
            degree.index_add_(0, target, torch.ones(target.size(0), dtype=x.dtype, device=x.device))
            degree = degree.clamp_min(1.0).unsqueeze(-1)
            aggregated = (aggregated + x) / (degree + 1.0)
            return self.linear(aggregated)

    class GCNConv(_BaseConv):
        pass

    class GATConv(_BaseConv):
        pass

    def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if batch.numel() == 0:
            return x.mean(dim=0, keepdim=True)
        num_graphs = int(batch.max().item()) + 1
        pooled = []
        for graph_idx in range(num_graphs):
            mask = batch == graph_idx
            pooled.append(x[mask].mean(dim=0))
        return torch.stack(pooled, dim=0)

    data_module.Data = Data
    nn_module.GCNConv = GCNConv
    nn_module.GATConv = GATConv
    nn_module.global_mean_pool = global_mean_pool

    tg_module.data = data_module
    tg_module.nn = nn_module

    sys.modules["torch_geometric"] = tg_module
    sys.modules["torch_geometric.data"] = data_module
    sys.modules["torch_geometric.nn"] = nn_module


_install_fake_torch_geometric()

from cage.evaluation import Evaluator
from cage.gcn import PolicyValueGCN
from cage.graph import DynamicEvidenceGraph
from cage.mcts import MCTSConfig, MCTSSearcher


class DummyController:
    def __init__(self, actions: list[str]) -> None:
        self.actions = actions
        self.calls = 0

    def available_actions(
        self,
        _query: str,
        _history: list[dict],
        _graph_state: DynamicEvidenceGraph,
    ) -> list[str]:
        return list(self.actions)

    def propose_next(self, query: str, _history: list[dict]) -> tuple[str, str, str]:
        return "Need one more tool call to resolve the claim.", self.actions[0], query

    def propose_step(
        self,
        query: str,
        _history: list[dict],
        preferred_action: str,
        _graph_state: DynamicEvidenceGraph,
    ) -> tuple[str, str, str]:
        return f"Use {preferred_action} to inspect the claim.", preferred_action, query

    def execute_action(self, action_name: str, _action_input: str):
        self.calls += 1
        payloads = {
            "web_search": {
                "text": "Retrieved article matched the claim context and source metadata.",
                "credibility_score": 0.85,
                "entities": [{"text": "bridge", "modality": "text"}],
            },
            "forgery_detection": {
                "text": "Forgery detector found inconsistent compression artifacts and possible manipulation.",
                "credibility_score": 0.9,
                "entities": [{"text": "bridge-visual", "modality": "visual"}],
                "visual_feature": torch.ones(512),
                "aligned_pairs": [("bridge", "bridge-visual")],
            },
            "vqa": {
                "text": "VQA matched the scene content with the headline description.",
                "credibility_score": 0.75,
                "visual_feature": torch.full((512,), 0.5),
            },
        }
        return payloads.get(
            action_name,
            {
                "text": f"{action_name} returned neutral evidence.",
                "credibility_score": 0.6,
            },
        )


def test_graph_gcn_mcts_decision_pipeline_runs_end_to_end() -> None:
    tool_vocab = ["web_search", "forgery_detection", "vqa"]

    root_graph = DynamicEvidenceGraph(tool_vocab=tool_vocab)
    model = PolicyValueGCN(
        input_dim=root_graph.node_feature_dim,
        hidden_dim=64,
        tool_vocab=tool_vocab,
        num_layers=2,
        conv_type="gcn",
        dropout=0.0,
    )
    controller = DummyController(tool_vocab)
    evaluator = Evaluator(graph_model=model)
    searcher = MCTSSearcher(
        controller=controller,
        evaluator=evaluator,
        policy_value_net=model,
        config=MCTSConfig(max_iterations=4, max_depth=2, prune_siblings_on_decisive=False),
        tool_vocab=tool_vocab,
    )

    best_node = searcher.search(
        "Is the news image manipulated or authentic?",
        root_graph=root_graph,
    )
    assert best_node is not None
    assert controller.calls > 0

    root = searcher.last_root
    assert root is not None
    assert root.graph_state.num_claims() >= 1
    assert len(root.children) >= 1

    direct_output = model.predict_graph(root.graph_state)
    assert direct_output.policy_probs.shape == (1, len(tool_vocab))
    assert 0.0 <= float(direct_output.conflict_prob.item()) <= 1.0

    grown_graph_sizes = [child.graph_state.graph.number_of_nodes() for child in root.children]
    assert max(grown_graph_sizes) > root.graph_state.graph.number_of_nodes()

    decision = searcher.final_decision(root, "Is the news image manipulated or authentic?")
    assert decision.verdict in {"True", "Fake", "Uncertain"}
    assert 0.0 <= decision.fake_probability <= 1.0
    assert 0.0 <= decision.confidence <= 1.0
    assert len(decision.path_breakdown) >= 1
