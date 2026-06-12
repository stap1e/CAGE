from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

from cage.graph import DynamicEvidenceGraph


@dataclass
class GCNOutput:
    """Typed output bundle for policy/value inference."""

    policy_logits: Tensor
    policy_probs: Tensor
    conflict_logit: Tensor
    conflict_prob: Tensor
    node_embeddings: Tensor
    graph_embedding: Tensor
    current_embedding: Tensor


class PolicyValueGCN(nn.Module):
    """Graph policy-value network for adaptive MCTS.

    Inputs:
    - PyG `Data` object produced by `DynamicEvidenceGraph.to_pyg_data()`

    Outputs:
    - Policy head: tool prior P_GCN(a_t | s_t)
    - Conflict head: graph-level conflict score in [0, 1]
    """

    def __init__(
        self,
        input_dim: int = 528,
        hidden_dim: int = 256,
        tool_vocab: Optional[Sequence[str]] = None,
        num_tools: Optional[int] = None,
        num_layers: int = 3,
        conv_type: str = "gat",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        self.tool_vocab = list(tool_vocab or [])
        self.num_tools = num_tools or len(self.tool_vocab)
        if self.num_tools <= 0:
            raise ValueError("PolicyValueGCN requires a non-empty tool vocabulary")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_type = conv_type.lower()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = input_dim
        for _ in range(num_layers):
            conv = self._build_conv(in_dim, hidden_dim)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_tools),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _build_conv(self, in_dim: int, out_dim: int) -> nn.Module:
        if self.conv_type == "gcn":
            return GCNConv(in_dim, out_dim, add_self_loops=False, normalize=True)
        if self.conv_type == "gat":
            return GATConv(in_dim, out_dim, heads=4, concat=False, add_self_loops=False, dropout=self.dropout)
        raise ValueError(f"Unsupported conv_type: {self.conv_type}")

    def forward(self, data: Data) -> GCNOutput:
        x = data.x
        edge_index = data.edge_index

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        graph_embedding = global_mean_pool(h, batch)

        current_node_index = getattr(data, "current_node_index", None)
        if current_node_index is None:
            current_embedding = graph_embedding
        else:
            current_node_index = current_node_index.to(h.device).view(-1)
            current_embedding = h[current_node_index].mean(dim=0, keepdim=True)

        policy_input = torch.cat([current_embedding, graph_embedding], dim=-1)
        policy_logits = self.policy_head(policy_input)
        policy_probs = torch.softmax(policy_logits, dim=-1)

        value_input = torch.cat(
            [
                graph_embedding,
                current_embedding,
                torch.abs(graph_embedding - current_embedding),
            ],
            dim=-1,
        )
        conflict_logit = self.value_head(value_input).squeeze(-1)
        conflict_prob = torch.sigmoid(conflict_logit)

        return GCNOutput(
            policy_logits=policy_logits,
            policy_probs=policy_probs,
            conflict_logit=conflict_logit,
            conflict_prob=conflict_prob,
            node_embeddings=h,
            graph_embedding=graph_embedding,
            current_embedding=current_embedding,
        )

    @torch.no_grad()
    def predict_graph(
        self,
        graph: DynamicEvidenceGraph,
        current_node_id: Optional[str] = None,
    ) -> GCNOutput:
        was_training = self.training
        self.eval()
        data = graph.to_pyg_data(current_node_id=current_node_id, tool_vocab=self.tool_vocab)
        output = self.forward(data)
        if was_training:
            self.train()
        return output

    @torch.no_grad()
    def policy_dict(
        self,
        graph: DynamicEvidenceGraph,
        current_node_id: Optional[str] = None,
    ) -> dict[str, float]:
        output = self.predict_graph(graph, current_node_id=current_node_id)
        probs = output.policy_probs.squeeze(0).detach().cpu()
        return {tool: float(prob) for tool, prob in zip(self.tool_vocab, probs)}

    @torch.no_grad()
    def conflict_score(
        self,
        graph: DynamicEvidenceGraph,
        current_node_id: Optional[str] = None,
    ) -> float:
        output = self.predict_graph(graph, current_node_id=current_node_id)
        return float(output.conflict_prob.squeeze().detach().cpu())


class GraphTensorizer:
    """Compatibility adapter for code that previously expected a tensorizer."""

    def tensorize(
        self,
        graph: DynamicEvidenceGraph,
        current_node_id: Optional[str] = None,
        tool_vocab: Optional[Sequence[str]] = None,
    ) -> Data:
        return graph.to_pyg_data(current_node_id=current_node_id, tool_vocab=tool_vocab)


class CAGEGraphScorerGCN(nn.Module):
    """Compatibility wrapper for older graph-level scoring code."""

    def __init__(
        self,
        node_feature_dim: int = 528,
        hidden_dim: int = 256,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.backbone = PolicyValueGCN(
            input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            tool_vocab=["web_search", "vqa", "forgery_detection"],
            num_layers=2,
            conv_type="gcn",
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, data: Data) -> Tensor:
        output = self.backbone(data)
        features = torch.cat([output.current_embedding, output.graph_embedding], dim=-1)
        return self.classifier(features)
