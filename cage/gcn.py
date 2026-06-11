from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import ClaimModality, ClaimNode, EvidenceGraph, EvidenceNode


class SimpleGraphConvolution(nn.Module):
    """Minimal GCN layer stub without PyTorch Geometric dependency."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.linear(adjacency @ x)


class CAGEGraphScorerGCN(nn.Module):
    """Graph-level scorer stub: logits for True/Fake/Uncertain."""

    def __init__(self, node_feature_dim: int, hidden_dim: int = 128, num_classes: int = 3) -> None:
        super().__init__()
        self.gcn1 = SimpleGraphConvolution(node_feature_dim, hidden_dim)
        self.gcn2 = SimpleGraphConvolution(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gcn1(node_features, adjacency))
        h = F.relu(self.gcn2(h, adjacency))
        graph_embedding = h.mean(dim=0)
        return self.classifier(graph_embedding)


class GraphTensorizer:
    """Convert EvidenceGraph into tensors for the GCN stub.

    Replace handcrafted features with text/VLM/source/relation embeddings in a
    full CAGE implementation.
    """

    def __init__(self, feature_dim: int = 16) -> None:
        self.feature_dim = feature_dim

    def tensorize(self, graph: EvidenceGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        node_ids = list(graph.graph.nodes())
        node_index = {node_id: i for i, node_id in enumerate(node_ids)}
        n = len(node_ids)
        x = torch.zeros(n, self.feature_dim)

        for node_id, idx in node_index.items():
            node = graph.get_node(node_id)
            if isinstance(node, ClaimNode):
                x[idx, 0] = 1.0
                x[idx, 1] = node.uncertainty_score
                x[idx, 2] = 1.0 if node.verified else 0.0
                if node.modality == ClaimModality.TEXT:
                    x[idx, 3] = 1.0
                elif node.modality == ClaimModality.VISUAL:
                    x[idx, 4] = 1.0
                elif node.modality == ClaimModality.CROSS_MODAL:
                    x[idx, 5] = 1.0
            elif isinstance(node, EvidenceNode):
                x[idx, 6] = 1.0
                x[idx, 7] = node.credibility_score

        adjacency = torch.eye(n)
        for source_id, target_id in graph.graph.edges():
            i = node_index[source_id]
            j = node_index[target_id]
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0
        return x, self._normalize_adjacency(adjacency)

    def _normalize_adjacency(self, adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        d_inv_sqrt = torch.diag(degree_inv_sqrt)
        return d_inv_sqrt @ adjacency @ d_inv_sqrt
