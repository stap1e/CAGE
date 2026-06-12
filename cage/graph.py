from __future__ import annotations

import copy
import hashlib
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple, Union

import networkx as nx
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data


class RelationType(str, Enum):
    """Typed relations used inside the dynamic evidence graph."""

    ROOT = "root"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    SUPPORT = "support"
    REFUTE = "refute"
    CONFLICT = "conflict"
    NEUTRAL = "neutral"
    DERIVED_FROM = "derived_from"
    ALIGNS_WITH = "aligns_with"
    MENTIONS = "mentions"
    QUERIES = "queries"
    OBSERVES = "observes"
    SELF = "self"


class ClaimModality(str, Enum):
    """Legacy modality enum kept for compatibility with the existing codebase."""

    TEXT = "text"
    VISUAL = "visual"
    CROSS_MODAL = "cross-modal"


class NodeType(str, Enum):
    ROOT = "root"
    CLAIM = "claim"
    EVIDENCE = "evidence"
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    ENTITY = "entity"


@dataclass
class ClaimNode:
    """Atomic claim node used by the old CAGE graph API and the new dynamic graph."""

    claim_text: str
    uncertainty_score: float
    modality: Union[str, ClaimModality]
    node_id: str = field(default_factory=lambda: f"claim:{uuid.uuid4()}")
    verified: bool = False
    feature: Optional[Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.modality, str):
            self.modality = ClaimModality(self.modality)
        self.uncertainty_score = min(max(float(self.uncertainty_score), 0.0), 1.0)


@dataclass
class EvidenceNode:
    """Evidence node used by the old CAGE graph API and the new dynamic graph."""

    content: str
    source: str
    credibility_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    node_id: str = field(default_factory=lambda: f"evidence:{uuid.uuid4()}")
    feature: Optional[Tensor] = None

    def __post_init__(self) -> None:
        self.credibility_score = min(max(float(self.credibility_score), 0.0), 1.0)


@dataclass
class GraphRecord:
    """Unified internal node record stored in NetworkX."""

    node_id: str
    node_type: NodeType
    modality: str
    text: str = ""
    feature: Optional[Tensor] = None
    score: float = 0.0
    step_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_data: Optional[Union[ClaimNode, EvidenceNode]] = None


class EmbeddingBackend(Protocol):
    """Pluggable lightweight embedding backend.

    TODO:
    Replace the mock implementation below with Sentence-BERT / CLIP / EVA-CLIP /
    SigLIP / BLIP-2 features in production.
    """

    def embed_text(self, text: str) -> Tensor:
        ...

    def embed_image(
        self,
        image_feature: Optional[Tensor] = None,
        image_payload: Optional[Any] = None,
    ) -> Tensor:
        ...


class MockFeatureEmbedder:
    """Deterministic mock feature embedder.

    This keeps the graph pipeline runnable before a real multimodal backbone is
    wired in. Text is hashed into a stable pseudo-random vector. Visual features
    are passed through if already available; otherwise a zero vector is used.
    """

    def __init__(self, feature_dim: int = 512, device: Optional[torch.device] = None) -> None:
        self.feature_dim = feature_dim
        self.device = device or torch.device("cpu")

    def embed_text(self, text: str) -> Tensor:
        text = text or ""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        seed = int(digest[:16], 16) % (2**31 - 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        vector = torch.randn(self.feature_dim, generator=generator, dtype=torch.float32)
        return F.normalize(vector, p=2, dim=0).to(self.device)

    def embed_image(
        self,
        image_feature: Optional[Tensor] = None,
        image_payload: Optional[Any] = None,
    ) -> Tensor:
        if image_feature is None:
            vector = torch.zeros(self.feature_dim, dtype=torch.float32, device=self.device)
        else:
            vector = image_feature.detach().clone().to(self.device, dtype=torch.float32).view(-1)
            if vector.numel() < self.feature_dim:
                pad = torch.zeros(self.feature_dim - vector.numel(), device=self.device)
                vector = torch.cat([vector, pad], dim=0)
            elif vector.numel() > self.feature_dim:
                vector = vector[: self.feature_dim]
        norm = torch.norm(vector, p=2)
        if norm > 0:
            vector = vector / norm
        return vector


GraphNode = Union[ClaimNode, EvidenceNode]


class DynamicEvidenceGraph:
    """Dynamic evidence graph that grows with every MCTS step.

    Internally it uses a NetworkX MultiDiGraph for flexible typed edges, and it
    exposes a `to_pyg_data()` adapter for PyTorch Geometric.
    """

    _NODE_TYPE_TO_INDEX: Dict[NodeType, int] = {
        NodeType.ROOT: 0,
        NodeType.CLAIM: 1,
        NodeType.EVIDENCE: 2,
        NodeType.THOUGHT: 3,
        NodeType.ACTION: 4,
        NodeType.OBSERVATION: 5,
        NodeType.ENTITY: 6,
    }

    _MODALITY_TO_INDEX: Dict[str, int] = {
        "none": 0,
        "text": 1,
        "visual": 2,
        "cross-modal": 3,
    }

    _RELATION_TO_INDEX: Dict[RelationType, int] = {
        RelationType.ROOT: 0,
        RelationType.TEMPORAL: 1,
        RelationType.CAUSAL: 2,
        RelationType.SUPPORT: 3,
        RelationType.REFUTE: 4,
        RelationType.CONFLICT: 5,
        RelationType.NEUTRAL: 6,
        RelationType.DERIVED_FROM: 7,
        RelationType.ALIGNS_WITH: 8,
        RelationType.MENTIONS: 9,
        RelationType.QUERIES: 10,
        RelationType.OBSERVES: 11,
        RelationType.SELF: 12,
    }

    def __init__(
        self,
        embedder: Optional[EmbeddingBackend] = None,
        feature_dim: int = 512,
        structural_dim: int = 16,
        device: Optional[torch.device] = None,
        tool_vocab: Optional[Sequence[str]] = None,
        initialize_root: bool = True,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.feature_dim = feature_dim
        self.structural_dim = structural_dim
        self.embedder = embedder or MockFeatureEmbedder(feature_dim=feature_dim, device=self.device)
        self.tool_vocab = list(tool_vocab or [])
        self.graph = nx.MultiDiGraph()
        self.root_node_id: Optional[str] = None
        self.current_node_id: Optional[str] = None
        self.step_counter: int = 0

        if initialize_root:
            root_feature = torch.zeros(self.feature_dim, dtype=torch.float32, device=self.device)
            self.root_node_id = self.add_node(
                node_type=NodeType.ROOT,
                text="ROOT",
                modality="none",
                feature=root_feature,
                score=0.0,
                metadata={"is_root": True},
                parent_node_id=None,
                relation_type=None,
                node_id="root:0",
            )
            self.current_node_id = self.root_node_id

    def copy(self) -> "DynamicEvidenceGraph":
        cloned = DynamicEvidenceGraph(
            embedder=self.embedder,
            feature_dim=self.feature_dim,
            structural_dim=self.structural_dim,
            device=self.device,
            tool_vocab=self.tool_vocab,
            initialize_root=False,
        )
        cloned.graph = copy.deepcopy(self.graph)
        cloned.root_node_id = self.root_node_id
        cloned.current_node_id = self.current_node_id
        cloned.step_counter = self.step_counter
        return cloned

    @property
    def node_feature_dim(self) -> int:
        return self.feature_dim + self.structural_dim

    def add_node(
        self,
        node_type: NodeType,
        text: str = "",
        modality: str = "text",
        feature: Optional[Tensor] = None,
        score: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
        parent_node_id: Optional[str] = None,
        relation_type: Optional[RelationType] = RelationType.TEMPORAL,
        node_id: Optional[str] = None,
        raw_data: Optional[GraphNode] = None,
    ) -> str:
        resolved_node_id = node_id or f"{node_type.value}:{uuid.uuid4()}"
        if feature is None:
            if modality == "visual":
                feature = self.embedder.embed_image()
            else:
                feature = self.embedder.embed_text(text)
        feature = self._prepare_feature(feature)

        record = GraphRecord(
            node_id=resolved_node_id,
            node_type=node_type,
            modality=modality,
            text=text,
            feature=feature,
            score=float(score),
            step_index=self.step_counter,
            metadata=dict(metadata or {}),
            raw_data=raw_data,
        )

        self.graph.add_node(
            resolved_node_id,
            data=record,
            node_type=node_type.value,
            modality=modality,
            text=text,
            score=float(score),
            step_index=self.step_counter,
            metadata=dict(metadata or {}),
        )

        if parent_node_id is not None and relation_type is not None:
            self.add_relation(parent_node_id, resolved_node_id, relation_type)

        self.current_node_id = resolved_node_id
        return resolved_node_id

    def add_claim(
        self,
        claim: ClaimNode,
        parent_node_id: Optional[str] = None,
        relation_type: RelationType = RelationType.DERIVED_FROM,
    ) -> str:
        return self.add_node(
            node_type=NodeType.CLAIM,
            text=claim.claim_text,
            modality=claim.modality.value,
            feature=claim.feature,
            score=1.0 - claim.uncertainty_score,
            metadata={
                "uncertainty_score": claim.uncertainty_score,
                "verified": claim.verified,
                **claim.metadata,
            },
            parent_node_id=parent_node_id or self.current_node_id or self.root_node_id,
            relation_type=relation_type,
            node_id=claim.node_id,
            raw_data=claim,
        )

    def add_evidence(
        self,
        evidence: EvidenceNode,
        parent_node_id: Optional[str] = None,
        relation_type: RelationType = RelationType.OBSERVES,
    ) -> str:
        return self.add_node(
            node_type=NodeType.EVIDENCE,
            text=evidence.content,
            modality="text",
            feature=evidence.feature,
            score=evidence.credibility_score,
            metadata={"source": evidence.source, **evidence.metadata},
            parent_node_id=parent_node_id or self.current_node_id or self.root_node_id,
            relation_type=relation_type,
            node_id=evidence.node_id,
            raw_data=evidence,
        )

    def add_entity_node(
        self,
        entity_text: str,
        modality: str = "text",
        parent_node_id: Optional[str] = None,
        feature: Optional[Tensor] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        return self.add_node(
            node_type=NodeType.ENTITY,
            text=entity_text,
            modality=modality,
            feature=feature,
            score=0.5,
            metadata=metadata,
            parent_node_id=parent_node_id or self.current_node_id or self.root_node_id,
            relation_type=RelationType.MENTIONS,
        )

    def add_relation(
        self,
        source_node: Union[str, GraphNode],
        target_node: Union[str, GraphNode],
        relation_type: RelationType,
        weight: float = 1.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        source_id = self._resolve_node_id(source_node)
        target_id = self._resolve_node_id(target_node)
        if source_id not in self.graph:
            raise KeyError(f"Source node does not exist: {source_id}")
        if target_id not in self.graph:
            raise KeyError(f"Target node does not exist: {target_id}")

        self.graph.add_edge(
            source_id,
            target_id,
            relation_type=relation_type,
            weight=float(weight),
            metadata=dict(metadata or {}),
        )

    def connect_claim_evidence(
        self,
        claim: Union[str, ClaimNode],
        evidence: Union[str, EvidenceNode],
        relation_type: RelationType,
        weight: float = 1.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        evidence_id = self._resolve_node_id(evidence)
        claim_id = self._resolve_node_id(claim)
        self.add_relation(evidence_id, claim_id, relation_type, weight=weight, metadata=metadata)

    def link_cross_modal_entities(
        self,
        text_entity_id: str,
        visual_entity_id: str,
        alignment_score: float = 1.0,
    ) -> None:
        self.add_relation(
            text_entity_id,
            visual_entity_id,
            RelationType.ALIGNS_WITH,
            weight=alignment_score,
            metadata={"cross_modal_alignment": True},
        )
        self.add_relation(
            visual_entity_id,
            text_entity_id,
            RelationType.ALIGNS_WITH,
            weight=alignment_score,
            metadata={"cross_modal_alignment": True},
        )

    def add_step(
        self,
        thought: str,
        action: str,
        observation: str,
        parent_node_id: Optional[str] = None,
        action_input: str = "",
        thought_feature: Optional[Tensor] = None,
        action_feature: Optional[Tensor] = None,
        observation_feature: Optional[Tensor] = None,
        observation_visual_feature: Optional[Tensor] = None,
        action_metadata: Optional[Mapping[str, Any]] = None,
        observation_metadata: Optional[Mapping[str, Any]] = None,
        extracted_entities: Optional[Sequence[Mapping[str, Any]]] = None,
        aligned_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> Dict[str, str]:
        self.step_counter += 1
        anchor_id = parent_node_id or self.current_node_id or self.root_node_id
        if anchor_id is None:
            raise RuntimeError("Graph root is missing; cannot append a new step.")

        thought_id = self.add_node(
            node_type=NodeType.THOUGHT,
            text=thought,
            modality="text",
            feature=thought_feature,
            score=0.5,
            metadata={"step_index": self.step_counter},
            parent_node_id=anchor_id,
            relation_type=RelationType.TEMPORAL,
        )

        action_id = self.add_node(
            node_type=NodeType.ACTION,
            text=action,
            modality="text",
            feature=action_feature,
            score=0.5,
            metadata={"step_index": self.step_counter, "action_input": action_input, **dict(action_metadata or {})},
            parent_node_id=thought_id,
            relation_type=RelationType.CAUSAL,
        )

        observation_id = self.add_node(
            node_type=NodeType.OBSERVATION,
            text=observation,
            modality="text",
            feature=observation_feature,
            score=0.5,
            metadata={"step_index": self.step_counter, **dict(observation_metadata or {})},
            parent_node_id=action_id,
            relation_type=RelationType.CAUSAL,
        )

        if observation_visual_feature is not None:
            visual_node_id = self.add_node(
                node_type=NodeType.OBSERVATION,
                text="[visual_observation]",
                modality="visual",
                feature=observation_visual_feature,
                score=0.5,
                metadata={"step_index": self.step_counter, "is_visual_proxy": True},
                parent_node_id=action_id,
                relation_type=RelationType.CAUSAL,
            )
            self.link_cross_modal_entities(observation_id, visual_node_id, alignment_score=1.0)

        entity_ids: Dict[str, str] = {}
        for entity in extracted_entities or []:
            entity_name = str(entity.get("text", "")).strip()
            if not entity_name:
                continue
            entity_modality = str(entity.get("modality", "text"))
            entity_feature = entity.get("feature")
            entity_id = self.add_entity_node(
                entity_text=entity_name,
                modality=entity_modality,
                parent_node_id=observation_id,
                feature=entity_feature,
                metadata=dict(entity),
            )
            entity_ids[entity_name] = entity_id

        for left_name, right_name in aligned_pairs or []:
            if left_name in entity_ids and right_name in entity_ids:
                self.link_cross_modal_entities(entity_ids[left_name], entity_ids[right_name], alignment_score=1.0)

        self.current_node_id = observation_id
        return {
            "parent": anchor_id,
            "thought": thought_id,
            "action": action_id,
            "observation": observation_id,
        }

    def get_node(self, node_id: str) -> GraphRecord:
        return self.graph.nodes[node_id]["data"]

    def get_claim_nodes(self) -> list[ClaimNode]:
        claims: list[ClaimNode] = []
        for _, attrs in self.graph.nodes(data=True):
            data = attrs["data"]
            if data.node_type != NodeType.CLAIM:
                continue
            if isinstance(data.raw_data, ClaimNode):
                claims.append(data.raw_data)
            else:
                claims.append(
                    ClaimNode(
                        claim_text=data.text,
                        uncertainty_score=1.0 - data.score,
                        modality=data.modality,
                        node_id=data.node_id,
                        verified=bool(data.metadata.get("verified", False)),
                        feature=data.feature,
                        metadata=dict(data.metadata),
                    )
                )
        return claims

    def get_evidence_nodes(self) -> list[EvidenceNode]:
        evidences: list[EvidenceNode] = []
        for _, attrs in self.graph.nodes(data=True):
            data = attrs["data"]
            if data.node_type != NodeType.EVIDENCE:
                continue
            if isinstance(data.raw_data, EvidenceNode):
                evidences.append(data.raw_data)
            else:
                evidences.append(
                    EvidenceNode(
                        content=data.text,
                        source=str(data.metadata.get("source", "unknown")),
                        credibility_score=float(data.score),
                        metadata=dict(data.metadata),
                        node_id=data.node_id,
                        feature=data.feature,
                    )
                )
        return evidences

    def get_unverified_claims(self) -> list[ClaimNode]:
        return [claim for claim in self.get_claim_nodes() if not claim.verified]

    def mark_claim_verified(self, claim: Union[str, ClaimNode]) -> None:
        claim_id = self._resolve_node_id(claim)
        record = self.get_node(claim_id)
        if record.node_type != NodeType.CLAIM:
            raise TypeError(f"Node is not a claim node: {claim_id}")
        record.metadata["verified"] = True
        self.graph.nodes[claim_id]["metadata"]["verified"] = True
        if isinstance(record.raw_data, ClaimNode):
            record.raw_data.verified = True

    def incoming_evidence_edges(
        self,
        claim: Union[str, ClaimNode],
        relation_types: Optional[Sequence[RelationType]] = None,
    ) -> list[Tuple[EvidenceNode, Dict[str, Any]]]:
        claim_id = self._resolve_node_id(claim)
        results: list[Tuple[EvidenceNode, Dict[str, Any]]] = []
        for source_id, _, _, edge_attr in self.graph.in_edges(claim_id, keys=True, data=True):
            source_record = self.get_node(source_id)
            if source_record.node_type != NodeType.EVIDENCE:
                continue
            relation = edge_attr.get("relation_type")
            if relation_types is not None and relation not in relation_types:
                continue
            if isinstance(source_record.raw_data, EvidenceNode):
                evidence = source_record.raw_data
            else:
                evidence = EvidenceNode(
                    content=source_record.text,
                    source=str(source_record.metadata.get("source", "unknown")),
                    credibility_score=float(source_record.score),
                    metadata=dict(source_record.metadata),
                    node_id=source_record.node_id,
                    feature=source_record.feature,
                )
            results.append((evidence, dict(edge_attr)))
        return results

    def claim_has_unresolved_conflict(self, claim: Union[str, ClaimNode]) -> bool:
        support_edges = self.incoming_evidence_edges(claim, [RelationType.SUPPORT])
        refute_edges = self.incoming_evidence_edges(claim, [RelationType.REFUTE])
        conflict_edges = self.incoming_evidence_edges(claim, [RelationType.CONFLICT])

        credible_support = [e for e, _ in support_edges if e.credibility_score >= 0.6]
        credible_refute = [e for e, _ in refute_edges if e.credibility_score >= 0.6]
        credible_conflict = [e for e, _ in conflict_edges if e.credibility_score >= 0.6]
        return bool(credible_conflict or (credible_support and credible_refute))

    def num_claims(self) -> int:
        return sum(1 for _, attrs in self.graph.nodes(data=True) if attrs["data"].node_type == NodeType.CLAIM)

    def num_evidence(self) -> int:
        return sum(1 for _, attrs in self.graph.nodes(data=True) if attrs["data"].node_type == NodeType.EVIDENCE)

    def to_pyg_data(
        self,
        current_node_id: Optional[str] = None,
        tool_vocab: Optional[Sequence[str]] = None,
    ) -> Data:
        node_ids = list(self.graph.nodes())
        node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        num_nodes = len(node_ids)

        x = torch.zeros(num_nodes, self.node_feature_dim, dtype=torch.float32, device=self.device)
        for node_id, idx in node_index.items():
            record = self.get_node(node_id)
            base = record.feature if record.feature is not None else self.embedder.embed_text(record.text)
            structural = self._structural_features(record)
            x[idx] = torch.cat([base, structural], dim=0)

        edge_pairs: list[Tuple[int, int]] = []
        edge_types: list[int] = []
        edge_weights: list[float] = []

        for source_id, target_id, _, edge_attr in self.graph.edges(keys=True, data=True):
            edge_pairs.append((node_index[source_id], node_index[target_id]))
            relation = edge_attr.get("relation_type", RelationType.NEUTRAL)
            edge_types.append(self._RELATION_TO_INDEX[relation])
            edge_weights.append(float(edge_attr.get("weight", 1.0)))

        for node_id, idx in node_index.items():
            edge_pairs.append((idx, idx))
            edge_types.append(self._RELATION_TO_INDEX[RelationType.SELF])
            edge_weights.append(1.0)

        edge_index = torch.tensor(edge_pairs, dtype=torch.long, device=self.device).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long, device=self.device)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32, device=self.device)

        resolved_current_node = current_node_id or self.current_node_id or self.root_node_id
        if resolved_current_node is None:
            raise RuntimeError("Current node is undefined.")
        current_node_index = torch.tensor([node_index[resolved_current_node]], dtype=torch.long, device=self.device)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_weight=edge_weight,
            current_node_index=current_node_index,
        )
        data.num_nodes = num_nodes
        data.num_tools = len(tool_vocab or self.tool_vocab)
        return data

    def _prepare_feature(self, feature: Tensor) -> Tensor:
        feature = feature.detach().clone().to(self.device, dtype=torch.float32).view(-1)
        if feature.numel() < self.feature_dim:
            pad = torch.zeros(self.feature_dim - feature.numel(), dtype=torch.float32, device=self.device)
            feature = torch.cat([feature, pad], dim=0)
        elif feature.numel() > self.feature_dim:
            feature = feature[: self.feature_dim]
        norm = torch.norm(feature, p=2)
        if norm > 0:
            feature = feature / norm
        return feature

    def _structural_features(self, record: GraphRecord) -> Tensor:
        features = torch.zeros(self.structural_dim, dtype=torch.float32, device=self.device)

        node_type_idx = self._NODE_TYPE_TO_INDEX[record.node_type]
        if node_type_idx < 7:
            features[node_type_idx] = 1.0

        modality_idx = self._MODALITY_TO_INDEX.get(record.modality, 0)
        modality_offset = 7
        if modality_idx < 4 and modality_offset + modality_idx < self.structural_dim:
            features[modality_offset + modality_idx] = 1.0

        features[11] = float(record.score)
        features[12] = float(self.graph.in_degree(record.node_id))
        features[13] = float(self.graph.out_degree(record.node_id))
        features[14] = float(record.step_index)
        features[15] = 1.0 if record.node_id == self.current_node_id else 0.0
        return features

    def _resolve_node_id(self, node: Union[str, GraphNode]) -> str:
        if isinstance(node, str):
            return node
        return node.node_id

    def __len__(self) -> int:
        return self.graph.number_of_nodes()

    def __repr__(self) -> str:
        return (
            f"DynamicEvidenceGraph(num_nodes={self.graph.number_of_nodes()}, "
            f"num_edges={self.graph.number_of_edges()}, current_node={self.current_node_id})"
        )


EvidenceGraph = DynamicEvidenceGraph
