from __future__ import annotations

import json
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from cage.agent import AgentDecision, T2Agent
from cage.data import MMFakeBenchDataset
from cage.experiments.metrics import compute_binary_metrics, grouped_binary_metrics


@dataclass
class PredictionRecord:
    """One JSONL prediction record for MMFakeBench."""

    sample_id: str
    text: str
    image_path: Optional[str]
    label: Optional[int]
    label_name: Optional[str]
    prediction: Optional[int]
    prediction_name: str
    fake_probability: float
    confidence: float
    value: float
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    initialization: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    latency_sec: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class MMFakeBenchRunConfig:
    root: str = "/data/lhy_data/MMFakeBench"
    split: str = "val"
    output_path: str = "outputs/mmfakebench_val_predictions.jsonl"
    metrics_path: Optional[str] = None
    max_samples: Optional[int] = None
    start_index: int = 0
    sample_mode: Literal["sequential", "balanced"] = "sequential"
    seed: int = 0
    resume: bool = True
    load_image: bool = False
    pass_image_to_agent: bool = False
    prediction_strategy: str = "fake_cls_baseline"
    flush_every: int = 1


class MMFakeBenchRunner:
    """Minimal experiment loop: MMFakeBench sample -> T2Agent -> JSONL -> metrics."""

    def __init__(self, agent: Optional[T2Agent] = None, config: Optional[MMFakeBenchRunConfig] = None) -> None:
        self.agent = agent or T2Agent()
        self.config = config or MMFakeBenchRunConfig()

    def run(self) -> Dict[str, Any]:
        dataset = MMFakeBenchDataset(
            root=self.config.root,
            split=self.config.split,  # type: ignore[arg-type]
            load_image=self.config.load_image,
        )
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path = Path(self.config.metrics_path) if self.config.metrics_path else output_path.with_suffix(".metrics.json")

        processed_ids = self._load_processed_ids(output_path) if self.config.resume else set()
        records: List[PredictionRecord] = self._load_existing_records(output_path) if self.config.resume else []

        sample_indices = self._build_sample_indices(dataset)

        mode = "a" if self.config.resume and output_path.exists() else "w"
        written_since_flush = 0
        total_selected = len(sample_indices)
        with output_path.open(mode, encoding="utf-8") as f:
            for progress, idx in enumerate(sample_indices, start=1):
                sample = dataset[idx]
                if sample.sample_id in processed_ids:
                    continue

                start = time.perf_counter()
                record: PredictionRecord
                try:
                    image_path = sample.image_path if self.config.pass_image_to_agent else None
                    decision = self.agent.run(query=sample.text, image_path=image_path)
                    prediction, prediction_name, fake_probability = self._predict_sample(decision, sample.metadata)
                    record = PredictionRecord(
                        sample_id=sample.sample_id,
                        text=sample.text,
                        image_path=sample.image_path,
                        label=sample.label,
                        label_name=sample.label_name,
                        prediction=prediction,
                        prediction_name=prediction_name,
                        fake_probability=fake_probability,
                        confidence=decision.confidence,
                        value=decision.value,
                        trajectory=decision.trajectory,
                        evidence=decision.evidence,
                        initialization=decision.initialization,
                        metadata={**sample.metadata, **sample.source, "index": idx},
                        latency_sec=time.perf_counter() - start,
                    )
                except Exception as exc:  # noqa: BLE001 - keep batch jobs alive
                    record = PredictionRecord(
                        sample_id=sample.sample_id,
                        text=sample.text,
                        image_path=sample.image_path,
                        label=sample.label,
                        label_name=sample.label_name,
                        prediction=None,
                        prediction_name="error",
                        fake_probability=0.5,
                        confidence=0.0,
                        value=0.0,
                        metadata={**sample.metadata, **sample.source, "index": idx},
                        error=f"{exc}\n{traceback.format_exc()}",
                        latency_sec=time.perf_counter() - start,
                    )

                f.write(record.to_json() + "\n")
                records.append(record)
                written_since_flush += 1
                if written_since_flush >= self.config.flush_every:
                    f.flush()
                    written_since_flush = 0

                print(
                    f"[{progress}/{total_selected}] {record.sample_id} "
                    f"label={record.label_name} pred={record.prediction_name} "
                    f"conf={record.confidence:.3f} err={record.error is not None}",
                    file=sys.stderr,
                )

        metrics = self.compute_metrics(records)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return metrics

    def compute_metrics(self, records: List[PredictionRecord]) -> Dict[str, Any]:
        valid = [r for r in records if r.label is not None and r.prediction is not None and r.error is None]
        failed = [r for r in records if r.error is not None or r.prediction is None]
        output: Dict[str, Any] = {
            "num_records": len(records),
            "num_valid": len(valid),
            "num_failed": len(failed),
        }
        if not valid:
            output["metrics"] = None
            return output

        y_true = [int(r.label) for r in valid if r.label is not None]
        y_pred = [int(r.prediction) for r in valid if r.prediction is not None]
        output["metrics"] = compute_binary_metrics(y_true, y_pred).to_dict()
        output["by_fake_cls"] = grouped_binary_metrics(
            y_true,
            y_pred,
            [str(r.metadata.get("fake_cls")) if r.metadata.get("fake_cls") is not None else None for r in valid],
        )
        output["avg_confidence"] = sum(r.confidence for r in valid) / len(valid)
        output["avg_value"] = sum(r.value for r in valid) / len(valid)
        output["avg_latency_sec"] = sum(r.latency_sec for r in valid) / len(valid)
        output["avg_trajectory_len"] = sum(len(r.trajectory) for r in valid) / len(valid)
        return output

    def _build_sample_indices(self, dataset: MMFakeBenchDataset) -> List[int]:
        if self.config.sample_mode == "sequential":
            return self._build_sequential_indices(len(dataset))
        if self.config.sample_mode == "balanced":
            if self.config.max_samples is None:
                return self._build_sequential_indices(len(dataset))
            return self._build_balanced_indices(dataset)
        raise ValueError(f"Unknown sample_mode: {self.config.sample_mode}")

    def _build_sequential_indices(self, dataset_len: int) -> List[int]:
        start = self.config.start_index
        if start < 0:
            raise ValueError(f"start_index must be non-negative; got {start}")
        end = dataset_len
        if self.config.max_samples is not None:
            if self.config.max_samples < 0:
                raise ValueError(f"max_samples must be non-negative; got {self.config.max_samples}")
            end = min(dataset_len, start + self.config.max_samples)
        return list(range(min(start, dataset_len), end))

    def _build_balanced_indices(self, dataset: MMFakeBenchDataset) -> List[int]:
        if self.config.max_samples is None:
            return self._build_sequential_indices(len(dataset))
        if self.config.max_samples < 0:
            raise ValueError(f"max_samples must be non-negative; got {self.config.max_samples}")
        if self.config.max_samples % 2 != 0:
            raise ValueError(
                "balanced mode requires an even --max-samples so true/fake counts can match; "
                f"got {self.config.max_samples}"
            )
        if self.config.start_index < 0:
            raise ValueError(f"start_index must be non-negative; got {self.config.start_index}")

        true_indices: List[int] = []
        fake_indices: List[int] = []
        for idx in range(min(self.config.start_index, len(dataset)), len(dataset)):
            raw_label = dataset.records[idx].get("gt_answers")
            if raw_label == "True":
                true_indices.append(idx)
            elif raw_label == "Fake":
                fake_indices.append(idx)
            else:
                raise ValueError(f"Unexpected MMFakeBench gt_answers at index {idx}: {raw_label!r}")

        per_class = self.config.max_samples // 2
        if len(true_indices) < per_class or len(fake_indices) < per_class:
            raise ValueError(
                "balanced mode requested "
                f"{per_class} true and {per_class} fake samples, but only "
                f"{len(true_indices)} true and {len(fake_indices)} fake are available "
                f"from start_index={self.config.start_index}"
            )

        rng = random.Random(self.config.seed)
        rng.shuffle(true_indices)
        rng.shuffle(fake_indices)
        return sorted(true_indices[:per_class] + fake_indices[:per_class])

    def _predict_sample(self, decision: AgentDecision, metadata: Dict[str, Any]) -> tuple[int, str, float]:
        """Convert an agent decision into a binary MMFakeBench prediction.

        Strategies:
            - fake_cls_baseline: sanity baseline using MMFakeBench fake_cls
              (original -> true, all other fake types -> fake). This validates
              the dataset/metrics loop before real model integration.
            - agent: use the current T2Agent heuristic decision.
        """
        if self.config.prediction_strategy == "fake_cls_baseline":
            return self._fake_cls_baseline_prediction(metadata)
        if self.config.prediction_strategy == "agent":
            return self._decision_to_prediction(decision)
        raise ValueError(f"Unknown prediction_strategy: {self.config.prediction_strategy}")

    def _fake_cls_baseline_prediction(self, metadata: Dict[str, Any]) -> tuple[int, str, float]:
        fake_cls = str(metadata.get("fake_cls", "")).strip().lower()
        prediction = 0 if fake_cls == "original" else 1
        return prediction, "true" if prediction == 0 else "fake", 0.0 if prediction == 0 else 1.0

    def _decision_to_prediction(self, decision: AgentDecision) -> tuple[int, str, float]:
        answer = decision.answer.lower()
        fake_prior = max(
            float(decision.initialization.get("text_authenticity_fake_prob", 0.0)),
            float(decision.initialization.get("image_authenticity_fake_prob", 0.0)),
            float(decision.initialization.get("cross_modal_inconsistency_prob", 0.0)),
        )
        fake_probability = max(0.0, min(1.0, 0.4 * fake_prior + 0.6 * (1.0 - decision.value)))
        if "fake_or_misleading" in answer:
            return 1, "fake", fake_probability
        if "likely_true" in answer:
            return 0, "true", fake_probability
        prediction = 1 if fake_probability >= 0.5 else 0
        return prediction, "fake" if prediction == 1 else "true", fake_probability

    def _load_processed_ids(self, output_path: Path) -> set[str]:
        return {record.sample_id for record in self._load_existing_records(output_path)}

    def _load_existing_records(self, output_path: Path) -> List[PredictionRecord]:
        if not output_path.exists():
            return []
        records: List[PredictionRecord] = []
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(PredictionRecord(**data))
                except Exception:
                    continue
        return records
