from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class BinaryClassificationMetrics:
    """Binary classification metrics for MMFakeBench.

    Label convention:
        0 = true / original
        1 = fake / misinformation
    """

    accuracy: float
    macro_f1: float
    precision_true: float
    recall_true: float
    f1_true: float
    precision_fake: float
    recall_fake: float
    f1_fake: float
    tp: int
    tn: int
    fp: int
    fn: int
    total: int

    def to_dict(self) -> Dict[str, float | int]:
        return asdict(self)


def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> Tuple[int, int, int, int]:
    """Return (tp, tn, fp, fn), treating label 1 as the positive fake class."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred length mismatch: {len(y_true)} vs {len(y_pred)}")
    tp = tn = fp = fn = 0
    for target, pred in zip(y_true, y_pred):
        if target == 1 and pred == 1:
            tp += 1
        elif target == 0 and pred == 0:
            tn += 1
        elif target == 0 and pred == 1:
            fp += 1
        elif target == 1 and pred == 0:
            fn += 1
        else:
            raise ValueError(f"Labels must be 0/1, got target={target}, pred={pred}")
    return tp, tn, fp, fn


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def f1_score(precision: float, recall: float) -> float:
    return safe_div(2.0 * precision * recall, precision + recall)


def compute_binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> BinaryClassificationMetrics:
    """Compute accuracy, per-class precision/recall/F1, and macro-F1."""
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    total = len(y_true)
    accuracy = safe_div(tp + tn, total)

    precision_fake = safe_div(tp, tp + fp)
    recall_fake = safe_div(tp, tp + fn)
    f1_fake = f1_score(precision_fake, recall_fake)

    precision_true = safe_div(tn, tn + fn)
    recall_true = safe_div(tn, tn + fp)
    f1_true = f1_score(precision_true, recall_true)

    macro_f1 = (f1_true + f1_fake) / 2.0
    return BinaryClassificationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        precision_true=precision_true,
        recall_true=recall_true,
        f1_true=f1_true,
        precision_fake=precision_fake,
        recall_fake=recall_fake,
        f1_fake=f1_fake,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        total=total,
    )


def grouped_binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    groups: Sequence[Optional[str]],
) -> Dict[str, Dict[str, float | int]]:
    """Compute metrics per group, e.g. MMFakeBench fake_cls."""
    grouped: Dict[str, List[int]] = {}
    for idx, group in enumerate(groups):
        grouped.setdefault(group or "<missing>", []).append(idx)

    output: Dict[str, Dict[str, float | int]] = {}
    for group, indices in grouped.items():
        metrics = compute_binary_metrics([y_true[i] for i in indices], [y_pred[i] for i in indices])
        output[group] = metrics.to_dict()
    return output
