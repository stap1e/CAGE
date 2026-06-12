# scripts/tool_selector.py
from __future__ import annotations

import argparse
from typing import Iterable, List, Sequence, Set


D_BASE: List[str] = ["web_search", "text_verifier"]
CANDIDATES: List[str] = [
    "forgery_detection",
    "vqa",
    "counterfactual",
    "entity_recognition",
]


def evaluate_accuracy(toolset: Sequence[str]) -> float:
    """Simulated validation accuracy for a toolset.

    Replace this helper with real validation over MMFakeBench / Hints-of-Truth.
    The function should run the agent with the provided tool subset and return a
    scalar accuracy. This deterministic placeholder makes the selector runnable.
    """
    weights = {
        "web_search": 0.52,
        "text_verifier": 0.06,
        "forgery_detection": 0.07,
        "vqa": 0.05,
        "counterfactual": 0.04,
        "entity_recognition": 0.03,
    }
    selected: Set[str] = set(toolset)
    accuracy = 0.0
    for tool in selected:
        accuracy += weights.get(tool, 0.0)

    # Small synergy bonuses reflecting multimodal complementarity.
    if {"vqa", "forgery_detection"}.issubset(selected):
        accuracy += 0.02
    if {"web_search", "entity_recognition"}.issubset(selected):
        accuracy += 0.015
    if {"counterfactual", "vqa"}.issubset(selected):
        accuracy += 0.015
    return min(accuracy, 0.99)


def greedy_select_tools(
    base_tools: Sequence[str],
    candidates: Sequence[str],
    min_delta: float = 0.0,
) -> List[str]:
    """Greedily add tools when validation accuracy improves.

    Args:
        base_tools: Initial D_base tool set.
        candidates: Candidate tools to evaluate.
        min_delta: Required improvement threshold. The paper-style condition is
            Delta Acc > 0; set min_delta > 0 for stricter selection.
    """
    selected = list(dict.fromkeys(base_tools))
    current_acc = evaluate_accuracy(selected)
    print(f"Initial D_base={selected}, Acc={current_acc:.4f}")

    for tool in candidates:
        if tool in selected:
            continue
        trial = selected + [tool]
        trial_acc = evaluate_accuracy(trial)
        delta = trial_acc - current_acc
        print(f"Try add {tool:>18s}: Acc={trial_acc:.4f}, Delta={delta:+.4f}")
        if delta > min_delta:
            selected.append(tool)
            current_acc = trial_acc
            print(f"  -> accepted. New D_base={selected}")
        else:
            print("  -> rejected.")

    print(f"Final toolset={selected}, Acc={current_acc:.4f}")
    return selected


def parse_csv_tools(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy tool selector for CAGE/T2 Agent.")
    parser.add_argument("--base", type=parse_csv_tools, default=D_BASE, help="Comma-separated base tools.")
    parser.add_argument(
        "--candidates",
        type=parse_csv_tools,
        default=CANDIDATES,
        help="Comma-separated candidate tools.",
    )
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum accuracy gain required.")
    args = parser.parse_args()
    greedy_select_tools(args.base, args.candidates, args.min_delta)


if __name__ == "__main__":
    main()
