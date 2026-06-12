# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment and setup

This is a Python research/skeleton repository. Dependencies are declared in `requirements.txt`; there is no `pyproject.toml` or test/lint configuration in the current tree.

```bash
# Create/use a local virtualenv with uv
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Equivalent pip install if not using uv
pip install -r requirements.txt
```

Main dependencies include `networkx`, `torch`, `pillow`, `pandas`, `pyarrow`, `aiohttp`, `google-api-python-client`, `wikipedia`, and `requests`.

## Common commands

```bash
# Syntax/import smoke check for package and scripts
python -m compileall cage scripts

# Minimal CAGE/T2 examples from README
python examples_cage_run.py
python examples_agent_mcts_run.py

# Greedy tool-selection placeholder
python scripts/tool_selector.py
python scripts/tool_selector.py --min-delta 0.01

# Run a small MMFakeBench experiment; requires local dataset files
python scripts/run_mmfakebench.py --root /data/lhy_data/MMFakeBench --split val --max-samples 10

# Use the agent prediction strategy instead of the fake_cls sanity baseline
python scripts/run_mmfakebench.py --prediction-strategy agent --max-samples 10
```

There are currently no committed tests or pytest configuration. If tests are added, prefer commands that target a single test directly, e.g. `pytest path/to/test_file.py::test_name`.

Note: at the time this file was created, `python -m compileall cage scripts` succeeds, but the two root-level example scripts are stale relative to the current package exports/imports and may need updates before they run.

## High-level architecture

CAGE stands for Claim-driven Active Graph Exploration. The code models multimodal misinformation reasoning as evidence gathering over claims rather than as only a linear Thought/Action/Observation trace.

- `cage/graph.py` defines the graph state: `ClaimNode`, `EvidenceNode`, `RelationType`, `ClaimModality`, and `EvidenceGraph`. `EvidenceGraph` wraps a `networkx.DiGraph`, stores the original dataclass nodes under node attributes, and tracks typed evidence-to-claim relations such as support, refute, conflict, and neutral.
- `cage/mcts.py` contains a generic MCTS searcher over tool-use trajectories. `MCTSSearcher` depends only on a controller protocol (`propose_next`, `execute_action`) and an evaluator, so controller/model integration is intentionally decoupled from tree search.
- `cage/agent.py` provides the top-level T2-style multimodal agent. `LLMController` builds provider-neutral prompts/payloads and has a mock `complete()` hook; real Claude/OpenAI/vLLM/Qwen/LLaVA integrations should override that method rather than changing MCTS. `T2Agent` wires default tools, controller, evaluator, and MCTS, then turns the best trajectory into an `AgentDecision`.
- `cage/tools.py` defines the tool abstraction and registry. `BaseTool` is string-in/string-out with optional async execution; `ToolRegistry` is a singleton registry. Default tools include web search, forgery detection, VQA/counterfactual OpenAI-compatible vision calls, Baidu entity recognition, and a placeholder text verifier. Most external services are optional and return descriptive errors when required environment variables are missing.
- `cage/evaluation.py` contains the dual evaluator. It can use an injected LLM scoring function, but defaults to deterministic heuristics so the package can run without model credentials. The final value combines trajectory correctness and evidence confidence using `alpha`.
- `cage/gcn.py` is a PyTorch GCN scorer stub and tensorizer for `EvidenceGraph`; it is not required by the default agent loop.
- `cage/data/` normalizes local dataset access into `CAGESample`. `MMFakeBenchDataset` reads JSON metadata and lazily loads images from split zip files. `HintsOfTruthDataset` reads local parquet shards and decodes Hugging Face-style image dictionaries. Use `cage_collate_fn` with PyTorch `DataLoader` so PIL images remain lists unless transforms produce stackable tensors.
- `cage/experiments/` provides a minimal MMFakeBench JSONL runner and binary metrics. The default `fake_cls_baseline` prediction strategy is a dataset/metrics sanity baseline; `agent` uses the current heuristic/mock `T2Agent`.

## Dataset conventions

Do not merge label spaces across the included dataset wrappers:

- MMFakeBench: misinformation truthfulness, `0=true`, `1=fake`; valid splits are `val` and `test`.
- Hints-of-Truth: check-worthiness, `0=not_check_worthy`, `1=check_worthy`; valid splits are `dev1`, `dev2`, and `test`.

Default local roots used by the code/README are `/data/lhy_data/MMFakeBench` and `/data/lhy_data/hints_of_truth`. Large datasets, media, model weights, and experiment outputs are intentionally ignored by `.gitignore`.

## External service configuration

The tool layer reads these optional environment variables:

- Web search: `GOOGLE_API_KEY`, `GOOGLE_CSE_ID`; without both, `WebSearchTool` falls back to Wikipedia search and records a Google configuration error.
- OpenAI-compatible vision tools: `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, optional `OPENAI_VISION_MODEL`.
- Baidu entity recognition: `BAIDU_ERNIE_API_URL` or `BAIDU_ENTITY_API_URL`, optional `BAIDU_ACCESS_TOKEN`.
- Forgery model: optional `FORGERY_MODEL_CKPT`; otherwise `ForgeryDetectionTool` uses an identity/model placeholder and deterministic mock score.

## Development notes

- The package API is exported from `cage/__init__.py`; update exports when adding public classes used by examples or downstream scripts.
- Keep provider-specific LLM/VLM calls isolated in controller/tool subclasses. The existing search/evaluation layers are designed around provider-neutral interfaces.
- `ToolRegistry` is a singleton. Tests or scripts that register tools should call `clear()` or use `overwrite=True` to avoid cross-run contamination within the same Python process.
- MMFakeBench image paths are zip members by default. `scripts/run_mmfakebench.py --pass-image-to-agent` only passes `sample.image_path`; for zipped images this is usually not a real filesystem path unless the dataset has been extracted or adapted.
