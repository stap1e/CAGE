# Problem Notes

## 1. `claude` command not found after reconnecting VSCode over SSH

### Symptom

After the machine slept and the VSCode SSH session disconnected, reconnecting to the remote host caused:

```bash
claude
# claude: command not found
```

### Root Cause

The `claude` CLI was not actually deleted. It had been installed under an older `nvm` Node.js version:

- Installed and available in `node v18.20.8`
- Missing in the current default `node v24.16.0`

Because `nvm` keeps global npm packages isolated per Node version, when the new SSH shell started with `v24.16.0`, the old `claude` binary from `v18.20.8` was no longer on `PATH`.

### Evidence

Current default Node version:

```bash
nvm current
# v24.16.0
```

Claude available in Node 18:

```bash
nvm use 18.20.8
command -v claude
# /home/hongyu/.nvm/versions/node/v18.20.8/bin/claude
```

Claude missing in Node 24:

```bash
nvm use 24.16.0
command -v claude
# no output
```

### Fix

Temporary fix for the current shell:

```bash
nvm use 18.20.8
claude
```

If `v18.20.8` should remain the default:

```bash
nvm alias default 18.20.8
```

If `v24.16.0` should remain the default, reinstall Claude there:

```bash
nvm use 24.16.0
npm install -g @anthropic-ai/claude-code
```

### Summary

This was caused by a Node version switch after reconnecting, not by SSH itself deleting Claude.

## 2. How to install this repository environment with `uv`

This repository currently defines dependencies in `requirements.txt` rather than `pyproject.toml`, so the simplest `uv` workflow is:

### Create and activate a virtual environment

```bash
cd /home/hongyu/CAGE
uv venv .venv
source .venv/bin/activate
```

### Install dependencies from `requirements.txt`

```bash
uv pip install -r requirements.txt
```

### Verify installation

```bash
python --version
uv pip list
```

### One-line setup after entering the repo

```bash
cd /home/hongyu/CAGE && uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt
```

### Notes

- `uv` is already installed in this environment.
- The current system Python is `3.12.7`.
- Main dependencies in this repo include `torch`, `pandas`, `pyarrow`, `aiohttp`, and `networkx`.
- If GPU-specific `torch` wheels are needed later, installation may need an extra index URL depending on your CUDA setup.
