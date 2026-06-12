from __future__ import annotations

import asyncio
import base64
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

import aiohttp

try:
    import torch
except Exception:  # pragma: no cover - torch is optional at import time
    torch = None  # type: ignore[assignment]


class BaseTool(ABC):
    """Abstract base class for all CAGE tools.

    Tools expose a minimal string-in/string-out contract to keep the MCTS and
    agent layers independent from HTTP clients, PyTorch versions, and vendor SDKs.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def execute(self, input: str) -> str:
        """Synchronous execution entry point used by MCTS."""
        raise NotImplementedError

    async def aexecute(self, input: str) -> str:
        """Async execution entry point. Override for real network I/O."""
        return await asyncio.to_thread(self.execute, input)

    def schema_text(self) -> str:
        return f"Tool(name={self.name!r}, description={self.description!r})"


class ToolRegistry:
    """Thread-safe singleton tool registry."""

    _instance: Optional["ToolRegistry"] = None
    _lock = RLock()

    def __new__(cls) -> "ToolRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._tools = {}
            return cls._instance

    def register(self, tool: BaseTool, overwrite: bool = False) -> None:
        with self._lock:
            if tool.name in self._tools and not overwrite:
                raise ValueError(f"Tool already registered: {tool.name}")
            self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        with self._lock:
            self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool:
        with self._lock:
            if name not in self._tools:
                raise KeyError(f"Tool not found: {name}. Available tools: {sorted(self._tools)}")
            return self._tools[name]

    def list_tools(self) -> List[BaseTool]:
        with self._lock:
            return list(self._tools.values())

    def list_tool_names(self) -> List[str]:
        with self._lock:
            return sorted(self._tools)

    def describe_tools(self) -> str:
        with self._lock:
            if not self._tools:
                return "<no tools registered>"
            return "\n".join(tool.schema_text() for tool in self._tools.values())

    def execute(self, name: str, input: str) -> str:
        return self.get(name).execute(input)

    async def aexecute(self, name: str, input: str) -> str:
        return await self.get(name).aexecute(input)

    def clear(self) -> None:
        with self._lock:
            self._tools.clear()


class RetryMixin:
    """Small async retry helper for network-bound tools."""

    def __init__(self, max_retries: int = 3, retry_backoff: float = 1.0) -> None:
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    async def _request_json(
        self,
        session: aiohttp.ClientSession,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                async with session.request(method, url, **kwargs) as response:
                    text = await response.text()
                    if response.status >= 400:
                        raise RuntimeError(f"HTTP {response.status}: {text[:500]}")
                    return json.loads(text) if text else {}
            except Exception as exc:  # noqa: BLE001 - tool should return robust errors
                last_error = exc
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(self.retry_backoff * (2**attempt))
        raise RuntimeError(f"Request failed after {self.max_retries} attempts: {last_error}")


class WebSearchTool(BaseTool, RetryMixin):
    """Google Custom Search + Wikipedia search tool.

    Environment variables:
        GOOGLE_API_KEY: Google Custom Search API key.
        GOOGLE_CSE_ID: Programmable Search Engine ID.

    If keys are absent, the tool falls back to Wikipedia-only retrieval; if
    optional dependencies are absent, it returns a descriptive error string.
    """

    def __init__(
        self,
        google_api_key: Optional[str] = None,
        google_cse_id: Optional[str] = None,
        max_results: int = 5,
        max_retries: int = 3,
    ) -> None:
        RetryMixin.__init__(self, max_retries=max_retries)
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = google_cse_id or os.getenv("GOOGLE_CSE_ID")
        self.max_results = max_results

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search Google Custom Search and Wikipedia for factual evidence. "
            "Call this for source grounding, entity/event verification, or missing background knowledge."
        )

    def execute(self, input: str) -> str:
        return asyncio.run(self.aexecute(input))

    async def aexecute(self, input: str) -> str:
        results: Dict[str, Any] = {"query": input, "google": [], "wikipedia": []}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            if self.google_api_key and self.google_cse_id:
                try:
                    params = {
                        "key": self.google_api_key,
                        "cx": self.google_cse_id,
                        "q": input,
                        "num": min(self.max_results, 10),
                    }
                    data = await self._request_json(
                        session,
                        "GET",
                        "https://www.googleapis.com/customsearch/v1",
                        params=params,
                    )
                    results["google"] = [
                        {
                            "title": item.get("title"),
                            "link": item.get("link"),
                            "snippet": item.get("snippet"),
                        }
                        for item in data.get("items", [])[: self.max_results]
                    ]
                except Exception as exc:  # noqa: BLE001
                    results["google_error"] = str(exc)
            else:
                results["google_error"] = "GOOGLE_API_KEY or GOOGLE_CSE_ID not configured"

            try:
                wiki = await self._search_wikipedia(session, input)
                results["wikipedia"] = wiki
            except Exception as exc:  # noqa: BLE001
                results["wikipedia_error"] = str(exc)

        return json.dumps(results, ensure_ascii=False, indent=2)

    async def _search_wikipedia(self, session: aiohttp.ClientSession, query: str) -> List[Dict[str, str]]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": 1,
            "srlimit": self.max_results,
        }
        data = await self._request_json(session, "GET", "https://en.wikipedia.org/w/api.php", params=params)
        output: List[Dict[str, str]] = []
        for item in data.get("query", {}).get("search", []):
            title = str(item.get("title", ""))
            page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            snippet = str(item.get("snippet", ""))
            output.append({"title": title, "url": page_url, "snippet": snippet})
        return output


class ForgeryDetectionTool(BaseTool):
    """PyTorch local-model skeleton for image forgery detection.

    Args:
        checkpoint_path: Path to a local model checkpoint, e.g. PSCC-Net weights.
        device: "cuda", "cpu", or None for auto selection.

    The default `_build_model` is an identity placeholder. Replace it with your
    actual architecture construction and checkpoint loading logic.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None) -> None:
        self.checkpoint_path = checkpoint_path or os.getenv("FORGERY_MODEL_CKPT")
        if device is not None:
            self.device = device
        elif torch is not None and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self._model: Optional[Any] = None

    @property
    def name(self) -> str:
        return "forgery_detection"

    @property
    def description(self) -> str:
        return (
            "Run local PyTorch image-forgery detection. Call this when a claim depends on image authenticity, "
            "tampering, synthetic media, or visual forensics. Input should be an image path or JSON containing image_path."
        )

    def execute(self, input: str) -> str:
        if torch is None:
            return "[ForgeryDetectionError] PyTorch is not installed."
        try:
            image_path = self._parse_image_path(input)
            model = self._load_model()
            # Placeholder inference. Replace preprocessing and forward pass with
            # PSCC-Net-specific transforms and output parsing.
            with torch.no_grad():
                score = self._mock_forward(model, image_path)
            return json.dumps(
                {
                    "tool": self.name,
                    "image_path": image_path,
                    "manipulation_score": score,
                    "verdict": "suspicious" if score >= 0.5 else "no_decisive_manipulation",
                },
                ensure_ascii=False,
            )
        except Exception as exc:  # noqa: BLE001
            return f"[ForgeryDetectionError] {exc}"
        finally:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        model = self._build_model()
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            if hasattr(model, "load_state_dict"):
                model.load_state_dict(state_dict, strict=False)
        if hasattr(model, "to"):
            model = model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()
        self._model = model
        return model

    def _build_model(self) -> Any:
        """Build the local forgery model. Replace with PSCC-Net architecture."""
        return torch.nn.Identity()

    def _mock_forward(self, model: Any, image_path: str) -> float:
        _ = model
        if not image_path:
            return 0.0
        # Deterministic pseudo-score for smoke tests.
        return (sum(bytearray(image_path.encode("utf-8"))) % 100) / 100.0

    def _parse_image_path(self, input: str) -> str:
        try:
            data = json.loads(input)
            if isinstance(data, dict):
                return str(data.get("image_path") or data.get("path") or data.get("image") or "")
        except json.JSONDecodeError:
            pass
        return input.strip()


class OpenAIVisionTool(BaseTool, RetryMixin):
    """Base class for GPT-4o-style vision tools.

    Environment variables:
        OPENAI_API_KEY: API key for OpenAI-compatible chat completions.
        OPENAI_BASE_URL: Optional endpoint, defaults to OpenAI public API.
        OPENAI_VISION_MODEL: Optional model name, defaults to gpt-4o.
    """

    tool_name = "openai_vision"
    tool_description = "Call an OpenAI-compatible vision model with an image and question."

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
    ) -> None:
        RetryMixin.__init__(self, max_retries=max_retries)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.getenv("OPENAI_VISION_MODEL") or "gpt-4o"

    @property
    def name(self) -> str:
        return self.tool_name

    @property
    def description(self) -> str:
        return self.tool_description

    def execute(self, input: str) -> str:
        return asyncio.run(self.aexecute(input))

    async def aexecute(self, input: str) -> str:
        if not self.api_key:
            return f"[{self.name}Error] OPENAI_API_KEY is not configured."
        try:
            payload = self._parse_payload(input)
            image_b64 = payload.get("image_base64") or self._image_path_to_base64(str(payload.get("image_path", "")))
            question = str(payload.get("question") or payload.get("query") or input)
            request_body = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 512,
            }
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                data = await self._request_json(
                    session,
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=request_body,
                )
            return data.get("choices", [{}])[0].get("message", {}).get("content", json.dumps(data, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            return f"[{self.name}Error] {exc}"

    def _parse_payload(self, input: str) -> Dict[str, Any]:
        try:
            data = json.loads(input)
            return data if isinstance(data, dict) else {"question": input}
        except json.JSONDecodeError:
            return {"question": input}

    def _image_path_to_base64(self, image_path: str) -> str:
        if not image_path:
            raise ValueError("image_path or image_base64 is required")
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


class CounterfactualTool(OpenAIVisionTool):
    """Generate counterfactual visual/text consistency analysis using GPT-4o."""

    tool_name = "counterfactual"
    tool_description = (
        "Use a vision-language model to reason about counterfactual image/text consistency. "
        "Input JSON should contain image_path or image_base64 plus question/query."
    )


class VQATool(OpenAIVisionTool):
    """Visual question answering via GPT-4o-compatible API."""

    tool_name = "vqa"
    tool_description = (
        "Answer visual questions using a vision-language model. Input JSON should contain "
        "image_path or image_base64 plus question/query."
    )


class EntityRecognitionTool(BaseTool, RetryMixin):
    """Baidu entity recognition API wrapper.

    Environment variables:
        BAIDU_ERNIE_API_URL or BAIDU_ENTITY_API_URL: endpoint URL.
        BAIDU_ACCESS_TOKEN: access token, if the endpoint expects it.

    The exact Baidu entity API shape differs by product/version; this wrapper
    keeps request construction isolated so it can be adapted without touching MCTS.
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        access_token: Optional[str] = None,
        max_retries: int = 3,
    ) -> None:
        RetryMixin.__init__(self, max_retries=max_retries)
        self.api_url = api_url or os.getenv("BAIDU_ERNIE_API_URL") or os.getenv("BAIDU_ENTITY_API_URL")
        self.access_token = access_token or os.getenv("BAIDU_ACCESS_TOKEN")

    @property
    def name(self) -> str:
        return "entity_recognition"

    @property
    def description(self) -> str:
        return (
            "Extract entities from text using Baidu entity recognition. Call this for people, places, "
            "organizations, events, and other factual anchors that need verification."
        )

    def execute(self, input: str) -> str:
        return asyncio.run(self.aexecute(input))

    async def aexecute(self, input: str) -> str:
        if not self.api_url:
            return "[EntityRecognitionError] BAIDU_ENTITY_API_URL or BAIDU_ERNIE_API_URL is not configured."
        params = {"access_token": self.access_token} if self.access_token else None
        body = {"text": input}
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                data = await self._request_json(session, "POST", self.api_url, params=params, json=body)
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            return f"[EntityRecognitionError] {exc}"


class TextVerifierTool(BaseTool):
    """Lightweight placeholder textual verifier/NLI tool."""

    @property
    def name(self) -> str:
        return "text_verifier"

    @property
    def description(self) -> str:
        return "Estimate whether evidence supports, refutes, or is neutral to a text claim."

    def execute(self, input: str) -> str:
        return f"[MockTextVerifier] Relation estimated for: {input}"


def register_default_tools(registry: Optional[ToolRegistry] = None, overwrite: bool = True) -> ToolRegistry:
    """Register default tools for server-side development."""
    registry = registry or ToolRegistry()
    registry.register(WebSearchTool(), overwrite=overwrite)
    registry.register(ForgeryDetectionTool(), overwrite=overwrite)
    registry.register(CounterfactualTool(), overwrite=overwrite)
    registry.register(VQATool(), overwrite=overwrite)
    registry.register(EntityRecognitionTool(), overwrite=overwrite)
    registry.register(TextVerifierTool(), overwrite=overwrite)
    return registry
