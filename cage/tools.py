from __future__ import annotations

from abc import ABC, abstractmethod
from threading import RLock
from typing import List, Optional


class BaseTool(ABC):
    """Abstract base class for tools used by the LLM-driven MCTS agent."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name used by the controller in an Action."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description, including when the LLM should call it."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, input: str) -> str:
        """Execute the tool and return a textual observation."""
        raise NotImplementedError

    def schema_text(self) -> str:
        """Compact tool schema injected into prompts."""
        return f"Tool(name={self.name!r}, description={self.description!r})"


class ToolRegistry:
    """Singleton dynamic tool registry.

    The registry decouples MCTS and LLM control from concrete tool backends, so
    simple API wrappers and heavy PyTorch model inference modules can be swapped
    without changing the search algorithm.
    """

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

    def clear(self) -> None:
        with self._lock:
            self._tools.clear()


class WebSearchTool(BaseTool):
    """Mock web-search tool. Replace with real search/retrieval API later."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search external web or document sources for factual evidence. "
            "Call this when the query requires source grounding, background "
            "knowledge, or current information."
        )

    def execute(self, input: str) -> str:
        return f"[MockWebSearch] Retrieved candidate evidence for query: {input}"


class ForgeryDetectionTool(BaseTool):
    """Mock image forgery detector.

    Later this can wrap a PyTorch model such as PSCC-Net, ManTraNet, or a custom
    tampering detector. The input can be an image path, URI, or serialized sample.
    """

    @property
    def name(self) -> str:
        return "forgery_detection"

    @property
    def description(self) -> str:
        return (
            "Detect image manipulation or forgery cues. Call this when a claim "
            "depends on image authenticity, tampering, synthetic media, or visual "
            "forensics."
        )

    def execute(self, input: str) -> str:
        return f"[MockForgeryDetection] No decisive manipulation signal found. Input={input}"


class VQATool(BaseTool):
    """Mock visual question answering tool."""

    @property
    def name(self) -> str:
        return "vqa"

    @property
    def description(self) -> str:
        return (
            "Answer questions about image content. Call this when a textual claim "
            "must be checked against visible objects, scenes, attributes, or "
            "cross-modal consistency."
        )

    def execute(self, input: str) -> str:
        return f"[MockVQA] Visual answer generated for request: {input}"


class TextVerifierTool(BaseTool):
    """Mock textual verifier / NLI tool."""

    @property
    def name(self) -> str:
        return "text_verifier"

    @property
    def description(self) -> str:
        return (
            "Assess whether evidence supports, refutes, or is neutral to a text "
            "claim. Call this after retrieving evidence or when relation judgment "
            "is needed."
        )

    def execute(self, input: str) -> str:
        return f"[MockTextVerifier] Neutral-to-support relation estimated for: {input}"


def register_default_tools(registry: Optional[ToolRegistry] = None, overwrite: bool = True) -> ToolRegistry:
    """Register default mock tools and return the singleton registry."""
    registry = registry or ToolRegistry()
    registry.register(WebSearchTool(), overwrite=overwrite)
    registry.register(ForgeryDetectionTool(), overwrite=overwrite)
    registry.register(VQATool(), overwrite=overwrite)
    registry.register(TextVerifierTool(), overwrite=overwrite)
    return registry
