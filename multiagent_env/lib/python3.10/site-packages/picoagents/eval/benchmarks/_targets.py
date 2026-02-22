"""
Benchmark targets - what we're benchmarking.

This module defines BenchmarkTarget (abstract base) and concrete implementations
for running benchmark tasks against different agent systems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..._cancellation_token import CancellationToken
from ...types import EvalTrajectory, Usage
from ._config import AgentConfig
from ._dataset import BenchmarkTask


@dataclass
class TargetResult:
    """Standardized result from any benchmark target."""

    # Output
    output: str
    success: bool
    error: Optional[str] = None

    # Token metrics
    input_tokens: int = 0
    output_tokens: int = 0

    # Execution metrics
    iterations: int = 0
    duration_ms: int = 0

    # Full message history from the agent run
    messages: List[Any] = None  # List of Message objects

    # All events from the run (tool calls, model calls, etc.)
    events: List[Any] = None  # List of AgentEvent objects

    # Additional data from middleware or target-specific tracking
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.messages is None:
            self.messages = []
        if self.events is None:
            self.events = []

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BenchmarkTarget(ABC):
    """Abstract base for anything that can run benchmark tasks.

    Implementations wrap different agent systems (PicoAgents, Claude Code,
    LangGraph, etc.) and provide a uniform interface for benchmarking.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def run(
        self,
        task: BenchmarkTask,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> TargetResult:
        """Execute a benchmark task.

        Args:
            task: The benchmark task to run
            cancellation_token: Optional cancellation support

        Returns:
            TargetResult with output and metrics
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class PicoAgentTarget(BenchmarkTarget):
    """Run benchmark tasks with PicoAgents.

    This target creates an agent from an AgentConfig and runs tasks,
    capturing metrics via optional middleware.
    """

    def __init__(
        self,
        config: AgentConfig,
        middlewares: Optional[List] = None,
    ):
        """Initialize PicoAgent target.

        Args:
            config: Agent configuration
            middlewares: Optional middleware list (BenchmarkMiddleware added by runner)
        """
        super().__init__(config.name)
        self.config = config
        self.middlewares = middlewares or []
        self._agent = None

    def _get_agent(self, extra_middlewares: Optional[List] = None):
        """Create or return cached agent."""
        all_middlewares = self.middlewares + (extra_middlewares or [])
        return self.config.to_agent(middlewares=all_middlewares)

    async def run(
        self,
        task: BenchmarkTask,
        cancellation_token: Optional[CancellationToken] = None,
        middlewares: Optional[List] = None,
    ) -> TargetResult:
        """Execute task with PicoAgents using run_stream to capture full trace.

        Args:
            task: Benchmark task to run
            cancellation_token: Optional cancellation
            middlewares: Additional middleware (e.g., BenchmarkMiddleware)

        Returns:
            TargetResult with execution data including full message/event history
        """
        from ...messages import AssistantMessage, Message, ToolMessage, UserMessage
        from ...types import AgentEvent, AgentResponse

        agent = self._get_agent(middlewares)

        try:
            # Use run_stream to capture EVERYTHING
            all_messages = []
            all_events = []
            response = None
            output = ""

            async for item in agent.run_stream(
                task.prompt,
                cancellation_token=cancellation_token,
                verbose=True,  # Get all events
            ):
                # Capture the final AgentResponse
                if isinstance(item, AgentResponse):
                    response = item
                # Capture messages
                elif isinstance(item, Message):
                    all_messages.append(item)
                    # Track the latest assistant message content as output
                    if isinstance(item, AssistantMessage) and item.content:
                        output = item.content
                # Capture events
                elif isinstance(item, AgentEvent):
                    all_events.append(item)

            if response is None:
                # No AgentResponse yielded - create fallback
                return TargetResult(
                    output=output,
                    success=False,
                    error="No response from agent",
                    messages=all_messages,
                    events=all_events,
                    metadata={"exception_type": "NoResponse"},
                )

            # Get messages from context if available (more complete)
            context_messages = list(response.context.messages) if response.context else []

            return TargetResult(
                output=output,
                success=response.finish_reason == "stop",
                error=None if response.finish_reason == "stop" else response.finish_reason,
                input_tokens=response.usage.tokens_input,
                output_tokens=response.usage.tokens_output,
                iterations=response.usage.llm_calls,
                duration_ms=response.usage.duration_ms,
                messages=context_messages if context_messages else all_messages,
                events=all_events,
                metadata={
                    "finish_reason": response.finish_reason,
                    "tool_calls": response.usage.tool_calls,
                },
            )

        except Exception as e:
            return TargetResult(
                output="",
                success=False,
                error=str(e),
                messages=[],
                events=[],
                metadata={"exception_type": type(e).__name__},
            )


class ClaudeCodeTarget(BenchmarkTarget):
    """Run benchmark tasks with Claude Code SDK.

    Requires `claude-code-sdk` package to be installed.
    """

    def __init__(
        self,
        name: str = "claude_code",
        max_turns: int = 30,
        allowed_tools: Optional[List[str]] = None,
    ):
        """Initialize Claude Code target.

        Args:
            name: Target name for results
            max_turns: Maximum conversation turns
            allowed_tools: Tools to enable (default: Read, Bash, Glob, Grep)
        """
        super().__init__(name)
        self.max_turns = max_turns
        self.allowed_tools = allowed_tools or ["Read", "Bash", "Glob", "Grep"]

    async def run(
        self,
        task: BenchmarkTask,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> TargetResult:
        """Execute task with Claude Code SDK.

        Args:
            task: Benchmark task to run
            cancellation_token: Optional cancellation (limited support)

        Returns:
            TargetResult with execution data
        """
        try:
            from claude_code_sdk import (
                AssistantMessage,
                ClaudeCodeOptions,
                ResultMessage,
                TextBlock,
                ToolResultBlock,
                ToolUseBlock,
                UserMessage,
                query,
            )
        except ImportError:
            return TargetResult(
                output="",
                success=False,
                error="claude-code-sdk not installed. Install with: pip install claude-code-sdk",
            )

        options = ClaudeCodeOptions(
            allowed_tools=self.allowed_tools,
            max_turns=self.max_turns,
        )

        response_text = ""
        iterations = 0
        input_tokens = 0
        output_tokens = 0
        duration_ms = 0
        success = False
        error = None

        try:
            async for message in query(prompt=task.prompt, options=options):
                if isinstance(message, AssistantMessage):
                    iterations += 1
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text

                elif isinstance(message, ResultMessage):
                    success = not message.is_error
                    duration_ms = message.duration_ms
                    if message.usage:
                        input_tokens = message.usage.get("input_tokens", 0)
                        output_tokens = message.usage.get("output_tokens", 0)
                    if message.is_error:
                        error = "Claude Code returned error"

        except Exception as e:
            error = str(e)

        return TargetResult(
            output=response_text,
            success=success,
            error=error,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            iterations=iterations,
            duration_ms=duration_ms,
        )


class CallableTarget(BenchmarkTarget):
    """Wrap any async callable as a benchmark target.

    Useful for custom agent implementations or quick testing.
    """

    def __init__(self, name: str, func):
        """Initialize callable target.

        Args:
            name: Target name
            func: Async callable(task: BenchmarkTask) -> TargetResult
        """
        super().__init__(name)
        self.func = func

    async def run(
        self,
        task: BenchmarkTask,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> TargetResult:
        """Execute the wrapped callable."""
        return await self.func(task)
