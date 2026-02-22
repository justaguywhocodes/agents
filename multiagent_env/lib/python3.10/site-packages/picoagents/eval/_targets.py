"""
Evaluation targets for different picoagents components.

This module provides adapters that wrap picoagents components 
(agents, models, orchestrators) as evaluation targets.
"""

import time
from typing import Optional

from .._cancellation_token import CancellationToken
from ..agents import BaseAgent
from ..llm import BaseChatCompletionClient
from ..messages import SystemMessage, UserMessage
from ..orchestration import BaseOrchestrator
from ..types import EvalTask, EvalTrajectory, Usage
from ._base import BaseEvalTarget


class AgentEvalTarget(BaseEvalTarget):
    """Evaluation target for picoagents agents."""

    def __init__(self, agent: BaseAgent, name: Optional[str] = None):
        """Initialize the agent evaluation target.

        Args:
            agent: The picoagents agent to evaluate
            name: Optional custom name (defaults to agent name)
        """
        super().__init__(name or getattr(agent, "name", "Agent"))
        self.agent = agent

    async def run(
        self, task: EvalTask, cancellation_token: Optional[CancellationToken] = None
    ) -> EvalTrajectory:
        """Execute the task using the agent.

        Args:
            task: The evaluation task to execute
            cancellation_token: Optional token to cancel execution

        Returns:
            EvalTrajectory containing the agent's execution sequence
        """
        start_time = time.time()

        try:
            # Execute the agent with the task input
            response = await self.agent.run(
                task.input, cancellation_token=cancellation_token
            )

            end_time = time.time()

            return EvalTrajectory(
                task=task,
                messages=response.messages,
                success=True,
                error=None,
                usage=response.usage,
                metadata={
                    "target_type": "agent",
                    "target_name": self.name,
                    "execution_time_ms": int((end_time - start_time) * 1000),
                },
            )

        except Exception as e:
            end_time = time.time()

            return EvalTrajectory(
                task=task,
                messages=[],
                success=False,
                error=str(e),
                usage=Usage(
                    duration_ms=int((end_time - start_time) * 1000),
                    llm_calls=0,
                    tokens_input=0,
                    tokens_output=0,
                ),
                metadata={
                    "target_type": "agent",
                    "target_name": self.name,
                    "execution_time_ms": int((end_time - start_time) * 1000),
                },
            )


class ModelEvalTarget(BaseEvalTarget):
    """Evaluation target for direct LLM model calls."""

    def __init__(
        self,
        client: BaseChatCompletionClient,
        system_message: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """Initialize the model evaluation target.

        Args:
            client: The LLM client to evaluate
            system_message: Optional system message to prepend
            name: Optional custom name (defaults to model name)
        """
        super().__init__(name or getattr(client, "model", "Model"))
        self.client = client
        self.system_message = system_message

    async def run(
        self, task: EvalTask, cancellation_token: Optional[CancellationToken] = None
    ) -> EvalTrajectory:
        """Execute the task using direct model calls.

        Args:
            task: The evaluation task to execute
            cancellation_token: Optional token to cancel execution

        Returns:
            EvalTrajectory containing the model's response
        """
        start_time = time.time()

        try:
            # Prepare messages
            messages = []
            if self.system_message:
                messages.append(
                    SystemMessage(content=self.system_message, source="system")
                )
            messages.append(UserMessage(content=task.input, source="user"))

            # Execute the model (note: cancellation_token not passed to client as it's not part of the base interface)
            result = await self.client.create(messages)

            end_time = time.time()

            # Build the complete message sequence
            response_messages = messages + [result.message]

            return EvalTrajectory(
                task=task,
                messages=response_messages,
                success=True,
                error=None,
                usage=result.usage,
                metadata={
                    "target_type": "model",
                    "target_name": self.name,
                    "model": result.model,
                    "finish_reason": result.finish_reason,
                    "execution_time_ms": int((end_time - start_time) * 1000),
                },
            )

        except Exception as e:
            end_time = time.time()

            return EvalTrajectory(
                task=task,
                messages=[],
                success=False,
                error=str(e),
                usage=Usage(
                    duration_ms=int((end_time - start_time) * 1000),
                    llm_calls=0,
                    tokens_input=0,
                    tokens_output=0,
                ),
                metadata={
                    "target_type": "model",
                    "target_name": self.name,
                    "execution_time_ms": int((end_time - start_time) * 1000),
                },
            )


class OrchestratorEvalTarget(BaseEvalTarget):
    """Evaluation target for picoagents orchestrators."""

    def __init__(self, orchestrator: BaseOrchestrator, name: Optional[str] = None):
        """Initialize the orchestrator evaluation target.

        Args:
            orchestrator: The orchestrator to evaluate (should already be initialized with agents)
            name: Optional custom name
        """
        super().__init__(name or f"{orchestrator.__class__.__name__}")
        self.orchestrator = orchestrator

    async def run(
        self, task: EvalTask, cancellation_token: Optional[CancellationToken] = None
    ) -> EvalTrajectory:
        """Execute the task using the orchestrator.

        Args:
            task: The evaluation task to execute
            cancellation_token: Optional token to cancel execution

        Returns:
            EvalTrajectory containing the orchestrator's execution sequence
        """
        start_time = time.time()

        try:
            # Execute the orchestrator with the task input
            response = await self.orchestrator.run(
                task.input, cancellation_token=cancellation_token
            )

            end_time = time.time()

            return EvalTrajectory(
                task=task,
                messages=response.messages,
                success=True,
                error=None,
                usage=response.usage,
                metadata={
                    "target_type": "orchestrator",
                    "target_name": self.name,
                    "pattern": response.pattern_metadata.get("pattern", "unknown"),
                    "iterations": response.pattern_metadata.get(
                        "iterations_completed", 0
                    ),
                    "stop_reason": response.stop_message.source,
                    "execution_time_ms": int((end_time - start_time) * 1000),
                },
            )

        except Exception as e:
            end_time = time.time()

            return EvalTrajectory(
                task=task,
                messages=[],
                success=False,
                error=str(e),
                usage=Usage(
                    duration_ms=int((end_time - start_time) * 1000),
                    llm_calls=0,
                    tokens_input=0,
                    tokens_output=0,
                ),
                metadata={
                    "target_type": "orchestrator",
                    "target_name": self.name,
                    "execution_time_ms": int((end_time - start_time) * 1000),
                },
            )
