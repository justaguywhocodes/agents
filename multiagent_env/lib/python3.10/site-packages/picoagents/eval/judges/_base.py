"""
Base judge class for evaluation scoring.

This module defines the abstract base class for all evaluation judges.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..._cancellation_token import CancellationToken
from ...messages import AssistantMessage
from ...types import EvalScore, EvalTrajectory


class BaseEvalJudge(ABC):
    """Abstract base class for evaluation judges."""

    def __init__(self, name: str, answer_strategy: str = "last_non_empty"):
        """Initialize the judge.

        Args:
            name: Human-readable name for this judge
            answer_strategy: Strategy for extracting answers from trajectories
                - "last_non_empty": Last message with non-empty content (default)
                - "last_assistant": Last AssistantMessage (skips tool results)
                - "last_content": Last message's content, even if empty
                - "all_assistant": Concatenate all AssistantMessages
        """
        self.name = name
        self.answer_strategy = answer_strategy

    def extract_answer(self, trajectory: EvalTrajectory) -> str:
        """Extract the agent's answer from trajectory.

        Uses the configured answer_strategy. Override this method for
        custom extraction logic beyond the built-in strategies.

        Built-in strategies:
        - last_non_empty: Works for most single-turn cases (default)
        - last_assistant: Good for tool-using agents (skips tool results)
        - all_assistant: Good for multi-turn explanations
        - last_content: Just use last message, even if empty

        Limitations:
        - Cannot distinguish "answer" from "thinking out loud"
        - Multi-agent: cannot filter by specific agent source
        - For complex scenarios, override this method

        Args:
            trajectory: The execution trajectory

        Returns:
            Extracted answer string (empty string if no answer found)
        """
        if not trajectory.messages:
            return ""

        if self.answer_strategy == "last_non_empty":
            for msg in reversed(trajectory.messages):
                content = getattr(msg, "content", "")
                if content and content.strip():
                    return content.strip()
            return ""

        elif self.answer_strategy == "last_assistant":
            for msg in reversed(trajectory.messages):
                if isinstance(msg, AssistantMessage):
                    return getattr(msg, "content", "").strip()
            return ""

        elif self.answer_strategy == "last_content":
            last_msg = trajectory.messages[-1]
            return getattr(last_msg, "content", "").strip()

        elif self.answer_strategy == "all_assistant":
            parts = []
            for msg in trajectory.messages:
                if isinstance(msg, AssistantMessage):
                    content = getattr(msg, "content", "").strip()
                    if content:
                        parts.append(content)
            return "\n".join(parts)

        else:
            raise ValueError(f"Unknown answer_strategy: {self.answer_strategy}")

    @abstractmethod
    async def score(
        self,
        trajectory: EvalTrajectory,
        criteria: Optional[List[str]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> EvalScore:
        """Score an evaluation trajectory.

        Args:
            trajectory: The execution trajectory to score
            criteria: Optional list of evaluation dimensions to score
            cancellation_token: Optional token to cancel scoring

        Returns:
            EvalScore with overall and dimensional scores
        """
        pass
