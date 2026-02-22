"""
Base classes for the evaluation system.

This module defines the abstract base classes that all evaluation components inherit from.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from .._cancellation_token import CancellationToken
from ..types import EvalScore, EvalTask, EvalTrajectory


class BaseEvalTarget(ABC):
    """Abstract base class for anything that can be evaluated."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def run(
        self, task: EvalTask, cancellation_token: Optional[CancellationToken] = None
    ) -> EvalTrajectory:
        """Execute the task and return the complete trajectory.

        Args:
            task: The evaluation task to execute
            cancellation_token: Optional token to cancel execution

        Returns:
            EvalTrajectory containing the complete execution sequence
        """
        pass


class BaseEvalJudge(ABC):
    """Abstract base class for evaluation judges."""

    def __init__(self, name: str):
        self.name = name

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


class BaseEvalRunner(ABC):
    """Abstract base class for evaluation runners."""

    def __init__(self, judge: BaseEvalJudge):
        self.judge = judge

    @abstractmethod
    async def evaluate(
        self,
        target: BaseEvalTarget,
        tasks: List[EvalTask],
        criteria: Optional[List[str]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> List[EvalScore]:
        """Evaluate a target on multiple tasks.

        Args:
            target: The evaluation target to test
            tasks: List of tasks to evaluate
            criteria: Optional evaluation criteria
            cancellation_token: Optional token to cancel evaluation

        Returns:
            List of evaluation scores, one per task
        """
        pass
