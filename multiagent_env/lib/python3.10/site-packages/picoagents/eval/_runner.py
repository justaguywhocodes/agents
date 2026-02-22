"""
Simple evaluation runner implementation.

This module provides a straightforward evaluation runner that coordinates
the execution of tasks and scoring by judges.
"""

import asyncio
from typing import List, Optional

from .._cancellation_token import CancellationToken
from ..types import EvalScore, EvalTask
from ._base import BaseEvalJudge, BaseEvalRunner, BaseEvalTarget


class EvalRunner(BaseEvalRunner):
    """Simple evaluation runner that coordinates target execution and judging."""

    def __init__(self, judge: BaseEvalJudge, parallel: bool = True):
        """Initialize the evaluation runner.

        Args:
            judge: The judge to use for scoring trajectories
            parallel: Whether to run evaluations in parallel (default: True)
        """
        super().__init__(judge)
        self.parallel = parallel

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
        if self.parallel:
            # Run evaluations in parallel
            eval_tasks = [
                self._evaluate_single_task(target, task, criteria, cancellation_token)
                for task in tasks
            ]
            return await asyncio.gather(*eval_tasks)
        else:
            # Run evaluations sequentially
            scores = []
            for task in tasks:
                if cancellation_token and cancellation_token.is_cancelled():
                    break
                score = await self._evaluate_single_task(
                    target, task, criteria, cancellation_token
                )
                scores.append(score)
            return scores

    async def _evaluate_single_task(
        self,
        target: BaseEvalTarget,
        task: EvalTask,
        criteria: Optional[List[str]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> EvalScore:
        """Evaluate a single task.

        Args:
            target: The evaluation target
            task: The task to evaluate
            criteria: Optional evaluation criteria
            cancellation_token: Optional cancellation token

        Returns:
            EvalScore for the task
        """
        try:
            # Execute the task
            trajectory = await target.run(task, cancellation_token)

            # Score the trajectory
            score = await self.judge.score(trajectory, criteria, cancellation_token)

            return score

        except Exception as e:
            # Return a zero score for failed evaluations
            from ..types import EvalScore, EvalTrajectory, Usage

            # Create a failed trajectory for the error case
            failed_trajectory = EvalTrajectory(
                task=task,
                messages=[],
                success=False,
                error=str(e),
                usage=Usage(
                    duration_ms=0, llm_calls=0, tokens_input=0, tokens_output=0
                ),
                metadata={"error": str(e)},
            )

            # Return a zero score
            return EvalScore(
                overall=0.0,
                dimensions={dim: 0.0 for dim in (criteria or ["accuracy"])},
                reasoning={
                    dim: f"Execution failed: {str(e)}"
                    for dim in (criteria or ["accuracy"])
                },
                trajectory=failed_trajectory,
                metadata={"error": str(e), "judge": self.judge.name},
            )
