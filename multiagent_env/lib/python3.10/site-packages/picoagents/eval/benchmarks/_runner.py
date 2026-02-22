"""
Benchmark runner - orchestrates benchmark execution.

This module provides BenchmarkRunner which executes datasets against
multiple targets and collects results.
"""

import asyncio
from typing import Callable, List, Optional

from ..._cancellation_token import CancellationToken
from .._base import BaseEvalJudge
from ...types import EvalScore, EvalTask, EvalTrajectory, Usage
from ._config import AgentConfig
from ._dataset import BenchmarkDataset, BenchmarkTask
from ._middleware import BenchmarkMiddleware
from ._results import BenchmarkResults, TaskResult
from ._targets import BenchmarkTarget, PicoAgentTarget, TargetResult


class BenchmarkRunner:
    """Runs benchmark datasets against multiple targets.

    The runner orchestrates:
    1. Creating targets from configurations
    2. Executing tasks against each target
    3. Scoring results with a judge
    4. Collecting and aggregating metrics

    Example:
        >>> runner = BenchmarkRunner(judge=my_judge)
        >>> results = await runner.run(
        ...     dataset=my_dataset,
        ...     targets=[
        ...         PicoAgentTarget(config_baseline),
        ...         PicoAgentTarget(config_optimized),
        ...     ]
        ... )
    """

    def __init__(
        self,
        judge: BaseEvalJudge,
        parallel_tasks: bool = False,
        parallel_targets: bool = False,
    ):
        """Initialize benchmark runner.

        Args:
            judge: Judge to score task outputs
            parallel_tasks: Run tasks in parallel (default: False for fair comparison)
            parallel_targets: Run targets in parallel (default: False)
        """
        self.judge = judge
        self.parallel_tasks = parallel_tasks
        self.parallel_targets = parallel_targets

    async def run(
        self,
        dataset: BenchmarkDataset,
        targets: List[BenchmarkTarget],
        task_filter: Optional[Callable[[BenchmarkTask], bool]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> BenchmarkResults:
        """Execute benchmark.

        Args:
            dataset: Dataset of tasks to run
            targets: Targets to benchmark
            task_filter: Optional filter to select subset of tasks
            cancellation_token: For cancellation support

        Returns:
            BenchmarkResults with full results matrix
        """
        tasks = list(dataset.tasks)
        if task_filter:
            tasks = [t for t in tasks if task_filter(t)]

        results = BenchmarkResults(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
        )

        if self.parallel_targets:
            # Run all targets in parallel
            target_tasks = [
                self._run_target(target, tasks, dataset, cancellation_token)
                for target in targets
            ]
            target_results = await asyncio.gather(*target_tasks, return_exceptions=True)

            for target, target_result in zip(targets, target_results):
                if isinstance(target_result, Exception):
                    # Handle target failure
                    continue
                for task_result in target_result:
                    results.add_result(task_result)
        else:
            # Run targets sequentially
            for target in targets:
                if cancellation_token and cancellation_token.is_cancelled():
                    break

                task_results = await self._run_target(
                    target, tasks, dataset, cancellation_token
                )
                for task_result in task_results:
                    results.add_result(task_result)

        return results

    async def _run_target(
        self,
        target: BenchmarkTarget,
        tasks: List[BenchmarkTask],
        dataset: BenchmarkDataset,
        cancellation_token: Optional[CancellationToken],
    ) -> List[TaskResult]:
        """Run all tasks for a single target."""
        results = []

        if self.parallel_tasks:
            task_coroutines = [
                self._run_single_task(target, task, dataset, cancellation_token)
                for task in tasks
            ]
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)
            # Filter out exceptions and convert to list
            results = [r for r in results if isinstance(r, TaskResult)]
        else:
            for task in tasks:
                if cancellation_token and cancellation_token.is_cancelled():
                    break

                result = await self._run_single_task(
                    target, task, dataset, cancellation_token
                )
                results.append(result)

        return results

    async def _run_single_task(
        self,
        target: BenchmarkTarget,
        task: BenchmarkTask,
        dataset: BenchmarkDataset,
        cancellation_token: Optional[CancellationToken],
    ) -> TaskResult:
        """Run a single task and score it."""
        # Create middleware for metrics
        middleware = BenchmarkMiddleware()

        # Execute task
        if isinstance(target, PicoAgentTarget):
            # Pass middleware to PicoAgent targets
            target_result = await target.run(
                task,
                cancellation_token=cancellation_token,
                middlewares=[middleware],
            )
        else:
            # Other targets don't support middleware
            target_result = await target.run(
                task,
                cancellation_token=cancellation_token,
            )

        # Build trajectory for judge
        trajectory = self._build_trajectory(task, target_result)

        # Score with judge
        criteria = task.eval_criteria or dataset.default_eval_criteria
        score = await self._score_trajectory(trajectory, criteria, cancellation_token)

        # Get metrics from middleware
        metrics = middleware.get_metrics()

        # Build task result
        return TaskResult(
            task_id=task.id,
            target_name=target.name,
            trajectory=trajectory,
            score=score,
            total_tokens=target_result.total_tokens or metrics.get("total_tokens", 0),
            input_tokens=target_result.input_tokens or metrics.get("input_tokens", 0),
            output_tokens=target_result.output_tokens or metrics.get("output_tokens", 0),
            iterations=target_result.iterations or metrics.get("iterations", 0),
            duration_ms=target_result.duration_ms,
            files_read=metrics.get("file_reads", {}),
            unique_files=metrics.get("unique_files", 0),
            duplicate_reads=metrics.get("duplicate_reads", 0),
            compaction_events=metrics.get("compaction_events", 0),
            tokens_saved=metrics.get("tokens_saved", 0),
            metrics=metrics,
        )

    def _build_trajectory(
        self,
        task: BenchmarkTask,
        result: TargetResult,
    ) -> EvalTrajectory:
        """Build EvalTrajectory from target result with full message history."""
        from ...messages import AssistantMessage, UserMessage

        # Use actual messages from the result if available
        if result.messages:
            messages = result.messages
        else:
            # Fallback to minimal message history
            messages = [
                UserMessage(content=task.prompt, source="user"),
                AssistantMessage(content=result.output, source="assistant"),
            ]

        # Include events in metadata for full trace
        metadata = result.metadata.copy() if result.metadata else {}
        if result.events:
            # Store event summaries (full events may be large)
            metadata["events"] = [
                {
                    "type": type(e).__name__,
                    "source": getattr(e, "source", None),
                    **{k: v for k, v in vars(e).items() if k != "source" and not k.startswith("_")}
                }
                for e in result.events
            ]
            metadata["event_count"] = len(result.events)

        return EvalTrajectory(
            task=task.to_eval_task(),
            messages=messages,
            success=result.success,
            error=result.error,
            usage=Usage(
                duration_ms=result.duration_ms,
                llm_calls=result.iterations,
                tokens_input=result.input_tokens,
                tokens_output=result.output_tokens,
            ),
            metadata=metadata,
        )

    async def _score_trajectory(
        self,
        trajectory: EvalTrajectory,
        criteria: List[str],
        cancellation_token: Optional[CancellationToken],
    ) -> EvalScore:
        """Score trajectory with judge."""
        try:
            return await self.judge.score(
                trajectory,
                criteria=criteria,
                cancellation_token=cancellation_token,
            )
        except Exception as e:
            # Return zero score on judge failure
            return EvalScore(
                overall=0.0,
                dimensions={c: 0.0 for c in criteria},
                reasoning={c: f"Judge error: {str(e)}" for c in criteria},
                trajectory=trajectory,
                metadata={"judge_error": str(e)},
            )

    async def run_configs(
        self,
        dataset: BenchmarkDataset,
        configs: List[AgentConfig],
        task_filter: Optional[Callable[[BenchmarkTask], bool]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> BenchmarkResults:
        """Convenience method to run with AgentConfigs directly.

        Args:
            dataset: Dataset of tasks
            configs: Agent configurations to compare
            task_filter: Optional task filter
            cancellation_token: For cancellation

        Returns:
            BenchmarkResults
        """
        targets = [PicoAgentTarget(config) for config in configs]
        return await self.run(dataset, targets, task_filter, cancellation_token)
