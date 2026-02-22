"""
Benchmark system for PicoAgents.

This module provides tools for benchmarking agent configurations against
datasets of tasks with evaluation criteria.

Example:
    >>> from picoagents.eval.benchmarks import (
    ...     AgentConfig,
    ...     BenchmarkDataset,
    ...     BenchmarkRunner,
    ...     PicoAgentTarget,
    ...     load_builtin_dataset,
    ... )
    >>> from picoagents.eval import LLMEvalJudge
    >>>
    >>> # Load dataset
    >>> dataset = load_builtin_dataset("coding_v1")
    >>>
    >>> # Define configurations to compare
    >>> configs = [
    ...     AgentConfig(name="baseline", compaction=None),
    ...     AgentConfig(name="head_tail", compaction="head_tail"),
    ... ]
    >>>
    >>> # Run benchmark
    >>> runner = BenchmarkRunner(judge=LLMEvalJudge(model_client))
    >>> results = await runner.run_configs(dataset, configs)
    >>>
    >>> # Analyze results
    >>> print_results(results)
"""

from ._analysis import (
    format_file_read_analysis,
    format_summary_table,
    format_task_breakdown,
    format_token_growth,
    print_results,
)
from ._config import AgentConfig
from ._dataset import (
    BenchmarkDataset,
    BenchmarkTask,
    list_builtin_datasets,
    load_builtin_dataset,
)
from ._middleware import BenchmarkMiddleware
from ._results import (
    BenchmarkResults,
    TaskResult,
    TargetSummary,
    list_benchmark_results,
    load_benchmark_results,
)
from ._runner import BenchmarkRunner
from ._targets import (
    BenchmarkTarget,
    CallableTarget,
    ClaudeCodeTarget,
    PicoAgentTarget,
    TargetResult,
)

__all__ = [
    # Configuration
    "AgentConfig",
    # Dataset
    "BenchmarkTask",
    "BenchmarkDataset",
    "load_builtin_dataset",
    "list_builtin_datasets",
    # Targets
    "BenchmarkTarget",
    "PicoAgentTarget",
    "ClaudeCodeTarget",
    "CallableTarget",
    "TargetResult",
    # Middleware
    "BenchmarkMiddleware",
    # Results
    "TaskResult",
    "TargetSummary",
    "BenchmarkResults",
    "load_benchmark_results",
    "list_benchmark_results",
    # Runner
    "BenchmarkRunner",
    # Analysis
    "format_summary_table",
    "format_task_breakdown",
    "format_file_read_analysis",
    "format_token_growth",
    "print_results",
]
