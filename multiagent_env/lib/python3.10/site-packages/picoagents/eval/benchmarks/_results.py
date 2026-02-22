"""
Benchmark results and storage.

This module defines TaskResult and BenchmarkResults - the data structures
for storing and analyzing benchmark execution results.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ...types import EvalScore, EvalTrajectory


@dataclass
class TaskResult:
    """Result of running one task with one configuration/target.

    Captures both the evaluation score and execution metrics like
    token usage, file reads, and compaction events.
    """

    # Identification
    task_id: str
    target_name: str

    # Execution data
    trajectory: EvalTrajectory
    score: EvalScore

    # Metrics
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 0
    duration_ms: int = 0

    # File access patterns
    files_read: Dict[str, int] = field(default_factory=dict)  # path -> count
    unique_files: int = 0
    duplicate_reads: int = 0

    # Context compaction metrics
    compaction_events: int = 0
    tokens_saved: int = 0

    # Additional metrics from middleware
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_trace: bool = False) -> Dict[str, Any]:
        """Serialize result to dictionary.

        Args:
            include_trace: If True, include full message history and events
        """
        result = {
            "task_id": self.task_id,
            "target_name": self.target_name,
            "score": {
                "overall": self.score.overall,
                "dimensions": self.score.dimensions,
                "reasoning": self.score.reasoning,
            },
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "iterations": self.iterations,
            "duration_ms": self.duration_ms,
            "files_read": self.files_read,
            "unique_files": self.unique_files,
            "duplicate_reads": self.duplicate_reads,
            "compaction_events": self.compaction_events,
            "tokens_saved": self.tokens_saved,
            "metrics": self.metrics,
            "success": self.trajectory.success,
            "error": self.trajectory.error,
        }

        if include_trace:
            # Serialize full message history
            result["trace"] = {
                "messages": self._serialize_messages(self.trajectory.messages),
                "events": self.trajectory.metadata.get("events", []),
                "event_count": self.trajectory.metadata.get("event_count", 0),
            }

        return result

    def _serialize_messages(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Serialize messages to JSON-safe format."""
        serialized = []
        for msg in messages:
            msg_dict = {
                "type": type(msg).__name__,
                "content": getattr(msg, "content", None),
                "source": getattr(msg, "source", None),
            }

            # Handle tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "tool_name": tc.tool_name if hasattr(tc, "tool_name") else str(tc),
                        "parameters": tc.parameters if hasattr(tc, "parameters") else {},
                        "call_id": tc.call_id if hasattr(tc, "call_id") else None,
                    }
                    for tc in msg.tool_calls
                ]

            # Handle tool messages
            if hasattr(msg, "tool_call_id"):
                msg_dict["tool_call_id"] = msg.tool_call_id
            if hasattr(msg, "tool_name"):
                msg_dict["tool_name"] = msg.tool_name
            if hasattr(msg, "success"):
                msg_dict["success"] = msg.success
            if hasattr(msg, "error") and msg.error:
                msg_dict["error"] = msg.error

            # Handle usage info if present
            if hasattr(msg, "usage") and msg.usage:
                msg_dict["usage"] = {
                    "tokens_input": getattr(msg.usage, "tokens_input", 0),
                    "tokens_output": getattr(msg.usage, "tokens_output", 0),
                }

            serialized.append(msg_dict)

        return serialized

    def save_trace(self, path: Path) -> Path:
        """Save full trace to a separate JSON file.

        Args:
            path: Path to save the trace file

        Returns:
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        trace_data = {
            "task_id": self.task_id,
            "target_name": self.target_name,
            "score": self.score.overall,
            "success": self.trajectory.success,
            "error": self.trajectory.error,
            "iterations": self.iterations,
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms,
            "messages": self._serialize_messages(self.trajectory.messages),
            "events": self.trajectory.metadata.get("events", []),
            "metrics": self.metrics,
        }

        path.write_text(json.dumps(trace_data, indent=2, default=str))
        return path

    @classmethod
    def from_dict(cls, data: Dict[str, Any], trajectory: EvalTrajectory, score: EvalScore) -> "TaskResult":
        """Create result from dictionary."""
        return cls(
            task_id=data["task_id"],
            target_name=data["target_name"],
            trajectory=trajectory,
            score=score,
            total_tokens=data.get("total_tokens", 0),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            iterations=data.get("iterations", 0),
            duration_ms=data.get("duration_ms", 0),
            files_read=data.get("files_read", {}),
            unique_files=data.get("unique_files", 0),
            duplicate_reads=data.get("duplicate_reads", 0),
            compaction_events=data.get("compaction_events", 0),
            tokens_saved=data.get("tokens_saved", 0),
            metrics=data.get("metrics", {}),
        )

    def __repr__(self) -> str:
        return (
            f"TaskResult(task={self.task_id!r}, target={self.target_name!r}, "
            f"score={self.score.overall:.1f}, tokens={self.total_tokens})"
        )


@dataclass
class TargetSummary:
    """Aggregated statistics for a single target across all tasks."""

    target_name: str
    task_count: int = 0

    # Aggregated scores
    avg_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0

    # Aggregated tokens
    total_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_tokens_per_task: float = 0.0

    # Aggregated iterations
    total_iterations: int = 0
    avg_iterations_per_task: float = 0.0

    # Aggregated time
    total_duration_ms: int = 0
    avg_duration_per_task_ms: float = 0.0

    # File access
    total_unique_files: int = 0
    total_duplicate_reads: int = 0
    duplicate_read_ratio: float = 0.0

    # Compaction
    total_compaction_events: int = 0
    total_tokens_saved: int = 0

    # Success rate
    success_count: int = 0
    success_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "target_name": self.target_name,
            "task_count": self.task_count,
            "avg_score": self.avg_score,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_task": self.avg_tokens_per_task,
            "total_iterations": self.total_iterations,
            "avg_iterations_per_task": self.avg_iterations_per_task,
            "total_duration_ms": self.total_duration_ms,
            "avg_duration_per_task_ms": self.avg_duration_per_task_ms,
            "total_unique_files": self.total_unique_files,
            "total_duplicate_reads": self.total_duplicate_reads,
            "duplicate_read_ratio": self.duplicate_read_ratio,
            "total_compaction_events": self.total_compaction_events,
            "total_tokens_saved": self.total_tokens_saved,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
        }


@dataclass
class BenchmarkResults:
    """Complete results from a benchmark run.

    Stores the results matrix (target x task) along with aggregated
    summaries and comparison utilities.
    """

    # Run metadata
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    dataset_name: str = ""
    dataset_version: str = ""

    # Target names (for ordering)
    target_names: List[str] = field(default_factory=list)

    # Task IDs (for ordering)
    task_ids: List[str] = field(default_factory=list)

    # Results matrix: target_name -> task_id -> TaskResult
    results: Dict[str, Dict[str, TaskResult]] = field(default_factory=dict)

    # Summaries (computed lazily)
    _summaries: Optional[Dict[str, TargetSummary]] = field(default=None, repr=False)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: TaskResult) -> None:
        """Add a task result.

        Args:
            result: TaskResult to add
        """
        if result.target_name not in self.results:
            self.results[result.target_name] = {}
            if result.target_name not in self.target_names:
                self.target_names.append(result.target_name)

        self.results[result.target_name][result.task_id] = result

        if result.task_id not in self.task_ids:
            self.task_ids.append(result.task_id)

        # Invalidate cached summaries
        self._summaries = None

    def get_result(self, target_name: str, task_id: str) -> Optional[TaskResult]:
        """Get a specific result.

        Args:
            target_name: Target/config name
            task_id: Task ID

        Returns:
            TaskResult or None if not found
        """
        return self.results.get(target_name, {}).get(task_id)

    def get_summaries(self) -> Dict[str, TargetSummary]:
        """Compute and return summaries for each target.

        Returns:
            Dict mapping target name to TargetSummary
        """
        if self._summaries is not None:
            return self._summaries

        summaries = {}

        for target_name in self.target_names:
            target_results = list(self.results.get(target_name, {}).values())
            if not target_results:
                continue

            scores = [r.score.overall for r in target_results]
            tokens = [r.total_tokens for r in target_results]
            iterations = [r.iterations for r in target_results]
            durations = [r.duration_ms for r in target_results]
            unique_files = [r.unique_files for r in target_results]
            duplicate_reads = [r.duplicate_reads for r in target_results]
            compaction_events = [r.compaction_events for r in target_results]
            tokens_saved = [r.tokens_saved for r in target_results]
            successes = [1 if r.trajectory.success else 0 for r in target_results]

            total_files = sum(unique_files) + sum(duplicate_reads)

            summaries[target_name] = TargetSummary(
                target_name=target_name,
                task_count=len(target_results),
                avg_score=sum(scores) / len(scores) if scores else 0,
                min_score=min(scores) if scores else 0,
                max_score=max(scores) if scores else 0,
                total_tokens=sum(tokens),
                total_input_tokens=sum(r.input_tokens for r in target_results),
                total_output_tokens=sum(r.output_tokens for r in target_results),
                avg_tokens_per_task=sum(tokens) / len(tokens) if tokens else 0,
                total_iterations=sum(iterations),
                avg_iterations_per_task=sum(iterations) / len(iterations) if iterations else 0,
                total_duration_ms=sum(durations),
                avg_duration_per_task_ms=sum(durations) / len(durations) if durations else 0,
                total_unique_files=sum(unique_files),
                total_duplicate_reads=sum(duplicate_reads),
                duplicate_read_ratio=sum(duplicate_reads) / total_files if total_files > 0 else 0,
                total_compaction_events=sum(compaction_events),
                total_tokens_saved=sum(tokens_saved),
                success_count=sum(successes),
                success_rate=sum(successes) / len(successes) if successes else 0,
            )

        self._summaries = summaries
        return summaries

    def compare_targets(self, baseline: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Generate comparison metrics vs baseline.

        Args:
            baseline: Target name to use as baseline (default: first target)

        Returns:
            Dict mapping target name to comparison metrics
        """
        summaries = self.get_summaries()

        if not summaries:
            return {}

        if baseline is None:
            baseline = self.target_names[0] if self.target_names else None

        if baseline not in summaries:
            return {}

        baseline_summary = summaries[baseline]
        comparison = {}

        for target_name, summary in summaries.items():
            comp = {
                "target_name": target_name,
                "is_baseline": target_name == baseline,
            }

            # Token comparison
            if baseline_summary.total_tokens > 0:
                token_diff = summary.total_tokens - baseline_summary.total_tokens
                token_pct = (token_diff / baseline_summary.total_tokens) * 100
                comp["token_diff"] = token_diff
                comp["token_diff_pct"] = token_pct
            else:
                comp["token_diff"] = 0
                comp["token_diff_pct"] = 0

            # Score comparison
            score_diff = summary.avg_score - baseline_summary.avg_score
            comp["score_diff"] = score_diff

            # Iteration comparison
            if baseline_summary.total_iterations > 0:
                iter_diff = summary.total_iterations - baseline_summary.total_iterations
                iter_pct = (iter_diff / baseline_summary.total_iterations) * 100
                comp["iteration_diff"] = iter_diff
                comp["iteration_diff_pct"] = iter_pct
            else:
                comp["iteration_diff"] = 0
                comp["iteration_diff_pct"] = 0

            # Duration comparison
            if baseline_summary.total_duration_ms > 0:
                dur_diff = summary.total_duration_ms - baseline_summary.total_duration_ms
                dur_pct = (dur_diff / baseline_summary.total_duration_ms) * 100
                comp["duration_diff_ms"] = dur_diff
                comp["duration_diff_pct"] = dur_pct
            else:
                comp["duration_diff_ms"] = 0
                comp["duration_diff_pct"] = 0

            comparison[target_name] = comp

        return comparison

    def to_dict(self) -> Dict[str, Any]:
        """Serialize results to dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "target_names": self.target_names,
            "task_ids": self.task_ids,
            "results": {
                target: {task_id: result.to_dict() for task_id, result in tasks.items()}
                for target, tasks in self.results.items()
            },
            "summaries": {name: s.to_dict() for name, s in self.get_summaries().items()},
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize results to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def save(self, path: Optional[Path] = None, include_traces: bool = False) -> Path:
        """Save results to JSON file.

        Args:
            path: Output path (default: .picoagents/benchmarks/{run_id}.json)
            include_traces: If True, also save full traces to a traces/ subdirectory

        Returns:
            Path to saved file
        """
        if path is None:
            output_dir = Path.cwd() / ".picoagents" / "benchmarks"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
            path = output_dir / f"benchmark_{self.run_id}_{timestamp_str}.json"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

        # Save individual traces if requested
        if include_traces:
            traces_dir = path.parent / f"traces_{self.run_id}"
            traces_dir.mkdir(parents=True, exist_ok=True)

            for target_name, tasks in self.results.items():
                for task_id, result in tasks.items():
                    trace_path = traces_dir / f"{target_name}_{task_id}.json"
                    result.save_trace(trace_path)

        return path

    def save_traces(self, output_dir: Optional[Path] = None) -> Path:
        """Save all traces to a directory.

        Args:
            output_dir: Directory to save traces (default: .picoagents/benchmarks/traces_{run_id}/)

        Returns:
            Path to traces directory
        """
        if output_dir is None:
            output_dir = Path.cwd() / ".picoagents" / "benchmarks" / f"traces_{self.run_id}"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for target_name, tasks in self.results.items():
            for task_id, result in tasks.items():
                trace_path = output_dir / f"{target_name}_{task_id}.json"
                result.save_trace(trace_path)

        return output_dir

    def __repr__(self) -> str:
        return (
            f"BenchmarkResults(run_id={self.run_id!r}, dataset={self.dataset_name!r}, "
            f"targets={len(self.target_names)}, tasks={len(self.task_ids)})"
        )


def load_benchmark_results(path: Path) -> BenchmarkResults:
    """Load benchmark results from JSON file.

    Note: This loads summary data only. Full trajectories are not preserved
    in JSON serialization to keep file sizes manageable.

    Args:
        path: Path to JSON file

    Returns:
        BenchmarkResults instance (without full trajectories)
    """
    path = Path(path)
    data = json.loads(path.read_text())

    results = BenchmarkResults(
        run_id=data["run_id"],
        timestamp=datetime.fromisoformat(data["timestamp"]),
        dataset_name=data["dataset_name"],
        dataset_version=data.get("dataset_version", ""),
        target_names=data["target_names"],
        task_ids=data["task_ids"],
        metadata=data.get("metadata", {}),
    )

    # Note: We don't fully reconstruct TaskResults since trajectories aren't serialized
    # This is for viewing/comparing saved results, not re-running

    return results


def list_benchmark_results(
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """List saved benchmark result files.

    Args:
        output_dir: Directory to search (default: .picoagents/benchmarks/)

    Returns:
        List of paths, newest first
    """
    output_dir = output_dir or Path.cwd() / ".picoagents" / "benchmarks"
    if not output_dir.exists():
        return []

    return sorted(
        output_dir.glob("benchmark_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
