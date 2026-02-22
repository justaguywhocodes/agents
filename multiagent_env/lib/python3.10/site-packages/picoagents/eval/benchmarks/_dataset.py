"""
Benchmark dataset definitions.

This module defines BenchmarkTask and BenchmarkDataset - collections of tasks
with evaluation criteria that can be used to compare agent configurations.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class BenchmarkTask:
    """A single task in a benchmark dataset.

    A benchmark task includes not just a prompt, but also evaluation criteria,
    expected behavior characteristics, and optional scoring rubrics.

    Example:
        >>> task = BenchmarkTask(
        ...     id="exhaustive_review",
        ...     name="Exhaustive Code Review",
        ...     prompt="Review all files in the codebase...",
        ...     category="coding",
        ...     eval_criteria=["completeness", "accuracy"],
        ... )
    """

    # Identification
    id: str
    name: str
    prompt: str
    category: str = "general"

    # Evaluation
    eval_criteria: List[str] = field(default_factory=lambda: ["task_completion"])
    expected_output: Optional[str] = None
    rubric: Dict[str, str] = field(default_factory=dict)

    # Task characteristics (for analysis)
    expected_iterations: int = 10
    min_files_to_read: Optional[int] = None
    target_context_tokens: int = 50_000

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt,
            "category": self.category,
            "eval_criteria": self.eval_criteria,
            "expected_output": self.expected_output,
            "rubric": self.rubric,
            "expected_iterations": self.expected_iterations,
            "min_files_to_read": self.min_files_to_read,
            "target_context_tokens": self.target_context_tokens,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkTask":
        """Create task from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            prompt=data["prompt"],
            category=data.get("category", "general"),
            eval_criteria=data.get("eval_criteria", ["task_completion"]),
            expected_output=data.get("expected_output"),
            rubric=data.get("rubric", {}),
            expected_iterations=data.get("expected_iterations", 10),
            min_files_to_read=data.get("min_files_to_read"),
            target_context_tokens=data.get("target_context_tokens", 50_000),
            metadata=data.get("metadata", {}),
        )

    def to_eval_task(self):
        """Convert to picoagents EvalTask for compatibility."""
        from ...types import EvalTask

        return EvalTask(
            name=self.id,
            input=self.prompt,
            expected_output=self.expected_output,
            metadata={
                "benchmark_task": self.to_dict(),
                "eval_criteria": self.eval_criteria,
                "rubric": self.rubric,
            },
        )

    def __repr__(self) -> str:
        return f"BenchmarkTask(id={self.id!r}, category={self.category!r})"


@dataclass
class BenchmarkDataset:
    """A collection of benchmark tasks.

    Datasets group related tasks together and provide common evaluation
    criteria. They can be loaded from JSON files or defined in code.

    Example:
        >>> dataset = BenchmarkDataset.from_json("context_engineering.json")
        >>> for task in dataset.tasks:
        ...     print(task.id)
    """

    # Dataset identification
    name: str
    version: str = "1.0.0"
    description: str = ""

    # Tasks
    tasks: List[BenchmarkTask] = field(default_factory=list)

    # Dataset-level settings
    categories: List[str] = field(default_factory=list)
    default_eval_criteria: List[str] = field(default_factory=lambda: ["task_completion"])

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Populate categories from tasks if not provided."""
        if not self.categories:
            self.categories = list(set(t.category for t in self.tasks))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize dataset to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "categories": self.categories,
            "default_eval_criteria": self.default_eval_criteria,
            "tasks": [t.to_dict() for t in self.tasks],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkDataset":
        """Create dataset from dictionary."""
        tasks = [BenchmarkTask.from_dict(t) for t in data.get("tasks", [])]
        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            tasks=tasks,
            categories=data.get("categories", []),
            default_eval_criteria=data.get("default_eval_criteria", ["task_completion"]),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, path: Path) -> "BenchmarkDataset":
        """Load dataset from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            BenchmarkDataset instance
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_json(self, path: Path) -> None:
        """Save dataset to JSON file.

        Args:
            path: Path to write JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def filter_by_category(self, category: str) -> "BenchmarkDataset":
        """Return subset of tasks matching category.

        Args:
            category: Category to filter by

        Returns:
            New BenchmarkDataset with filtered tasks
        """
        filtered_tasks = [t for t in self.tasks if t.category == category]
        return BenchmarkDataset(
            name=f"{self.name}_{category}",
            version=self.version,
            description=f"{self.description} (filtered: {category})",
            tasks=filtered_tasks,
            categories=[category],
            default_eval_criteria=self.default_eval_criteria,
            metadata={**self.metadata, "filtered_from": self.name},
        )

    def filter_by_ids(self, task_ids: List[str]) -> "BenchmarkDataset":
        """Return subset of tasks matching IDs.

        Args:
            task_ids: List of task IDs to include

        Returns:
            New BenchmarkDataset with filtered tasks
        """
        filtered_tasks = [t for t in self.tasks if t.id in task_ids]
        return BenchmarkDataset(
            name=f"{self.name}_subset",
            version=self.version,
            description=f"{self.description} (subset)",
            tasks=filtered_tasks,
            categories=list(set(t.category for t in filtered_tasks)),
            default_eval_criteria=self.default_eval_criteria,
            metadata={**self.metadata, "filtered_from": self.name},
        )

    def filter(self, predicate: Callable[[BenchmarkTask], bool]) -> "BenchmarkDataset":
        """Return subset of tasks matching predicate.

        Args:
            predicate: Function that returns True for tasks to include

        Returns:
            New BenchmarkDataset with filtered tasks
        """
        filtered_tasks = [t for t in self.tasks if predicate(t)]
        return BenchmarkDataset(
            name=f"{self.name}_filtered",
            version=self.version,
            description=f"{self.description} (custom filter)",
            tasks=filtered_tasks,
            categories=list(set(t.category for t in filtered_tasks)),
            default_eval_criteria=self.default_eval_criteria,
            metadata={**self.metadata, "filtered_from": self.name},
        )

    def get_task(self, task_id: str) -> Optional[BenchmarkTask]:
        """Get task by ID.

        Args:
            task_id: Task ID to find

        Returns:
            BenchmarkTask or None if not found
        """
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def __repr__(self) -> str:
        return f"BenchmarkDataset(name={self.name!r}, tasks={len(self.tasks)}, categories={self.categories})"


def load_builtin_dataset(name: str) -> BenchmarkDataset:
    """Load a built-in benchmark dataset.

    Args:
        name: Dataset name (e.g., "coding_v1")

    Returns:
        BenchmarkDataset instance

    Raises:
        ValueError: If dataset not found
    """
    datasets_dir = Path(__file__).parent / "datasets"

    # Try exact name first
    path = datasets_dir / f"{name}.json"
    if path.exists():
        return BenchmarkDataset.from_json(path)

    # List available datasets
    available = [p.stem for p in datasets_dir.glob("*.json")]
    raise ValueError(f"Dataset '{name}' not found. Available: {available}")


def list_builtin_datasets() -> List[str]:
    """List available built-in datasets.

    Returns:
        List of dataset names
    """
    datasets_dir = Path(__file__).parent / "datasets"
    if not datasets_dir.exists():
        return []
    return [p.stem for p in datasets_dir.glob("*.json")]
