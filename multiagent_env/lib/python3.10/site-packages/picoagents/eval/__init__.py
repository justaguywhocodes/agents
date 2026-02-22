"""
Evaluation system for picoagents.

This module provides a simple, type-safe evaluation framework for testing
and comparing different picoagents components (agents, models, orchestrators).

Includes a benchmark system for comparing agent configurations:

    >>> from picoagents.eval import benchmarks
    >>> from picoagents.eval.benchmarks import (
    ...     AgentConfig, BenchmarkRunner, load_builtin_dataset
    ... )
"""

from ._base import BaseEvalJudge, BaseEvalRunner, BaseEvalTarget
from ._runner import EvalRunner
from ._targets import AgentEvalTarget, ModelEvalTarget, OrchestratorEvalTarget
from .judges import BaseEvalJudge as BaseJudge
from .judges import (
    CompositeJudge,
    ContainsJudge,
    ExactMatchJudge,
    FuzzyMatchJudge,
    LLMEvalJudge,
)

# Benchmark system - import as submodule
from . import benchmarks

__all__ = [
    # Base classes
    "BaseEvalTarget",
    "BaseEvalJudge",
    "BaseEvalRunner",
    # Implementations
    "EvalRunner",
    # Evaluation targets
    "AgentEvalTarget",
    "ModelEvalTarget",
    "OrchestratorEvalTarget",
    # Judges
    "BaseJudge",
    "LLMEvalJudge",
    "ExactMatchJudge",
    "FuzzyMatchJudge",
    "ContainsJudge",
    "CompositeJudge",
    # Benchmark system
    "benchmarks",
]
