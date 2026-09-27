"""Reproducible ASR benchmark runner."""

from .metrics import error_rate, normalize_text
from .models import AsrBenchmarkReport, BenchmarkSample
from .runner import AsrBenchmarkRunner, load_manifest

__all__ = [
    "AsrBenchmarkReport",
    "AsrBenchmarkRunner",
    "BenchmarkSample",
    "error_rate",
    "load_manifest",
    "normalize_text",
]

__version__ = "0.9.0"
