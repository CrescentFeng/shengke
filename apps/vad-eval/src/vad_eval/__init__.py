"""Manifest-driven VAD calibration and evaluation."""

from .models import VadBenchmarkReport, VadBenchmarkSample, VadCandidate
from .runner import VadBenchmarkRunner, load_manifest

__all__ = [
    "VadBenchmarkReport",
    "VadBenchmarkRunner",
    "VadBenchmarkSample",
    "VadCandidate",
    "load_manifest",
]

__version__ = "0.9.0"
