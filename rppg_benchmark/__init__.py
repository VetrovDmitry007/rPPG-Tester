"""Помощники высокого уровня для сравнения и использования моделей RPPG."""
from .interfaces import IRPPGModel  # noqa: F401
from .benchmark import RPPGBenchmark  # noqa: F401

__all__ = [
    "IRPPGModel",
    "RPPGBenchmark",
]