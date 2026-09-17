"""
src/evaluation
==============
Public API for the ALL Detection evaluation module.

Typical usage
-------------
    from src.evaluation import EvaluationEngine

    engine  = EvaluationEngine(model, device, cfg)
    results = engine.run(test_loader, split="test")
    engine.save(results, output_dir="logs/evaluation")
"""

from src.evaluation.engine import EvaluationEngine

__all__ = ["EvaluationEngine"]
