"""Minimal subset of the author's `ackaudit` project, vendored so the experiment scripts run unchanged.

Only the model builders used by the paper are included: `hf_models.py` is copied verbatim, and `capture._resolve`
is reduced to the part that returns those builders.
"""
