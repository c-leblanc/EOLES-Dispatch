"""Pyomo optimization model definitions."""

from .default import build_model as build_default_model
from .static_thermal import build_model as build_static_thermal_model

MODEL_REGISTRY = {
    "standard": build_default_model,
    "static_thermal": build_static_thermal_model,
}
