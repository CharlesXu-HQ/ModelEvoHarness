"""Agent-led model iteration across recommendation, search, ads and marketing."""

from .catalog import (applicability, catalog_digest, coverage_report, decision_applicability,
                      implementation_digest, load_catalog, load_guide,
                      common_knowledge, read_references, model_api, method_applicability,
                      training_applicability, validate_research)
from .engine import run_search, validate_data_request, validate_reflection
from .provider import REFERENCE_INSTRUCTIONS, propose_with_references

__all__ = ["applicability", "catalog_digest", "coverage_report", "decision_applicability",
           "implementation_digest", "load_catalog", "load_guide", "model_api", "method_applicability",
           "common_knowledge", "read_references",
           "REFERENCE_INSTRUCTIONS", "propose_with_references",
           "training_applicability", "run_search", "validate_data_request",
           "validate_reflection", "validate_research"]
