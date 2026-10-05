"""Agent-led model iteration across recommendation, search, ads and marketing."""

from .catalog import (applicability, catalog_digest, coverage_report, decision_applicability,
                      implementation_digest, load_catalog, load_guide,
                      common_knowledge, read_references, model_api, method_applicability,
                      training_applicability, validate_research)
from .composition import composition_sources, model_design_identity, validate_model_design
from .engine import run_search, validate_data_request, validate_reflection
from .evidence import validate_audit_recommendations, validate_host_evidence
from .provider import COMPOSITION_INSTRUCTIONS, REFERENCE_INSTRUCTIONS, propose_with_references

__all__ = ["applicability", "catalog_digest", "coverage_report", "decision_applicability",
           "implementation_digest", "load_catalog", "load_guide", "model_api", "method_applicability",
           "common_knowledge", "read_references",
           "COMPOSITION_INSTRUCTIONS", "composition_sources", "model_design_identity", "validate_model_design",
           "REFERENCE_INSTRUCTIONS", "propose_with_references",
           "training_applicability", "run_search", "validate_data_request",
           "validate_reflection", "validate_research", "validate_host_evidence",
           "validate_audit_recommendations"]
