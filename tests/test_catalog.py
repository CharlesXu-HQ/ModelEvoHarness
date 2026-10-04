import copy
import unittest

from model_evo_harness.catalog import (applicability, catalog_digest, coverage_report,
                                       implementation_digest, load_catalog, load_inventory,
                                       method_applicability,
                                       validate_research)


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog()
        self.task = {"task_id": "coupon", "dataset_digest": "dataset-v1", "stage": "policy",
                     "capabilities": ["tabular_features", "assignment_or_exposure_propensity"],
                     "fields": ["age", "past_spend"], "objective": {"direction": "max"}}

    def test_every_upstream_topic_and_model_is_accounted_for(self):
        report = coverage_report(self.catalog, load_inventory())
        self.assertEqual(report["document_count"], 55)
        self.assertEqual(report["model_count"], 38)
        self.assertEqual(report["support_count"], 54)
        self.assertEqual(report["project_count"], 50)
        for key in ("unmapped_docs", "unmapped_models", "unknown_docs", "unknown_models",
                    "duplicate_docs", "duplicate_models", "unmapped_support", "unknown_support",
                    "duplicate_support", "unmapped_project", "unknown_project",
                    "duplicate_project"):
            self.assertEqual(report[key], [])
        for key in ("unmapped_method_models", "unknown_method_models",
                    "duplicate_method_models", "misassigned_method_models"):
            self.assertEqual(report[key], [])
        self.assertEqual(len(self.catalog["method_cards"]), 38)

    def test_capability_and_stage_screening(self):
        screen = {entry["family_id"]: entry for entry in applicability(self.task, self.catalog)}
        self.assertEqual(screen["feature_interactions"]["status"], "ready")
        self.assertEqual(screen["sequence_ranking"]["status"], "needs_data")
        self.assertIn("event_sequence", screen["sequence_ranking"]["missing_capabilities"])
        self.assertEqual(screen["collaborative_retrieval"]["status"], "other_stage")
        methods = {entry["method_id"]: entry for entry in method_applicability(self.task, self.catalog)}
        self.assertEqual(methods["fm"]["status"], "needs_data")
        self.assertIn("observed_outcome_labels", methods["fm"]["missing_capabilities"])
        self.assertEqual(methods["din"]["status"], "needs_data")

    def test_known_unavailable_family_is_blocked_but_new_direction_is_allowed(self):
        research = {"direction": "new idea", "mechanism": "transform available raw fields",
                    "why_now": "baseline misses this pattern", "data_rationale": "age is available",
                    "comparison": "same estimator without transformation",
                    "expected_result": "score increases", "falsification": "score unchanged",
                    "input_fields": ["age"], "alternatives": [{"direction": "other", "mechanism": "pooling",
                                                               "reason": "less relevant now"}]}
        proposal = {"research": research}
        self.assertEqual(validate_research(proposal, self.task, self.catalog)["direction"], "new idea")
        blocked = copy.deepcopy(proposal)
        blocked["research"]["family_id"] = "sequence_ranking"
        with self.assertRaisesRegex(ValueError, "needs_data"):
            validate_research(blocked, self.task, self.catalog)
        missing = copy.deepcopy(proposal)
        missing["research"]["input_fields"] = ["unknown"]
        with self.assertRaisesRegex(ValueError, "task fields"):
            validate_research(missing, self.task, self.catalog)
        selected_method = copy.deepcopy(proposal)
        selected_method["research"]["method_id"] = "fm"
        with self.assertRaisesRegex(ValueError, "method fm is needs_data"):
            validate_research(selected_method, self.task, self.catalog)
        self.task["capabilities"].append("observed_outcome_labels")
        self.assertEqual(validate_research(selected_method, self.task, self.catalog)["method_id"], "fm")

    def test_digest_tracks_catalog_content(self):
        original = catalog_digest(self.catalog)
        changed = copy.deepcopy(self.catalog)
        changed["families"][0]["question"] += " more"
        self.assertNotEqual(original, catalog_digest(changed))
        self.assertEqual(len(implementation_digest()), 64)


if __name__ == "__main__":
    unittest.main()
