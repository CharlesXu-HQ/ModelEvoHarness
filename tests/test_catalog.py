import copy
import unittest

from model_evo_harness.catalog import (applicability, catalog_digest, coverage_report,
                                       decision_applicability, implementation_digest, load_catalog,
                                       load_inventory, method_applicability,
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
        self.assertEqual(sum(card.get("source_repo", "funrec") == "funrec"
                             for card in self.catalog["method_cards"]), 38)
        self.assertGreater(len(self.catalog["method_cards"]), 38)
        linked = [method_id for pattern in self.catalog["structure_patterns"]
                  for method_id in pattern["method_ids"]]
        self.assertEqual(set(linked), {card["id"] for card in self.catalog["method_cards"]})
        self.assertEqual(len(linked), len(set(linked)))

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
        self.assertEqual(methods["dcn_v2"]["status"], "needs_data")
        self.assertIn("typed_feature_schema", methods["dcn_v2"]["missing_capabilities"])
        self.assertEqual(methods["mlr"]["status"], "needs_data")
        self.assertIn("observed_outcome_labels", methods["mlr"]["missing_capabilities"])
        self.assertEqual(methods["aitm"]["status"], "needs_data")
        self.assertIn("ordered_outcome_labels", methods["aitm"]["missing_capabilities"])
        checks = {entry["check_id"]: entry for entry in decision_applicability(self.task,
                                                                               self.catalog)}
        self.assertEqual(checks["evaluation_protocol"]["status"], "ready")
        self.assertEqual(checks["negative_sampling"]["status"], "other_stage")
        self.assertEqual(checks["shared_resource_interference"]["status"], "not_triggered")

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

    def test_external_method_preserves_funrec_inventory_checks(self):
        external = copy.deepcopy(next(card for card in self.catalog["method_cards"]
                                      if card["id"] == "dcn"))
        external.update(id="external_cross", source_repo="other_project",
                        source_model="models/cross.py",
                        source_url="https://github.com/example/project/blob/abcdef/models/cross.py")
        extended = copy.deepcopy(self.catalog)
        extended["method_cards"].append(external)
        report = coverage_report(extended, load_inventory())
        self.assertEqual(report["unknown_method_models"], [])
        self.assertEqual(report["duplicate_method_models"], [])
        self.assertEqual(report["misassigned_method_models"], [])
        self.assertEqual(report["unmapped_method_models"], [])
        self.assertEqual({entry["method_id"] for entry in method_applicability(self.task, extended)} &
                         {"external_cross"}, {"external_cross"})
        del external["source_url"]
        with self.assertRaisesRegex(ValueError, "source_url"):
            catalog_digest(extended)

    def test_external_family_paths_do_not_pollute_funrec_inventory(self):
        extended = copy.deepcopy(self.catalog)
        external = copy.deepcopy(extended["families"][0])
        external.update(id="external_family", source_repo="other_project",
                        source_url="https://github.com/example/project/tree/abcdef",
                        source_docs=["docs/other.md"], source_models=["models/other.py"],
                        source_support=[], source_project=[])
        extended["families"].append(external)
        report = coverage_report(extended, load_inventory())
        self.assertEqual(report["unknown_docs"], [])
        self.assertEqual(report["unknown_models"], [])


if __name__ == "__main__":
    unittest.main()
