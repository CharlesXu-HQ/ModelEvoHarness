import copy
import unittest

from model_evo_harness.catalog import (applicability, catalog_digest, coverage_report,
                                       decision_applicability, implementation_digest, load_catalog,
                                       load_inventory, load_guide, model_api, method_applicability,
                                       training_applicability,
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
        self.assertEqual(report["missing_local_guides"], [])
        self.assertEqual(report["missing_implementation_files"], [])
        self.assertEqual(report["missing_card_implementations"], [])
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
        self.assertEqual(methods["fm"]["frameworks"], ["pytorch", "tensorflow"])
        self.assertEqual(methods["hstu"]["frameworks"], ["pytorch", "tensorflow"])
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

    def test_research_bottleneck_cites_host_observed_fact(self):
        evidence = [
            {"id": "encoding-width", "statement": "The encoded training matrix has 7 columns",
             "source": "preprocessing check", "status": "observed", "scope": "task", "value": 7},
            {"id": "timing", "statement": "Feature timing is declared only",
             "source": "dataset manifest", "status": "declared", "scope": "task"},
        ]
        research = {"direction": "try a cross", "mechanism": "Change the interaction layer",
                    "why_now": "The measured representation is narrow", "data_rationale": "age exists",
                    "comparison": "same fixed split", "expected_result": "score rises",
                    "falsification": "score does not rise", "input_fields": ["age"],
                    "alternatives": [{"direction": "keep baseline", "mechanism": "same model",
                                      "reason": "control"}], "change_factors": ["interaction layer"]}
        with self.assertRaisesRegex(ValueError, "evidence_ids"):
            validate_research({"research": research}, self.task, self.catalog, evidence=evidence)
        research["evidence_ids"] = ["invented-50k-columns"]
        with self.assertRaisesRegex(ValueError, "host evidence") as caught:
            validate_research({"research": research}, self.task, self.catalog, evidence=evidence)
        self.assertIn("invented-50k-columns", str(caught.exception))
        self.assertIn("encoding-width", str(caught.exception))
        self.assertIn("timing", str(caught.exception))
        self.assertNotIn("The encoded training matrix", str(caught.exception))
        research["evidence_ids"] = ["timing"]
        with self.assertRaisesRegex(ValueError, "observed"):
            validate_research({"research": research}, self.task, self.catalog, evidence=evidence)
        research["evidence_ids"] = ["encoding-width"]
        checked = validate_research({"research": research}, self.task, self.catalog,
                                    evidence=evidence)
        self.assertEqual(checked["evidence_ids"], ["encoding-width"])
        self.assertEqual(checked["change_factors"], ["interaction layer"])

        research["evidence_ids"] = [f"invented-{index}-" + "x" * 1000 for index in range(100)]
        many_facts = [{**evidence[0], "id": f"available-{index}-" + "y" * 1000} for index in range(100)]
        with self.assertRaises(ValueError) as caught:
            validate_research({"research": research}, self.task, self.catalog, evidence=many_facts)
        self.assertIn("invented-0-", str(caught.exception))
        self.assertIn("available-0-", str(caught.exception))
        self.assertIn("more", str(caught.exception))
        self.assertLess(len(str(caught.exception)), 1600)

    def test_digest_tracks_catalog_content(self):
        original = catalog_digest(self.catalog)
        changed = copy.deepcopy(self.catalog)
        changed["families"][0]["question"] += " more"
        self.assertNotEqual(original, catalog_digest(changed))
        self.assertEqual(len(implementation_digest()), 64)

    def test_all_families_have_substantive_local_guides(self):
        for family in self.catalog["families"]:
            with self.subTest(family=family["id"]):
                guide = load_guide(family["id"])
                self.assertGreater(len(guide.split()), 150)
                self.assertIn("##", guide)
                self.assertIn("## Research lineage", guide)

    def test_training_methods_screen_actual_data_contracts(self):
        patterns = self.catalog["training_patterns"]
        self.assertTrue({"loss", "sampling", "optimization", "regularization", "calibration"}
                        <= {item["category"] for item in patterns})
        self.assertTrue(any(item["id"] == "hard_negative_mining" for item in patterns))
        screen = {item["pattern_id"]: item for item in
                  training_applicability(self.task, self.catalog)}
        self.assertEqual(screen["hard_negative_mining"]["status"], "other_stage")
        self.assertEqual(screen["focal_loss"]["status"], "needs_data")
        self.task["capabilities"].append("observed_outcome_labels")
        screen = {item["pattern_id"]: item for item in
                  training_applicability(self.task, self.catalog)}
        self.assertEqual(screen["focal_loss"]["status"], "ready")

    def test_retrieval_mining_does_not_require_impression_logs_for_declared_sampled_feedback(self):
        task = {"stage": "retrieval", "capabilities": ["implicit_feedback",
                "negative_sampler_definition", "item_catalog"]}
        screen = {item["pattern_id"]: item for item in training_applicability(task, self.catalog)}
        self.assertEqual(screen["hard_negative_mining"]["status"], "ready")

    def test_model_manifest_distinguishes_code_from_guidance(self):
        implementations = self.catalog["model_implementations"]
        card_ids = {card["id"] for card in self.catalog["method_cards"]}
        self.assertEqual(len(card_ids), 44)
        self.assertEqual(len(implementations), 2 * (len(card_ids) + 1))
        self.assertEqual({item["id"] for item in implementations}, card_ids | {"two_tower"})
        for method_id in card_ids:
            with self.subTest(method=method_id):
                self.assertEqual({item["framework"] for item in implementations
                                  if item["id"] == method_id}, {"pytorch", "tensorflow"})
        for framework in ("pytorch", "tensorflow"):
            with self.subTest(framework=framework):
                self.assertEqual(set(model_api(self.catalog, framework=framework,
                                               method_ids=card_ids | {"two_tower"})),
                                 card_ids | {"two_tower"})

    def test_agent_can_read_local_model_api_without_importing_frameworks(self):
        pytorch = model_api(self.catalog, framework="pytorch",
                            method_ids={"fm", "mlr", "biassvd"})
        self.assertEqual(set(pytorch), {"fm", "mlr", "biassvd"})
        self.assertIn("cardinalities", pytorch["fm"]["constructor"])
        self.assertIn("forward", pytorch["fm"]["methods"])
        self.assertIn("num_users", pytorch["biassvd"]["constructor"])
        self.assertIn("forward", pytorch["biassvd"]["methods"])
        self.assertEqual(pytorch["mlr"]["symbol"],
                         "model_evo_harness.models.pytorch.segmentation:MLR")
        tensorflow = model_api(self.catalog, framework="tensorflow", method_ids={"mlr"})
        self.assertIn("call", tensorflow["mlr"]["methods"])

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
