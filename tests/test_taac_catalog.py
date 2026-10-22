"""TAAC components must reach the Agent without importing a DL framework."""

import unittest

from model_evo_harness import load_catalog, model_api, paper_coverage_report, read_references
from model_evo_harness.catalog import method_applicability


METHODS = {"rankmixer", "unidot_conditioned_gate", "unidot_multichannel_pool", "unidot_fm_highway"}


class TaacCatalogTests(unittest.TestCase):
    def test_all_components_deliver_source_contract_and_attribution(self):
        catalog = load_catalog()
        cards = {card["id"]: card for card in catalog["method_cards"]}
        self.assertTrue(METHODS <= cards.keys(), "TAAC components are missing from the catalog")
        for framework in ("pytorch", "tensorflow"):
            apis = model_api(catalog, framework=framework, method_ids=METHODS)
            references = read_references(catalog, {"framework": framework, "method_ids": sorted(METHODS)})
            self.assertEqual(set(apis), METHODS)
            for method in METHODS:
                with self.subTest(framework=framework, method=method):
                    api = apis[method]
                    self.assertTrue(api["structure_pattern"])
                    self.assertTrue(api["local_guide"])
                    self.assertEqual(references["contracts"][method]["output_contract"], api["output_contract"])
                    source = references["files"][api["file"]]["content"]
                    compile(source, api["file"], "exec")
                    if method.startswith("unidot_"):
                        self.assertIn("Copyright (c) 2026 Rongcheng Lin", source)
                        self.assertIn("Permission is hereby granted", source)
                        self.assertTrue(cards[method]["source_references"])

    def test_field_components_do_not_inherit_sequence_requirements(self):
        catalog = load_catalog()
        status = {item["method_id"]: item for item in method_applicability({
            "stage": "ranking", "capabilities": ["tabular_features", "field_tokens", "layer_user_item_tokens"]}, catalog)}
        self.assertTrue(METHODS <= status.keys(), "TAAC applicability mappings are missing")
        for method in ("rankmixer", "unidot_fm_highway"):
            self.assertEqual(status[method]["status"], "ready")
        for method in ("unidot_conditioned_gate", "unidot_multichannel_pool"):
            self.assertEqual(status[method]["status"], "needs_data")
            self.assertIn("event_sequence", status[method]["missing_capabilities"])

    def test_taac_does_not_change_reco_papers_coverage(self):
        report = paper_coverage_report(load_catalog())
        self.assertEqual(report["pdf_count"], 115)
        self.assertEqual(report["added_method_count"], 25)
