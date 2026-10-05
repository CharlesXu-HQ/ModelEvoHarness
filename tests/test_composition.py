import copy
import unittest

from model_evo_harness.composition import composition_sources, validate_model_design


SNAPSHOT = {"fields": ["spend", "visits"], "capabilities": ["tabular"]}


def component(identifier="cross"):
    return {"id": identifier, "mechanism": "Explicit feature interactions", "code_sections": ["Cross.forward"],
            "input_fields": ["spend", "visits"], "required_capabilities": ["tabular"],
            "reference_method_id": "fm"}


def design():
    return {"estimator": "T-learner", "backbone": "MLP", "change_scope": "initialize",
            "parent_trial_id": None, "rationale": "Start a tracked design from the legacy seed",
            "data_fit": "Tabular aggregate features support interactions",
            "comparison_plan": "Keep the split and objective fixed", "components": [component()],
            "inheritance": []}


def source(identifier="trial_001", **extra):
    return {"id": identifier, "status": "evaluated", "research": {"model_design": design()}, **extra}


def inherit(identifier="cross", decision="retain", source_id="trial_001"):
    item = {"source_trial_id": source_id, "component_id": identifier, "decision": decision,
            "reason": "Preserve this isolated tested interaction", "compatibility": "Same fields and outputs",
            "validation_plan": "Compare a component ablation on the same split"}
    if decision != "drop":
        item["target_component_id"] = identifier
    return item


def local():
    return {**design(), "change_scope": "local", "parent_trial_id": "trial_001",
            "inheritance": [inherit()]}


class CompositionTests(unittest.TestCase):
    def test_initialization_preserves_reference_and_normalizes_text(self):
        value = design()
        value["backbone"] = " MLP "
        canonical = validate_model_design(value, SNAPSHOT, [])
        self.assertEqual(canonical["backbone"], "MLP")
        self.assertEqual(canonical["components"][0]["reference_method_id"], "fm")

    def test_parent_components_cannot_disappear(self):
        value = local()
        value["inheritance"] = []
        with self.assertRaisesRegex(ValueError, "parent component"):
            validate_model_design(value, SNAPSHOT, [source()])

    def test_missing_source_and_missing_component_rejected(self):
        for key, value in [("source_trial_id", "invented"), ("component_id", "invented")]:
            candidate = local()
            candidate["inheritance"][0][key] = value
            with self.assertRaisesRegex(ValueError, "source|component"):
                validate_model_design(candidate, SNAPSHOT, [source()])

    def test_switch_cannot_be_disguised_as_local(self):
        candidate = local()
        candidate["estimator"] = "DR-learner"
        with self.assertRaisesRegex(ValueError, "local"):
            validate_model_design(candidate, SNAPSHOT, [source()])
        candidate["change_scope"] = "switch"
        self.assertEqual(validate_model_design(candidate, SNAPSHOT, [source()])["change_scope"], "switch")

    def test_switch_requires_changed_estimator_or_backbone(self):
        candidate = local()
        candidate["change_scope"] = "switch"
        with self.assertRaisesRegex(ValueError, "switch"):
            validate_model_design(candidate, SNAPSHOT, [source()])

    def test_unavailable_fields_and_sequence_capability_rejected(self):
        for key, value in [("input_fields", ["history"]), ("required_capabilities", ["sequence"])]:
            candidate = design()
            candidate["components"][0][key] = value
            with self.assertRaisesRegex(ValueError, "fields|capabilities"):
                validate_model_design(candidate, SNAPSHOT, [])

    def test_selective_transfer_keeps_adapts_drops_and_borrows(self):
        parent = source()
        parent["research"]["model_design"]["components"] += [component("loss"), component("sequence")]
        parent["research"]["model_design"]["components"][2]["required_capabilities"] = ["sequence"]
        borrowed = source("trial_002")
        borrowed["research"]["model_design"]["components"] = [component("regularizer")]
        candidate = local()
        candidate.update(change_scope="switch", backbone="DCN")
        candidate["components"] += [component("loss"), component("regularizer")]
        candidate["components"][1]["mechanism"] = "Adapt the loss to a pseudo outcome"
        candidate["inheritance"] += [inherit("loss", "adapt"), inherit("sequence", "drop"),
                                     inherit("regularizer", "retest", "trial_002")]
        self.assertEqual(len(validate_model_design(candidate, SNAPSHOT, [parent, borrowed])["inheritance"]), 4)

    def test_invalid_sources_allow_only_drop_or_retest(self):
        for metadata in [{"status": "failed"}, {"status": "pending"}, {"status": None},
                         {"eligibility": "blocked_implementation"}, {"eligibility": "blocked_analysis_error"},
                         {"eligibility": {"status": "blocked_leakage"}},
                         {"reflection": {"verdict": "invalid"}},
                         {"reflection": {"technical_experience": {"component_assessments": [
                             {"component_id": "cross", "outcome": "harmful"}]}}},
                         {"reflection": {"technical_experience": {"component_assessments": [
                             {"component_id": "cross", "outcome": "invalid"}]}}},
                         {"implementation_check": {"status": "contradicted"}},
                         {"analysis": {"max": {"implementation_check": {"status": "contradicted"}}}}]:
            for decision in ["retain", "adapt"]:
                candidate = local()
                candidate["inheritance"] = [inherit(decision=decision)]
                with self.assertRaisesRegex(ValueError, "retest|invalid"):
                    validate_model_design(candidate, SNAPSHOT, [source(**metadata)])
            candidate["inheritance"] = [inherit(decision="retest")]
            self.assertEqual(validate_model_design(candidate, SNAPSHOT, [source(**metadata)])["inheritance"][0]["decision"], "retest")

    def test_retain_cannot_silently_change_component(self):
        for key, value in [("mechanism", "Different interaction"), ("code_sections", ["Other.forward"]),
                           ("input_fields", ["spend"]), ("required_capabilities", []),
                           ("reference_method_id", "dcn")]:
            candidate = local()
            candidate["components"][0][key] = value
            with self.assertRaisesRegex(ValueError, "retain"):
                validate_model_design(candidate, SNAPSHOT, [source()])

    def test_other_harmful_component_does_not_invalidate_retained_component(self):
        parent = source(reflection={"technical_experience": {"component_assessments": [
            {"component_id": "other", "outcome": "harmful"},
            {"component_id": "cross", "outcome": "inconclusive"}]}})
        current = validate_model_design(local(), SNAPSHOT, [parent])
        self.assertEqual(current["inheritance"][0]["decision"], "retain")

    def test_incompatible_source_needs_redeclared_legal_inputs(self):
        parent = source()
        old = parent["research"]["model_design"]["components"][0]
        old.update(input_fields=["history"], required_capabilities=["sequence"])
        for decision in ("adapt", "retest"):
            candidate = local()
            candidate["inheritance"] = [inherit(decision=decision)]
            current = validate_model_design(candidate, SNAPSHOT, [parent])
            self.assertEqual(current["components"][0]["input_fields"], ["spend", "visits"])
        with self.assertRaisesRegex(ValueError, "retain"):
            validate_model_design(local(), SNAPSHOT, [parent])
        candidate["components"][0]["input_fields"] = ["history"]
        with self.assertRaisesRegex(ValueError, "fields"):
            validate_model_design(candidate, SNAPSHOT, [parent])

    def test_inheritance_requires_a_current_target_and_explanation(self):
        for key, value in [("target_component_id", "invented"), ("reason", "  "),
                           ("compatibility", ""), ("validation_plan", "")]:
            candidate = local()
            candidate["inheritance"][0][key] = value
            with self.assertRaises(ValueError):
                validate_model_design(candidate, SNAPSHOT, [source()])

    def test_missing_parent_and_duplicate_sources_rejected(self):
        with self.assertRaisesRegex(ValueError, "source"):
            validate_model_design(local(), SNAPSHOT, [])
        with self.assertRaisesRegex(ValueError, "duplicate source"):
            validate_model_design(local(), SNAPSHOT, [source(), source()])
        candidate = local()
        candidate["parent_trial_id"] = None
        with self.assertRaisesRegex(ValueError, "parent source"):
            validate_model_design(candidate, SNAPSHOT, [source()])

    def test_drop_cannot_map_to_a_target(self):
        candidate = local()
        candidate["inheritance"][0]["decision"] = "drop"
        with self.assertRaisesRegex(ValueError, "drop"):
            validate_model_design(candidate, SNAPSHOT, [source()])

    def test_tracked_history_cannot_be_reset_as_initialize(self):
        with self.assertRaisesRegex(ValueError, "initialize"):
            validate_model_design(design(), SNAPSHOT, [source()])
        seed = {"id": "seed", "research": {}}
        candidate = {**design(), "parent_trial_id": "seed"}
        self.assertEqual(validate_model_design(candidate, SNAPSHOT, [seed])["change_scope"], "initialize")

    def test_duplicate_component_or_inheritance_rejected(self):
        candidate = local()
        candidate["components"].append(copy.deepcopy(candidate["components"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_model_design(candidate, SNAPSHOT, [source()])
        candidate = local()
        candidate["inheritance"].append(copy.deepcopy(candidate["inheritance"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_model_design(candidate, SNAPSHOT, [source()])

    def test_source_normalization_handles_core_and_coupon_without_report_bloat(self):
        core = {"id": "one", "proposal": {"research": {"model_design": design()}},
                "status": "evaluated", "evaluation": {"implementation_check": {"status": "verified"}}}
        coupon = {"id": "two", "research": {"model_design": design()}, "status": "evaluated",
                  "eligibility": "blocked_implementation", "report": {"large": "ignored"},
                  "analysis": {"high": {"implementation_check": {"status": "verified"}},
                               "max": {"implementation_check": {"status": "contradicted"}}},
                  "reflection": {"lesson": "Repair output"}}
        normalized = composition_sources([core, coupon])
        self.assertEqual(normalized[0]["research"]["model_design"]["backbone"], "MLP")
        self.assertEqual(normalized[1]["implementation_check"]["status"], "contradicted")
        self.assertNotIn("report", normalized[1])
        self.assertEqual(normalized[1]["reflection"]["lesson"], "Repair output")


if __name__ == "__main__":
    unittest.main()
