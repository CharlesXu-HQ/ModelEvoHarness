import copy
import unittest

from model_evo_harness import composition


SNAPSHOT = {"fields": ["spend", "visits"], "capabilities": ["tabular"],
            "horizontal_expansion_required": True}


def component(identifier, method="fm"):
    return {"id": identifier, "mechanism": "Learn interactions from selected fields",
            "reference_method_id": method, "code_sections": [f"Model.{identifier}.forward"],
            "input_fields": ["spend", "visits"], "required_capabilities": ["tabular"],
            "instance_path": f"model.{identifier}",
            "output_contract": "Batch by 8 float representation on the model device"}


def group(identifier="parallel", branches=None, fusion="merge"):
    return {"id": identifier, "branch_ids": branches or ["left", "right"],
            "fusion_id": fusion, "parameter_sharing": []}


def design():
    return {"estimator": "Any estimator", "backbone": "Any backbone", "change_scope": "initialize",
            "parent_trial_id": None, "rationale": "Evaluate complementary interactions",
            "data_fit": "Available aggregate features", "comparison_plan": "Fixed split and objective",
            "components": [component("left"), component("right"), component("merge", "custom")],
            "inheritance": [], "horizontal_expansion": {
                "decision": "expand", "rationale": "Compare different field specializations",
                "comparison_plan": "Ablate each branch under the same training budget",
                "groups": [group()]}}


def sharing():
    return {"component_ids": ["left", "right"], "code_sections": ["model.embedding"],
            "rationale": "Both branches reuse the declared input embedding"}


class HorizontalTests(unittest.TestCase):
    def test_same_method_and_heterogeneous_branches_preserve_declared_structure(self):
        for method in ("fm", "din", "unregistered_custom_method"):
            with self.subTest(method=method):
                candidate = design()
                candidate["components"][1]["reference_method_id"] = method
                candidate["horizontal_expansion"]["groups"][0]["parameter_sharing"] = [sharing()]
                original = copy.deepcopy(candidate)
                result = composition.validate_model_design(candidate, SNAPSHOT, [])
                self.assertEqual(result["horizontal_expansion"], original["horizontal_expansion"])
                self.assertEqual(result["components"], original["components"])
                self.assertEqual(candidate, original)
                result["horizontal_expansion"]["groups"][0]["branch_ids"].append("later")
                result["components"][0]["input_fields"].append("later")
                self.assertEqual(candidate, original)

    def test_required_horizontal_expansion_cannot_be_omitted(self):
        candidate = design()
        del candidate["horizontal_expansion"]
        with self.assertRaisesRegex(ValueError, "horizontal_expansion"):
            composition.validate_model_design(candidate, SNAPSHOT, [])

    def test_legacy_design_and_source_are_not_augmented_with_horizontal_claims(self):
        candidate = design()
        del candidate["horizontal_expansion"]
        for item in candidate["components"]:
            del item["instance_path"]
            del item["output_contract"]
        original = copy.deepcopy(candidate)
        result = composition.validate_model_design(candidate, {"fields": ["spend", "visits"],
                                                              "capabilities": ["tabular"]}, [])
        self.assertNotIn("horizontal_expansion", result)
        source = {"id": "old", "research": {"model_design": candidate}}
        normalized = composition.composition_sources([source])
        self.assertNotIn("horizontal_expansion", normalized[0]["research"]["model_design"])
        self.assertEqual(candidate, original)

    def test_horizontal_expansion_requires_evidence_and_comparison_even_when_deferred(self):
        for decision in ("expand", "defer", "not_applicable"):
            for field in ("rationale", "comparison_plan"):
                with self.subTest(decision=decision, field=field):
                    candidate = design()
                    declaration = candidate["horizontal_expansion"]
                    declaration.update(decision=decision, groups=[] if decision != "expand" else [group()])
                    declaration[field] = " "
                    with self.assertRaisesRegex(ValueError, field):
                        composition.validate_model_design(candidate, SNAPSHOT, [])

    def test_decision_and_groups_must_agree(self):
        for decision, groups in (("invented", []), ("expand", []), ("not_applicable", [group()])):
            with self.subTest(decision=decision):
                candidate = design()
                candidate["horizontal_expansion"].update(decision=decision, groups=groups)
                with self.assertRaisesRegex(ValueError, "decision|groups|expand|not_applicable"):
                    composition.validate_model_design(candidate, SNAPSHOT, [])

    def test_defer_preserves_existing_groups_and_can_also_have_no_groups(self):
        for groups in ([], [group()]):
            candidate = design()
            candidate["horizontal_expansion"].update(decision="defer", groups=groups)
            result = composition.validate_model_design(candidate, SNAPSHOT, [])
            self.assertEqual(result["horizontal_expansion"]["groups"], groups)

    def test_not_applicable_accepts_no_groups(self):
        candidate = design()
        candidate["horizontal_expansion"].update(decision="not_applicable", groups=[])
        self.assertEqual(composition.validate_model_design(candidate, SNAPSHOT, [])[
            "horizontal_expansion"]["decision"], "not_applicable")

    def test_topology_fields_and_references_are_validated(self):
        invalid = [("id", " "), ("branch_ids", []), ("branch_ids", ["left"]),
                   ("branch_ids", ["left", "left"]), ("branch_ids", ["left", "missing"]),
                   ("fusion_id", "missing"), ("fusion_id", "left"),
                   ("parameter_sharing", None)]
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                candidate = design()
                candidate["horizontal_expansion"]["groups"][0][field] = value
                with self.assertRaises(ValueError):
                    composition.validate_model_design(candidate, SNAPSHOT, [])
        for field in ("decision", "rationale", "comparison_plan", "groups"):
            candidate = design()
            del candidate["horizontal_expansion"][field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                composition.validate_model_design(candidate, SNAPSHOT, [])

    def test_duplicate_group_and_duplicate_fusion_are_rejected(self):
        for duplicate_id in (True, False):
            candidate = design()
            extra = group("parallel" if duplicate_id else "other")
            candidate["horizontal_expansion"]["groups"].append(extra)
            with self.subTest(duplicate_id=duplicate_id), self.assertRaisesRegex(ValueError, "duplicate"):
                composition.validate_model_design(candidate, SNAPSHOT, [])

    def test_acyclic_multistage_fusions_are_supported_but_cross_group_cycles_are_rejected(self):
        candidate = design()
        candidate["components"] += [component("extra"), component("final")]
        groups = candidate["horizontal_expansion"]["groups"]
        groups.append(group("stage_two", ["merge", "extra"], "final"))
        result = composition.validate_model_design(candidate, SNAPSHOT, [])
        self.assertEqual(result["horizontal_expansion"]["groups"][1]["branch_ids"], ["merge", "extra"])
        groups[0]["branch_ids"] = ["left", "final"]
        with self.assertRaisesRegex(ValueError, "cycle|acyclic"):
            composition.validate_model_design(candidate, SNAPSHOT, [])

    def test_branch_and_fusion_require_instance_output_and_execution_locations(self):
        for index in (0, 2):
            for field in ("instance_path", "output_contract"):
                candidate = design()
                del candidate["components"][index][field]
                with self.subTest(index=index, field=field), self.assertRaisesRegex(ValueError, field):
                    composition.validate_model_design(candidate, SNAPSHOT, [])
            candidate = design()
            candidate["components"][index]["code_sections"] = ["Model.__init__"]
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "forward|call"):
                composition.validate_model_design(candidate, SNAPSHOT, [])

    def test_tensorflow_call_is_an_execution_location(self):
        candidate = design()
        for item in candidate["components"]:
            item["code_sections"] = [f"Model::{item['id']}::call"]
        result = composition.validate_model_design(candidate, SNAPSHOT, [])
        self.assertEqual(result["components"][0]["code_sections"], ["Model::left::call"])

    def test_distinct_branches_cannot_reuse_one_instance_path(self):
        candidate = design()
        candidate["components"][1]["instance_path"] = "model.left"
        with self.assertRaisesRegex(ValueError, "instance_path"):
            composition.validate_model_design(candidate, SNAPSHOT, [])

    def test_fusion_cannot_claim_the_same_instance_path_as_another_component(self):
        candidate = design()
        candidate['components'][2]['instance_path'] = candidate['components'][0]['instance_path']
        with self.assertRaisesRegex(ValueError, 'instance_path'):
            composition.validate_model_design(candidate, SNAPSHOT, [])

    def test_retain_checks_connections_and_sharing_even_when_component_fields_match(self):
        previous = design()
        source = {'id': 'parent', 'status': 'evaluated', 'research': {'model_design': previous},
                  'reflection': {'technical_experience': {'component_assessments': [
                      {'component_id': c['id'], 'outcome': 'inconclusive'} for c in previous['components']]}}}
        local = copy.deepcopy(previous)
        local.update(change_scope='local', parent_trial_id='parent', inheritance=[
            {'source_trial_id': 'parent', 'component_id': c['id'], 'target_component_id': c['id'],
             'decision': 'retain', 'reason': 'Same implementation', 'compatibility': 'Same inputs',
             'validation_plan': 'Ablate with fixed training'} for c in previous['components']])
        for mode in ('rewire', 'share', 'erase'):
            candidate = copy.deepcopy(local)
            horizontal = candidate['horizontal_expansion']
            if mode == 'rewire':
                horizontal['groups'][0].update(branch_ids=['left', 'merge'], fusion_id='right')
            elif mode == 'share':
                horizontal['groups'][0]['parameter_sharing'] = [sharing()]
            else:
                horizontal.update(decision='not_applicable', groups=[])
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'retain.*|horizontal'):
                composition.validate_model_design(candidate, SNAPSHOT, [source])
            for entry in candidate['inheritance']:
                entry['decision'] = 'adapt'
            composition.validate_model_design(candidate, SNAPSHOT, [source])
        # Editorial changes and renaming component IDs do not rewire instance paths.
        local['horizontal_expansion'].update(decision='defer', rationale='Test the loss first')
        local['horizontal_expansion']['groups'][0].update(id='renamed', branch_ids=['right', 'renamed_left'])
        local['components'][0]['id'] = 'renamed_left'
        local['inheritance'][0]['target_component_id'] = 'renamed_left'
        composition.validate_model_design(local, SNAPSHOT, [source])

    def test_sharing_members_sections_and_rationale_are_validated(self):
        for field, value in (("component_ids", ["left"]), ("component_ids", ["left", "left"]),
                             ("component_ids", ["left", "merge"]),
                             ("component_ids", ["left", "missing"]), ("code_sections", []),
                             ("code_sections", [" "]), ("rationale", " ")):
            candidate = design()
            declaration = sharing()
            declaration[field] = value
            candidate["horizontal_expansion"]["groups"][0]["parameter_sharing"] = [declaration]
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                composition.validate_model_design(candidate, SNAPSHOT, [])

    def test_duplicate_sharing_is_order_independent_but_distinct_shared_parts_are_allowed(self):
        candidate = design()
        first, second = sharing(), sharing()
        second.update(component_ids=["right", "left"], rationale="Another explanation")
        declared = candidate["horizontal_expansion"]["groups"][0]["parameter_sharing"]
        declared.extend([first, second])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            composition.validate_model_design(candidate, SNAPSHOT, [])
        second["code_sections"] = ["model.shared_projection"]
        self.assertEqual(len(composition.validate_model_design(candidate, SNAPSHOT, [])[
            "horizontal_expansion"]["groups"][0]["parameter_sharing"]), 2)


class FeatureGroupTests(unittest.TestCase):
    def validate(self, groups):
        validator = getattr(composition, "validate_feature_groups", None)
        self.assertIsNotNone(validator, "composition must expose validate_feature_groups")
        return validator(groups, ["spend", "visits"])

    def test_missing_groups_are_compatible_and_overlapping_groups_normalize_without_mutation(self):
        self.assertEqual(self.validate(None), [])
        groups = [{"id": " money ", "fields": [" spend "], "rationale": " Spend block "},
                  {"id": "activity", "fields": ["spend", "visits"], "rationale": "Joint view"}]
        original = copy.deepcopy(groups)
        result = self.validate(groups)
        self.assertEqual(result, [{"id": "money", "fields": ["spend"], "rationale": "Spend block"},
                                  {"id": "activity", "fields": ["spend", "visits"], "rationale": "Joint view"}])
        result[0]["fields"].append("visits")
        self.assertEqual(groups, original)

    def test_invalid_groups_are_rejected(self):
        for groups in ({}, [None], [{"id": "", "fields": ["spend"], "rationale": "Known"}],
                       [{"id": "one", "fields": [], "rationale": "Known"}],
                       [{"id": "one", "fields": ["spend", "spend"], "rationale": "Known"}],
                       [{"id": "one", "fields": ["absent"], "rationale": "Known"}],
                       [{"id": "one", "fields": ["spend"], "rationale": " "}],
                       [{"id": "one", "fields": ["spend"], "rationale": "Known"},
                        {"id": " one ", "fields": ["visits"], "rationale": "Known"}]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                self.validate(groups)


if __name__ == "__main__":
    unittest.main()
