import copy
import unittest
import tempfile
from pathlib import Path

from model_evo_harness.catalog import load_catalog, validate_research
from model_evo_harness.engine import validate_reflection
from model_evo_harness.provider import propose_with_references


def design():
    return {'estimator': 'supervised', 'backbone': 'mlp', 'change_scope': 'initialize',
            'parent_trial_id': None, 'rationale': 'Start a tracked baseline',
            'data_fit': 'Use observed tabular fields', 'comparison_plan': 'Fixed split control',
            'components': [{'id': 'cross', 'mechanism': 'Field interactions before the head',
                            'code_sections': ['Encoder.forward'], 'input_fields': ['x'],
                            'required_capabilities': ['tabular_features'],
                            'reference_method_id': 'fm'}], 'inheritance': []}


def research():
    return {**{key: 'A measurable hypothesis' for key in
               ('direction', 'mechanism', 'why_now', 'data_rationale', 'comparison',
                'expected_result', 'falsification')}, 'input_fields': ['x'],
            'alternatives': [{'direction': 'loss', 'mechanism': 'change loss', 'reason': 'control first'}],
            'model_design': design()}


class CompositionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog()
        self.snapshot = {'stage': 'ranking', 'framework': 'pytorch', 'fields': ['x'],
                         'capabilities': ['tabular_features'], 'model_design_required': True}

    def test_enabled_host_rejects_missing_design_but_legacy_adapter_remains_compatible(self):
        proposal = {'research': research()}
        del proposal['research']['model_design']
        with self.assertRaisesRegex(ValueError, 'model_design'):
            validate_research(proposal, self.snapshot, self.catalog)
        legacy = {key: value for key, value in self.snapshot.items() if key != 'model_design_required'}
        self.assertNotIn('model_design', validate_research(proposal, legacy, self.catalog))

    def test_canonical_research_preserves_component_contract(self):
        actual = validate_research({'research': research()}, self.snapshot, self.catalog)
        self.assertEqual(actual['model_design'], design())

    def test_component_reference_must_exist_for_host_framework(self):
        value = research()
        value['model_design']['components'][0]['reference_method_id'] = 'invented'
        with self.assertRaisesRegex(ValueError, 'reference_method_id'):
            validate_research({'research': value}, self.snapshot, self.catalog)

    def test_component_sources_are_loaded_even_without_top_level_model_selection(self):
        contexts = []
        value = research()
        second = copy.deepcopy(value['model_design']['components'][0])
        second.update(id='attention', reference_method_id='afm')
        value['model_design']['components'].append(second)
        def complete(context):
            contexts.append(context)
            return {'action': 'experiment', 'research': value, 'candidate': {},
                    'reference_reads': {'fake': 'invented'}}
        result = propose_with_references(complete, {}, catalog=self.catalog, framework='pytorch')
        self.assertEqual(len(contexts), 2)
        self.assertIn('models/pytorch/architectures.py', result['reference_reads'])
        self.assertIn('models/pytorch/interactions.py', result['reference_reads'])
        self.assertNotIn('fake', result['reference_reads'])
        self.assertIn('class FM', contexts[1]['reference_material']['models/pytorch/architectures.py']['content'])

    def test_engine_tracks_local_composition_then_selective_backbone_switch(self):
        from model_evo_harness.engine import run_search
        snapshot = self.snapshot
        class Task:
            def snapshot(self):
                return {**snapshot, "task_id": "composition", "dataset_digest": "frozen",
                        "objective": {"name": "score", "direction": "max"}}
            def baseline(self, path):
                return {"score": 0.0, "metrics": {}}
            def evaluate(self, proposal, path):
                return {"score": float(proposal["candidate"]["trial"]), "metrics": {}}
        class Agent:
            requires_model_design = True
            def propose(self, context):
                value = research()
                index = len(context["steps"])
                if index:
                    value["model_design"] = copy.deepcopy(context["steps"][-1]["proposal"]["research"]["model_design"])
                    current = value["model_design"]
                    current.update(change_scope="local" if index == 1 else "switch",
                                   parent_trial_id=f"trial_{index:03d}")
                    if index == 2:
                        current["backbone"] = "shared_encoder"
                    current["inheritance"] = [
                        {"source_trial_id": f"trial_{index:03d}", "component_id": item["id"],
                         "decision": "retain", "target_component_id": item["id"],
                         "reason": "Same input path", "compatibility": "Same interface",
                         "validation_plan": "Compare frozen controls"}
                        for item in current["components"]]
                    if index == 1:
                        current["components"].append({"id": "loss", "mechanism": "regularized loss",
                            "code_sections": ["train.loss"], "input_fields": [], "required_capabilities": []})
                    else:
                        current["inheritance"][1]["decision"] = "drop"
                        del current["inheritance"][1]["target_component_id"]
                        current["components"] = current["components"][:1]
                return {"action": "experiment", "research": value, "candidate": {"trial": index + 1}}
            def reflect(self, observation):
                current = observation["trial"]["proposal"]["research"]["model_design"]
                return {"technical_experience": {"lesson": "Joint recipe observation", "evidence": "host score",
                    "uncertainty": "component attribution unresolved", "next_test": "ablate",
                    "component_assessments": [{"component_id": c["id"], "outcome": "inconclusive",
                        "evidence": "combined recipe", "compatibility_limits": "this task", "next_test": "ablate"}
                        for c in current["components"]]},
                    "business_experience": {"status": "not_observable", "reason": "no business labels"}}
        with tempfile.TemporaryDirectory() as directory:
            result = run_search(Task(), Agent(), output=Path(directory), catalog=self.catalog, max_steps=3)
        designs = [step["proposal"]["research"]["model_design"] for step in result["steps"]]
        self.assertEqual([d["change_scope"] for d in designs], ["initialize", "local", "switch"])
        self.assertEqual([entry["decision"] for entry in designs[-1]["inheritance"]], ["retain", "drop"])
        self.assertEqual(result["steps"][-1]["reflection"]["technical_experience"]["component_assessments"][0]["attribution"], "unverified")

    def test_component_experience_cannot_upgrade_joint_evidence(self):
        reflection = {'technical_experience': {'lesson': 'Promising combined recipe',
                      'evidence': 'Joint score', 'uncertainty': 'not isolated', 'next_test': 'ablate',
                      'attribution': 'joint', 'component_assessments': [
                          {'component_id': 'cross', 'outcome': 'promising', 'evidence': 'joint trial',
                           'compatibility_limits': 'tabular fields', 'next_test': 'ablate cross',
                           'attribution': 'isolated'}]},
                      'business_experience': {'status': 'not_observable', 'reason': 'no business observations'}}
        evaluation = {'research': research(), 'implementation_check': {'status': 'verified'},
                      'change_audit': {'status': 'verified', 'changed_factors': ['cross', 'loss']}}
        with self.assertRaisesRegex(ValueError, 'component.*attribution'):
            validate_reflection(reflection, evaluation, self.snapshot, [])
        reflection['technical_experience']['component_assessments'][0]['attribution'] = 'joint'
        self.assertEqual(validate_reflection(reflection, evaluation, self.snapshot, []), reflection)

    def test_component_reflection_cannot_drop_or_invent_component(self):
        reflection = {'technical_experience': {'lesson': 'no result', 'evidence': 'trial failed',
                      'uncertainty': 'not tested', 'next_test': 'repair', 'component_assessments': []},
                      'business_experience': {'status': 'not_observable', 'reason': 'none'}}
        evaluation = {'research': research()}
        with self.assertRaisesRegex(ValueError, 'component_assessments'):
            validate_reflection(reflection, evaluation, self.snapshot, [])


if __name__ == '__main__':
    unittest.main()
