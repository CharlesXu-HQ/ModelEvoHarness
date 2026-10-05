import copy
import json
import tempfile
import unittest
from pathlib import Path

from model_evo_harness import catalog, provider
from model_evo_harness.engine import _snapshot, run_search
from test_composition_integration import research


class HorizontalFlowTests(unittest.TestCase):
    def setUp(self):
        self.catalog = catalog.load_catalog()

    def test_composition_source_can_be_read_without_a_catalog_model(self):
        for framework in ('pytorch', 'tensorflow'):
            bundle = catalog.read_references(self.catalog, {
                'framework': framework, 'include_composition': True})
            path = f'models/{framework}/composition.py'
            self.assertEqual(set(bundle['files']), {path})
            self.assertIn('ParallelBranches', bundle['files'][path]['content'])
            self.assertEqual(len(bundle['files'][path]['sha256']), 64)
        with self.assertRaises(ValueError):
            catalog.read_references(self.catalog, {'framework': 'pytorch', 'include_composition': 'yes'})

    def test_parallel_proposal_reads_composition_source_before_returning(self):
        contexts = []
        proposal = {'action': 'experiment', 'research': {'model_design': {
            'horizontal_expansion': {'groups': [{'id': 'encoders'}]}, 'components': []}},
                    'candidate': {}}
        def complete(context):
            contexts.append(context)
            return copy.deepcopy(proposal)
        actual = provider.propose_with_references(complete, {}, catalog=self.catalog, framework='pytorch')
        self.assertEqual(len(contexts), 2)
        self.assertIn('models/pytorch/composition.py', actual['reference_reads'])
        self.assertIn('ParallelBranches', contexts[-1]['reference_material']['models/pytorch/composition.py']['content'])

    def test_snapshot_validates_explicit_feature_groups_without_inventing_capabilities(self):
        class Task:
            def snapshot(self):
                return {'task_id': 'grouped', 'dataset_digest': 'v1', 'stage': 'ranking',
                        'fields': ['left', 'right'], 'capabilities': ['tabular_features'],
                        'objective': {'direction': 'max'}, 'feature_groups': self.groups}
        task = Task()
        task.groups = [{'id': 'observed', 'fields': ['left'], 'rationale': 'Measured attributes'}]
        actual = _snapshot(task)
        self.assertEqual(actual['feature_groups'], task.groups)
        self.assertEqual(actual['capabilities'], ['tabular_features'])
        task.groups[0]['fields'] = ['invented_history']
        with self.assertRaisesRegex(ValueError, 'feature_groups'):
            _snapshot(task)

    def test_opted_in_agent_must_assess_horizontal_expansion_before_evaluation(self):
        class Task:
            def snapshot(self):
                return {'task_id': 'horizontal', 'dataset_digest': 'v1', 'stage': 'ranking',
                        'framework': 'pytorch', 'fields': ['x'], 'capabilities': ['tabular_features'],
                        'objective': {'direction': 'max'}}
            def baseline(self, path):
                return {'score': 0.0, 'metrics': {}}
            def evaluate(self, proposal, path):
                return {'score': 0.0, 'metrics': {}}
        class Agent:
            requires_model_design = True
            requires_horizontal_expansion = True
            def propose(self, context):
                return {'action': 'experiment', 'candidate': {}, 'research': research()}
            def reflect(self, observation):
                return {'technical_experience': {'lesson': 'No claim', 'evidence': 'Mock host score',
                        'uncertainty': 'Not isolated', 'next_test': 'Ablate', 'component_assessments': [
                        {'component_id': 'cross', 'outcome': 'inconclusive', 'evidence': 'No gain',
                         'compatibility_limits': 'Unknown', 'next_test': 'Ablate'}]},
                        'business_experience': {'status': 'not_observable', 'reason': 'No semantic outcome'}}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'horizontal_expansion'):
                run_search(Task(), Agent(), output=Path(directory), catalog=self.catalog, max_steps=1)

    def test_provider_enables_horizontal_contract(self):
        agent = provider.OpenAICompatibleAgent('https://example.test', 'key', 'model')
        self.assertTrue(getattr(agent, 'requires_horizontal_expansion', False))

    def test_parallel_recipe_and_component_lessons_survive_a_local_iteration(self):
        contexts = []

        class Task:
            def snapshot(self):
                return {'task_id': 'parallel-contract', 'dataset_digest': 'frozen-v1',
                        'stage': 'ranking', 'framework': 'pytorch', 'fields': ['x'],
                        'capabilities': ['tabular_features'], 'objective': {'direction': 'max'}}
            def baseline(self, path):
                return {'score': 0.0, 'metrics': {}}
            def evaluate(self, proposal, path):
                # Mock evaluator tests persistence, not model quality or runtime execution.
                return {'score': 0.0, 'metrics': {}}

        class Agent:
            requires_model_design = True
            requires_horizontal_expansion = True
            def propose(self, context):
                contexts.append(copy.deepcopy(context))
                value = research()
                design = value['model_design']
                if not context['steps']:
                    branch = design['components'][0]
                    design['components'] = [dict(branch, id=name, instance_path=f'model.{name}',
                        output_contract='Batch by 4 representation') for name in ('left', 'right')]
                    design['components'].append({'id': 'merge', 'mechanism': 'Concatenate representations',
                        'code_sections': ['Fusion.forward'], 'input_fields': ['x'],
                        'required_capabilities': [], 'instance_path': 'model.fusion',
                        'output_contract': 'Batch by 8 representation'})
                    design['horizontal_expansion'] = {'decision': 'expand',
                        'rationale': 'Compare independent complementary parameterizations of x',
                        'comparison_plan': 'Same-input capacity control and branch ablations',
                        'groups': [{'id': 'views', 'branch_ids': ['left', 'right'],
                                    'fusion_id': 'merge', 'parameter_sharing': []}]}
                else:
                    design = copy.deepcopy(context['steps'][-1]['proposal']['research']['model_design'])
                    value['model_design'] = design
                    design.update(change_scope='local', parent_trial_id='trial_001',
                                  backbone='Same backbone with changed objective')
                    design['horizontal_expansion'].update(decision='defer',
                        rationale='Preserve branches while investigating the objective')
                    design['inheritance'] = [{'source_trial_id': 'trial_001', 'component_id': c['id'],
                        'decision': 'retain', 'target_component_id': c['id'], 'reason': 'Unchanged path',
                        'compatibility': 'Same frozen inputs and output contract',
                        'validation_plan': 'Fixed split, budget and architecture'} for c in design['components']]
                    design['components'].append({'id': 'loss', 'mechanism': 'Reweight existing objective',
                        'code_sections': ['train.loss'], 'input_fields': [], 'required_capabilities': []})
                return {'action': 'experiment', 'candidate': {}, 'research': value}

            def reflect(self, observation):
                design = observation['trial']['proposal']['research']['model_design']
                return {'technical_experience': {'lesson': 'No causal or model gain established',
                    'evidence': 'Mock evaluator', 'uncertainty': 'No model execution', 'next_test': 'Run host control',
                    'component_assessments': [{'component_id': c['id'], 'outcome': 'inconclusive',
                        'evidence': 'Mock contract trial', 'compatibility_limits': 'frozen-v1 only',
                        'next_test': 'Measure ablation with the real host'} for c in design['components']]},
                    'business_experience': {'status': 'not_observable', 'reason': 'No business observations'}}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result = run_search(Task(), Agent(), output=path, catalog=self.catalog, max_steps=2)
            persisted = json.loads((path / 'journal.json').read_text())
        self.assertEqual(result, persisted)
        first, second = [s['proposal']['research']['model_design'] for s in result['steps']]
        self.assertEqual(first['backbone_id'], second['backbone_id'])
        self.assertEqual(first['horizontal_expansion']['groups'], second['horizontal_expansion']['groups'])
        self.assertEqual(second['horizontal_expansion']['decision'], 'defer')
        self.assertEqual(len(contexts[1]['composition_sources']), 1)
        history = contexts[1]['composition_sources'][0]
        self.assertEqual(history['research']['model_design'], first)
        assessments = history['reflection']['technical_experience']['component_assessments']
        self.assertEqual({a['component_id'] for a in assessments}, {'left', 'right', 'merge'})
        self.assertTrue(all(a['attribution'] == 'unverified' for a in assessments))
