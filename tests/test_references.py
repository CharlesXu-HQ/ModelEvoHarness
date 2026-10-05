import ast
import hashlib
import unittest

from model_evo_harness import catalog
from model_evo_harness import provider


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.catalog = catalog.load_catalog()

    def test_bundle_contains_whole_modules_and_training_without_importing_framework(self):
        self.assertTrue(hasattr(catalog, 'read_references'), 'Agent needs a local source reader')
        bundle = catalog.read_references(self.catalog, {
            'framework': 'pytorch', 'method_ids': ['afm', 'deepfm'], 'include_training': True})
        sources = bundle['files']
        self.assertIn('models/pytorch/interactions.py', sources)
        self.assertIn('models/pytorch/architectures.py', sources)
        self.assertIn('models/pytorch/training.py', sources)
        self.assertIn('class _Fields', sources['models/pytorch/interactions.py']['content'])
        for entry in sources.values():
            ast.parse(entry['content'])
            self.assertEqual(entry['sha256'], hashlib.sha256(entry['content'].encode()).hexdigest())
        self.assertFalse(any('tensorflow' in path for path in sources))

    def test_model_ids_cannot_be_used_as_paths_and_unknown_ids_are_not_silently_dropped(self):
        self.assertTrue(hasattr(catalog, 'read_references'))
        for request in ({'framework': 'pytorch', 'method_ids': ['../../provider']},
                        {'framework': 'numpy', 'method_ids': ['fm']},
                        {'framework': 'pytorch', 'method_ids': 'fm'}):
            with self.subTest(request=request), self.assertRaises(ValueError):
                catalog.read_references(self.catalog, request)

    def test_provider_reads_sources_before_final_candidate_and_records_real_hashes(self):
        calls = []
        agent = provider.OpenAICompatibleAgent('https://example.test/v1', 'key', 'model')
        def complete(instruction, context, effort):
            calls.append(context)
            if len(calls) == 1:
                return {'action': 'read_reference', 'framework': 'pytorch',
                        'method_ids': ['afm'], 'include_training': True}
            self.assertIn('def forward', str(context['reference_material']))
            return {'action': 'experiment', 'candidate': {}, 'reference_reads': ['invented']}
        agent._complete = complete
        result = agent.propose({'catalog': self.catalog, 'task': {'framework': 'pytorch'}})
        self.assertEqual(len(calls), 2)
        self.assertEqual(result['action'], 'experiment')
        self.assertIn('models/pytorch/interactions.py', result['reference_reads'])
        self.assertEqual(len(result['reference_reads']['models/pytorch/training.py']), 64)

    def test_declared_bundled_method_is_read_even_if_first_draft_skips_request(self):
        agent = provider.OpenAICompatibleAgent('https://example.test/v1', 'key', 'model')
        calls = []
        def complete(instruction, context, effort):
            calls.append(context)
            return {'action': 'experiment', 'research': {'method_id': 'fm'}, 'candidate': {}}
        agent._complete = complete
        result = agent.propose({'catalog': self.catalog, 'task': {'framework': 'pytorch'}})
        self.assertEqual(len(calls), 2)
        self.assertIn('class FM', str(calls[1]['reference_material']))
        self.assertIn('models/pytorch/architectures.py', result['reference_reads'])

    def test_source_reads_are_bounded_and_cannot_switch_host_framework(self):
        agent = provider.OpenAICompatibleAgent('https://example.test/v1', 'key', 'model')
        agent._complete = lambda *args: {'action': 'read_reference', 'framework': 'tensorflow',
                                        'method_ids': ['afm']}
        with self.assertRaisesRegex(ValueError, 'framework'):
            agent.propose({'catalog': self.catalog, 'task': {'framework': 'pytorch'}})
        agent._complete = lambda *args: {'action': 'read_reference', 'framework': 'pytorch',
                                        'method_ids': ['afm']}
        with self.assertRaisesRegex(ValueError, 'reference.*limit'):
            agent.propose({'catalog': self.catalog, 'task': {'framework': 'pytorch'}})

    def test_read_limit_survives_host_contract_retry(self):
        state = {}
        responses = iter([
            {'action': 'read_reference', 'framework': 'pytorch', 'method_ids': ['afm']},
            {'action': 'read_reference', 'framework': 'pytorch', 'method_ids': ['esmm']},
            {'action': 'experiment'},
        ])
        provider.propose_with_references(lambda _: next(responses), {}, catalog=self.catalog,
                                         framework='pytorch', read_state=state)
        self.assertEqual(state['rounds'], 2)
        with self.assertRaisesRegex(ValueError, 'round limit'):
            provider.propose_with_references(lambda _: {'action': 'read_reference',
                'framework': 'pytorch', 'method_ids': ['fm']}, {}, catalog=self.catalog,
                framework='pytorch', read_state=state)
        self.assertEqual(state['rounds'], 2)
        self.assertNotIn('models/pytorch/architectures.py', state['files'])

    def test_common_guides_are_exposed(self):
        self.assertTrue(hasattr(catalog, 'common_knowledge'))
        guides = catalog.common_knowledge()
        self.assertTrue({'training_objectives', 'sampling_and_hard_examples',
                         'optimization_and_regularization', 'feature_gap_decisions',
                         'business_insight_synthesis'} <= set(guides))


if __name__ == '__main__':
    unittest.main()
