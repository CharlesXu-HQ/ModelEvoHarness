"""New mechanisms and composition dependencies reach the Agent as local sources."""
import ast
import copy
import hashlib
import sys

import pytest

from model_evo_harness import load_catalog, model_api, read_references
from model_evo_harness.catalog import common_knowledge, validate_catalog
from model_evo_harness.provider import OpenAICompatibleAgent

NEW = {'crossnet_mix', 'ifm', 'difm', 'dice_activation',
       'local_activation_pooling', 'masked_sequence_pooling'}


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
def test_new_mechanisms_have_independent_native_sources(framework):
    catalog = load_catalog()
    assert NEW <= {card['id'] for card in catalog['method_cards']}
    api = model_api(catalog, framework=framework, method_ids=NEW)
    assert set(api) == NEW
    allowed = sys.stdlib_module_names | {'torch' if framework == 'pytorch' else 'tensorflow'}
    for method in NEW:
        assert api[method]['constructor'] != 'inherited'
        assert api[method]['limitations'] and api[method]['output_contract']
        refs = read_references(catalog, {'framework': framework, 'method_ids': [method]})
        for path, source in refs['files'].items():
            assert path.startswith(f'models/{framework}/')
            assert source['sha256'] == hashlib.sha256(source['content'].encode()).hexdigest()
            for node in ast.walk(ast.parse(source['content'])):
                if isinstance(node, ast.Import):
                    assert {item.name.split('.')[0] for item in node.names} <= allowed
                elif isinstance(node, ast.ImportFrom):
                    assert not node.level and node.module.split('.')[0] in allowed


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
def test_dien_selection_delivers_auxiliary_and_attention_sources(framework):
    catalog = load_catalog()
    calls = []
    agent = OpenAICompatibleAgent('https://example.test/v1', 'test', 'model')
    def complete(instruction, context, effort):
        calls.append(context)
        return {'action': 'experiment', 'research': {'method_id': 'dien'}, 'candidate': {}}
    agent._complete = complete
    result = agent.propose({'catalog': catalog, 'task': {'framework': framework}})
    assert len(calls) == 2
    for name in ('sequence.py', 'sequence_attention.py', 'ctr_training.py', 'sequence_inputs.py'):
        path = f'models/{framework}/{name}'
        assert path in calls[1]['reference_material']
        assert path in result['reference_reads']


@pytest.mark.parametrize('bad', [['../../provider.py'], ['models/tensorflow/training.py'],
                                 'models/pytorch/training.py', ['models/pytorch/missing.py']])
def test_companion_sources_reject_wrong_framework_or_unregistered_paths(bad):
    catalog = copy.deepcopy(load_catalog())
    entry = next(x for x in catalog['model_implementations']
                 if x['id'] == 'dien' and x['framework'] == 'pytorch')
    entry['companion_files'] = bad
    with pytest.raises(ValueError, match='companion'):
        validate_catalog(catalog)


def test_agent_guide_distinguishes_structure_training_and_correctness():
    guide = common_knowledge()['native_ctr_refinements']
    for term in ('CrossNetMix', 'IFM', 'DIFM', 'Dice', 'prefix', 'valid_count',
                 'alpha=0', 'one mechanism', 'two_tower'):
        assert term in guide
