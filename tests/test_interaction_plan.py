import copy
import unittest

from model_evo_harness.interaction import (
    interaction_context, validate_interaction_plan, validate_interaction_views)


def snapshot():
    return {'fields': ['I1', 'I2', 'C1', 'C2'], 'interaction_plan_required': True,
            'interaction_views': [
                {'id': 'numeric', 'kind': 'numeric', 'fields': ['I1', 'I2'], 'representation': 'train-standardized'},
                {'id': 'category', 'kind': 'categorical', 'fields': ['C1', 'C2'], 'representation': 'field embeddings'}]}


def plan():
    return {'decision': 'test', 'rationale': 'Test a currently implicit mixed relation',
            'coverage': [{**pair, 'status': 'implicit', 'basis': 'Current pooled MLP'}
                         for pair in interaction_context(snapshot(), [])['coverage_pairs']],
            'candidates': [{'id': 'mixed', 'views': ['numeric', 'category'], 'order': 2,
                            'mechanism': 'scaled numeric vectors crossed with category vectors',
                            'priority': 1, 'reason': 'Available inputs, untested explicit relation',
                            'cost': 'one candidate and one same-code control',
                            'risks': 'scale and shared-gradient interference', 'evidence_ids': ['baseline']}],
            'selected_id': 'mixed',
            'control': {'config_patch': {'model': {'use_mixed': False}},
                        'expected_effect': 'Remove only the mixed term from the logit'}}


EVIDENCE = [{'id': 'baseline', 'statement': 'Validation result', 'source': 'host',
             'scope': 'task', 'status': 'observed', 'value': .45}]


class InteractionPlanTests(unittest.TestCase):
    def test_views_keep_derived_fields_tied_to_original_inputs(self):
        views = snapshot()['interaction_views']
        views.append({'id': 'missing', 'kind': 'binary', 'fields': ['I1', 'I2'],
                      'representation': 'observed missing indicators'})
        self.assertEqual(len(validate_interaction_views(views, snapshot()['fields'])), 3)
        views[-1]['fields'] = ['invented_history']
        with self.assertRaisesRegex(ValueError, 'fields'):
            validate_interaction_views(views, snapshot()['fields'])

    def test_plan_expresses_priority_without_fixed_model_or_field_ranking(self):
        value = plan()
        actual = validate_interaction_plan(value, snapshot(), evidence=EVIDENCE)
        self.assertEqual(actual['selected_id'], 'mixed')
        self.assertEqual(len(actual['coverage']), 3)
        self.assertEqual(value, plan())

    def test_missing_unknown_and_duplicate_pair_coverage_are_rejected(self):
        for mode in ('missing', 'unknown', 'duplicate'):
            value = plan()
            if mode == 'missing':
                value['coverage'].pop()
            elif mode == 'unknown':
                value['coverage'][0]['left'] = 'unknown'
            else:
                value['coverage'].append(copy.deepcopy(value['coverage'][0]))
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'coverage'):
                validate_interaction_plan(value, snapshot(), evidence=EVIDENCE)

    def test_control_cannot_change_training_protocol_or_be_empty(self):
        for patch in ({'epochs': 2}, {'model': {}}, {'lr': .1, 'model': {'use_mixed': False}}):
            value = plan()
            value['control']['config_patch'] = patch
            with self.subTest(patch=patch), self.assertRaisesRegex(ValueError, 'control'):
                validate_interaction_plan(value, snapshot(), evidence=EVIDENCE)

    def test_priority_and_evidence_must_match_selection(self):
        value = plan()
        value['candidates'].append({**value['candidates'][0], 'id': 'other', 'priority': 2})
        value['selected_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'priority'):
            validate_interaction_plan(value, snapshot(), evidence=EVIDENCE)
        value = plan()
        value['candidates'][0]['evidence_ids'] = ['invented']
        with self.assertRaises(ValueError):
            validate_interaction_plan(value, snapshot(), evidence=EVIDENCE)

    def test_defer_keeps_loss_and_other_research_available(self):
        value = plan()
        value.update(decision='defer', selected_id=None, candidates=[])
        del value['control']
        self.assertEqual(validate_interaction_plan(value, snapshot(), evidence=EVIDENCE)['decision'], 'defer')

    def test_history_retains_scope_control_and_uncertainty(self):
        history = interaction_context(snapshot(), [{
            'id': 'trial_001', 'status': 'evaluated', 'proposal': {'research': {'interaction_plan': plan()}},
            'evaluation': {'score': .46, 'interaction_control': {'status': 'failed'},
                           'change_audit': {'status': 'unverified'}}}])['history']
        self.assertEqual(history[0]['candidate']['views'], ['numeric', 'category'])
        self.assertEqual(history[0]['control']['status'], 'failed')
        self.assertEqual(history[0]['attribution'], 'unverified')

    def test_research_validator_requires_and_preserves_plan_for_opted_in_host(self):
        from model_evo_harness import load_catalog, validate_research
        research = {key: 'Bounded research reason' for key in
                    ('direction', 'mechanism', 'why_now', 'data_rationale', 'comparison', 'expected_result', 'falsification')}
        research.update(input_fields=['I1', 'C1'], alternatives=[{
            'direction': 'diagnosis', 'mechanism': 'measure scales', 'reason': 'cheap check'}],
            evidence_ids=['baseline'], change_factors=['interaction scope'])
        task = {**snapshot(), 'stage': 'ranking', 'capabilities': ['tabular_features']}
        with self.assertRaisesRegex(ValueError, 'interaction_plan'):
            validate_research({'research': research}, task, load_catalog(), evidence=EVIDENCE)
        research['interaction_plan'] = plan()
        result = validate_research({'research': research}, task, load_catalog(), evidence=EVIDENCE)
        self.assertEqual(result['interaction_plan'], plan())

    def test_malformed_view_names_are_validation_errors(self):
        for target, key, value in [('coverage', 'left', []), ('coverage', 'right', {}),
                                   ('candidates', 'views', [{}])]:
            proposal = plan()
            proposal[target][0][key] = value
            with self.subTest(target=target, key=key), self.assertRaises(ValueError):
                validate_interaction_plan(proposal, snapshot(), evidence=EVIDENCE)

    def test_verified_audit_does_not_become_gain_attribution(self):
        step = {'id': 'trial_001', 'status': 'evaluated',
                'proposal': {'research': {'interaction_plan': plan()}},
                'evaluation': {'change_audit': {'status': 'verified'}}}
        self.assertEqual(interaction_context(snapshot(), [step])['history'][0]['attribution'], 'unverified')
        step['reflection'] = {'technical_experience': {'attribution': 'joint'}}
        self.assertEqual(interaction_context(snapshot(), [step])['history'][0]['attribution'], 'joint')

    def test_deferred_candidate_without_selected_id_survives_next_round(self):
        value = plan()
        value.update(decision='defer')
        del value['control'], value['selected_id']
        value = validate_interaction_plan(value, snapshot(), evidence=EVIDENCE)
        history = interaction_context(snapshot(), [{'id': 'trial_001', 'status': 'evaluated',
            'proposal': {'research': {'interaction_plan': value}}}])['history']
        self.assertIsNone(history[0]['candidate'])
