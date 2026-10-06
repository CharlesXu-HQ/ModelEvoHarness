"""Typed field views and falsifiable explicit-interaction research plans."""

from copy import deepcopy
from itertools import combinations_with_replacement

from .evidence import cited_facts


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_interaction_views(views, fields):
    if not isinstance(views, list) or not views:
        raise ValueError('interaction_views must be a nonempty list')
    seen = set()
    for view in views:
        if not isinstance(view, dict) or any(not _text(view.get(key))
                for key in ('id', 'kind', 'representation')):
            raise ValueError('interaction view needs id, kind and representation')
        names = view.get('fields')
        if (not isinstance(names, list) or not names or any(not _text(name) or name not in fields for name in names)
                or len(set(names)) != len(names)):
            raise ValueError('interaction view fields must be distinct available original fields')
        if view['id'] in seen:
            raise ValueError('duplicate interaction view id')
        seen.add(view['id'])
    return deepcopy(views)


def _pairs(views):
    return [(left['id'], right['id']) for left, right in combinations_with_replacement(views, 2)
            if left['id'] != right['id'] or len(left['fields']) > 1]


def interaction_context(snapshot, steps):
    views = snapshot.get('interaction_views', [])
    history = []
    for step in steps:
        plan = step.get('proposal', {}).get('research', {}).get('interaction_plan')
        if not plan:
            continue
        evaluation = step.get('evaluation') or {}
        history.append({'trial_id': step['id'], 'status': step['status'],
            'candidate': next((item for item in plan['candidates'] if item['id'] == plan.get('selected_id')), None),
            'decision': plan['decision'], 'coverage': plan['coverage'], 'score': evaluation.get('score'),
            'control': evaluation.get('interaction_control', {'status': 'not_run'}),
            'attribution': (step.get('reflection', {}).get('technical_experience') or {}).get('attribution', 'unverified')})
    return {'views': deepcopy(views), 'coverage_pairs': [{'left': a, 'right': b} for a, b in _pairs(views)],
            'coverage_provenance': 'Agent assessments, not verified host coverage', 'history': deepcopy(history)}


def validate_interaction_plan(plan, snapshot, *, evidence=None):
    views = validate_interaction_views(snapshot.get('interaction_views'), snapshot.get('fields', []))
    known = {view['id'] for view in views}
    if (not isinstance(plan, dict) or plan.get('decision') not in ('test', 'defer')
            or not _text(plan.get('rationale'))):
        raise ValueError('interaction_plan needs test/defer decision and rationale')
    expected = {tuple(sorted(pair)) for pair in _pairs(views)}
    seen = set()
    coverage = plan.get('coverage')
    if not isinstance(coverage, list):
        raise ValueError('interaction coverage must list every view pair')
    for item in coverage:
        if (not isinstance(item, dict) or not _text(item.get('left')) or not _text(item.get('right'))
                or item['left'] not in known or item['right'] not in known
                or item.get('status') not in ('implicit', 'explicit', 'mixed', 'none', 'unknown')
                or not _text(item.get('basis'))):
            raise ValueError('interaction coverage needs known views, status and basis')
        pair = tuple(sorted((item['left'], item['right'])))
        if pair in seen or pair not in expected:
            raise ValueError('duplicate or unavailable interaction coverage pair')
        seen.add(pair)
    if seen != expected:
        raise ValueError('interaction coverage must account for every host view pair')
    candidates = plan.get('candidates')
    if not isinstance(candidates, list):
        raise ValueError('interaction candidates must be a list')
    ids, priorities = set(), set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or any(not _text(candidate.get(key))
                for key in ('id', 'mechanism', 'reason', 'cost', 'risks')):
            raise ValueError('interaction candidate needs id, mechanism, reason, cost and risks')
        scope = candidate.get('views')
        if (not isinstance(scope, list) or not scope or any(not _text(view) or view not in known for view in scope)
                or len(set(scope)) != len(scope)):
            raise ValueError('interaction candidate views must name distinct host views')
        if type(candidate.get('order')) is not int or candidate['order'] < 2:
            raise ValueError('interaction order must be an integer >= 2')
        rank = candidate.get('priority')
        if type(rank) is not int or rank < 1 or rank in priorities or candidate['id'] in ids:
            raise ValueError('interaction candidate ids and positive priorities must be unique')
        ids.add(candidate['id'])
        priorities.add(rank)
        if evidence is not None:
            facts = cited_facts(candidate.get('evidence_ids'), evidence)
            if not any(fact['status'] == 'observed' for fact in facts):
                raise ValueError('interaction priority needs observed host evidence')
        elif candidate.get('evidence_ids'):
            raise ValueError('interaction evidence_ids need host evidence')
    if plan['decision'] == 'test':
        chosen = next((item for item in candidates if item['id'] == plan.get('selected_id')), None)
        if chosen is None or chosen['priority'] != min(priorities):
            raise ValueError('selected interaction must name the highest-priority candidate')
        control = plan.get('control')
        patch = control.get('config_patch') if isinstance(control, dict) else None
        if (not isinstance(patch, dict) or set(patch) != {'model'} or
                not isinstance(patch['model'], dict) or not patch['model'] or
                not _text(control.get('expected_effect'))):
            raise ValueError('interaction control needs a nonempty model-only config_patch and expected_effect')
    elif plan.get('selected_id') is not None or plan.get('control') is not None:
        raise ValueError('deferred interaction has no selected_id or executable control')
    return deepcopy(plan)


INTERACTION_INSTRUCTIONS = """
When task.interaction_plan_required is true, include research.interaction_plan. First inspect
interaction_context.views and all coverage_pairs. Embeddings represent fields; they are not a
separate source. An MLP gives implicit interactions, while an existing Cross network may already
explicitly involve numeric and categorical views. Missing indicators are derived views, not new fields.
Plan format: {decision:'test'|'defer', rationale, coverage:[{left,right,status,basis}],
candidates:[{id,views:[view_id],order:2,mechanism,priority:1,reason,cost,risks,evidence_ids:[host_id]}],
selected_id:id_or_null, control:{config_patch:{model:{toggle:false}},expected_effect}}.
Cover each supplied unordered view pair once; status is implicit/explicit/mixed/none/unknown and
basis identifies the current code/evidence. Coverage is your assessment, not a verified host fact.
Candidates have unique positive priority ranks; choose the smallest rank. One informative candidate
is enough; mechanisms are open-ended. A test needs a nonempty same-source model-config control.
For defer, explain the higher-value diagnosis/loss/other work, use selected_id:null and omit control;
candidates may be empty. Do not force numerical fields, FM, all pairs, or higher capacity to be first.
Read include_interactions=true before a test. Numeric x_i*v_i, categorical embeddings, masks,
grouped products, bilinear/cross networks, learned selection and higher-order designs are options.
Keep field identity; do not treat coordinates of a flattened embedding as independent raw fields.
Check branch scale, fusion, shared gradients, sparse support and duplicate/source-overlapping views.
Fit scaling, vocabulary and binning only on train. A failed categorical-only recipe does not reject
untested numeric/mixed relations; an untested relation alone does not prove priority. Use measured
evidence and the planned control to distinguish explanations. Existing-field code crosses do not
require a human dataset update. Do not invent sequences/aggregates unavailable in the snapshot.
Report planned controls separately from observed control results; gains belong to the evaluated
recipe until attribution is supported. Preserve this uncertainty in dataset-bound experience.
"""
