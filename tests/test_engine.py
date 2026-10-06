import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_evo_harness.catalog import load_catalog
from model_evo_harness.engine import run_search, validate_reflection


def experiment(direction="try a linear model", fields=None):
    return {
        "action": "experiment",
        "research": {
            "direction": direction,
            "mechanism": "Use a smaller model to reduce variance",
            "why_now": "The baseline has high validation variance",
            "data_rationale": "The numeric feature is available in training",
            "comparison": "Same split and score as the baseline",
            "expected_result": "Higher score",
            "falsification": "No score gain on validation",
            "input_fields": ["feature"] if fields is None else fields,
            "alternatives": [{
                "direction": "increase regularization",
                "mechanism": "Shrink coefficients",
                "reason": "Could address the same variance",
            }],
        },
        "candidate": {"model": direction},
    }


class Task:
    def __init__(self, scores=(0.3, 0.6)):
        self.scores = iter(scores)
        self.evaluated = []
        self.baseline_calls = 0
        self.dataset_digest = "dataset-a"

    def snapshot(self):
        return {
            "task_id": "task-a",
            "dataset_digest": self.dataset_digest,
            "objective": {"metric": "score", "direction": "max"},
            "fields": ["feature", "label"],
            "capabilities": ["tabular"],
            "stage": "ranking",
        }

    def baseline(self, trial_dir):
        self.baseline_calls += 1
        return {"score": 0.4, "metrics": {"score": 0.4}}

    def evaluate(self, proposal, trial_dir):
        self.evaluated.append((proposal, trial_dir))
        score = next(self.scores)
        return {"score": score, "metrics": {"score": score}}


class Agent:
    def __init__(self, proposals):
        self.proposals = iter(proposals)
        self.contexts = []
        self.observations = []

    def propose(self, context):
        self.contexts.append(context)
        return next(self.proposals)

    def reflect(self, observation):
        self.observations.append(observation)
        return {"lesson": "Keep the strongest measured candidate"}


def structured_reflection(*, observed=False, observation_id="policy_net"):
    business = ({"status": "observed", "observation_id": observation_id,
                 "insight": "The validation policy has measured net value",
                 "limitations": "Only this randomized validation population"}
                if observed else {"status": "not_observable",
                                  "reason": "No semantically defined business observation"})
    return {"technical_experience": {
                "lesson": "The tested mechanism did not establish a gain",
                "evidence": "The same validation split gives a lower score",
                "uncertainty": "The host supplied no interval",
                "next_test": "Try a different loss at the same inputs"},
            "business_experience": business}


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.catalog = {"schema_version": 1, "families": [{
            "id": "simple", "name": "Simple models", "question": "Can capacity help?",
            "experiment": "Fit one candidate", "pitfalls": "Overfitting",
            "stages": ["ranking"], "requires": ["tabular"],
            "source_docs": [], "source_models": [],
        }]}

    def test_promotion_policy_uses_host_implementation_status_without_attribution_gate(self):
        for strict, status, expected in (
                (False, None, "promoted"), (False, "unverified", "promoted"),
                (False, "contradicted", "contradicted"),
                (True, None, "unverified"), (True, "unverified", "unverified"),
                (True, "contradicted", "contradicted"), (True, "verified", "promoted")):
            with self.subTest(strict=strict, status=status):
                class CheckedTask(Task):
                    def evaluate(self, proposal, trial_dir):
                        result = super().evaluate(proposal, trial_dir)
                        if status is not None:
                            result["implementation_check"] = {"status": status}
                        result["change_audit"] = {"status": "unverified"}
                        return result

                agent = Agent([experiment()])
                state = run_search(CheckedTask(scores=(0.6,)), agent,
                                   output=self.output / f"{strict}-{status}", catalog=self.catalog,
                                   max_steps=1, require_verified_implementation=strict)
                promoted = expected == "promoted"
                self.assertEqual(state["best_id"], "trial_001" if promoted else "baseline")
                self.assertEqual(state["steps"][0]["promotion"]["reason"], expected)
                self.assertEqual(state["steps"][0]["promotion"]["promoted"], promoted)
                policy = {"require_verified_implementation": strict}
                self.assertEqual(state["promotion_policy"], policy)
                self.assertEqual(agent.contexts[0]["promotion_policy"], policy)
                self.assertEqual(agent.observations[0]["promotion_policy"], policy)

    def test_verified_lower_score_is_eligible_but_not_promoted(self):
        class CheckedTask(Task):
            def evaluate(self, proposal, trial_dir):
                return {**super().evaluate(proposal, trial_dir),
                        "implementation_check": {"status": "verified"}}

        state = run_search(CheckedTask(scores=(0.3,)), Agent([experiment()]),
                           output=self.output, catalog=self.catalog, max_steps=1,
                           require_verified_implementation=True)
        decision = state["steps"][0]["promotion"]
        self.assertEqual(decision["reason"], "no_gain")
        self.assertTrue(decision["eligible"])
        self.assertFalse(decision["promoted"])

    def test_null_implementation_check_is_treated_as_unverified(self):
        self.catalog['experience_schema_version'] = 2
        class CheckedTask(Task):
            def evaluate(self, proposal, trial_dir):
                return {**super().evaluate(proposal, trial_dir), "implementation_check": None,
                        "change_audit": None}
        class StructuredAgent(Agent):
            def reflect(self, observation):
                return structured_reflection()
        state = run_search(CheckedTask(scores=(0.6,)), StructuredAgent([experiment()]),
                           output=self.output, catalog=self.catalog, max_steps=1,
                           require_verified_implementation=True)
        self.assertEqual(state["best_id"], "baseline")
        self.assertEqual(state["steps"][0]["promotion"]["reason"], "unverified")

    def test_promotion_policy_is_frozen_for_resume(self):
        task = Task()
        run_search(task, Agent([]), output=self.output, catalog=self.catalog, max_steps=0,
                   require_verified_implementation=True)
        with self.assertRaisesRegex(ValueError, "promotion policy"):
            run_search(task, Agent([]), output=self.output, catalog=self.catalog, max_steps=0,
                       resume=True)
        state = run_search(task, Agent([]), output=self.output, catalog=self.catalog,
                           max_steps=0, resume=True, require_verified_implementation=True)
        self.assertEqual(state["baseline"]["score"], 0.4)
        self.assertEqual(task.baseline_calls, 1)

    def test_forged_agent_reference_metadata_never_enters_official_journal(self):
        proposal = experiment()
        proposal["reference_reads"] = {"invented.py": "a" * 64}
        proposal["reference_events"] = [{"event": "read", "path": "invented.py"}]
        state = run_search(Task(), Agent([proposal]), output=self.output,
                           catalog=self.catalog, max_steps=1)
        recorded = state["steps"][0]["proposal"]
        self.assertNotIn("reference_reads", recorded)
        self.assertNotIn("reference_events", recorded)

    def test_reference_reads_and_delivery_are_host_observed_and_proposal_scoped(self):
        from model_evo_harness import read_references, call_with_references
        full_catalog = load_catalog()
        path = "models/pytorch/training.py"

        class ReadingAgent(Agent):
            def propose(self, context):
                if context["steps"]:
                    return experiment()
                bundle = read_references(full_catalog, {
                    "framework": "pytorch", "include_training": True})
                self.expected_hash = bundle["files"][path]["sha256"]

                def complete(delivered):
                    delivered["reference_material"][path]["sha256"] = "mutated"
                    return {**experiment(), "reference_reads": {"fake.py": "forged"}}

                return call_with_references(complete, {**context,
                    "reference_material": bundle["files"]})

        agent = ReadingAgent([])
        state = run_search(Task(), agent, output=self.output, catalog=self.catalog, max_steps=2)
        recorded = state["steps"][0]["proposal"]
        self.assertEqual(recorded["reference_reads"], {path: agent.expected_hash})
        self.assertEqual([event["event"] for event in recorded["reference_events"]],
                         ["read", "delivered"])
        self.assertEqual(recorded["reference_events"][1]["files"], {path: agent.expected_hash})
        self.assertNotIn("reference_reads", state["steps"][1]["proposal"])

    def test_reference_wrapper_read_is_distinct_from_delivery_and_does_not_trust_cached_claims(self):
        from model_evo_harness import read_references, call_with_references
        full_catalog = load_catalog()

        class ReadOnlyAgent(Agent):
            def propose(self, context):
                read_references(full_catalog, {"framework": "pytorch", "include_training": True})
                return experiment()

        state = run_search(Task(), ReadOnlyAgent([]), output=self.output,
                           catalog=self.catalog, max_steps=1)
        events = state["steps"][0]["proposal"]["reference_events"]
        self.assertEqual([event["event"] for event in events], ["read"])

        class ForgedDeliveryAgent(Agent):
            def propose(self, context):
                return call_with_references(lambda _: experiment(), {
                    "reference_material": {"fake.py": {"content": "fake", "sha256": "forged"}}})

        with self.assertRaisesRegex(ValueError, "host-observed read"):
            run_search(Task(), ForgedDeliveryAgent([]), output=self.output / "forged",
                       catalog=self.catalog, max_steps=1)

    def test_failed_proposal_reference_scope_is_not_reused_by_resume(self):
        from model_evo_harness import read_references
        full_catalog = load_catalog()

        class BrokenAgent(Agent):
            def propose(self, context):
                read_references(full_catalog, {"framework": "pytorch", "include_training": True})
                raise RuntimeError("provider unavailable")

        task = Task()
        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            run_search(task, BrokenAgent([]), output=self.output,
                       catalog=self.catalog, max_steps=1)
        state = run_search(task, Agent([experiment()]), output=self.output,
                           catalog=self.catalog, max_steps=1, resume=True)
        self.assertNotIn("reference_reads", state["steps"][0]["proposal"])

    def test_changed_reference_content_cannot_be_recorded_as_delivered(self):
        from model_evo_harness import read_references, call_with_references
        full_catalog = load_catalog()

        class MutatingAgent(Agent):
            def propose(self, context):
                bundle = read_references(full_catalog, {
                    "framework": "pytorch", "include_training": True})
                bundle["files"]["models/pytorch/training.py"]["content"] = "fake content"
                return call_with_references(lambda _: experiment(), {
                    "reference_material": bundle["files"]})

        with self.assertRaisesRegex(ValueError, "host-observed read"):
            run_search(Task(), MutatingAgent([]), output=self.output,
                       catalog=self.catalog, max_steps=1)

    def test_reference_retry_keeps_actual_reads_and_records_each_delivery(self):
        from model_evo_harness import propose_with_references
        full_catalog = load_catalog()

        class RetryingAgent(Agent):
            def propose(self, context):
                read_state = {}
                responses = iter([
                    {"action": "read_reference", "framework": "pytorch", "include_training": True},
                    {"action": "experiment", "candidate": None},
                ])
                propose_with_references(lambda _: next(responses), context,
                                        catalog=full_catalog, framework="pytorch", read_state=read_state)
                return propose_with_references(lambda _: experiment(), context,
                    catalog=full_catalog, framework="pytorch", read_state=read_state)

        state = run_search(Task(), RetryingAgent([]), output=self.output,
                           catalog=self.catalog, max_steps=1)
        events = state["steps"][0]["proposal"]["reference_events"]
        self.assertEqual([event["event"] for event in events], ["read", "delivered", "delivered"])

    def test_builtin_reference_loop_records_delivery_without_trusting_result(self):
        from model_evo_harness import propose_with_references
        full_catalog = load_catalog()
        calls = []

        class ReadingAgent(Agent):
            def propose(self, context):
                def complete(current):
                    calls.append(current)
                    if len(calls) == 1:
                        return {"action": "read_reference", "framework": "pytorch",
                                "include_training": True}
                    return experiment()
                return propose_with_references(complete, context, catalog=full_catalog,
                                               framework="pytorch")

        state = run_search(Task(), ReadingAgent([]), output=self.output,
                           catalog=self.catalog, max_steps=1)
        recorded = state["steps"][0]["proposal"]
        self.assertIn("models/pytorch/training.py", recorded["reference_reads"])
        self.assertEqual([event["event"] for event in recorded["reference_events"]],
                         ["read", "delivered"])

    def test_two_trials_record_real_history_and_best_result(self):
        task = Task()
        agent = Agent([experiment(), experiment("try a tree")])

        result = run_search(task, agent, output=self.output, catalog=self.catalog, max_steps=2)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["best_id"], "trial_002")
        self.assertEqual([step["evaluation"]["score"] for step in result["steps"]], [0.3, 0.6])
        self.assertEqual(len(agent.observations), 2)
        self.assertEqual(agent.contexts[0]["steps"], [])
        self.assertEqual(len(agent.contexts[1]["steps"]), 1)
        self.assertEqual(agent.contexts[1]["steps"][0]["id"], "trial_001")
        self.assertEqual(len(agent.contexts[1]["steps"][0]["proposal"]["research"]["alternatives"]), 1)
        self.assertEqual(len(result["steps"]), 2)  # An alternative is not an evaluated trial.
        self.assertEqual(json.loads((self.output / "journal.json").read_text()), result)
        self.assertEqual(task.baseline_calls, 1)

    def test_agent_sees_eligible_local_model_api_for_selected_framework(self):
        class ModelTask(Task):
            def snapshot(self):
                snapshot = super().snapshot()
                snapshot.update(framework="pytorch", capabilities=[
                    "tabular_features", "categorical_field_identities",
                    "observed_outcome_labels"])
                return snapshot

        agent = Agent([{"action": "stop", "reason": "Inspect available structures"}])
        run_search(ModelTask(), agent, output=self.output, catalog=load_catalog(), max_steps=1)
        api = agent.contexts[0]["model_api"]
        self.assertIn("fm", api["pytorch"])
        self.assertNotIn("tensorflow", api)
        self.assertIn("cardinalities", api["pytorch"]["fm"]["constructor"])

    def test_unknown_field_is_rejected_before_evaluation(self):
        task = Task()
        agent = Agent([experiment(fields=["missing_field"])])

        with self.assertRaisesRegex(ValueError, "missing_field"):
            run_search(task, agent, output=self.output, catalog=self.catalog, max_steps=1)

        self.assertEqual(task.evaluated, [])
        journal = json.loads((self.output / "journal.json").read_text())
        self.assertEqual(journal["steps"], [])
        self.assertEqual(journal["baseline"]["score"], 0.4)

    def test_resume_checks_task_and_catalog_identity(self):
        task = Task()
        run_search(task, Agent([experiment()]), output=self.output, catalog=self.catalog, max_steps=1)
        resumed = run_search(task, Agent([]), output=self.output, catalog=self.catalog, max_steps=1, resume=True)
        self.assertEqual(resumed["best_id"], "baseline")
        self.assertEqual(task.baseline_calls, 1)

        task.dataset_digest = "dataset-b"
        with self.assertRaisesRegex(ValueError, "identity"):
            run_search(task, Agent([]), output=self.output, catalog=self.catalog, max_steps=1, resume=True)
        task.dataset_digest = "dataset-a"
        changed_catalog = json.loads(json.dumps(self.catalog))
        changed_catalog["families"][0]["name"] = "Changed family"
        with self.assertRaisesRegex(ValueError, "identity"):
            run_search(task, Agent([]), output=self.output, catalog=changed_catalog, max_steps=1, resume=True)

    def test_evaluation_protocol_is_exposed_and_freezes_comparison(self):
        class ProtocolTask(Task):
            def __init__(self):
                super().__init__()
                self.candidate_universe = "all-eligible-items"

            def snapshot(self):
                return {**super().snapshot(), "evaluation_protocol": {
                    "unit": "user", "split": "fixed-v1", "metric": "ndcg_at_10",
                    "candidate_universe": self.candidate_universe,
                    "negative_source": "none-full-catalog"}}

        task = ProtocolTask()
        agent = Agent([experiment()])
        result = run_search(task, agent, output=self.output, catalog=self.catalog, max_steps=1)
        self.assertEqual(len(result["identity"]["evaluation_protocol_sha256"]), 64)
        self.assertEqual(agent.contexts[0]["task"]["evaluation_protocol"]["candidate_universe"],
                         "all-eligible-items")
        task.candidate_universe = "sampled-100"
        with self.assertRaisesRegex(ValueError, "evaluation protocol"):
            run_search(task, Agent([]), output=self.output, catalog=self.catalog, max_steps=1,
                       resume=True)

    def test_partial_evaluation_protocol_is_rejected(self):
        class PartialProtocolTask(Task):
            def snapshot(self):
                return {**super().snapshot(), "evaluation_protocol": {"split": "fixed-v1"}}

        with self.assertRaisesRegex(ValueError, "evaluation_protocol"):
            run_search(PartialProtocolTask(), Agent([]), output=self.output,
                       catalog=self.catalog, max_steps=0)

    def test_retrieval_protocol_needs_candidate_and_negative_definitions(self):
        class RetrievalTask(Task):
            def snapshot(self):
                return {**super().snapshot(), "stage": "retrieval",
                        "capabilities": ["implicit_feedback"],
                        "evaluation_protocol": {"unit": "user", "split": "fixed-v1",
                                                "metric": "recall_at_10"}}

        with self.assertRaisesRegex(ValueError, "candidate_universe, negative_source"):
            run_search(RetrievalTask(), Agent([]), output=self.output,
                       catalog=self.catalog, max_steps=0)

    def test_resume_rejects_changed_installed_implementation(self):
        task = Task()
        run_search(task, Agent([experiment()]), output=self.output,
                   catalog=self.catalog, max_steps=1)

        with patch("model_evo_harness.catalog.implementation_digest", return_value="changed-code"):
            with self.assertRaisesRegex(ValueError, "identity"):
                run_search(task, Agent([]), output=self.output, catalog=self.catalog,
                           max_steps=1, resume=True)

    def test_request_data_stops_without_evaluation(self):
        task = Task()
        request = {"fields": ["item_sequence"], "reason": "Sequential model needs user history"}
        result = run_search(task, Agent([{"action": "request_data", "request": request}]),
                            output=self.output, catalog=self.catalog, max_steps=3)

        self.assertEqual(result["status"], "needs_data")
        self.assertEqual(result["data_request"], request)
        self.assertEqual(result["steps"], [])
        self.assertEqual(task.evaluated, [])

    def test_feature_request_needs_two_distinct_evaluated_mechanisms(self):
        self.catalog["experience_schema_version"] = 2
        request = {"basis": "experimental_evidence", "fields": ["prior_use"],
                   "source": "pre-assignment event log", "as_of": "before assignment",
                   "reason": "Residual gap remains", "evidence": "Two controlled trials retain the gap",
                   "validation_plan": "Audit timestamps and coverage before publishing a new dataset",
                   "trial_ids": ["trial_001", "trial_002"],
                   "alternatives_considered": "Other available pre-treatment fields were tested"}

        class StructuredAgent(Agent):
            def reflect(self, observation):
                return structured_reflection()

        task = Task(scores=(0.3, 0.35))
        second = experiment("change architecture")
        second["research"]["mechanism"] = "Add explicit field-cross network"
        agent = StructuredAgent([experiment("cross existing fields"), second,
                                 {"action": "request_data", "request": request}])
        result = run_search(task, agent, output=self.output, catalog=self.catalog, max_steps=3)
        self.assertEqual(result["status"], "needs_data")
        self.assertEqual(result["data_request"], request)
        self.assertEqual(len(task.evaluated), 2)
        self.assertIn("technical_experience", result["steps"][0]["reflection"])
        self.assertIn("business_experience", result["steps"][0]["reflection"])

    def test_host_evidence_reaches_agent_and_rejects_uncited_bottleneck(self):
        self.catalog["experience_schema_version"] = 2

        class EvidenceTask(Task):
            def baseline(self, trial_dir):
                return {"score": 0.4, "metrics": {"score": 0.4}, "evidence": [{
                    "id": "baseline-width", "statement": "Encoded training input has 7 columns",
                    "source": "preprocessing audit", "status": "observed", "scope": "task",
                    "value": 7}]}

        proposal = experiment()
        proposal["research"]["evidence_ids"] = ["invented-50k-columns"]
        proposal["research"]["change_factors"] = ["outcome model"]
        agent = Agent([proposal])
        with self.assertRaisesRegex(ValueError, "host evidence"):
            run_search(EvidenceTask(), agent, output=self.output,
                       catalog=self.catalog, max_steps=1)
        self.assertEqual(agent.contexts[0]["evidence"][0]["value"], 7)
        self.assertEqual(json.loads((self.output / "journal.json").read_text())["steps"], [])

    def test_data_request_needs_trial_specific_gap_diagnostics(self):
        self.catalog["experience_schema_version"] = 2

        class EvidenceTask(Task):
            def snapshot(self):
                return {**super().snapshot(), "evidence": [{
                    "id": "timing-declared", "statement": "Timing is declared only",
                    "source": "manifest", "status": "declared", "scope": "task"}]}

            def baseline(self, trial_dir):
                return {"score": 0.4, "metrics": {"score": 0.4}, "evidence": [{
                    "id": "baseline-score", "statement": "Baseline score is 0.4",
                    "source": "fixed validation", "status": "observed", "scope": "task"}]}

        class EvidenceAgent(Agent):
            def reflect(self, observation):
                return structured_reflection()

        first, second = experiment(), experiment("try interactions")
        for proposal in (first, second):
            proposal["research"]["evidence_ids"] = ["baseline-score"]
            proposal["research"]["change_factors"] = ["architecture"]
        second["research"]["mechanism"] = "Use explicit interactions"
        request = {"basis": "experimental_evidence", "fields": ["history"],
                   "source": "event log", "as_of": "before decision",
                   "reason": "Need history", "evidence": "Timing is declared only",
                   "validation_plan": "Audit timestamps", "trial_ids": ["trial_001", "trial_002"],
                   "alternatives_considered": "Two model structures tested",
                   "evidence_ids": ["timing-declared"]}
        with self.assertRaisesRegex(ValueError, "trial-specific"):
            run_search(EvidenceTask(scores=(0.3, 0.35)),
                       EvidenceAgent([first, second, {"action": "request_data", "request": request}]),
                       output=self.output, catalog=self.catalog, max_steps=3)

    def test_data_request_accepts_two_measured_gap_diagnostics(self):
        self.catalog["experience_schema_version"] = 2

        class EvidenceTask(Task):
            def baseline(self, trial_dir):
                return {"score": 0.4, "metrics": {"score": 0.4}, "evidence": [{
                    "id": "baseline-score", "statement": "Baseline score is 0.4",
                    "source": "fixed validation", "status": "observed", "scope": "task"}]}

            def evaluate(self, proposal, trial_dir):
                result = super().evaluate(proposal, trial_dir)
                trial_id = trial_dir.name
                result["evidence"] = [{"id": f"{trial_id}-gap",
                                       "statement": "Residual gap persists in the same slice",
                                       "source": "held-out slice report", "status": "observed",
                                       "scope": "trial", "trial_id": trial_id,
                                       "data_gap_candidate": True}]
                return result

        class EvidenceAgent(Agent):
            def reflect(self, observation):
                return structured_reflection()

        first, second = experiment(), experiment("try interactions")
        for proposal in (first, second):
            proposal["research"]["evidence_ids"] = ["baseline-score"]
            proposal["research"]["change_factors"] = ["architecture"]
        second["research"]["mechanism"] = "Use explicit interactions"
        request = {"basis": "experimental_evidence", "fields": ["history"],
                   "source": "event log", "as_of": "before decision", "reason": "Residual gap",
                   "evidence": "Two trial-specific diagnostics retain the gap",
                   "validation_plan": "Audit timing and coverage",
                   "trial_ids": ["trial_001", "trial_002"],
                   "alternatives_considered": "Available representations were tested",
                   "evidence_ids": ["trial_001-gap", "trial_002-gap"]}
        result = run_search(EvidenceTask(scores=(0.3, 0.35)),
                            EvidenceAgent([first, second, {"action": "request_data", "request": request}]),
                            output=self.output, catalog=self.catalog, max_steps=3)
        self.assertEqual(result["status"], "needs_data")

    def test_attribution_requires_verified_implementation_and_change_audit(self):
        self.catalog["experience_schema_version"] = 2

        class AuditedTask(Task):
            def evaluate(self, proposal, trial_dir):
                result = super().evaluate(proposal, trial_dir)
                result["implementation_check"] = {"status": "verified"}
                result["change_audit"] = {"status": "verified", "changed_factors":
                                          ["architecture", "loss"]}
                return result

        class IsolatedAgent(Agent):
            def reflect(self, observation):
                result = structured_reflection()
                result["technical_experience"]["attribution"] = "isolated"
                return result

        with self.assertRaisesRegex(ValueError, "joint"):
            run_search(AuditedTask(scores=(0.6,)), IsolatedAgent([experiment()]),
                       output=self.output, catalog=self.catalog, max_steps=1)

        class JointAgent(Agent):
            def reflect(self, observation):
                result = structured_reflection()
                result["technical_experience"]["attribution"] = "joint"
                return result

        result = run_search(AuditedTask(), JointAgent([]), output=self.output,
                            catalog=self.catalog, max_steps=1, resume=True)
        self.assertEqual(result["steps"][0]["reflection"]["technical_experience"]
                         ["attribution"], "joint")

    def test_agent_intended_change_does_not_verify_attribution(self):
        reflection = structured_reflection()
        reflection["technical_experience"]["attribution"] = "isolated"
        with self.assertRaisesRegex(ValueError, "unverified"):
            validate_reflection(reflection, {"implementation_check": {"status": "verified"}},
                                Task().snapshot(), [{"id": "trial_001", "status": "evaluated",
                                "proposal": {"research": {"mechanism": "new model",
                                                           "change_factors": ["architecture"]}}}])

    def test_contradicted_implementation_cannot_be_champion_and_audit_is_nonblocking(self):
        self.catalog["experience_schema_version"] = 2

        class AuditedTask(Task):
            def snapshot(self):
                return {**super().snapshot(), "evidence": [{
                    "id": "timing-declared", "statement": "Pre-decision timing is declared only",
                    "source": "manifest", "status": "declared", "scope": "task"}]}

            def baseline(self, trial_dir):
                return {"score": 0.4, "metrics": {"score": 0.4}, "evidence": [{
                    "id": "baseline-score", "statement": "Baseline score is 0.4",
                    "source": "fixed validation", "status": "observed", "scope": "task"}]}

            def evaluate(self, proposal, trial_dir):
                result = super().evaluate(proposal, trial_dir)
                result["implementation_check"] = {"status": "contradicted"}
                return result

        class AuditAgent(Agent):
            def reflect(self, observation):
                result = structured_reflection()
                result["technical_experience"]["attribution"] = "unverified"
                result["audit_recommendations"] = [{
                    "issue": "Verify feature measurement time", "evidence_ids": ["timing-declared"],
                    "validation_plan": "Compare record time with decision time"}]
                return result

        proposal = experiment()
        proposal["research"]["evidence_ids"] = ["baseline-score"]
        proposal["research"]["change_factors"] = ["architecture"]
        result = run_search(AuditedTask(scores=(0.9,)), AuditAgent([proposal]),
                            output=self.output, catalog=self.catalog, max_steps=1)
        self.assertEqual(result["best_id"], "baseline")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps"][0]["reflection"]["audit_recommendations"][0]
                         ["evidence_ids"], ["timing-declared"])
        self.assertEqual(result["steps"][0]["reflection"]["technical_experience"]
                         ["implementation_status"], "contradicted")
        self.assertEqual(result["steps"][0]["reflection"]["technical_experience"]
                         ["attribution"], "unverified")
        self.assertNotIn("data_request", result)

    def test_stop_can_persist_audit_without_a_data_request(self):
        class AuditTask(Task):
            def snapshot(self):
                return {**super().snapshot(), "evidence": [{
                    "id": "timing-declared", "statement": "Time cutoff is declared only",
                    "source": "task contract", "status": "declared", "scope": "task"}]}

        audit = {"issue": "Check event timing", "evidence_ids": ["timing-declared"],
                 "validation_plan": "Compare event times with decision cutoff"}
        result = run_search(AuditTask(), Agent([{"action": "stop", "reason": "Budget exhausted",
                                                 "audit_recommendations": [audit]}]),
                            output=self.output, catalog=self.catalog, max_steps=1)
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["audit_recommendations"], [audit])
        self.assertNotIn("data_request", result)

    def test_feature_request_rejects_one_trial_as_insufficient_evidence(self):
        self.catalog["experience_schema_version"] = 2
        request = {"basis": "experimental_evidence", "fields": ["prior_use"],
                   "source": "pre-assignment event log", "as_of": "before assignment",
                   "reason": "Residual gap remains", "evidence": "One trial retains the gap",
                   "validation_plan": "Audit timestamps and coverage",
                   "trial_ids": ["trial_001"],
                   "alternatives_considered": "No other available method tried"}

        class StructuredAgent(Agent):
            def reflect(self, observation):
                return structured_reflection()

        with self.assertRaisesRegex(ValueError, "two distinct"):
            run_search(Task(scores=(0.3,)), StructuredAgent([experiment(),
                {"action": "request_data", "request": request}]), output=self.output,
                catalog=self.catalog, max_steps=2)

    def test_feature_request_without_trials_needs_host_domain_requirement(self):
        self.catalog["experience_schema_version"] = 2
        request = {"basis": "domain_requirement", "requirement_id": "coupon_history",
                   "fields": ["prior_use"], "source": "coupon event log",
                   "as_of": "before assignment", "reason": "Eligibility requires prior use",
                   "evidence": "Documented coupon eligibility rule",
                   "validation_plan": "Audit join cutoff and coverage"}
        task = Task()
        with self.assertRaisesRegex(ValueError, "domain requirement"):
            run_search(task, Agent([{"action": "request_data", "request": request}]),
                       output=self.output, catalog=self.catalog, max_steps=1)
        self.assertEqual(task.evaluated, [])

    def test_host_domain_requirement_allows_immediate_feature_request(self):
        self.catalog["experience_schema_version"] = 2

        class ExpertTask(Task):
            def snapshot(self):
                return {**super().snapshot(), "domain_requirements": [{
                    "id": "coupon_history", "source": "coupon event log",
                    "fields": ["prior_use"], "as_of": "before assignment"}]}

        request = {"basis": "domain_requirement", "requirement_id": "coupon_history",
                   "fields": ["prior_use"], "source": "coupon event log",
                   "as_of": "before assignment", "reason": "Eligibility requires prior use",
                   "evidence": "Documented coupon eligibility rule",
                   "validation_plan": "Audit join cutoff and coverage"}
        task = ExpertTask()
        result = run_search(task, Agent([{"action": "request_data", "request": request}]),
                            output=self.output, catalog=self.catalog, max_steps=1)
        self.assertEqual(result["status"], "needs_data")
        self.assertEqual(task.evaluated, [])

    def test_business_experience_must_cite_measured_host_observation(self):
        self.catalog["experience_schema_version"] = 2

        class BusinessTask(Task):
            def evaluate(self, proposal, trial_dir):
                return {"score": 0.5, "metrics": {"score": 0.5},
                        "business_observations": [{"id": "policy_net", "population": "validation users",
                                                   "metric": "net_value", "estimate": 0.5,
                                                   "uncertainty": "paired interval unavailable"}]}

        class BusinessAgent(Agent):
            def reflect(self, observation):
                self.observations.append(observation)
                return structured_reflection(observed=True)

        agent = BusinessAgent([experiment()])
        result = run_search(BusinessTask(), agent, output=self.output,
                            catalog=self.catalog, max_steps=1)
        self.assertEqual(result["steps"][0]["reflection"]["business_experience"]
                         ["observation_id"], "policy_net")
        self.assertEqual(agent.observations[0]["trial"]["evaluation"]
                         ["business_observations"][0]["metric"], "net_value")

    def test_future_feature_suggestion_waits_for_distinct_experiments(self):
        self.catalog["experience_schema_version"] = 2
        idea = {"field": "prior_use", "source": "coupon event log",
                "as_of": "before assignment", "evidence": "Residual gap after tests",
                "validation_plan": "Audit timestamp and coverage"}

        class EarlyAgent(Agent):
            def reflect(self, observation):
                return {**structured_reflection(), "future_feature_suggestions": [idea]}

        with self.assertRaisesRegex(ValueError, "two distinct"):
            run_search(Task(scores=(0.3,)), EarlyAgent([experiment()]),
                       output=self.output, catalog=self.catalog, max_steps=1)

    def test_future_feature_suggestion_accepts_experiments_or_explicit_domain_rule(self):
        self.catalog["experience_schema_version"] = 2
        idea = {"field": "prior_use", "source": "coupon event log",
                "as_of": "before assignment", "evidence": "Residual gap after tests",
                "validation_plan": "Audit timestamp and coverage"}

        class LaterAgent(Agent):
            def reflect(self, observation):
                result = structured_reflection()
                if observation["trial"]["id"] == "trial_002":
                    result["future_feature_suggestions"] = [idea]
                return result

        second = experiment("try explicit crosses")
        second["research"]["mechanism"] = "Use explicit field interactions"
        result = run_search(Task(scores=(0.3, 0.35)),
                            LaterAgent([experiment(), second]), output=self.output,
                            catalog=self.catalog, max_steps=2)
        self.assertEqual(result["steps"][1]["reflection"]["future_feature_suggestions"],
                         [idea])

        class ExpertTask(Task):
            def snapshot(self):
                return {**super().snapshot(), "domain_requirements": [{
                    "id": "coupon_history", "source": "coupon event log",
                    "fields": ["prior_use"], "as_of": "before assignment"}]}

        class ExpertAgent(Agent):
            def reflect(self, observation):
                return {**structured_reflection(), "future_feature_suggestions": [idea]}

        expert_dir = self.output / "expert"
        result = run_search(ExpertTask(scores=(0.3,)),
                            ExpertAgent([experiment()]), output=expert_dir,
                            catalog=self.catalog, max_steps=1)
        self.assertEqual(result["steps"][0]["reflection"]["future_feature_suggestions"],
                         [idea])

    def test_business_experience_rejects_unmeasured_observation(self):
        self.catalog["experience_schema_version"] = 2

        class UnsupportedAgent(Agent):
            def reflect(self, observation):
                return structured_reflection(observed=True, observation_id="invented_cohort")

        with self.assertRaisesRegex(ValueError, "business observation"):
            run_search(Task(scores=(0.3,)), UnsupportedAgent([experiment()]),
                       output=self.output, catalog=self.catalog, max_steps=1)

    def test_stop_stops_without_evaluation(self):
        task = Task()
        result = run_search(task, Agent([{"action": "stop", "reason": "No useful hypothesis"}]),
                            output=self.output, catalog=self.catalog, max_steps=3)

        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["stop_reason"], "No useful hypothesis")
        self.assertEqual(task.evaluated, [])

    def test_invalid_score_is_rejected(self):
        task = Task(scores=(math.nan,))
        with self.assertRaisesRegex(ValueError, "score"):
            run_search(task, Agent([experiment()]), output=self.output, catalog=self.catalog, max_steps=1)
        self.assertEqual(json.loads((self.output / "journal.json").read_text())["steps"], [])

    def test_feature_free_candidate_can_be_evaluated(self):
        task = Task(scores=(0.6,))
        result = run_search(task, Agent([experiment(fields=[])]), output=self.output,
                            catalog=self.catalog, max_steps=1)

        self.assertEqual(result["best_id"], "trial_001")
        self.assertEqual(result["steps"][0]["proposal"]["research"]["input_fields"], [])

    def test_resume_reflects_saved_evaluation_without_retraining(self):
        class ReflectFailsOnce(Agent):
            def reflect(self, observation):
                raise RuntimeError("reflection transport failed")

        task = Task(scores=(0.6,))
        with self.assertRaisesRegex(RuntimeError, "reflection transport failed"):
            run_search(task, ReflectFailsOnce([experiment()]), output=self.output,
                       catalog=self.catalog, max_steps=1)

        pending = json.loads((self.output / "journal.json").read_text())
        self.assertEqual(pending["steps"][0]["evaluation"]["score"], 0.6)
        self.assertNotIn("reflection", pending["steps"][0])
        second_agent = Agent([])
        result = run_search(task, second_agent, output=self.output,
                            catalog=self.catalog, max_steps=1, resume=True)
        self.assertEqual(len(task.evaluated), 1)
        self.assertEqual(task.baseline_calls, 1)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps"][0]["reflection"]["lesson"],
                         "Keep the strongest measured candidate")
        self.assertEqual(second_agent.observations[0]["trial"]["evaluation"]["score"], 0.6)

    def test_failed_trial_is_reflected_and_next_trial_can_win(self):
        class FailThenPassTask(Task):
            def evaluate(self, proposal, trial_dir):
                self.evaluated.append((proposal, trial_dir))
                if len(self.evaluated) == 1:
                    raise RuntimeError("provider_api_key=super-secret; " + "x" * 300)
                return {"score": 0.6, "metrics": {"score": 0.6}}

        task = FailThenPassTask()
        agent = Agent([experiment("first attempt"), experiment("revised attempt")])
        result = run_search(task, agent, output=self.output,
                            catalog=self.catalog, max_steps=2)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["best_id"], "trial_002")
        self.assertEqual([step["status"] for step in result["steps"]], ["failed", "evaluated"])
        self.assertEqual(agent.observations[0]["trial"]["status"], "failed")
        self.assertEqual(agent.contexts[1]["steps"][0]["status"], "failed")
        self.assertIn("reflection", agent.contexts[1]["steps"][0])
        self.assertLessEqual(len(result["steps"][0]["error"]), 120)
        self.assertNotIn("super-secret", (self.output / "journal.json").read_text())
        self.assertEqual(len(task.evaluated), 2)

    def test_resume_reflects_saved_failed_trial_without_evaluating_again(self):
        class FailingTask(Task):
            def evaluate(self, proposal, trial_dir):
                self.evaluated.append((proposal, trial_dir))
                raise RuntimeError("private token")

        class ReflectFails(Agent):
            def reflect(self, observation):
                raise RuntimeError("reflection unavailable")

        task = FailingTask()
        with self.assertRaisesRegex(RuntimeError, "reflection unavailable"):
            run_search(task, ReflectFails([experiment()]), output=self.output,
                       catalog=self.catalog, max_steps=1)

        pending = json.loads((self.output / "journal.json").read_text())
        self.assertEqual(pending["steps"][0]["status"], "failed")
        self.assertNotIn("reflection", pending["steps"][0])
        second_agent = Agent([])
        result = run_search(task, second_agent, output=self.output,
                            catalog=self.catalog, max_steps=1, resume=True)
        self.assertEqual(len(task.evaluated), 1)
        self.assertEqual(result["steps"][0]["status"], "failed")
        self.assertEqual(second_agent.observations[0]["trial"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
