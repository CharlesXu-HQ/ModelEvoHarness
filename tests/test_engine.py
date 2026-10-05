import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_evo_harness.catalog import load_catalog
from model_evo_harness.engine import run_search


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
