import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
