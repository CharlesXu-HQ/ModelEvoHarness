import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

from model_evo_harness import cli
from model_evo_harness.provider import OpenAICompatibleAgent


class ProviderTests(unittest.TestCase):
    def test_proposal_and_reflection_use_separate_efforts_and_json_context(self):
        requests = []
        responses = iter([
            {"action": "request_data", "request": {"fields": ["sequence"],
                                                   "reason": "Need event order"}},
            {"lesson": "The measured score did not improve"},
        ])

        def fake_open(request, timeout):
            requests.append((request, timeout))
            content = json.dumps(next(responses))
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

        agent = OpenAICompatibleAgent("https://api.example.test/v1", "secret-key", "model-x",
                                      thinking="enabled", iteration_effort="high",
                                      review_effort="max")
        context = {"task": {"objective": {"metric": "score", "direction": "max"},
                            "fields": ["age"], "constraints": ["fixed split"]},
                   "catalog": {"families": [{"id": "feature_interactions"}]},
                   "applicability": [{"family_id": "sequence_ranking", "status": "needs_data"}],
                   "steps": [{"evaluation": {"score": 0.3}}]}
        observation = {"task": context["task"], "trial": {"evaluation": {"score": 0.2}}}

        with patch("model_evo_harness.provider.urlopen", side_effect=fake_open):
            self.assertEqual(agent.propose(context)["action"], "request_data")
            self.assertEqual(agent.reflect(observation)["lesson"], "The measured score did not improve")

        first_request, first_timeout = requests[0]
        second_request, _ = requests[1]
        self.assertEqual(first_request.full_url, "https://api.example.test/v1/chat/completions")
        self.assertEqual(first_request.get_method(), "POST")
        self.assertEqual(first_request.get_header("Authorization"), "Bearer secret-key")
        self.assertGreater(first_timeout, 0)
        first_body = json.loads(first_request.data)
        second_body = json.loads(second_request.data)
        self.assertEqual(first_body["reasoning_effort"], "high")
        self.assertEqual(second_body["reasoning_effort"], "max")
        self.assertEqual(first_body["thinking"], {"type": "enabled"})
        self.assertEqual(first_body["response_format"], {"type": "json_object"})
        self.assertEqual(first_body["model"], "model-x")
        self.assertIn('"feature_interactions"', first_body["messages"][1]["content"])
        self.assertIn('"score": 0.2', second_body["messages"][1]["content"])
        self.assertIn("falsification", first_body["messages"][0]["content"])
        self.assertIn("not a closed", first_body["messages"][0]["content"])
        self.assertIn("measured", second_body["messages"][0]["content"])

    def test_rejects_invalid_url_and_invalid_json_without_exposing_key(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleAgent("ftp://api.example.test/v1", "secret-key", "model-x")
        with self.assertRaises(ValueError):
            OpenAICompatibleAgent("https://user:pass@api.example.test/v1", "secret-key", "model-x")
        with self.assertRaises(ValueError):
            OpenAICompatibleAgent("https://api.example.test/v1?token=x", "secret-key", "model-x")
        with self.assertRaises(ValueError):
            OpenAICompatibleAgent("https://api.example.test:invalid/v1", "secret-key", "model-x")

        agent = OpenAICompatibleAgent("https://api.example.test/v1/chat/completions",
                                      "secret-key", "model-x", thinking="omit")
        with patch("model_evo_harness.provider.urlopen", return_value=io.BytesIO(b"not json")):
            with self.assertRaisesRegex(ValueError, "invalid JSON") as error:
                agent.propose({})
        self.assertNotIn("secret-key", str(error.exception))

        malformed = json.dumps({"choices": [{"message": {"content": "[]"}}]}).encode()
        with patch("model_evo_harness.provider.urlopen", return_value=io.BytesIO(malformed)):
            with self.assertRaisesRegex(ValueError, "JSON object"):
                agent.propose({})

        error = HTTPError(agent.url, 401, "secret-key", {}, None)
        with patch("model_evo_harness.provider.urlopen", side_effect=error):
            with self.assertRaises(RuntimeError) as raised:
                agent.propose({})
        self.assertNotIn("secret-key", str(raised.exception))


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        module = types.ModuleType("harness_cli_test_adapter")

        class Task:
            def snapshot(self):
                return {"task_id": "cli", "dataset_digest": "dataset-1", "stage": "ranking",
                        "objective": {"metric": "score", "direction": "max"},
                        "fields": ["age"], "capabilities": ["tabular_features"]}

            def baseline(self, trial_dir):
                return {"score": 0.1, "metrics": {"score": 0.1}}

            def evaluate(self, proposal, trial_dir):
                raise AssertionError("unexpected trial")

        class Agent:
            def propose(self, context):
                return {"action": "stop", "reason": "No trial to run"}

            def reflect(self, observation):
                return {"lesson": "unused"}

        module.Task = Task
        module.agent = Agent()
        self.module_patch = patch.dict(sys.modules, {module.__name__: module})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def test_run_imports_task_factory_and_local_agent(self):
        output = Path(self.temp.name) / "local"
        result = cli.main(["run", "--task", "harness_cli_test_adapter:Task",
                           "--agent", "harness_cli_test_adapter:agent", "--output", str(output),
                           "--max-steps", "1"])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads((output / "journal.json").read_text())["status"], "stopped")
        resumed = cli.main(["run", "--task", "harness_cli_test_adapter:Task",
                            "--agent", "harness_cli_test_adapter:agent", "--output", str(output),
                            "--max-steps", "1", "--resume"])
        self.assertEqual(resumed, 0)
        with self.assertRaises(SystemExit) as error:
            cli.main(["run", "--task", "harness_cli_test_adapter:Task",
                      "--agent", "harness_cli_test_adapter:agent", "--output", str(output),
                      "--max-steps", "1"])
        self.assertEqual(error.exception.code, 1)

    def test_run_reads_provider_config_and_env_key(self):
        config = Path(self.temp.name) / "agent.json"
        config.write_text(json.dumps({"provider_url": "https://api.example.test/v1",
                                      "api_key_env": "MODEL_EVO_TEST_KEY", "model": "model-x",
                                      "thinking": "enabled", "iteration_effort": "high",
                                      "review_effort": "max"}))
        output = Path(self.temp.name) / "provider"

        def fake_open(request, timeout):
            body = json.loads(request.data)
            self.assertEqual(request.get_header("Authorization"), "Bearer key-from-env")
            self.assertEqual(body["reasoning_effort"], "high")
            answer = {"action": "request_data", "request": {"fields": ["sequence"],
                                                         "reason": "Need event order"}}
            return io.BytesIO(json.dumps({"choices": [{"message": {
                "content": json.dumps(answer)}}]}).encode())

        with patch.dict(os.environ, {"MODEL_EVO_TEST_KEY": "key-from-env"}), \
             patch("model_evo_harness.provider.urlopen", side_effect=fake_open):
            result = cli.main(["run", "--task", "harness_cli_test_adapter:Task",
                               "--agent-config", str(config), "--output", str(output),
                               "--max-steps", "2"])
            self.assertEqual(os.environ["MODEL_EVO_TEST_KEY"], "key-from-env")
        self.assertEqual(result, 0)
        self.assertEqual(json.loads((output / "journal.json").read_text())["status"], "needs_data")

    def test_key_is_hidden_during_task_execution_and_restored_for_next_run(self):
        config = Path(self.temp.name) / "agent.json"
        config.write_text(json.dumps({"provider_url": "https://api.example.test/v1",
                                      "api_key_env": "MODEL_EVO_TEST_KEY", "model": "model-x"}))

        def fake_run(task, agent, **kwargs):
            self.assertNotIn("MODEL_EVO_TEST_KEY", os.environ)
            return {"status": "stopped", "best_id": "baseline", "steps": []}

        with patch.dict(os.environ, {"MODEL_EVO_TEST_KEY": "key-from-env"}), \
             patch.object(cli, "run_search", side_effect=fake_run) as run:
            for index in range(2):
                self.assertEqual(cli.main(["run", "--task", "harness_cli_test_adapter:Task",
                                           "--agent-config", str(config), "--output",
                                           str(Path(self.temp.name) / str(index)), "--max-steps", "1"]), 0)
                self.assertEqual(os.environ["MODEL_EVO_TEST_KEY"], "key-from-env")
            self.assertEqual(run.call_count, 2)

    def test_catalog_check_reports_and_fails_for_unmapped_path(self):
        self.assertEqual(cli.main(["catalog-check"]), 0)
        with patch.object(cli, "coverage_report", return_value={"source_tree_sha": "abc",
                "document_count": 1, "model_count": 1, "unmapped_docs": ["doc.html"],
                "unmapped_models": [], "unknown_docs": [], "unknown_models": [],
                "duplicate_docs": [], "duplicate_models": []}):
            self.assertEqual(cli.main(["catalog-check"]), 1)
        with patch.object(cli, "coverage_report", return_value={"unmapped_support": ["module.py"]}):
            self.assertEqual(cli.main(["catalog-check"]), 1)


if __name__ == "__main__":
    unittest.main()
