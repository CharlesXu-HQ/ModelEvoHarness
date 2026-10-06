"""Command line runner for importable host tasks and Agents."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path

from .catalog import coverage_report, load_catalog
from .engine import run_search
from .provider import OpenAICompatibleAgent


def _import_adapter(spec: str, methods: tuple[str, ...]) -> object:
    module_name, separator, symbol = spec.partition(":")
    if not module_name or not separator or not symbol:
        raise ValueError("adapter must be module:symbol")
    value = importlib.import_module(module_name)
    for name in symbol.split("."):
        value = getattr(value, name)
    if inspect.isclass(value) or (callable(value) and not all(hasattr(value, m) for m in methods)):
        value = value()
    if not all(callable(getattr(value, method, None)) for method in methods):
        raise ValueError(f"{spec} does not provide {', '.join(methods)}")
    return value


def _provider_agent(config_path: Path) -> tuple[OpenAICompatibleAgent, str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "api_key" in config:
        raise ValueError("agent config must be an object using api_key_env")
    env_name = config.get("api_key_env")
    if not isinstance(env_name, str) or not env_name:
        raise ValueError("agent config needs api_key_env")
    api_key = os.environ.get(env_name)
    if not api_key:
        raise ValueError(f"environment variable {env_name} is unset or empty")
    return (OpenAICompatibleAgent(config.get("provider_url"), api_key, config.get("model"),
                                  thinking=config.get("thinking", "omit"),
                                  iteration_effort=config.get("iteration_effort", "high"),
                                  review_effort=config.get("review_effort", "max")), env_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model-evo-harness")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a task with an importable Agent or provider")
    run.add_argument("--task", required=True, metavar="module:symbol")
    agent_choice = run.add_mutually_exclusive_group(required=True)
    agent_choice.add_argument("--agent", metavar="module:symbol")
    agent_choice.add_argument("--agent-config", type=Path, metavar="config.json")
    run.add_argument("--output", type=Path, required=True, metavar="DIR")
    run.add_argument("--max-steps", type=int, required=True, metavar="N")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--require-verified-implementation", action="store_true",
                     help="promote only candidates with a host-verified implementation")
    commands.add_parser("catalog-check", help="check research source coverage")
    args = parser.parse_args(argv)

    if args.command == "catalog-check":
        report = coverage_report(load_catalog())
        print(json.dumps(report, indent=2, ensure_ascii=False))
        issue_keys = tuple(key for key in report if key.startswith(
            ("unmapped_", "unknown_", "duplicate_", "misassigned_", "missing_")))
        return int(any(report[key] for key in issue_keys))

    try:
        task = _import_adapter(args.task, ("snapshot", "baseline", "evaluate"))
        agent, key_env = ((_import_adapter(args.agent, ("propose", "reflect")), None)
                          if args.agent else _provider_agent(args.agent_config))
        # Generated candidate subprocesses must not inherit the provider credential.
        key = os.environ.pop(key_env) if key_env else None
        try:
            result = run_search(task, agent, output=args.output, catalog=load_catalog(),
                                max_steps=args.max_steps, resume=args.resume,
                                require_verified_implementation=args.require_verified_implementation)
        finally:
            if key_env:
                os.environ[key_env] = key
    except (ValueError, FileNotFoundError, FileExistsError, ImportError, AttributeError) as error:
        parser.exit(1, f"model-evo-harness: {error}\n")
    print(json.dumps({"status": result["status"], "best_id": result["best_id"],
                      "steps": len(result["steps"]),
                      "journal": str(args.output / "journal.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
