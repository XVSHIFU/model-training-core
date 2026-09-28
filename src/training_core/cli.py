"""Thin command layer; task resolution is supplied by the composition root."""
import argparse
import json
from pathlib import Path
import sys

from training_core.contracts import DatasetSpec
from training_core.runner import Workflow, load_config


def build_parser():
    parser = argparse.ArgumentParser(prog="model-workflow", description="Explicit, reproducible model training workflow")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-data", "train"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
    for name in ("predict", "evaluate"):
        command = commands.add_parser(name)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--artifact", help="Run directory or its manifest.json")
        source.add_argument("--config", help="Legacy model config containing artifact_path")
        command.add_argument("--run-root", help="Directory for a new run (default: repository runs/)")
        if name == "predict":
            command.add_argument("--input", required=True)
        else:
            command.add_argument("--data", required=True)
            command.add_argument("--purpose", required=True, choices=["development", "historical_regression", "independent_test"])
            command.add_argument("--provenance", default="")
            command.add_argument("--independent", action="store_true", help="Attest independent collection; overlap checks still apply")
            command.add_argument("--mode", help="Task-specific evaluation mode")
    return parser


def main(resolve_task, argv=None):
    args = build_parser().parse_args(argv)
    workflow = Workflow(resolve_task)
    try:
        if args.command in ("validate-data", "train"):
            config = load_config(args.config)
            result = workflow.validate_data(config) if args.command == "validate-data" else workflow.train(config)
        else:
            options = {"artifact": args.artifact, "config": load_config(args.config) if args.config else None,
                       "run_root": args.run_root}
            if args.command == "predict":
                result = workflow.predict(args.input, **options)
            else:
                spec = DatasetSpec(path=str(Path(args.data).resolve()), purpose=args.purpose,
                                   provenance=args.provenance, independent=args.independent)
                result = workflow.evaluate(spec, mode=args.mode, **options)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted", "error": "Interrupted by user"}), file=sys.stderr)
        return 130
