"""训练、评估、研判、转换 CLI。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from training_core.artifacts import guard_output

from upload_judge.convert import convert_file, convert_http_dump
from upload_judge.acceptance import run_acceptance
from upload_judge.decision import Thresholds
from upload_judge.judge import UploadJudge
from upload_judge.metrics import build_evaluation, to_markdown
from upload_judge.ml_model import UploadClassifier
from upload_judge.schemas import UploadEvent


def load_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else [value]


def load_events(path: Path, labeled: bool = False) -> tuple[list[UploadEvent], list[str]]:
    events: list[UploadEvent] = []
    labels: list[str] = []
    for raw in load_records(path):
        item = dict(raw)
        label = str(item.pop("label", item.pop("gold_verdict", "")))
        item.pop("raw", None)
        events.append(UploadEvent.model_validate(item))
        if labeled:
            if not label:
                raise ValueError(f"缺少 label: {item.get('event_id')}")
            labels.append(label)
    return events, labels


def cmd_train(args: argparse.Namespace) -> int:
    manifest_path = Path(args.output).with_suffix(".manifest.json")
    _guard_outputs(Path(args.output), manifest_path)
    data_path = Path(args.data)
    events, labels = load_events(data_path, labeled=True)
    classifier = UploadClassifier(
        backend="logistic" if args.model_type == "b1" else "lightgbm",
        thresholds=Thresholds(args.confirmed_threshold, args.likely_threshold, args.failed_threshold),
        random_state=args.seed,
    )
    info = classifier.fit(events, labels)
    classifier.manifest["training_file"] = str(data_path.resolve())
    classifier.manifest["training_file_sha256"] = _sha256(data_path)
    classifier.save(args.output)
    _write_new_text(manifest_path, json.dumps(classifier.manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"saved": str(args.output), "manifest": str(manifest_path), **info}, ensure_ascii=False, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    _guard_outputs(*(Path(path) for path in (args.json_out, args.md_out) if path))
    events, labels = load_events(Path(args.data), labeled=True)
    judge = UploadJudge(model_path=args.model)
    predictions = [item.verdict.value for item in judge.judge_batch(events)]
    metrics = build_evaluation(events, labels, predictions, mode=args.mode)
    if args.json_out:
        _write_new_text(Path(args.json_out), json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    if args.md_out:
        _write_new_text(Path(args.md_out), to_markdown(metrics))
    summary_keys = (
        ("total", "accuracy_4", "semantic_confirmed_precision", "failed_to_success", "hard_gates")
        if args.mode == "semantic"
        else ("total", "confirmed_upload_success_precision", "failed_precision", "unknown_rate", "false_success_rate", "hard_gates")
    )
    print(json.dumps({key: metrics[key] for key in summary_keys}, ensure_ascii=False, indent=2))
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    if args.output:
        _guard_outputs(Path(args.output))
    events, _ = load_events(Path(args.input))
    output = [item.to_dict() for item in UploadJudge(model_path=args.model).judge_batch(events)]
    text = json.dumps(output[0] if len(output) == 1 else output, ensure_ascii=False, indent=2)
    if args.output:
        _write_new_text(Path(args.output), text + "\n")
    else:
        print(text)
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    _guard_outputs(Path(args.output))
    info = convert_file(Path(args.input), Path(args.output), args.label)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


def cmd_convert_alert(args: argparse.Namespace) -> int:
    _guard_outputs(Path(args.output))
    raw = Path(args.input).read_bytes()
    event = convert_http_dump(raw, event_id=args.event_id)
    out = Path(args.output)
    _write_new_text(out, json.dumps(event, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(event, ensure_ascii=False, indent=2))
    return 0


def cmd_verify_acceptance(args: argparse.Namespace) -> int:
    outputs = [Path(args.out), Path(args.json_out)]
    if not args.smoke:
        outputs.append(Path(args.json_out).parent / ".acceptance_v2_consumed.json")
    _guard_outputs(*outputs)
    report = run_acceptance(
        Path(args.model), Path(args.gold), Path(args.tdp), Path(args.out), Path(args.json_out), smoke=args.smoke
    )
    print(json.dumps({
        "mode": report["mode"],
        "status": report["status"],
        "all_gates_passed": report["all_gates_passed"],
    }, ensure_ascii=False, indent=2))
    return 0 if args.smoke or report["all_gates_passed"] else 2


def _guard_outputs(*paths: Path) -> None:
    """Check every destination before training or a multi-output operation starts."""
    resolved = [guard_output(path) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Output paths must be distinct")


def _write_new_text(path: Path, text: str) -> None:
    target = guard_output(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="upload-judge")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("-d", "--data", required=True)
    train.add_argument("-o", "--output", required=True)
    train.add_argument("--model-type", choices=["b1", "b2"], default="b2")
    train.add_argument("--cv", type=int, default=0, help="兼容计划书；v1 固定不做网格 CV")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--confirmed-threshold", type=float, default=0.85)
    train.add_argument("--likely-threshold", type=float, default=0.60)
    train.add_argument("--failed-threshold", type=float, default=0.85)
    train.set_defaults(func=cmd_train)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("-d", "--data", required=True)
    evaluate.add_argument("-m", "--model", required=True)
    evaluate.add_argument("--json-out")
    evaluate.add_argument("--md-out")
    evaluate.add_argument("--mode", choices=["semantic", "tdp_v1", "tdp_v2"], default="tdp_v2")
    evaluate.set_defaults(func=cmd_evaluate)
    judge = commands.add_parser("judge")
    judge.add_argument("-i", "--input", required=True)
    judge.add_argument("-m", "--model", required=True)
    judge.add_argument("-o", "--output")
    judge.set_defaults(func=cmd_judge)
    convert = commands.add_parser("convert")
    convert.add_argument("-i", "--input", required=True)
    convert.add_argument("-o", "--output", required=True)
    convert.add_argument("--label", required=True, choices=["confirmed_upload_success", "failed"])
    convert.set_defaults(func=cmd_convert)
    convert_alert = commands.add_parser("convert-alert", help="原始 HTTP 请求/响应文本 → 模型可研判 JSON")
    convert_alert.add_argument("-i", "--input", required=True, help="原始告警/抓包文本文件")
    convert_alert.add_argument("-o", "--output", required=True, help="输出的标准事件 JSON 路径")
    convert_alert.add_argument("--event-id", default="", help="事件 ID（默认按请求内容哈希生成）")
    convert_alert.set_defaults(func=cmd_convert_alert)
    acceptance = commands.add_parser("verify-acceptance")
    acceptance.add_argument("-m", "--model", required=True)
    acceptance.add_argument("--gold", required=True)
    acceptance.add_argument("--tdp", required=True)
    acceptance.add_argument("--out", required=True)
    acceptance.add_argument("--json-out", required=True)
    acceptance.add_argument("--smoke", action="store_true")
    acceptance.set_defaults(func=cmd_verify_acceptance)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return args.func(args)
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
