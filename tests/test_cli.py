from upload_judge.cli import build_parser


def test_required_commands_exist():
    parser = build_parser()
    for command in ("train", "evaluate", "judge", "convert", "verify-acceptance"):
        with __import__("pytest").raises(SystemExit) as exc:
            parser.parse_args([command, "--help"])
        assert exc.value.code == 0


def test_evaluate_defaults_to_v2_compatibility_mode():
    args = build_parser().parse_args(["evaluate", "-d", "data.json", "-m", "model.joblib"])
    assert args.mode == "tdp_v2"
