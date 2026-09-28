"""Explicit composition root; training_core never imports business tasks."""


def resolve_task(task_id):
    if task_id == "text_classification":
        from training_tasks.text_classification import TextClassificationTask
        return TextClassificationTask()
    if task_id == "upload":
        from training_tasks.upload.adapter import UploadTask
        return UploadTask()
    raise ValueError(f"Unsupported task_id: {task_id}")


def main(argv=None):
    from training_core.cli import main as run_cli
    return run_cli(resolve_task, argv)


if __name__ == "__main__":
    raise SystemExit(main())
