#!/usr/bin/env python3
"""Rewrite image paths in the DyRef RL train and test JSONL files."""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Callable


TARGET_KEYS = ("image", "edit_image")


def convert_test_path(path: str) -> str:
    if path.startswith("test_set/"):
        return "OmniRef-Bench/" + path.removeprefix("test_set/")
    if path.startswith("OmniRef-Bench/"):
        return path
    raise ValueError(f"unexpected test path prefix: {path!r}")


def convert_train_path(path: str) -> str:
    if path.startswith(("data_4500/", "data_2500/")):
        return "DyRef_training/" + path
    if path.startswith(("DyRef_training/data_4500/", "DyRef_training/data_2500/")):
        return path
    raise ValueError(f"unexpected train path prefix: {path!r}")


def transform_record(
    record: dict,
    convert_path: Callable[[str], str],
    line_number: int,
) -> tuple[dict, int]:
    image = record.get("image")
    edit_images = record.get("edit_image")
    if not isinstance(image, str):
        raise TypeError(f"line {line_number}: 'image' must be a string")
    if not isinstance(edit_images, list) or not all(isinstance(path, str) for path in edit_images):
        raise TypeError(f"line {line_number}: 'edit_image' must be a list of strings")

    untouched_fields = {key: value for key, value in record.items() if key not in TARGET_KEYS}
    converted_image = convert_path(image)
    converted_edit_images = [convert_path(path) for path in edit_images]

    changed_paths = int(converted_image != image)
    changed_paths += sum(new != old for new, old in zip(converted_edit_images, edit_images))
    record["image"] = converted_image
    record["edit_image"] = converted_edit_images

    assert untouched_fields == {
        key: value for key, value in record.items() if key not in TARGET_KEYS
    }, f"line {line_number}: a non-path field was modified"
    return record, changed_paths


def convert_jsonl(
    path: Path,
    convert_path: Callable[[str], str],
    dry_run: bool,
) -> tuple[int, int]:
    rows = 0
    changed_paths = 0
    temporary_path = None

    try:
        with path.open("r", encoding="utf-8") as source, tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            for line_number, line in enumerate(source, start=1):
                record = json.loads(line)
                record, line_changes = transform_record(record, convert_path, line_number)
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                rows += 1
                changed_paths += line_changes

        if dry_run or changed_paths == 0:
            temporary_path.unlink()
        else:
            os.chmod(temporary_path, path.stat().st_mode)
            os.replace(temporary_path, path)
        return rows, changed_paths
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report changes without rewriting the JSONL files.",
    )
    args = parser.parse_args()

    dataset_dir = Path(__file__).resolve().parent
    conversions = (
        (dataset_dir / "test.jsonl", convert_test_path),
        (dataset_dir / "train.jsonl", convert_train_path),
    )
    for path, converter in conversions:
        rows, changed_paths = convert_jsonl(path, converter, args.dry_run)
        action = "would change" if args.dry_run else "changed"
        print(f"{path.name}: validated {rows} rows, {action} {changed_paths} paths")


if __name__ == "__main__":
    main()
