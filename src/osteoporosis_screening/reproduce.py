"""Run the research workflow in a new directory with immutable provenance records."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def verify_inputs(root: Path, expected: dict[str, str]) -> None:
    mismatches = [
        name
        for name, digest in expected.items()
        if not (root / name).is_file() or sha256(root / name) != digest
    ]
    if mismatches:
        raise ValueError("Missing or changed reference inputs: " + ", ".join(mismatches))


def verify_run(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Run is not complete")
    verify_inputs(root / "source", manifest["source_sha256"])
    verify_inputs(root / "configs", manifest["config_sha256"])
    verify_inputs(root, manifest["environment_files_sha256"])
    for name, expected in manifest["input_sha256"].items():
        verify_inputs(root / "data" / name, expected)
    for name, expected in manifest["output_sha256"].items():
        verify_inputs(root / name, expected)


def verify_environment(source: Path) -> None:
    version_file = source / ".python-version"
    if version_file.exists() and platform.python_version() != version_file.read_text().strip():
        raise ValueError("Use the Python version specified in .python-version")
    lock = source / "requirements-lock.txt"
    if not lock.exists():
        return
    mismatches = []
    for line in lock.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "not installed"
        if actual != expected:
            mismatches.append(f"{name}: expected {expected}, found {actual}")
    if mismatches:
        raise ValueError("Install requirements-lock.txt first: " + "; ".join(mismatches))


def create_run(source: Path, destination: Path, data_root: Path) -> dict:
    source, destination, data_root = source.resolve(), destination.resolve(), data_root.resolve()
    if destination.exists():
        raise FileExistsError(f"Run directory already exists: {destination}")
    if not (source / "configs/model_config.yaml").is_file():
        raise FileNotFoundError("--source must be the repository containing configs and src")
    for protected in (source / "src", source / "configs", data_root):
        if destination.is_relative_to(protected):
            raise ValueError("Run directory must be outside source/config/data inputs")
    destination.mkdir(parents=True)
    shutil.copytree(source / "configs", destination / "configs")
    shutil.copytree(
        source / "src",
        destination / "source",
        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
    )
    for name in ("pyproject.toml", "requirements-lock.txt", ".python-version"):
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    for name in ("raw_xpt", "raw_knhanes"):
        path = data_root / name
        if path.exists():
            shutil.copytree(path, destination / "data" / name)
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "created",
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": dict(
            sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
        ),
        "source_sha256": inventory(destination / "source"),
        "config_sha256": inventory(destination / "configs"),
        "environment_files_sha256": {
            name: sha256(destination / name)
            for name in ("pyproject.toml", "requirements-lock.txt", ".python-version")
            if (destination / name).exists()
        },
        "stages": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, help="Directory containing raw_xpt/raw_knhanes")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--external", action="store_true")
    parser.add_argument("--experiments", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing completed run instead of executing it",
    )
    parser.add_argument(
        "--expected-inputs",
        type=Path,
        help="Optional JSON mapping of data-relative paths to SHA-256 hashes",
    )
    args = parser.parse_args()
    root = args.output.resolve()
    if args.verify:
        verify_run(root)
        print("Completed run matches its recorded hashes.")
        return
    if args.prepare_only and args.experiments:
        parser.error("--experiments cannot be combined with --prepare-only")
    verify_environment(args.source)
    manifest = create_run(args.source, root, args.data_root or args.source / "data")
    manifest_path = root / "manifest.json"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(root / "source"),
            "PYTHONHASHSEED": "42",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
        }
    )
    commands = ["download"] if args.download else []
    commands += ["prepare"] if args.prepare_only else ["run"]
    if args.external:
        commands += ["knhanes-harmonize" if args.prepare_only else "knhanes-validate"]
    if args.experiments and not args.prepare_only:
        commands += ["feature-combination-experiment", "target-definition-experiment"]
    try:
        manifest["status"] = "running"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for command in commands:
            if command != "download" and args.expected_inputs:
                expected = json.loads(args.expected_inputs.read_text(encoding="utf-8"))
                if not args.external:
                    expected = {k: v for k, v in expected.items() if k.startswith("raw_xpt/")}
                verify_inputs(root / "data", expected)
                manifest["reference_inputs_verified"] = True
                manifest["reference_inputs_sha256"] = sha256(args.expected_inputs)
            print(f"Running {command}; log: {root / (command + '.log')}", flush=True)
            with (root / f"{command}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "osteoporosis_screening.cli",
                        command,
                        "--root",
                        str(root),
                    ],
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            manifest["stages"].append({"command": command, "returncode": result.returncode})
            if result.returncode:
                raise RuntimeError(f"{command} failed; see {root / (command + '.log')}")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        manifest["input_sha256"] = {
            name: inventory(root / "data" / name) for name in ("raw_xpt", "raw_knhanes")
        }
        manifest["output_sha256"] = {
            name: inventory(root / name) for name in ("data/processed", "artifacts", "reports")
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
