import json
from pathlib import Path

import pytest

from osteoporosis_screening.reproduce import (
    create_run,
    inventory,
    verify_environment,
    verify_inputs,
    verify_run,
)


def test_isolated_run_preserves_inputs_and_refuses_overwrite(tmp_path):
    source = tmp_path / "repo"
    (source / "configs").mkdir(parents=True)
    (source / "configs/model_config.yaml").write_text("seed: 42")
    (source / "src").mkdir()
    (source / "src/example.py").write_text("VALUE = 1")
    raw = source / "data/raw_xpt"
    raw.mkdir(parents=True)
    (raw / "example.XPT").write_bytes(b"survey")
    output = tmp_path / "run"
    manifest = create_run(source, output, source / "data")
    assert manifest["source_sha256"] == inventory(source / "src")
    assert inventory(output / "data/raw_xpt") == inventory(raw)
    (output / "configs/model_config.yaml").write_text("seed: 7")
    assert (source / "configs/model_config.yaml").read_text() == "seed: 42"
    with pytest.raises(FileExistsError):
        create_run(source, output, source / "data")


def test_inventory_detects_changed_bytes(tmp_path):
    path = Path(tmp_path) / "input"
    path.write_bytes(b"before")
    before = inventory(tmp_path)
    path.write_bytes(b"after")
    assert before != inventory(tmp_path)
    with pytest.raises(ValueError, match="Missing or changed"):
        verify_inputs(tmp_path, before)
    verify_inputs(tmp_path, inventory(tmp_path))


def test_verify_run_rejects_incomplete_or_modified_artifacts(tmp_path):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"original")
    manifest = {
        "status": "failed",
        "source_sha256": {},
        "config_sha256": {},
        "environment_files_sha256": {},
        "input_sha256": {},
        "output_sha256": {".": inventory(tmp_path)},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not complete"):
        verify_run(tmp_path)
    manifest["status"] = "complete"
    path.write_text(json.dumps(manifest))
    verify_run(tmp_path)
    artifact.write_bytes(b"modified")
    with pytest.raises(ValueError, match="Missing or changed"):
        verify_run(tmp_path)


def test_environment_rejects_dependency_drift(tmp_path):
    (tmp_path / "requirements-lock.txt").write_text("pytest==0.0.0\n")
    with pytest.raises(ValueError, match="Install requirements-lock"):
        verify_environment(tmp_path)
