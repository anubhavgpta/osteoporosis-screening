import json

from osteoporosis_screening import ingest
from osteoporosis_screening.reproduce import sha256


def test_download_manifest_hashes_cached_and_downloaded_inputs(tmp_path, monkeypatch):
    existing = tmp_path / "DEMO_E.XPT"
    existing.write_bytes(b"cached data")
    calls = []

    def download(url, destination):
        calls.append(url)
        destination.write_bytes(b"new data")

    monkeypatch.setattr(ingest, "urlretrieve", download)
    ingest.download_files(
        tmp_path,
        {
            "cycles": [{"suffix": "E", "begin_year": 2007}],
            "files": {"DEMO_H": {"required": True}, "BMX_H": {"required": True}},
        },
    )
    records = json.loads((tmp_path / "download_manifest.json").read_text())
    assert len(calls) == 1
    assert len(records) == 2
    assert all(row["sha256"] == sha256(tmp_path / row["file"]) for row in records)
    assert existing.read_bytes() == b"cached data"
