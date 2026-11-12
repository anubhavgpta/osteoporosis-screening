from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import pandas as pd

_BONE_RISK_MEDICATION_TERMS = {
    "systemic_glucocorticoid_medication": (
        "prednisone",
        "prednisolone",
        "methylprednisolone",
        "dexamethasone",
        "cortisone",
    ),
    "anticonvulsant_medication": (
        "phenytoin",
        "fosphenytoin",
        "carbamazepine",
        "oxcarbazepine",
        "phenobarbital",
        "primidone",
        "valproic acid",
        "divalproex",
        "topiramate",
        "gabapentin",
        "pregabalin",
    ),
    "aromatase_inhibitor_medication": ("anastrozole", "letrozole", "exemestane"),
    "proton_pump_inhibitor_medication": (
        "omeprazole",
        "esomeprazole",
        "lansoprazole",
        "dexlansoprazole",
        "pantoprazole",
        "rabeprazole",
    ),
}


def _read_xport(path: Path) -> pd.DataFrame:
    try:
        return pd.read_sas(path, format="xport", encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_sas(path, format="xport", encoding="latin-1")


def read_prescription_component(raw_dir: Path, file_name: str) -> pd.DataFrame:
    """Aggregate the drug-level RXQ file to leakage-safe participant flags.

    NHANES releases one row per reported prescription. Only medication classes
    that can raise bone risk are retained. Osteoporosis treatments and the
    reported indication are deliberately not represented because they can reveal
    an existing diagnosis. RXDDAYS supplies duration, not dose.
    """
    path = raw_dir / f"{file_name}.XPT"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the download command first.")
    frame = _read_xport(path)
    required = {"SEQN", "RXDUSE", "RXDDRUG", "RXDDAYS", "RXDCOUNT"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{file_name} is missing prescription columns: {missing}")
    frame = frame[list(required)].rename(
        columns={
            "SEQN": "participant_id",
            "RXDUSE": "prescription_use_raw",
            "RXDDRUG": "prescription_generic_name",
            "RXDDAYS": "prescription_days",
            "RXDCOUNT": "prescription_count",
        }
    )
    frame["prescription_generic_name"] = (
        frame["prescription_generic_name"].astype("string").str.lower().str.strip()
    )
    frame["known_drug"] = frame["prescription_generic_name"].notna() & ~frame[
        "prescription_generic_name"
    ].isin(["55555", "77777", "99999"])
    for output, terms in _BONE_RISK_MEDICATION_TERMS.items():
        pattern = "|".join(re.escape(term) for term in terms)
        frame[output] = frame["prescription_generic_name"].str.contains(
            pattern, case=False, na=False
        )

    rows = []
    flag_columns = list(_BONE_RISK_MEDICATION_TERMS)
    for participant_id, group in frame.groupby("participant_id", sort=False):
        use_yes = group["prescription_use_raw"].eq(1).any()
        use_no = group["prescription_use_raw"].eq(2).any()
        complete_names = bool(use_yes and group.loc[group["prescription_use_raw"].eq(1), "known_drug"].all())
        record: dict[str, object] = {
            "participant_id": participant_id,
            "prescription_count": group["prescription_count"].dropna().max(),
        }
        for column in flag_columns:
            if group[column].any():
                record[column] = "yes"
            elif use_no or complete_names:
                record[column] = "no"
            else:
                record[column] = None
        risk_mask = group[flag_columns].any(axis=1)
        risk_days = group.loc[risk_mask, "prescription_days"]
        if risk_days.ge(90).any():
            record["long_term_bone_risk_medication"] = "yes"
        elif risk_mask.any() and risk_days.notna().any():
            record["long_term_bone_risk_medication"] = "no"
        elif use_no or (complete_names and not risk_mask.any()):
            record["long_term_bone_risk_medication"] = "no"
        else:
            record["long_term_bone_risk_medication"] = None
        rows.append(record)
    result = pd.DataFrame(rows)
    if result["participant_id"].duplicated().any():
        raise ValueError(f"{file_name} prescription aggregation is not one row per participant")
    return result


def _cycle_file_name(template: str, suffix: str) -> str:
    stem, _, _ = template.rpartition("_")
    return f"{stem}_{suffix}"


def download_files(raw_dir: Path, variable_config: dict, overwrite: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for cycle in variable_config["cycles"]:
        for template, details in variable_config["files"].items():
            name = _cycle_file_name(template, cycle["suffix"])
            destination = raw_dir / f"{name}.XPT"
            if destination.exists() and not overwrite:
                continue
            partial = destination.with_suffix(".XPT.part")
            partial.unlink(missing_ok=True)
            url = (
                "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/"
                f"{cycle['begin_year']}/DataFiles/{name}.XPT"
            )
            try:
                urlretrieve(url, partial)
            except URLError as error:
                if os.name != "nt":
                    if details.get("required", False):
                        raise
                    continue
                completed = subprocess.run(
                    [
                        "curl.exe",
                        "--fail",
                        "--location",
                        "--ssl-revoke-best-effort",
                        "--silent",
                        "--show-error",
                        "--output",
                        str(partial),
                        url,
                    ],
                    check=False,
                )
                if completed.returncode:
                    partial.unlink(missing_ok=True)
                    if details.get("required", False):
                        raise RuntimeError(f"Failed to download required {url}") from error
                    continue
            partial.replace(destination)

    from .reproduce import sha256

    records = []
    for cycle in variable_config["cycles"]:
        for template, details in variable_config["files"].items():
            name = _cycle_file_name(template, cycle["suffix"]) + ".XPT"
            path = raw_dir / name
            records.append({
                "file": name,
                "required": details.get("required", False),
                "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/"
                       f"{cycle['begin_year']}/DataFiles/{name}",
                "sha256": sha256(path) if path.exists() else None,
            })
    (raw_dir / "download_manifest.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )


def read_component(
    raw_dir: Path,
    file_name: str,
    columns: dict[str, str],
    allow_missing_columns: bool = False,
) -> pd.DataFrame:
    path = raw_dir / f"{file_name}.XPT"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the download command first.")
    frame = _read_xport(path)
    # pandas exposes SAS/XPORT numeric system-missing as a tiny positive sentinel.
    numeric = frame.select_dtypes(include="number").columns
    # CDC laboratory comment fields use a real zero for "at or above LOD".
    # Preserve those zeros before applying the general SAS-missing cleanup.
    lod_zero_masks = {
        source: frame[source].abs().lt(1e-70)
        for source, target in columns.items()
        if target.endswith("_lod_code") and source in numeric
    }
    frame[numeric] = frame[numeric].mask(frame[numeric].abs() < 1e-70)
    for source, zero_mask in lod_zero_masks.items():
        frame.loc[zero_mask, source] = 0.0
    missing = sorted(set(columns) - set(frame.columns))
    if missing and not allow_missing_columns:
        raise ValueError(f"{file_name} is missing configured columns: {missing}")
    if missing:
        absent = pd.DataFrame(float("nan"), index=frame.index, columns=missing)
        frame = pd.concat([frame, absent], axis=1)
    frame = frame[list(columns)].rename(columns=columns)
    if frame["participant_id"].duplicated().any():
        raise ValueError(f"{file_name} contains duplicate participant_id values")
    return frame


def _load_cycle(
    raw_dir: Path, variable_config: dict, cycle: dict, cycle_index: int
) -> tuple[pd.DataFrame, list[dict]]:
    files = variable_config["files"]
    suffix = cycle["suffix"]
    anchor_name = _cycle_file_name("DXXFEM_H", suffix)
    anchor = read_component(raw_dir, anchor_name, files["DXXFEM_H"]["columns"])
    merged = anchor
    flow = [
        {
            "cycle": cycle["label"],
            "stage": "femur_dxa_anchor",
            "participants": len(merged),
        }
    ]

    join_order = [
        "DXXSPN_H",
        "DEMO_H",
        "BMX_H",
        "DR1TOT_H",
        "PAQ_H",
        "ALQ_H",
        "SMQ_H",
        "OSQ_H",
        "RHQ_H",
        "MGX_H",
        "BIOPRO_H",
        "FOLFMS_H",
        "VID_H",
        "CBC_H",
        "GHB_H",
        "TCHOL_H",
        "DIQ_H",
        "MCQ_H",
        "PFQ_H",
        "RXQ_RX_H",
    ]
    for template in join_order:
        name = _cycle_file_name(template, suffix)
        path = raw_dir / f"{name}.XPT"
        details = files[template]
        if not path.exists() and not details.get("required", False):
            flow.append(
                {
                    "cycle": cycle["label"],
                    "stage": f"skip_missing_{name}",
                    "participants": len(merged),
                }
            )
            continue
        component = (
            read_prescription_component(raw_dir, name)
            if details.get("one_to_many", False)
            else read_component(
                raw_dir,
                name,
                details["columns"],
                allow_missing_columns=details.get("allow_missing_columns", False),
            )
        )
        merged = merged.merge(component, on="participant_id", how="left", validate="one_to_one")
        matched = int(merged[component.columns.difference(["participant_id"])].notna().any(axis=1).sum())
        flow.append(
            {
                "cycle": cycle["label"],
                "stage": f"left_join_{name}",
                "participants": len(merged),
                "matched": matched,
            }
        )
    merged["survey_cycle"] = cycle["label"]
    merged["survey_stratum"] = merged["survey_stratum"] + cycle_index * 1000
    cycles_pooled = len(variable_config["cycles"])
    merged["mec_weight"] = merged["mec_weight"] / cycles_pooled
    merged["dietary_day1_weight"] = merged["dietary_day1_weight"] / cycles_pooled
    return merged, flow


def load_and_merge(raw_dir: Path, variable_config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Harmonize cycles, anchor each on femur DXA, then pool participants."""
    pooled = []
    flow = []
    for cycle_index, cycle in enumerate(variable_config["cycles"]):
        frame, cycle_flow = _load_cycle(raw_dir, variable_config, cycle, cycle_index)
        pooled.append(frame)
        flow.extend(cycle_flow)

    merged = pd.concat(pooled, ignore_index=True, sort=False)
    if merged["participant_id"].duplicated().any():
        raise ValueError("participant_id is not unique after pooling NHANES cycles")
    flow.append({"cycle": "pooled", "stage": "pooled_cycles", "participants": len(merged)})
    return merged, pd.DataFrame(flow)
