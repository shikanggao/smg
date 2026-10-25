#!/usr/bin/env python3
"""Freeze deadline truth before artificial background termination changes pressure."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT / "analyze_wait_deadline.py"
if not source.exists():
    source = ROOT / "analyze_deadline.py"
spec = importlib.util.spec_from_file_location("deadline_scores", source)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
DEADLINE = 300.0


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def natural_completion(row):
    return (
        d.truth(row.get("complete"))
        and d.truth(row.get("done_seen"))
        and d.truth(row.get("finish_reason_seen"))
        and not d.truth(row.get("cancelled"))
        and not row.get("error")
    )


def pressure_cutoff(requests, unload):
    """Controlled admission rejection never withdraws admitted background work."""
    candidates = []
    for row in requests:
        end = number(row.get("completed_elapsed_seconds"))
        if (
            not row["case"].startswith("background")
            or str(row["status"]) == "503"
            or end is None
            or end >= unload
            or natural_completion(row)
        ):
            continue
        error = row.get("error", "")
        if d.truth(row.get("cancelled")):
            reason = "background_cancellation"
        elif "timeout" in error.lower():
            reason = "background_client_timeout"
        elif error:
            reason = "background_error"
        else:
            reason = "background_incomplete_stream"
        candidates.append(
            dict(
                elapsed_seconds=end,
                reason=reason,
                request_id=row["opc_request_id"],
                status=str(row["status"]),
            )
        )
    candidates.sort(key=lambda row: (row["elapsed_seconds"], row["request_id"]))
    first = (
        candidates[0]
        if candidates
        else dict(elapsed_seconds=unload, reason="planned_unload", request_id=None)
    )
    return first, candidates


def timing(row, cutoff):
    start = number(row.get("started_elapsed_seconds"))
    end = number(row.get("completed_elapsed_seconds"))
    ttft = number(row.get("client_ttft", row.get("ttft_seconds")))
    text = start + ttft if start is not None and ttft is not None else None
    complete = d.truth(row.get("complete"))
    eligible = start is not None and start <= cutoff and str(row.get("status", "200")) != "503"
    before_end = eligible and complete and end is not None and end <= cutoff
    before_text = eligible and text is not None and text <= cutoff
    # Cancellation ends observation: a later clock crossing proves no outcome.
    observed_until = cutoff if end is None else min(end, cutoff)
    crossed = eligible and start + DEADLINE <= observed_until
    completion = (end - start >= DEADLINE) if before_end else True if crossed else None
    first_text = (ttft >= DEADLINE) if before_text else True if crossed else None
    present = eligible and (end is None or end >= cutoff)
    return dict(
        client_first_text_elapsed_seconds=text,
        client_first_text_observed_before_cutoff=before_text,
        completed_before_cutoff=before_end,
        first_text_pending_at_cutoff=present and not before_text,
        completion_pending_at_cutoff=present and not before_end,
        completion_deadline_at_cutoff=completion,
        first_text_deadline_at_cutoff=first_text,
    )


def boundary(rows, low, high):
    return [r for r in rows if r["complete"] and low <= r["e2e_seconds"] < high]


def score_at_cutoff(rows):
    return [
        dict(
            r,
            completion_late=r["completion_deadline_at_cutoff"],
            first_text_late=r["first_text_deadline_at_cutoff"],
        )
        for r in rows
    ]


def stratify(path):
    names = ["requests.tsv", "paired-deadline.json", "stages.json"]
    with (path / "requests.tsv").open() as file:
        requests = list(csv.DictReader(file, delimiter="\t"))
    rows = json.loads((path / "paired-deadline.json").read_text())
    stages = json.loads((path / "stages.json").read_text())
    unload = next(
        number(s.get("elapsed_seconds", s.get("start_elapsed_seconds")))
        for s in stages
        if s["name"] == "unload-background"
    )
    assert unload is not None
    raw_by_id = {r["opc_request_id"]: r for r in requests}
    assert len(raw_by_id) == len(requests), "Duplicate raw request IDs"
    paired_by_id = {r["request_id"]: r for r in rows}
    assert len(paired_by_id) == len(rows) and paired_by_id.keys() <= raw_by_id.keys()
    first, interventions = pressure_cutoff(requests, unload)
    cutoff = first["elapsed_seconds"]
    for row in rows:
        row["status"] = raw_by_id[row["request_id"]]["status"]
        row.update(timing(row, cutoff))
        # Compatibility keys keep their literal planned-unload meaning.
        planned = timing(row, unload)
        row.update(
            {
                key.replace("cutoff", "unload"): value
                for key, value in planned.items()
                if key != "client_first_text_elapsed_seconds"
            }
        )
    sentinels = [r for r in rows if r["case"].startswith("sentinel")]
    before_end = [r for r in sentinels if r["completed_before_cutoff"]]
    before_text = [r for r in sentinels if r["client_first_text_observed_before_cutoff"]]
    proof = [
        r
        for r in before_end
        if r["actual_ttft"] is not None and r["actual_ttft"] >= DEADLINE and r["client_ttft"] >= 330
    ]
    cutoff_rows = score_at_cutoff(sentinels)
    legacy = [dict(r, **timing(r, unload)) for r in sentinels]
    legacy_end = [r for r in legacy if r["completed_before_cutoff"]]
    legacy_text = [r for r in legacy if r["client_first_text_observed_before_cutoff"]]
    legacy_proof = [
        r
        for r in legacy_end
        if r["actual_ttft"] is not None and r["actual_ttft"] >= DEADLINE and r["client_ttft"] >= 330
    ]
    healthy_boundary = boundary(before_end, 240, 300)
    result = dict(
        schema_version=2,
        cutoff_elapsed_seconds=cutoff,
        cutoff_reason=first["reason"],
        cutoff_request_id=first["request_id"],
        unload_elapsed_seconds=unload,
        cutoff_precedes_planned_unload=cutoff < unload,
        artificial_background_terminations_before_unload=interventions,
        raw_request_count=len(requests),
        paired_request_count=len(rows),
        sentinel_count=len(sentinels),
        rejected_sentinel_ids=[
            r["opc_request_id"]
            for r in requests
            if r["case"].startswith("sentinel") and str(r["status"]) == "503"
        ],
        completed_paired_proof_before_cutoff=len(proof),
        paired_proof_before_cutoff_ids=[r["request_id"] for r in proof],
        maximum_smg_ttft_completed_before_cutoff=max(
            (r["actual_ttft"] for r in before_end if r["actual_ttft"] is not None), default=None
        ),
        before_cutoff_completion=d.score(before_end, "completion_late"),
        before_cutoff_first_text=d.score(before_text, "first_text_late"),
        completion_deadline_at_cutoff=d.score(cutoff_rows, "completion_late"),
        first_text_deadline_at_cutoff=d.score(cutoff_rows, "first_text_late"),
        healthy_boundary_completed_before_cutoff=d.score(healthy_boundary, "completion_late"),
        healthy_boundary_completed_before_cutoff_ids=[r["request_id"] for r in healthy_boundary],
        late_boundary_completed_before_cutoff=d.score(
            boundary(before_end, 300, 360), "completion_late"
        ),
        before_cutoff_complete_ids=[r["request_id"] for r in before_end],
        before_cutoff_first_text_ids=[r["request_id"] for r in before_text],
        completed_paired_proof_before_unload=len(legacy_proof),
        paired_proof_ids=[r["request_id"] for r in proof],
        paired_proof_before_unload_ids=[r["request_id"] for r in legacy_proof],
        before_unload_complete_ids=[r["request_id"] for r in legacy_end],
        before_unload_completion=d.score(legacy_end, "completion_late"),
        before_unload_first_text=d.score(legacy_text, "first_text_late"),
        completion_deadline_ascertained_at_unload=d.score(
            score_at_cutoff(legacy), "completion_late"
        ),
        first_text_deadline_ascertained_at_unload=d.score(
            score_at_cutoff(legacy), "first_text_late"
        ),
        healthy_boundary_completed_before_unload=d.score(
            boundary(legacy_end, 240, 300), "completion_late"
        ),
        notes=[
            (
                "Primary scores freeze truth at the earliest artificial background "
                "termination or planned unload. Natural DONE + finish_reason "
                "completion and controlled HTTP503 admission rejection do "
                "not cut pressure short."
            ),
            (
                "An observed outcome before cutoff is known; a deadline crossed "
                "while still observed before cutoff is a known miss. A request "
                "canceled before its deadline remains unknown."
            ),
            (
                "First-text deadline truth uses the client clock; measured "
                "SMG TTFT proof is separate and requires the same completed "
                "request ID before cutoff."
            ),
            (
                "Rejected requests remain unknown counterfactuals and are "
                "excluded from accepted-sentinel FP/FN denominators."
            ),
            (
                "All raw requests are retained in requests-deadline-stratified.json, "
                "including rejected and unselected canceled requests; all "
                "paired rows and later actual outliers are retained."
            ),
            (
                "Legacy *_unload summaries retain planned-unload semantics. "
                "Use canonical *_cutoff fields for primary results when termination "
                "occurred earlier."
            ),
            "An empty healthy240–300s completed-before-cutoff bucket means specificity near the deadline is unproven.",
        ],
    )
    for outcome in ["completion", "first_text"]:
        key = outcome + "_deadline_at_cutoff"
        for value, label in [(True, "late"), (False, "healthy"), (None, "unknown")]:
            result[outcome + "_" + label + "_at_cutoff_ids"] = [
                r["request_id"] for r in sentinels if r[key] is value
            ]
    raw_rows = []
    for raw in requests:
        rid = raw["opc_request_id"]
        paired = paired_by_id.get(rid)
        row = (
            raw
            | (paired or {})
            | dict(
                request_id=rid,
                raw_request=dict(raw),
                paired_available=paired is not None,
                counterfactual_unknown=str(raw["status"]) == "503",
            )
        )
        row.update(timing(row, cutoff))
        raw_rows.append(row)
    (path / "paired-deadline-stratified.json").write_text(json.dumps(rows, indent=2) + "\n")
    (path / "requests-deadline-stratified.json").write_text(json.dumps(raw_rows, indent=2) + "\n")
    result["stratifier_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["input_sha256"] = {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in names
    }
    result["inputs"] = dict(result["input_sha256"])
    (path / "deadline-strata.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "cutoff_elapsed_seconds",
                    "cutoff_reason",
                    "completed_paired_proof_before_cutoff",
                    "completion_deadline_at_cutoff",
                ]
            },
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    stratify(p.parse_args().input)
