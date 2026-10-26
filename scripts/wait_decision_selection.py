"""Phase-wise Pareto selection on identical, whole-capture admission cohorts.

Rows carry capture, model, phase, request_id, truth (late TTFT bool/None), and
predictions (policy -> recorded or causally replayed decision bool/None).
This module never turns an unknown forecast or rejected outcome into a label.
"""

from collections import Counter


def score(rows, policy):
    counts = Counter(
        offered=len(rows), tp=0, fp=0, tn=0, fn=0, unknown=0, prediction_unknown=0, truth_unknown=0
    )
    known = set()
    for row in rows:
        truth, predicted = row["truth"], row["predictions"][policy]
        if any(value is not None and type(value) is not bool for value in (truth, predicted)):
            raise ValueError("Truth and prediction must be bool or None")
        counts["truth_unknown"] += truth is None
        counts["prediction_unknown"] += predicted is None
        if predicted is not None:
            known.add(row["request_id"])
        if truth is None or predicted is None:
            counts["unknown"] += 1
        else:
            counts[
                "tp" if predicted and truth else "fp" if predicted else "fn" if truth else "tn"
            ] += 1
    counts["healthy_denominator"] = counts["fp"] + counts["tn"]
    counts["late_denominator"] = counts["tp"] + counts["fn"]
    return dict(counts), known


def compare(rows, baseline, candidate):
    """Require no FP/FN regression in any capture/model/phase and one gain.

    All individual phases are retained. pressure + pressure-high is additionally
    reported as overload_combined; it never replaces the individual phase gates.
    An unchanged candidate is not selected. No weighted FP/FN cost is assumed.
    """
    identities = [(r["capture"], r["model"], r["request_id"]) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate admission identity")
    if not rows:
        raise ValueError("Cannot select from an empty evaluation")
    cells, violations, improvements = [], [], []
    for capture, model in sorted({(r["capture"], r["model"]) for r in rows}):
        cohort = [r for r in rows if (r["capture"], r["model"]) == (capture, model)]
        phases = sorted({r["phase"] for r in cohort})
        if any(p in phases for p in ("pressure", "pressure-high")):
            phases += ["overload_combined"]
        for phase in phases:
            subset = [
                r
                for r in cohort
                if r["phase"] == phase
                or phase == "overload_combined"
                and r["phase"] in ("pressure", "pressure-high")
            ]
            original, original_known = score(subset, baseline)
            current, current_known = score(subset, candidate)
            identity = dict(capture=capture, model=model, phase=phase)
            cells.append(dict(**identity, baseline=original, candidate=current))
            if original_known != current_known:
                violations.append(dict(**identity, reason="prediction coverage differs"))
            for metric in ("fp", "fn"):
                change = current[metric] - original[metric]
                if change > 0:
                    violations.append(dict(**identity, reason=metric + " increased", delta=change))
                elif change < 0:
                    improvements.append(dict(**identity, metric=metric, delta=change))
    return dict(
        baseline=baseline,
        candidate=candidate,
        eligible=not violations and bool(improvements),
        reason="regression_or_coverage_change"
        if violations
        else "pareto_improvement"
        if improvements
        else "no_strict_improvement",
        cells=cells,
        violations=violations,
        improvements=improvements,
    )
