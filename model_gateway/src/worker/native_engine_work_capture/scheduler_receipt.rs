//! Optional SG-only diagnostics. Invalid evidence never changes native acceptance.
use serde_json::{json, Value};

const MAX_EXACT: u64 = (1 << 53) - 1;
pub(super) const MAX_BYTES: usize = 4096;
const METHOD: &str = "sg_scheduler_decision_receipt_v1";
const SOURCE: &str = "c9e5799bd12fd5fefb0fb375ac537a6b7d53892e";
const CHOICES: &[&str] = &["prefill", "decode", "idle", "unsupported"];
const STOPS: &[&str] = &[
    "not_evaluated",
    "full_or_empty_no_chunk",
    "no_request_slots",
    "test_retract",
    "lora_limit",
    "running_full_no_preemption",
    "preemption_refused",
    "waiting_queue_exhausted",
    "empty_prefill_selection",
    "adder_no_token",
    "adder_other",
];
const VALUES: &[&str] = &[
    "running_bs",
    "running_full",
    "allocatable_slots",
    "rem_total_tokens",
    "cur_rem_tokens",
    "stored_rem_input_tokens",
    "stored_rem_chunk_tokens",
];
const REASONS: &[&str] = &[
    "identity_or_counter_range",
    "no_completed_scheduler_choice",
    "receipt_size_limit",
    "observer_failure",
    "scheduler_choice_raised",
    "invalid_evaluated_value",
    "unknown_stop_code",
    "attempt_counter_range",
    "unknown_adder_result",
    "prefetch_counter_range",
    "batch_link_tag_unavailable",
    "invalid_submission_link",
    "invalid_snapshot_clock_binding",
    "invalid_observation_clock",
    "reentrant_scheduler_observation",
];

fn count(v: Option<&Value>) -> Option<u64> {
    v?.as_u64().filter(|v| *v <= MAX_EXACT)
}
fn clock(v: Option<&Value>) -> Option<f64> {
    v?.as_f64()
        .filter(|v| v.is_finite() && *v >= 0.0 && *v <= MAX_EXACT as f64)
}
fn named(v: Option<&Value>, names: &[&str]) -> bool {
    v.and_then(Value::as_str)
        .is_some_and(|s| names.contains(&s))
}
fn nullable_named(v: Option<&Value>, names: &[&str]) -> bool {
    v.is_none_or(Value::is_null) || named(v, names)
}
fn counter_sum(v: Option<&Value>, names: &[&str]) -> Option<u64> {
    let map = v?.as_object()?;
    if map.len() != names.len() {
        return None;
    }
    names.iter().try_fold(0_u64, |sum, key| {
        sum.checked_add(count(map.get(*key))?)
            .filter(|v| *v <= MAX_EXACT)
    })
}
fn valid(value: &Value, epoch: &str, profile: &str, rank: u64, captured: f64) -> bool {
    let Some(map) = value.as_object() else {
        return false;
    };
    let keys = [
        "schema_version",
        "diagnostic_only",
        "method",
        "source_commit",
        "engine_epoch",
        "profile_id",
        "dp_rank",
        "available",
        "unavailable_reason",
        "counters_available",
        "decision_seq",
        "choice_counts",
        "prefill_stop_counts",
        "scheduler_exceptions",
        "diagnostic_error_count",
        "counter_scope",
        "values_scope",
        "decision_in_progress",
        "prefill_stop_code",
        "last_adder_result",
        "adder_attempt_seq",
        "prefetch_skips",
        "evaluated_values",
        "diagnostic_error",
        "selected_mode",
        "selected_forward_mode",
        "observed_at_monotonic_secs",
        "batch_sequence",
        "submission_observed_at_monotonic_secs",
    ];
    if map.keys().any(|k| !keys.contains(&k.as_str()))
        || value["schema_version"] != 1 || value["diagnostic_only"] != true
        || value["method"] != METHOD || value["source_commit"] != SOURCE
        || value["engine_epoch"] != epoch || value["profile_id"] != profile
        || value["dp_rank"].as_u64() != Some(rank) || rank != 0 || !profile.starts_with("sglang-c9e5799:")
        || value["counters_available"] != true
        || value["available"].as_bool().is_none() || value["decision_in_progress"].as_bool().is_none()
        || !nullable_named(value.get("unavailable_reason"), REASONS)
        || !nullable_named(value.get("diagnostic_error"), REASONS)
        || value["counter_scope"] != "observed scheduler returns; interval complete only if error count unchanged"
        || value["values_scope"] != "last evaluated or assigned at named site, not a binding constraint or snapshot gauge"
    { return false; }
    let Some(seq) = count(value.get("decision_seq")) else {
        return false;
    };
    if counter_sum(value.get("choice_counts"), CHOICES) != Some(seq)
        || counter_sum(value.get("prefill_stop_counts"), STOPS) != Some(seq)
        || count(value.get("scheduler_exceptions")).is_none()
        || count(value.get("diagnostic_error_count")).is_none()
    {
        return false;
    }
    if seq == 0 && value.get("observed_at_monotonic_secs").is_none() {
        return value["available"] == false
            && value["unavailable_reason"] == "no_completed_scheduler_choice"
            && [
                "selected_mode",
                "selected_forward_mode",
                "prefill_stop_code",
                "last_adder_result",
                "adder_attempt_seq",
                "prefetch_skips",
                "evaluated_values",
                "batch_sequence",
                "submission_observed_at_monotonic_secs",
                "diagnostic_error",
            ]
            .iter()
            .all(|key| !map.contains_key(*key))
            && serde_json::to_vec(value).is_ok_and(|b| b.len() <= MAX_BYTES);
    }
    let Some(observed) = clock(value.get("observed_at_monotonic_secs")) else {
        return false;
    };
    if !captured.is_finite() || observed > captured {
        return false;
    }
    let link = (
        value.get("batch_sequence"),
        value.get("submission_observed_at_monotonic_secs"),
    );
    if link.0.is_some_and(|v| !v.is_null()) || link.1.is_some_and(|v| !v.is_null()) {
        if count(link.0).is_none_or(|s| s == 0)
            || clock(link.1).is_none_or(|t| t < observed || t > captured)
        {
            return false;
        }
    }
    if value["available"] == true {
        let mode = value["selected_mode"].as_str();
        let forward = value["selected_forward_mode"].as_str();
        if seq == 0
            || mode.is_none_or(|m| value["choice_counts"][m].as_u64().unwrap_or(0) == 0)
            || value["prefill_stop_code"]
                .as_str()
                .is_none_or(|code| value["prefill_stop_counts"][code].as_u64().unwrap_or(0) == 0)
            || !matches!(
                (mode, forward),
                (Some("prefill"), Some("EXTEND"))
                    | (Some("decode"), Some("DECODE"))
                    | (Some("idle"), Some("IDLE"))
                    | (Some("unsupported"), Some("MIXED" | "UNSUPPORTED"))
            )
            || !named(value.get("prefill_stop_code"), STOPS)
            || !value["unavailable_reason"].is_null()
            || !value["diagnostic_error"].is_null()
        {
            return false;
        }
    } else if value
        .get("diagnostic_error")
        .and_then(Value::as_str)
        .is_none()
        || value.get("unavailable_reason") != value.get("diagnostic_error")
        || !nullable_named(value.get("selected_mode"), CHOICES)
        || !nullable_named(
            value.get("selected_forward_mode"),
            &["EXTEND", "DECODE", "IDLE", "MIXED", "UNSUPPORTED"],
        )
        || !nullable_named(value.get("prefill_stop_code"), STOPS)
    {
        return false;
    }
    let Some(attempt) = count(value.get("adder_attempt_seq")) else {
        return false;
    };
    if count(value.get("prefetch_skips")).is_none()
        || !nullable_named(
            value.get("last_adder_result"),
            &["CONTINUE", "NO_TOKEN", "OTHER"],
        )
    {
        return false;
    }
    let Some(values) = value["evaluated_values"].as_object() else {
        return false;
    };
    if values.len() > VALUES.len() {
        return false;
    }
    for (key, item) in values {
        let Some(entry) = item.as_object() else {
            return false;
        };
        if !VALUES.contains(&key.as_str())
            || entry.len() != 3
            || count(item.get("adder_attempt_seq")).is_none_or(|n| n > attempt)
        {
            return false;
        }
        let site = item["site"].as_str().unwrap_or("");
        if !STOPS.contains(&site)
            && ![
                "prefill_full_gate",
                "adder_init",
                "adder_attempt",
                "adder_return",
                "allocator_property_return",
                "get_num_allocatable_reqs_return",
            ]
            .contains(&site)
        {
            return false;
        }
        let v = &item["value"];
        if key == "running_full" {
            if !v.is_boolean() {
                return false;
            }
        } else if !v.is_null()
            && v.as_f64()
                .is_none_or(|n| !n.is_finite() || n.abs() > MAX_EXACT as f64)
        {
            return false;
        }
    }
    serde_json::to_vec(value).is_ok_and(|b| b.len() <= MAX_BYTES)
}

pub(super) fn sanitize(
    value: &Value,
    epoch: &str,
    profile: &str,
    rank: u64,
    captured: f64,
) -> Value {
    if valid(value, epoch, profile, rank, captured) {
        return value.clone();
    }
    let source_reason = value
        .get("unavailable_reason")
        .filter(|v| named(Some(v), REASONS))
        .cloned()
        .unwrap_or(Value::Null);
    json!({"schema_version":1,"diagnostic_only":true,"method":METHOD,"source_commit":SOURCE,
        "available":false,"counters_available":false,"unavailable_reason":"missing_invalid_or_unavailable_scheduler_receipt",
        "source_reported_reason":source_reason})
}

#[cfg(test)]
pub(super) fn fixture() -> Value {
    let choices = CHOICES
        .iter()
        .map(|k| ((*k).to_owned(), json!(if *k == "idle" { 1 } else { 0 })))
        .collect::<serde_json::Map<_, _>>();
    let stops = STOPS
        .iter()
        .map(|k| {
            (
                (*k).to_owned(),
                json!(if *k == "not_evaluated" { 1 } else { 0 }),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    json!({"schema_version":1,"diagnostic_only":true,"method":METHOD,"source_commit":SOURCE,"engine_epoch":"e","profile_id":"sglang-c9e5799:p","dp_rank":0,
            "available":true,"unavailable_reason":null,"counters_available":true,"decision_seq":1,"choice_counts":choices,"prefill_stop_counts":stops,"scheduler_exceptions":0,
            "diagnostic_error_count":0,"counter_scope":"observed scheduler returns; interval complete only if error count unchanged",
            "values_scope":"last evaluated or assigned at named site, not a binding constraint or snapshot gauge","decision_in_progress":false,
            "prefill_stop_code":"not_evaluated","last_adder_result":null,"adder_attempt_seq":0,"prefetch_skips":0,"evaluated_values":{},"diagnostic_error":null,
            "selected_mode":"idle","selected_forward_mode":"IDLE","observed_at_monotonic_secs":1.0,"batch_sequence":null,"submission_observed_at_monotonic_secs":null})
}

#[cfg(test)]
mod tests {
    use super::*;
    fn good() -> Value {
        fixture()
    }
    #[test]
    fn valid_and_linked_receipts_retain_exact_values() {
        let mut v = good();
        assert_eq!(sanitize(&v, "e", "sglang-c9e5799:p", 0, 3.0), v);
        v["batch_sequence"] = json!(7);
        v["submission_observed_at_monotonic_secs"] = json!(2.0);
        assert_eq!(sanitize(&v, "e", "sglang-c9e5799:p", 0, 3.0), v);
    }
    #[test]
    fn malformed_identity_counter_clock_and_partial_link_are_isolated() {
        for (key, bad) in [
            ("engine_epoch", json!("other")),
            ("decision_seq", json!(2)),
            ("observed_at_monotonic_secs", json!(4.0)),
            ("batch_sequence", json!(7)),
            ("submission_observed_at_monotonic_secs", json!(-1.0)),
            ("diagnostic_error_count", json!(MAX_EXACT + 1)),
            ("selected_mode", json!("prefill")),
        ] {
            let mut v = good();
            v[key] = bad;
            let out = sanitize(&v, "e", "sglang-c9e5799:p", 0, 3.0);
            assert_eq!(out["available"], false, "{key}");
            assert!(serde_json::to_vec(&out).unwrap().len() < MAX_BYTES);
        }
    }
    #[test]
    fn huge_unknown_values_and_inconsistent_stop_counts_never_escape_bound() {
        let mut v = good();
        v["ignored"] = json!("x".repeat(100_000));
        assert_eq!(
            sanitize(&v, "e", "sglang-c9e5799:p", 0, 3.0)["available"],
            false
        );
        let mut v = good();
        v["prefill_stop_counts"]["not_evaluated"] = json!(0);
        assert_eq!(
            sanitize(&v, "e", "sglang-c9e5799:p", 0, 3.0)["available"],
            false
        );
        let mut v = good();
        v["batch_sequence"] = json!(7);
        v["submission_observed_at_monotonic_secs"] = json!(0.5);
        assert_eq!(
            sanitize(&v, "e", "sglang-c9e5799:p", 0, 3.0)["available"],
            false
        );
    }
    #[test]
    fn initial_and_exception_unavailable_shapes_are_bounded_and_consistent() {
        let mut initial = good();
        initial["available"] = json!(false);
        initial["unavailable_reason"] = json!("no_completed_scheduler_choice");
        initial["decision_seq"] = json!(0);
        for map in ["choice_counts", "prefill_stop_counts"] {
            for v in initial[map].as_object_mut().unwrap().values_mut() {
                *v = json!(0);
            }
        }
        for key in [
            "selected_mode",
            "selected_forward_mode",
            "prefill_stop_code",
            "last_adder_result",
            "adder_attempt_seq",
            "prefetch_skips",
            "evaluated_values",
            "diagnostic_error",
            "observed_at_monotonic_secs",
            "batch_sequence",
            "submission_observed_at_monotonic_secs",
        ] {
            initial.as_object_mut().unwrap().remove(key);
        }
        assert_eq!(sanitize(&initial, "e", "sglang-c9e5799:p", 0, 3.0), initial);
        let mut huge = initial.clone();
        huge["evaluated_values"] = json!({"running_bs":"x".repeat(100_000)});
        let out = sanitize(&huge, "e", "sglang-c9e5799:p", 0, 3.0);
        assert_eq!(out["available"], false);
        assert!(serde_json::to_vec(&out).unwrap().len() < MAX_BYTES);
        let mut exception = good();
        exception["available"] = json!(false);
        exception["decision_seq"] = json!(0);
        exception["scheduler_exceptions"] = json!(1);
        for map in ["choice_counts", "prefill_stop_counts"] {
            for v in exception[map].as_object_mut().unwrap().values_mut() {
                *v = json!(0);
            }
        }
        for key in [
            "selected_mode",
            "selected_forward_mode",
            "prefill_stop_code",
        ] {
            exception[key] = Value::Null;
        }
        exception["diagnostic_error"] = json!("scheduler_choice_raised");
        exception["unavailable_reason"] = json!("scheduler_choice_raised");
        assert_eq!(
            sanitize(&exception, "e", "sglang-c9e5799:p", 0, 3.0),
            exception
        );
        for (reason, error) in [
            (Value::Null, Value::Null),
            (json!("observer_failure"), json!("scheduler_choice_raised")),
        ] {
            let mut bad = exception.clone();
            bad["unavailable_reason"] = reason;
            bad["diagnostic_error"] = error;
            assert_ne!(sanitize(&bad, "e", "sglang-c9e5799:p", 0, 3.0), bad);
        }
        let mut impossible = good();
        impossible["selected_mode"] = json!("prefill");
        impossible["selected_forward_mode"] = json!("EXTEND");
        assert_eq!(
            sanitize(&impossible, "e", "sglang-c9e5799:p", 0, 3.0)["available"],
            false
        );
        let mut zero = good();
        zero["decision_seq"] = json!(0);
        for map in ["choice_counts", "prefill_stop_counts"] {
            for v in zero[map].as_object_mut().unwrap().values_mut() {
                *v = json!(0);
            }
        }
        assert_eq!(
            sanitize(&zero, "e", "sglang-c9e5799:p", 0, 3.0)["available"],
            false
        );
    }
    #[test]
    fn actual_python_producer_fixture_is_retained_exactly() {
        let value: Value = serde_json::from_str(include_str!(
            "../../../../grpc_servicer/tests/fixtures/sg_scheduler_decision_receipt_v1.json"
        ))
        .unwrap();
        assert_eq!(sanitize(&value, "e", "sglang-c9e5799:p", 0, 2.0), value);
    }
}
