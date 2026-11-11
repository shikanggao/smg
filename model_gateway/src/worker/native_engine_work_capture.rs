//! Bounded, opt-in copies of existing native work diagnostics only.
//!
//! This module neither changes the typed load response nor interprets work as
//! physical GPU progress, accepted tokens, present allocation, or free capacity.
use std::collections::{BTreeMap, HashSet};

use serde::Serialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

pub(super) const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
const MAX_CAPTURE_BYTES: usize = 2 * 1024 * 1024;
const MAX_RANKS: usize = 256;
const MAX_REQUESTS: usize = 8192;
const MAX_ID_BYTES: usize = 256;
// Covers outer metadata and later changes to per-rank availability/counts.
const CAPTURE_RESERVE: usize = 32 * 1024;

pub(super) fn enabled() -> bool {
    std::env::var("SMG_NATIVE_ENGINE_WORK_DIAGNOSTICS").as_deref() == Ok("1")
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Capture {
    schema_version: u8,
    diagnostic_only: bool,
    http_status: Option<u16>,
    response_complete: bool,
    response_sha256: Option<String>,
    response_bytes: Option<usize>,
    bytes_observed: usize,
    available: bool,
    unavailable_reason: Option<&'static str>,
    // Counts observed failures per rank/row/capture, including retained rows
    // with unavailable diagnostics. Cap counts are per unit rejected by a cap.
    failure_reason_counts: BTreeMap<FailureReason, u64>,
    ranks_seen: Option<usize>,
    ranks_retained: usize,
    ranks_dropped: Option<usize>,
    requests_seen: Option<usize>,
    requests_retained: usize,
    requests_dropped: Option<usize>,
    ranks: Vec<Value>,
}

impl Capture {
    /// Caller invokes this only in its explicit, default-off capture branch.
    /// `body` must be the complete response; partial/transport/size-limit paths
    /// use `unavailable` and never claim a full-response digest.
    pub(super) fn from_body(body: &[u8], http_status: u16) -> Self {
        if body.len() > MAX_RESPONSE_BYTES {
            return Self::unavailable("response_body_size_limit", Some(http_status), body.len());
        }
        let mut out = Self::empty(Some(http_status), body.len());
        out.response_complete = true;
        out.response_bytes = Some(body.len());
        out.response_sha256 = Some(
            Sha256::digest(body)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
        );
        if http_status != 200 {
            out.mark("http_status_not_success");
            return out;
        }
        let Ok(value) = serde_json::from_slice::<Value>(body) else {
            out.mark("malformed_response");
            return out;
        };
        let Some(loads) = value.get("loads").and_then(Value::as_array) else {
            out.mark("malformed_response");
            return out;
        };
        out.ranks_seen = Some(loads.len());
        out.requests_seen = loads.iter().try_fold(0usize, |count, rank| {
            rank.get("deadline_recovery")?
                .get("active_requests")?
                .as_array()?
                .len()
                .checked_add(count)
        });
        out.available = true;
        out.unavailable_reason = None;
        if loads.is_empty()
            || value.get("dp_rank_count").and_then(Value::as_u64) != Some(loads.len() as u64)
        {
            out.mark("unsupported_rank_layout");
        }
        let mut budget = MAX_CAPTURE_BYTES - CAPTURE_RESERVE;
        let mut ranks = HashSet::new();
        for raw in loads.iter().take(MAX_RANKS) {
            let Some(rank) = raw.get("dp_rank").and_then(Value::as_u64) else {
                out.mark("malformed_rank_identity");
                continue;
            };
            if rank > i32::MAX as u64 || rank >= loads.len() as u64 || !ranks.insert(rank) {
                out.mark("duplicate_or_unsupported_rank");
                continue;
            }
            let Some(lane) = raw.get("deadline_recovery").and_then(Value::as_object) else {
                out.mark("missing_native_lane");
                continue;
            };
            let Some(identity) = Identity::read(lane, rank) else {
                out.mark("malformed_rank_identity");
                continue;
            };
            let (progress, progress_reason) =
                sanitize_progress(lane.get("engine_work_progress"), &identity);
            if let Some(why) = progress_reason {
                out.mark(why);
            }
            if lane
                .get("engine_work_progress")
                .and_then(|v| v.get("counters_available"))
                .and_then(Value::as_bool)
                == Some(false)
                && progress_reason != Some("source_progress_unavailable")
            {
                out.mark("source_progress_unavailable");
            }
            let raw_rows = lane.get("active_requests").and_then(Value::as_array);
            let seen = raw_rows.map(Vec::len);
            let mut reason = progress_reason;
            if raw_rows.is_none() {
                reason.get_or_insert("missing_active_inventory");
                out.mark("missing_active_inventory");
            }
            let mut result = json!({
                "engine_epoch":identity.epoch,"profile_id":identity.profile,"dp_rank":rank,
                "snapshot_seq":identity.sequence,"captured_at_monotonic_secs":identity.captured,
                "available":reason.is_none(),"unavailable_reason":reason,
                "engine_work_progress":progress,"requests_seen":seen,
                "requests_retained":0,"requests_dropped":seen,"active_requests":[]
            });
            if let Some(value) = lane.get("original_prefill_service_provenance") {
                let service = sanitize_sg_prefill(
                    value,
                    &identity,
                    None,
                    Some(&result["engine_work_progress"]),
                );
                result["original_prefill_service_provenance"] = service;
            }
            let header_bytes = encoded_len(&result);
            if header_bytes > budget {
                out.mark("capture_byte_limit");
                continue;
            }
            budget -= header_bytes;
            let mut retained = Vec::new();
            let mut native_ids = HashSet::new();
            let mut attempts = HashSet::new();
            for row in raw_rows.into_iter().flatten() {
                if out.requests_retained >= MAX_REQUESTS {
                    reason.get_or_insert("request_row_limit");
                    out.mark("request_row_limit");
                    continue;
                }
                let Some(row) = row.as_object() else {
                    reason.get_or_insert("malformed_request_row");
                    out.mark("malformed_request_row");
                    continue;
                };
                let (Some(native_id), Some(attempt), Some(state_seq)) = (
                    nullable_id(row.get("native_request_id")),
                    nullable_id(row.get("attempt_id")),
                    row.get("state_seq").and_then(Value::as_u64),
                ) else {
                    reason.get_or_insert("malformed_request_identity");
                    out.mark("malformed_request_identity");
                    continue;
                };
                if native_id.is_none() {
                    reason.get_or_insert("missing_request_identity");
                    out.mark("missing_request_identity");
                    continue;
                }
                if !native_ids.insert(native_id.clone())
                    || attempt
                        .as_ref()
                        .is_some_and(|id| !attempts.insert(id.clone()))
                {
                    reason.get_or_insert("duplicate_request_identity");
                    out.mark("duplicate_request_identity");
                    continue;
                }
                let (state, state_reason) =
                    sanitize_state(row.get("engine_work_state"), &native_id, &attempt);
                if let Some(why) = state_reason {
                    reason.get_or_insert(why);
                    out.mark(why);
                }
                if row
                    .get("engine_work_state")
                    .and_then(|v| v.get("available"))
                    .and_then(Value::as_bool)
                    == Some(false)
                    && state_reason != Some("source_request_unavailable")
                {
                    out.mark("source_request_unavailable");
                }
                let original_prefill = row.get("original_prefill_observation").map(|value| {
                    sanitize_sg_prefill(
                        value,
                        &identity,
                        Some((&native_id, &attempt, state_seq)),
                        None,
                    )
                });
                let mut row = json!({"native_request_id":native_id,"attempt_id":attempt,
                    "state_seq":state_seq,"engine_work_state":state});
                if let Some(value) = original_prefill {
                    row["original_prefill_observation"] = value;
                }
                let bytes = encoded_len(&row) + 1;
                if bytes > budget {
                    reason.get_or_insert("capture_byte_limit");
                    out.mark("capture_byte_limit");
                    continue;
                }
                budget -= bytes;
                retained.push(row);
                out.requests_retained += 1;
            }
            result["available"] = json!(reason.is_none());
            result["unavailable_reason"] = json!(reason);
            result["requests_retained"] = json!(retained.len());
            result["requests_dropped"] = json!(seen.map(|n| n - retained.len()));
            result["active_requests"] = Value::Array(retained);
            out.ranks.push(result);
        }
        out.ranks_retained = out.ranks.len();
        out.ranks_dropped = out.ranks_seen.map(|n| n - out.ranks_retained);
        out.requests_dropped = out.requests_seen.map(|n| n - out.requests_retained);
        if loads.len() > MAX_RANKS {
            out.mark_n("rank_limit", (loads.len() - MAX_RANKS) as u64);
        }
        if out.ranks_retained != loads.len() {
            out.mark_n(
                "rank_data_dropped",
                (loads.len() - out.ranks_retained) as u64,
            );
        }
        // An independent final bound remains even if schema overhead changes.
        if serde_json::to_vec(&out).map_or(true, |bytes| bytes.len() > MAX_CAPTURE_BYTES) {
            out.ranks.clear();
            out.ranks_retained = 0;
            out.ranks_dropped = out.ranks_seen;
            out.requests_retained = 0;
            out.requests_dropped = out.requests_seen;
            out.mark("capture_byte_limit");
        }
        out
    }

    pub(super) fn unavailable(
        reason: &'static str,
        http_status: Option<u16>,
        bytes_observed: usize,
    ) -> Self {
        let mut out = Self::empty(http_status, bytes_observed);
        out.mark(reason);
        out
    }

    fn empty(http_status: Option<u16>, bytes_observed: usize) -> Self {
        Self {
            schema_version: 1,
            diagnostic_only: true,
            http_status,
            response_complete: false,
            response_sha256: None,
            response_bytes: None,
            bytes_observed,
            available: false,
            unavailable_reason: None,
            failure_reason_counts: BTreeMap::new(),
            ranks_seen: None,
            ranks_retained: 0,
            ranks_dropped: None,
            requests_seen: None,
            requests_retained: 0,
            requests_dropped: None,
            ranks: Vec::new(),
        }
    }

    fn mark(&mut self, reason: &'static str) {
        self.mark_n(reason, 1);
    }

    fn mark_n(&mut self, reason: &'static str, count: u64) {
        if count == 0 {
            return;
        }
        self.available = false;
        self.unavailable_reason.get_or_insert(reason);
        let value = self
            .failure_reason_counts
            .entry(FailureReason::from_static(reason))
            .or_default();
        *value = value.saturating_add(count);
    }
}

// Serialized histogram keys are a fixed enum, never source or error strings.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum FailureReason {
    ResponseBodySizeLimit,
    MalformedResponse,
    HttpStatusNotSuccess,
    UnsupportedRankLayout,
    MalformedRankIdentity,
    DuplicateOrUnsupportedRank,
    MissingNativeLane,
    MissingActiveInventory,
    MissingProgressDiagnostics,
    MalformedProgressDiagnostics,
    ProgressIdentityMismatch,
    SourceProgressUnavailable,
    RequestRowLimit,
    MalformedRequestRow,
    MalformedRequestIdentity,
    MissingRequestIdentity,
    DuplicateRequestIdentity,
    RequestAssociationMismatch,
    MalformedRequestDiagnostics,
    MissingRequestDiagnostics,
    SourceRequestUnavailable,
    CaptureByteLimit,
    RankLimit,
    RankDataDropped,
    NativeHttpStatusUnavailable,
    NativeResponseSizeLimit,
    NativeTransportTimeout,
    NativeTransportError,
    NativeBodyReadTimeout,
    NativeBodyReadError,
    NativePollTaskDropped,
    TransportError,
    OtherCaptureUnavailable,
}

impl FailureReason {
    fn from_static(reason: &'static str) -> Self {
        match reason {
            "response_body_size_limit" => Self::ResponseBodySizeLimit,
            "malformed_response" => Self::MalformedResponse,
            "http_status_not_success" => Self::HttpStatusNotSuccess,
            "unsupported_rank_layout" => Self::UnsupportedRankLayout,
            "malformed_rank_identity" => Self::MalformedRankIdentity,
            "duplicate_or_unsupported_rank" => Self::DuplicateOrUnsupportedRank,
            "missing_native_lane" => Self::MissingNativeLane,
            "missing_active_inventory" => Self::MissingActiveInventory,
            "missing_progress_diagnostics" => Self::MissingProgressDiagnostics,
            "malformed_progress_diagnostics" => Self::MalformedProgressDiagnostics,
            "progress_identity_mismatch" => Self::ProgressIdentityMismatch,
            "source_progress_unavailable" => Self::SourceProgressUnavailable,
            "request_row_limit" => Self::RequestRowLimit,
            "malformed_request_row" => Self::MalformedRequestRow,
            "malformed_request_identity" => Self::MalformedRequestIdentity,
            "missing_request_identity" => Self::MissingRequestIdentity,
            "duplicate_request_identity" => Self::DuplicateRequestIdentity,
            "request_association_mismatch" => Self::RequestAssociationMismatch,
            "malformed_request_diagnostics" => Self::MalformedRequestDiagnostics,
            "missing_request_diagnostics" => Self::MissingRequestDiagnostics,
            "source_request_unavailable" => Self::SourceRequestUnavailable,
            "capture_byte_limit" => Self::CaptureByteLimit,
            "rank_limit" => Self::RankLimit,
            "rank_data_dropped" => Self::RankDataDropped,
            "native_http_status_unavailable" => Self::NativeHttpStatusUnavailable,
            "native_response_size_limit" => Self::NativeResponseSizeLimit,
            "native_transport_timeout" => Self::NativeTransportTimeout,
            "native_transport_error" => Self::NativeTransportError,
            "native_body_read_timeout" => Self::NativeBodyReadTimeout,
            "native_body_read_error" => Self::NativeBodyReadError,
            "native_poll_task_dropped" => Self::NativePollTaskDropped,
            "transport_error" => Self::TransportError,
            _ => Self::OtherCaptureUnavailable,
        }
    }
}

struct Identity {
    epoch: String,
    profile: String,
    rank: u64,
    sequence: u64,
    captured: f64,
}

impl Identity {
    fn read(lane: &Map<String, Value>, rank: u64) -> Option<Self> {
        if lane.get("schema_version")?.as_u64()? != 1 || lane.get("dp_rank")?.as_u64()? != rank {
            return None;
        }
        Some(Self {
            epoch: id(lane.get("engine_epoch")?)?,
            profile: id(lane.get("profile_id")?)?,
            rank,
            sequence: lane.get("snapshot_seq")?.as_u64()?,
            captured: number(lane.get("captured_at_monotonic_secs")?)?,
        })
    }
}

fn id(value: &Value) -> Option<String> {
    let text = value.as_str()?;
    (!text.is_empty()
        && text.len() <= MAX_ID_BYTES
        && text
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"-_.:/".contains(&c)))
    .then(|| text.to_owned())
}

// Outer None means malformed; inner None preserves a source null/missing value.
fn nullable_id(value: Option<&Value>) -> Option<Option<String>> {
    match value {
        None | Some(Value::Null) => Some(None),
        Some(value) => id(value).map(Some),
    }
}

fn number(value: &Value) -> Option<f64> {
    let value = value.as_f64()?;
    (value.is_finite() && value >= 0.0).then_some(value)
}

fn encoded_len(value: &Value) -> usize {
    serde_json::to_vec(value).map_or(MAX_CAPTURE_BYTES + 1, |bytes| bytes.len())
}

fn nullable_scalar(value: Option<&Value>, kind: fn(&Value) -> bool, bad: &mut bool) -> Value {
    match value {
        None | Some(Value::Null) => Value::Null,
        Some(value) if kind(value) => value.clone(),
        Some(_) => {
            *bad = true;
            Value::Null
        }
    }
}

fn unsigned(value: &Value) -> bool {
    value.as_u64().is_some()
}
fn boolean(value: &Value) -> bool {
    value.is_boolean()
}
fn clock(value: &Value) -> bool {
    number(value).is_some()
}

fn source_reason(value: Option<&Value>, bad: &mut bool) -> Value {
    match value {
        None | Some(Value::Null) => Value::Null,
        Some(Value::String(reason)) => {
            let known = match reason.as_str() {
                "unsupported_engine_configuration" => "unsupported_engine_configuration",
                "invalid_or_missing_scheduled_input_ranges" => {
                    "invalid_or_missing_scheduled_input_ranges"
                }
                "completion_without_submission_record" => "completion_without_submission_record",
                "invalid_completion_observation_clock" => "invalid_completion_observation_clock",
                "invalid_snapshot_observation_clock" => "invalid_snapshot_observation_clock",
                "unsupported_or_missing_input_work_metadata" => {
                    "unsupported_or_missing_input_work_metadata"
                }
                "nonstandard_or_missing_completed_request_output" => {
                    "nonstandard_or_missing_completed_request_output"
                }
                "engine_work_diagnostic_observation_failed" => {
                    "engine_work_diagnostic_observation_failed"
                }
                "native_schedule_or_result_processing_failed" => {
                    "native_schedule_or_result_processing_failed"
                }
                "native_model_execution_or_result_failed" => {
                    "native_model_execution_or_result_failed"
                }
                "unsupported_or_incomplete_batch_work_metadata" => {
                    "unsupported_or_incomplete_batch_work_metadata"
                }
                "native_batch_execution_failed" => "native_batch_execution_failed",
                "result_work_metadata_not_retained" => "result_work_metadata_not_retained",
                "native_result_processing_failed" => "native_result_processing_failed",
                "request_work_metadata_unavailable" => "request_work_metadata_unavailable",
                "request_work_diagnostic_unavailable" => "request_work_diagnostic_unavailable",
                "request_work_metadata_missing_or_invalid" => {
                    "request_work_metadata_missing_or_invalid"
                }
                "request_work_metadata_observation_failed" => {
                    "request_work_metadata_observation_failed"
                }
                _ => {
                    *bad = true;
                    "unrecognized_source_reason"
                }
            };
            json!(known)
        }
        Some(_) => {
            *bad = true;
            Value::Null
        }
    }
}

fn sanitize_progress(value: Option<&Value>, outer: &Identity) -> (Value, Option<&'static str>) {
    let Some(source) = value.and_then(Value::as_object) else {
        return (
            Value::Null,
            Some(if value.is_some_and(|v| !v.is_null()) {
                "malformed_progress_diagnostics"
            } else {
                "missing_progress_diagnostics"
            }),
        );
    };
    if source.get("schema_version").and_then(Value::as_u64) != Some(1)
        || source.get("engine_epoch").and_then(Value::as_str) != Some(outer.epoch.as_str())
        || source.get("profile_id").and_then(Value::as_str) != Some(outer.profile.as_str())
        || source.get("dp_rank").and_then(Value::as_u64) != Some(outer.rank)
    {
        return (Value::Null, Some("progress_identity_mismatch"));
    }
    let mut bad = false;
    let available = source.get("counters_available").and_then(Value::as_bool);
    if available.is_none() {
        bad = true;
    }
    let reason = source_reason(source.get("unavailable_reason"), &mut bad);
    let commit = source
        .get("source_commit")
        .and_then(Value::as_str)
        .filter(|s| s.len() == 40 && s.bytes().all(|c| c.is_ascii_hexdigit()));
    let boundary = source
        .get("completion_boundary")
        .and_then(Value::as_str)
        .filter(|s| {
            matches!(
                *s,
                "scheduler_update_from_output_after_native_model_future"
                    | "normal_process_batch_result_return_after_native_completion_wait"
            )
        });
    if source.get("source_commit").is_some_and(|v| !v.is_null()) && commit.is_none()
        || source
            .get("completion_boundary")
            .is_some_and(|v| !v.is_null())
            && boundary.is_none()
    {
        bad = true;
    }
    let observed = nullable_scalar(source.get("snapshot_monotonic_secs"), clock, &mut bad);
    let sequence = nullable_scalar(source.get("completion_sequence"), unsigned, &mut bad);
    let mut cumulative = Map::new();
    for field in [
        "completed_batches",
        "completed_uncached_prefill_tokens",
        "completed_generation_context_tokens",
        "completed_generation_context_attention_tokens",
    ] {
        cumulative.insert(
            field.into(),
            nullable_scalar(
                source.get("cumulative").and_then(|v| v.get(field)),
                unsigned,
                &mut bad,
            ),
        );
    }
    if source
        .get("cumulative")
        .is_some_and(|v| !v.is_null() && !v.is_object())
    {
        bad = true;
    }
    let last = match source.get("last_completion") {
        None | Some(Value::Null) => Value::Null,
        Some(Value::Object(last)) => {
            let mut out = Map::new();
            for field in [
                "batch_sequence",
                "request_count",
                "uncached_prefill_tokens",
                "generation_context_tokens",
                "generation_context_attention_tokens",
            ] {
                let value = nullable_scalar(last.get(field), unsigned, &mut bad);
                if value.is_null() {
                    bad = true;
                }
                out.insert(field.into(), value);
            }
            for field in [
                "schedule_observed_monotonic_secs",
                "completion_observed_monotonic_secs",
                "host_observation_span_seconds",
            ] {
                let value = nullable_scalar(last.get(field), clock, &mut bad);
                if value.is_null() {
                    bad = true;
                }
                out.insert(field.into(), value);
            }
            Value::Object(out)
        }
        Some(_) => {
            bad = true;
            Value::Null
        }
    };
    if available == Some(true)
        && (commit.is_none()
            || boundary.is_none()
            || observed.is_null()
            || sequence.is_null()
            || cumulative.values().any(Value::is_null)
            || !reason.is_null())
    {
        bad = true;
    }
    let why = if bad {
        Some("malformed_progress_diagnostics")
    } else if available != Some(true) {
        Some("source_progress_unavailable")
    } else {
        None
    };
    (
        json!({"schema_version":1,"engine_epoch":outer.epoch,"profile_id":outer.profile,"dp_rank":outer.rank,
        "source_commit":commit,"completion_boundary":boundary,"snapshot_monotonic_secs":observed,
        "counters_available":if bad {None} else {available},"unavailable_reason":reason,
        "completion_sequence":sequence,"cumulative":cumulative,"last_completion":last}),
        why,
    )
}

/// These opt-in SG facts are outside native parsing/equality, remaining work,
/// predictor input and ACK credit. Malformed new facts are rejected locally.
/// Their bytes share existing response/capture caps and can reduce coverage.
fn sanitize_sg_prefill(
    value: &Value,
    outer: &Identity,
    request: Option<(&Option<String>, &Option<String>, u64)>,
    progress: Option<&Value>,
) -> Value {
    let rejected = || {
        json!({"schema_version":1,"diagnostic_only":true,"available":false,
        "reason":"malformed_or_unassociated_sg_prefill_diagnostic"})
    };
    let Some(source) = value.as_object() else {
        return rejected();
    };
    let method = if request.is_some() {
        "sg_original_prefill_observation_v1"
    } else {
        "sg_completed_prefill_original_unit_guards_v1"
    };
    if source.get("schema_version").and_then(Value::as_u64) != Some(1)
        || source.get("diagnostic_only").and_then(Value::as_bool) != Some(true)
        || source.get("method").and_then(Value::as_str) != Some(method)
        || source.get("source_commit").and_then(Value::as_str)
            != Some("c9e5799bd12fd5fefb0fb375ac537a6b7d53892e")
    {
        return rejected();
    }
    if let Some((native, attempt, seq)) = request {
        if nullable_id(source.get("native_request_id")).as_ref() != Some(native)
            || nullable_id(source.get("attempt_id")).as_ref() != Some(attempt)
            || source.get("state_seq").and_then(Value::as_u64) != Some(seq)
        {
            return rejected();
        }
    } else if source.get("engine_epoch").and_then(Value::as_str) != Some(outer.epoch.as_str())
        || source.get("profile_id").and_then(Value::as_str) != Some(outer.profile.as_str())
        || source.get("dp_rank").and_then(Value::as_u64) != Some(outer.rank)
    {
        return rejected();
    }
    let mut bad = false;
    let mut out = Map::new();
    out.insert("schema_version".into(), json!(1));
    out.insert("diagnostic_only".into(), json!(true));
    out.insert("method".into(), json!(method));
    out.insert("source_commit".into(), source["source_commit"].clone());
    let observed = nullable_scalar(source.get("observed_at_monotonic_secs"), clock, &mut bad);
    if let Some(t) = observed.as_f64() {
        if request.is_some() && t > outer.captured || request.is_none() && t < outer.captured {
            bad = true;
        }
    }
    out.insert("observed_at_monotonic_secs".into(), observed);
    out.insert(
        "available".into(),
        nullable_scalar(source.get("available"), boolean, &mut bad),
    );
    let (numbers, flags): (&[&str], &[&str]) = if let Some((native, attempt, seq)) = request {
        out.insert("native_request_id".into(), json!(native));
        out.insert("attempt_id".into(), json!(attempt));
        out.insert("state_seq".into(), json!(seq));
        (
            &[
                "original_prompt_tokens",
                "normal_return_frontier_tokens",
                "committed_output_tokens",
                "lifetime_retraction_count",
            ],
            &[
                "supported_mode",
                "original_alias_verified",
                "input_embeds_none",
                "session_none",
                "multimodal_inputs_none",
                "is_retracted",
                "abort_pending",
                "qualified",
            ],
        )
    } else {
        out.insert("engine_epoch".into(), json!(outer.epoch));
        out.insert("profile_id".into(), json!(outer.profile));
        out.insert("dp_rank".into(), json!(outer.rank));
        if source.get("scope").and_then(Value::as_str)
            != Some("original_unpadded_alias_length_guards_only")
            || source
                .get("token_content_identity_verified")
                .is_some_and(|v| !v.is_null())
        {
            bad = true;
        }
        out.insert(
            "scope".into(),
            json!("original_unpadded_alias_length_guards_only"),
        );
        out.insert("token_content_identity_verified".into(), Value::Null);
        (
            &[
                "completion_sequence",
                "observed_completed_batches",
                "guarded_completed_batches",
                "counted_contributor_occurrences",
            ],
            &[],
        )
    };
    for field in numbers {
        out.insert(
            (*field).into(),
            nullable_scalar(source.get(*field), unsigned, &mut bad),
        );
    }
    for field in flags {
        out.insert(
            (*field).into(),
            nullable_scalar(source.get(*field), boolean, &mut bad),
        );
    }
    if request.is_none() {
        // The SG producer copies these two values from this exact completed-
        // work snapshot. SG's native timestamp is an earlier, separate read.
        let progress = progress.filter(|p| {
            p["engine_epoch"].as_str() == Some(outer.epoch.as_str())
                && p["profile_id"].as_str() == Some(outer.profile.as_str())
                && p["dp_rank"].as_u64() == Some(outer.rank)
                && p["source_commit"].as_str() == Some("c9e5799bd12fd5fefb0fb375ac537a6b7d53892e")
                && p["completion_boundary"].as_str()
                    == Some("normal_process_batch_result_return_after_native_completion_wait")
                && p["counters_available"].is_boolean()
        });
        if out["observed_at_monotonic_secs"].as_f64().is_some_and(|t| {
            progress.and_then(|p| p["snapshot_monotonic_secs"].as_f64()) != Some(t)
        }) || out["completion_sequence"].as_u64().is_some_and(|seq| {
            progress.and_then(|p| p["completion_sequence"].as_u64()) != Some(seq)
        }) || out["available"].as_bool() == Some(true)
            && progress.and_then(|p| p["counters_available"].as_bool()) != Some(true)
        {
            bad = true;
        }
    }
    let reason_key = if request.is_some() {
        "reason"
    } else {
        "unavailable_reason"
    };
    let reason = source.get(reason_key).filter(|v| {
        v.is_null()
            || v.as_str().is_some_and(|s| {
                matches!(
                    s,
                    "original_prefill_complete_witness"
                        | "request_guard_not_satisfied"
                        | "request_metadata_unavailable"
                        | "request_observation_failed"
                        | "submission_metadata_unavailable"
                        | "completion_metadata_unavailable"
                        | "contributor_original_unit_guard_unavailable"
                        | "completed_counter_scope_gap"
                        | "unsupported_engine_configuration"
                        | "legacy_completed_counters_unavailable"
                        | "no_counted_contributor_observation"
                        | "native_batch_execution_failed"
                        | "service_observation_failed"
                )
            })
    });
    if reason.is_none() {
        bad = true;
    }
    out.insert(reason_key.into(), reason.cloned().unwrap_or(Value::Null));
    if bad {
        rejected()
    } else {
        Value::Object(out)
    }
}

fn sanitize_state(
    value: Option<&Value>,
    native: &Option<String>,
    attempt: &Option<String>,
) -> (Value, Option<&'static str>) {
    let Some(source) = value.and_then(Value::as_object) else {
        return (
            Value::Null,
            Some(if value.is_some_and(|v| !v.is_null()) {
                "malformed_request_diagnostics"
            } else {
                "missing_request_diagnostics"
            }),
        );
    };
    if source.get("schema_version").and_then(Value::as_u64) != Some(1)
        || source.get("diagnostic_only").and_then(Value::as_bool) != Some(true)
        || !source.contains_key("native_request_id")
        || !source.contains_key("attempt_id")
        || nullable_id(source.get("native_request_id")).as_ref() != Some(native)
        || nullable_id(source.get("attempt_id")).as_ref() != Some(attempt)
    {
        return (Value::Null, Some("request_association_mismatch"));
    }
    let mut bad = false;
    let available = source.get("available").and_then(Value::as_bool);
    if available.is_none() {
        bad = true;
    }
    if available == Some(true)
        && [
            "unavailable_reason",
            "output_token_count",
            "retraction_count",
            "preemption_count",
            "abort_pending",
            "abort_pending_source",
            "cache_release_observed",
        ]
        .iter()
        .any(|field| !source.contains_key(*field))
    {
        bad = true;
    }
    let reason = source_reason(source.get("unavailable_reason"), &mut bad);
    let mut out = Map::new();
    out.insert("schema_version".into(), json!(1));
    out.insert("diagnostic_only".into(), json!(true));
    out.insert("native_request_id".into(), json!(native));
    out.insert("attempt_id".into(), json!(attempt));
    for field in ["output_token_count", "retraction_count", "preemption_count"] {
        out.insert(
            field.into(),
            nullable_scalar(source.get(field), unsigned, &mut bad),
        );
    }
    for field in ["abort_pending", "cache_release_observed"] {
        out.insert(
            field.into(),
            nullable_scalar(source.get(field), boolean, &mut bad),
        );
    }
    let abort_source = source
        .get("abort_pending_source")
        .and_then(Value::as_str)
        .filter(|s| {
            matches!(
                *s,
                "req_to_finish_FINISH_ABORT" | "native_aborted_status_or_pending_ack" | "unknown"
            )
        });
    if source
        .get("abort_pending_source")
        .is_some_and(|v| !v.is_null())
        && abort_source.is_none()
    {
        bad = true;
    }
    if !out["abort_pending"].is_null()
        && !matches!(
            abort_source,
            Some("req_to_finish_FINISH_ABORT" | "native_aborted_status_or_pending_ack")
        )
    {
        bad = true;
    }
    if available == Some(true)
        && (out["output_token_count"].is_null()
            || out["retraction_count"].is_null() && out["preemption_count"].is_null()
            || !reason.is_null())
    {
        bad = true;
    }
    out.insert("abort_pending_source".into(), json!(abort_source));
    out.insert(
        "available".into(),
        json!(if bad { None } else { available }),
    );
    out.insert("unavailable_reason".into(), reason);
    (
        Value::Object(out),
        if bad {
            Some("malformed_request_diagnostics")
        } else if available != Some(true) {
            Some("source_request_unavailable")
        } else {
            None
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sg_original_facts() -> Value {
        json!({"schema_version":1,"diagnostic_only":true,"method":"sg_original_prefill_observation_v1",
            "source_commit":"c9e5799bd12fd5fefb0fb375ac537a6b7d53892e",
            "native_request_id":"native","attempt_id":"attempt","state_seq":2,
            "observed_at_monotonic_secs":19.9,"available":true,"qualified":true,
            "reason":"original_prefill_complete_witness","supported_mode":true,
            "original_alias_verified":true,"original_prompt_tokens":100,"input_embeds_none":true,
            "session_none":true,"multimodal_inputs_none":true,"lifetime_retraction_count":0,"is_retracted":false,
            "normal_return_frontier_tokens":100,"committed_output_tokens":1,"abort_pending":false})
    }

    #[test]
    fn sg_original_facts_preserve_legacy_capture_fields_within_budget() {
        let old = fixture(true);
        let baseline =
            serde_json::to_value(Capture::from_body(&serde_json::to_vec(&old).unwrap(), 200))
                .unwrap();
        let mut changed = old;
        let mut facts = sg_original_facts();
        facts["prompt"] = json!("SECRET_CONTENT");
        facts["output_ids"] = json!([1, 2, 3]);
        lane(&mut changed)["active_requests"][0]["original_prefill_observation"] = facts;
        let captured = serde_json::to_value(Capture::from_body(
            &serde_json::to_vec(&changed).unwrap(),
            200,
        ))
        .unwrap();
        assert_eq!(captured["available"], baseline["available"]);
        assert_eq!(
            captured["failure_reason_counts"],
            baseline["failure_reason_counts"]
        );
        assert_eq!(
            captured["ranks"][0]["active_requests"][0]["engine_work_state"],
            baseline["ranks"][0]["active_requests"][0]["engine_work_state"]
        );
        let retained = &captured["ranks"][0]["active_requests"][0]["original_prefill_observation"];
        assert_eq!(retained["qualified"], true);
        assert_eq!(retained["lifetime_retraction_count"], 0);
        assert_eq!(retained["is_retracted"], false);
        assert!(retained.get("prompt").is_none());
        assert!(retained.get("output_ids").is_none());
        assert!(!serde_json::to_string(retained)
            .unwrap()
            .contains("SECRET_CONTENT"));
    }

    #[test]
    fn malformed_sg_facts_are_local_and_unknown_fields_remain_null() {
        let identity = Identity {
            epoch: "epoch".into(),
            profile: "profile".into(),
            rank: 0,
            sequence: 3,
            captured: 20.0,
        };
        let native = Some("native".into());
        let attempt = Some("attempt".into());
        let mut facts = sg_original_facts();
        facts["is_retracted"] = Value::Null;
        facts["lifetime_retraction_count"] = Value::Null;
        facts["available"] = json!(false);
        facts["qualified"] = Value::Null;
        facts["reason"] = json!("request_metadata_unavailable");
        let out = sanitize_sg_prefill(&facts, &identity, Some((&native, &attempt, 2)), None);
        assert!(out["is_retracted"].is_null());
        assert!(out["lifetime_retraction_count"].is_null());
        for field in ["lifetime_retraction_count", "state_seq"] {
            let mut bad = facts.clone();
            bad[field] = json!(true);
            assert_eq!(
                sanitize_sg_prefill(&bad, &identity, Some((&native, &attempt, 2)), None)
                    ["available"],
                false
            );
        }
        facts["attempt_id"] = json!("foreign");
        assert_eq!(
            sanitize_sg_prefill(&facts, &identity, Some((&native, &attempt, 2)), None)["available"],
            false
        );
    }

    #[test]
    fn sg_service_guard_scope_retains_unproven_content_identity() {
        let identity = Identity {
            epoch: "epoch".into(),
            profile: "profile".into(),
            rank: 0,
            sequence: 3,
            captured: 20.0,
        };
        let value = json!({"schema_version":1,"diagnostic_only":true,
            "method":"sg_completed_prefill_original_unit_guards_v1",
            "source_commit":"c9e5799bd12fd5fefb0fb375ac537a6b7d53892e",
            "engine_epoch":"epoch","profile_id":"profile","dp_rank":0,
            "observed_at_monotonic_secs":20.1,"available":null,
            "unavailable_reason":"no_counted_contributor_observation","completion_sequence":0,
            "observed_completed_batches":0,"guarded_completed_batches":0,"counted_contributor_occurrences":0,
            "token_content_identity_verified":null,"scope":"original_unpadded_alias_length_guards_only"});
        let mut progress =
            fixture(true)["loads"][0]["deadline_recovery"]["engine_work_progress"].clone();
        progress["snapshot_monotonic_secs"] = json!(20.1);
        progress["completion_sequence"] = json!(0);
        let out = sanitize_sg_prefill(&value, &identity, None, Some(&progress));
        assert!(out["available"].is_null());
        assert!(out["token_content_identity_verified"].is_null());
        let mut fabricated = value;
        fabricated["token_content_identity_verified"] = json!(true);
        assert_eq!(
            sanitize_sg_prefill(&fabricated, &identity, None, Some(&progress))["available"],
            false
        );
    }

    #[test]
    fn sg_service_clock_and_sequence_bind_to_same_lane_progress() {
        let mut value = fixture(true);
        lane(&mut value)["engine_work_progress"]["snapshot_monotonic_secs"] = json!(20.1);
        lane(&mut value)["engine_work_progress"]["completion_sequence"] = json!(3);
        let mut service = json!({"schema_version":1,"diagnostic_only":true,
            "method":"sg_completed_prefill_original_unit_guards_v1",
            "source_commit":"c9e5799bd12fd5fefb0fb375ac537a6b7d53892e",
            "engine_epoch":"epoch","profile_id":"profile","dp_rank":0,
            "observed_at_monotonic_secs":20.1,"completion_sequence":3,
            "available":true,"unavailable_reason":null});
        service.as_object_mut().unwrap().extend(
            json!({"observed_completed_batches":3,"guarded_completed_batches":3,
                "counted_contributor_occurrences":3,"token_content_identity_verified":null,
                "scope":"original_unpadded_alias_length_guards_only"})
            .as_object()
            .unwrap()
            .clone(),
        );
        lane(&mut value)["original_prefill_service_provenance"] = service;
        let baseline = capture(&value);
        assert_eq!(baseline["available"], true);
        assert_eq!(
            baseline["ranks"][0]["original_prefill_service_provenance"]["available"],
            true
        );
        // Native captured_at remains 20.0: SG uses separate same-event reads.
        assert_eq!(baseline["ranks"][0]["captured_at_monotonic_secs"], 20.0);
        for field in ["observed_at_monotonic_secs", "completion_sequence"] {
            let mut bad = value.clone();
            lane(&mut bad)["original_prefill_service_provenance"][field] = json!(999);
            let out = capture(&bad);
            assert_eq!(
                out["ranks"][0]["original_prefill_service_provenance"]["available"],
                false
            );
            assert_eq!(out["available"], baseline["available"]);
            assert_eq!(
                out["failure_reason_counts"],
                baseline["failure_reason_counts"]
            );
            assert_eq!(
                out["ranks"][0]["engine_work_progress"],
                baseline["ranks"][0]["engine_work_progress"]
            );
        }
        for missing in [false, true] {
            let mut unknown = value.clone();
            let service = &mut lane(&mut unknown)["original_prefill_service_provenance"];
            service["available"] = Value::Null;
            for field in ["observed_at_monotonic_secs", "completion_sequence"] {
                if missing {
                    service.as_object_mut().unwrap().remove(field);
                } else {
                    service[field] = Value::Null;
                }
            }
            let out = capture(&unknown);
            let retained = &out["ranks"][0]["original_prefill_service_provenance"];
            assert!(retained["available"].is_null());
            assert!(retained["observed_at_monotonic_secs"].is_null());
            assert!(retained["completion_sequence"].is_null());
        }
        for field in ["source_commit", "completion_boundary"] {
            let mut bad = value.clone();
            lane(&mut bad)["engine_work_progress"][field] = if field == "source_commit" {
                json!("2a69949d0833cd0d2fe5469a1c5f319185eba5f9")
            } else {
                json!("scheduler_update_from_output_after_native_model_future")
            };
            let out = capture(&bad);
            assert_eq!(out["available"], true);
            assert_eq!(
                out["ranks"][0]["original_prefill_service_provenance"]["available"],
                false
            );
        }
    }

    #[test]
    fn optional_sg_stage_facts_preserve_legacy_typed_equality() {
        use openai_protocol::worker::WorkerLoadResponse;
        let mut value = fixture(true);
        let base = json!({"snapshot_age_seconds":0.0,"active_inventory_complete":true,"native_population":1,
            "scheduler_semantics":"continuous_batching","deadline_boundary":"first_nonempty_text",
            "deadline_origin":"engine_http_receive_monotonic","remaining_work_source":"unknown",
            "effective_capacity_tokens_per_sec":null,"capacity_source":null,
            "hard_ttft_deadline_supported":true,"cancellation_ack_supported":true,
            "abort_enforcement_enabled":true,"abort_max_lag_seconds":null,"ack_floor_seq":0,"cancellation_acks":[]});
        lane(&mut value)
            .as_object_mut()
            .unwrap()
            .extend(base.as_object().unwrap().clone());
        let row = json!({"state":"decode","first_text_emitted":false,
            "remaining_prefill_work_tokens":null,"hard_ttft_deadline_monotonic_secs":30.,"queue_position":null});
        lane(&mut value)["active_requests"][0]
            .as_object_mut()
            .unwrap()
            .extend(row.as_object().unwrap().clone());
        let before: WorkerLoadResponse = serde_json::from_value(value.clone()).unwrap();
        assert!(before.loads[0].deadline_recovery.is_some());
        lane(&mut value)["active_requests"][0]["original_prefill_observation"] =
            sg_original_facts();
        lane(&mut value)["original_prefill_service_provenance"] = json!({"available":false});
        let after: WorkerLoadResponse = serde_json::from_value(value).unwrap();
        assert!(before.loads[0]
            .deadline_recovery
            .as_ref()
            .unwrap()
            .legacy_snapshot_eq(after.loads[0].deadline_recovery.as_ref().unwrap()));
        assert!(after.loads[0]
            .deadline_recovery
            .as_ref()
            .unwrap()
            .active_requests[0]
            .original_prefill_observation
            .is_some());
        // Only the separate SG shadow can consume this optional evidence.
        // Existing deadline/ACK equality deliberately excludes it.
    }

    #[test]
    fn sg_extra_facts_share_existing_caps_and_fail_closed_without_overflow() {
        let mut old = fixture(true);
        let template = lane(&mut old)["active_requests"][0].clone();
        let mut rows = Vec::new();
        for index in 0..4000 {
            let mut row = template.clone();
            let native = format!("native-{index}");
            let attempt = format!("attempt-{index}");
            row["native_request_id"] = json!(native);
            row["attempt_id"] = json!(attempt);
            row["engine_work_state"]["native_request_id"] = row["native_request_id"].clone();
            row["engine_work_state"]["attempt_id"] = row["attempt_id"].clone();
            rows.push(row);
        }
        lane(&mut old)["active_requests"] = Value::Array(rows);
        let mut added = old.clone();
        for row in lane(&mut added)["active_requests"].as_array_mut().unwrap() {
            let mut facts = sg_original_facts();
            facts["native_request_id"] = row["native_request_id"].clone();
            facts["attempt_id"] = row["attempt_id"].clone();
            row["original_prefill_observation"] = facts;
        }
        let old_raw = serde_json::to_vec(&old).unwrap();
        let added_raw = serde_json::to_vec(&added).unwrap();
        assert!(added_raw.len() < MAX_RESPONSE_BYTES);
        let baseline = serde_json::to_value(Capture::from_body(&old_raw, 200)).unwrap();
        let captured = serde_json::to_value(Capture::from_body(&added_raw, 200)).unwrap();
        assert!(serde_json::to_vec(&captured).unwrap().len() <= MAX_CAPTURE_BYTES);
        assert!(
            captured["requests_retained"].as_u64().unwrap()
                < baseline["requests_retained"].as_u64().unwrap()
        );
        assert!(captured["requests_dropped"].as_u64().unwrap() > 0);
        assert_eq!(captured["available"], false);
        assert_eq!(captured["response_complete"], true);
        let too_large = Capture::from_body(&vec![b' '; MAX_RESPONSE_BYTES + 1], 200);
        let rejected = serde_json::to_value(too_large).unwrap();
        assert_eq!(rejected["available"], false);
        assert_eq!(rejected["response_complete"], false);
        assert!(rejected["response_sha256"].is_null());
    }

    fn fixture(sglang: bool) -> Value {
        let progress = json!({"schema_version":1,"engine_epoch":"epoch","profile_id":"profile","dp_rank":0,
            "source_commit":"c9e5799bd12fd5fefb0fb375ac537a6b7d53892e",
            "completion_boundary":if sglang {"normal_process_batch_result_return_after_native_completion_wait"} else {"scheduler_update_from_output_after_native_model_future"},
            "snapshot_monotonic_secs":20.0,"counters_available":true,"unavailable_reason":null,
            "completion_sequence":2,"cumulative":{"completed_batches":2,"completed_uncached_prefill_tokens":100,
                "completed_generation_context_tokens":3,"completed_generation_context_attention_tokens":300},
            "last_completion":{"batch_sequence":2,"request_count":1,"uncached_prefill_tokens":0,
                "generation_context_tokens":1,"generation_context_attention_tokens":101,
                "schedule_observed_monotonic_secs":19.0,"completion_observed_monotonic_secs":19.5,"host_observation_span_seconds":0.5}});
        let state = json!({"schema_version":1,"diagnostic_only":true,"native_request_id":"native","attempt_id":"attempt",
            "available":true,"unavailable_reason":null,"output_token_count":3,
            "retraction_count":if sglang {Some(1)} else {None},"preemption_count":if sglang {None} else {Some(1)},
            "abort_pending":if sglang {Some(false)} else {None},
            "abort_pending_source":if sglang {"req_to_finish_FINISH_ABORT"} else {"unknown"},"cache_release_observed":false});
        json!({"dp_rank_count":1,"loads":[{"dp_rank":0,"deadline_recovery":{
            "schema_version":1,"engine_epoch":"epoch","profile_id":"profile","dp_rank":0,
            "snapshot_seq":3,"captured_at_monotonic_secs":20.0,"engine_work_progress":progress,
            "active_requests":[{"native_request_id":"native","attempt_id":"attempt","state_seq":2,"engine_work_state":state}]}}]})
    }

    fn capture(value: &Value) -> Value {
        serde_json::to_value(Capture::from_body(&serde_json::to_vec(value).unwrap(), 200)).unwrap()
    }

    fn lane(value: &mut Value) -> &mut Value {
        &mut value["loads"][0]["deadline_recovery"]
    }

    #[test]
    fn valid_vllm_and_sglang_preserve_counts_and_unknown_flags() {
        for sglang in [false, true] {
            let value = fixture(sglang);
            let out = capture(&value);
            assert_eq!(out["available"], true);
            assert_eq!(out["response_complete"], true);
            assert_eq!(out["ranks_seen"], 1);
            assert_eq!(out["requests_seen"], 1);
            assert_eq!(out["requests_retained"], 1);
            assert_eq!(out["requests_dropped"], 0);
            let state = &out["ranks"][0]["active_requests"][0]["engine_work_state"];
            assert_eq!(state["output_token_count"], 3);
            if sglang {
                assert!(state["preemption_count"].is_null());
                assert_eq!(state["abort_pending"], false);
            } else {
                assert!(state["retraction_count"].is_null());
                assert!(state["abort_pending"].is_null());
            }
        }
    }

    #[test]
    fn full_response_digest_is_exact_even_for_malformed_diagnostics() {
        let mut value = fixture(false);
        lane(&mut value)["engine_work_progress"] = json!("invalid");
        let raw = serde_json::to_vec(&value).unwrap();
        let out = serde_json::to_value(Capture::from_body(&raw, 200)).unwrap();
        assert_eq!(
            out["response_sha256"],
            Sha256::digest(&raw)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        );
        assert_eq!(out["response_bytes"], raw.len());
        assert_eq!(out["response_complete"], true);
        assert_eq!(out["available"], false);
        let invalid = serde_json::to_value(Capture::from_body(b"{not-json}", 200)).unwrap();
        assert_eq!(
            invalid["response_sha256"],
            "aa4b143348f3e7cc620a0b03986c3a428677bfccc9f998f5a7f447f832fcf389"
        );
        assert!(invalid["ranks_seen"].is_null());
    }

    #[test]
    fn transport_and_size_markers_never_claim_complete_body_hash() {
        let out = serde_json::to_value(Capture::unavailable("transport_error", None, 17)).unwrap();
        assert_eq!(out["response_complete"], false);
        assert!(out["response_sha256"].is_null());
        assert!(out["response_bytes"].is_null());
        assert_eq!(out["bytes_observed"], 17);
        assert!(out["requests_seen"].is_null());
        let out = Capture::from_body(&vec![b' '; MAX_RESPONSE_BYTES + 1], 200);
        assert!(!out.response_complete);
        assert!(out.response_sha256.is_none());
        let out = Capture::from_body(b"error", 503);
        assert!(out.response_complete);
        assert!(out.response_sha256.is_some());
        assert_eq!(out.unavailable_reason, Some("http_status_not_success"));
    }

    #[test]
    fn missing_null_and_malformed_diagnostics_remain_unavailable() {
        for missing in [Value::Null, json!({}), json!(false), json!([])] {
            let mut value = fixture(false);
            lane(&mut value)["engine_work_progress"] = missing;
            assert_eq!(capture(&value)["available"], false);
        }
        let mut value = fixture(false);
        lane(&mut value)
            .as_object_mut()
            .unwrap()
            .remove("engine_work_progress");
        let out = capture(&value);
        assert_eq!(out["unavailable_reason"], "missing_progress_diagnostics");
        let mut value = fixture(false);
        lane(&mut value)["active_requests"][0]["engine_work_state"] = Value::Null;
        let out = capture(&value);
        assert_eq!(out["requests_retained"], 1);
        assert_eq!(out["available"], false);
        assert!(out["ranks"][0]["active_requests"][0]["engine_work_state"].is_null());
    }

    #[test]
    fn nested_identity_and_request_association_mismatches_are_not_copied() {
        for field in ["engine_epoch", "profile_id", "dp_rank"] {
            let mut value = fixture(false);
            lane(&mut value)["engine_work_progress"][field] = if field == "dp_rank" {
                json!(1)
            } else {
                json!("other")
            };
            let out = capture(&value);
            assert_eq!(out["unavailable_reason"], "progress_identity_mismatch");
            assert!(out["ranks"][0]["engine_work_progress"].is_null());
        }
        for field in ["native_request_id", "attempt_id"] {
            let mut value = fixture(false);
            lane(&mut value)["active_requests"][0]["engine_work_state"][field] = json!("other");
            let out = capture(&value);
            assert_eq!(out["available"], false);
            assert!(out["ranks"][0]["active_requests"][0]["engine_work_state"].is_null());
        }
    }

    #[test]
    fn unknown_fields_content_ids_and_unrecognized_reasons_are_stripped() {
        let mut value = fixture(false);
        value["prompt"] = json!("SECRET_CONTENT");
        let lane = lane(&mut value);
        lane["output_ids"] = json!([123, 456]);
        lane["engine_work_progress"]["arbitrary"] = json!("SECRET_CONTENT");
        lane["active_requests"][0]["engine_work_state"]["text"] = json!("SECRET_CONTENT");
        lane["active_requests"][0]["engine_work_state"]["unavailable_reason"] =
            json!("SECRET_CONTENT");
        let text = serde_json::to_string(&capture(&value)).unwrap();
        assert!(!text.contains("SECRET_CONTENT"));
        assert!(!text.contains("output_ids"));
        assert!(text.contains("unrecognized_source_reason"));
    }

    #[test]
    fn unavailable_source_preserves_partial_counters_without_inventing_zeros() {
        let mut value = fixture(false);
        let progress = &mut lane(&mut value)["engine_work_progress"];
        progress["counters_available"] = json!(false);
        progress["unavailable_reason"] = json!("unsupported_engine_configuration");
        progress["cumulative"]["completed_generation_context_tokens"] = Value::Null;
        progress
            .as_object_mut()
            .unwrap()
            .remove("completion_sequence");
        let out = capture(&value);
        let progress = &out["ranks"][0]["engine_work_progress"];
        assert_eq!(out["available"], false);
        assert_eq!(progress["counters_available"], false);
        assert_eq!(
            progress["cumulative"]["completed_uncached_prefill_tokens"],
            100
        );
        assert!(progress["cumulative"]["completed_generation_context_tokens"].is_null());
        assert!(progress["completion_sequence"].is_null());
        assert_eq!(
            progress["unavailable_reason"],
            "unsupported_engine_configuration"
        );
    }

    #[test]
    fn available_request_requires_all_declared_scalar_flag_and_reason_fields() {
        for field in [
            "unavailable_reason",
            "output_token_count",
            "retraction_count",
            "preemption_count",
            "abort_pending",
            "abort_pending_source",
            "cache_release_observed",
        ] {
            let mut value = fixture(false);
            lane(&mut value)["active_requests"][0]["engine_work_state"]
                .as_object_mut()
                .unwrap()
                .remove(field);
            let out = capture(&value);
            assert_eq!(out["available"], false, "missing {field}");
            assert!(
                out["ranks"][0]["active_requests"][0]["engine_work_state"]["available"].is_null()
            );
        }
        // A present source null is distinct from a missing field.
        assert_eq!(capture(&fixture(false))["available"], true);
    }

    #[test]
    fn explicit_abort_flag_requires_recognized_nonunknown_source() {
        for source in [Value::Null, json!("unknown"), json!("not_a_source")] {
            for pending in [false, true] {
                let mut value = fixture(true);
                let state = &mut lane(&mut value)["active_requests"][0]["engine_work_state"];
                state["abort_pending"] = json!(pending);
                state["abort_pending_source"] = source.clone();
                assert_eq!(capture(&value)["available"], false);
            }
        }
        let mut value = fixture(false);
        let state = &mut lane(&mut value)["active_requests"][0]["engine_work_state"];
        state["abort_pending"] = json!(true);
        state["abort_pending_source"] = json!("native_aborted_status_or_pending_ack");
        assert_eq!(capture(&value)["available"], true);
    }

    #[test]
    fn pinned_sglang_observation_failure_reason_is_preserved() {
        let mut value = fixture(true);
        let state = &mut lane(&mut value)["active_requests"][0]["engine_work_state"];
        state["available"] = json!(false);
        state["unavailable_reason"] = json!("request_work_metadata_observation_failed");
        let out = capture(&value);
        assert_eq!(out["available"], false);
        assert_eq!(
            out["ranks"][0]["active_requests"][0]["engine_work_state"]["unavailable_reason"],
            "request_work_metadata_observation_failed"
        );
    }

    #[test]
    fn source_fallback_without_optional_provenance_is_explicitly_unavailable() {
        let mut value = fixture(true);
        lane(&mut value)["engine_work_progress"] = json!({"schema_version":1,"engine_epoch":"epoch","profile_id":"profile","dp_rank":0,
            "snapshot_monotonic_secs":20.0,"counters_available":false,"unavailable_reason":"engine_work_diagnostic_observation_failed"});
        let out = capture(&value);
        let progress = &out["ranks"][0]["engine_work_progress"];
        assert_eq!(progress["counters_available"], false);
        assert!(progress["source_commit"].is_null());
        assert!(progress["cumulative"]["completed_batches"].is_null());
    }

    #[test]
    fn uint64_max_is_exact_and_boolean_counts_are_not_numbers() {
        let mut value = fixture(false);
        lane(&mut value)["engine_work_progress"]["cumulative"]
            ["completed_uncached_prefill_tokens"] = json!(u64::MAX);
        lane(&mut value)["active_requests"][0]["engine_work_state"]["output_token_count"] =
            json!(u64::MAX);
        let out = capture(&value);
        assert_eq!(
            out["ranks"][0]["engine_work_progress"]["cumulative"]
                ["completed_uncached_prefill_tokens"]
                .as_u64(),
            Some(u64::MAX)
        );
        assert_eq!(
            out["ranks"][0]["active_requests"][0]["engine_work_state"]["output_token_count"]
                .as_u64(),
            Some(u64::MAX)
        );
        lane(&mut value)["active_requests"][0]["engine_work_state"]["output_token_count"] =
            json!(true);
        let out = capture(&value);
        assert_eq!(out["available"], false);
        assert!(
            out["ranks"][0]["active_requests"][0]["engine_work_state"]["output_token_count"]
                .is_null()
        );
    }

    #[test]
    fn duplicate_native_ids_or_attempts_are_explicit_drops() {
        for duplicate_native in [true, false] {
            let mut value = fixture(false);
            let mut row = lane(&mut value)["active_requests"][0].clone();
            let field = if duplicate_native {
                "attempt_id"
            } else {
                "native_request_id"
            };
            row[field] = json!("different");
            row["engine_work_state"][field] = json!("different");
            lane(&mut value)["active_requests"]
                .as_array_mut()
                .unwrap()
                .push(row);
            let out = capture(&value);
            assert_eq!(out["requests_seen"], 2);
            assert_eq!(out["requests_retained"], 1);
            assert_eq!(out["requests_dropped"], 1);
            assert_eq!(out["available"], false);
        }
    }

    #[test]
    fn rank_layout_duplicates_and_caps_are_explicit() {
        let mut value = fixture(false);
        value["dp_rank_count"] = json!(2);
        let duplicate = value["loads"][0].clone();
        value["loads"].as_array_mut().unwrap().push(duplicate);
        let out = capture(&value);
        assert_eq!(out["ranks_seen"], 2);
        assert_eq!(out["ranks_retained"], 1);
        assert_eq!(out["ranks_dropped"], 1);
        let mut value = fixture(false);
        value["loads"][0]["dp_rank"] = json!(1);
        assert_eq!(capture(&value)["available"], false);
        let value =
            json!({"dp_rank_count":MAX_RANKS+1,"loads":vec![json!({"dp_rank":0});MAX_RANKS+1]});
        let out = capture(&value);
        assert_eq!(out["ranks_seen"], MAX_RANKS + 1);
        assert_eq!(out["ranks_retained"], 0);
        assert!(out["requests_seen"].is_null());
        assert!(out["requests_dropped"].is_null());
    }

    #[test]
    fn row_and_serialized_byte_limits_never_claim_full_coverage() {
        let mut value = fixture(false);
        let original = lane(&mut value)["active_requests"][0].clone();
        let rows: Vec<_> = (0..MAX_REQUESTS + 1)
            .map(|i| {
                let mut row = original.clone();
                let native = format!("native-{i}");
                let attempt = format!("attempt-{i}");
                row["native_request_id"] = json!(native);
                row["attempt_id"] = json!(attempt);
                row["engine_work_state"]["native_request_id"] = json!(native);
                row["engine_work_state"]["attempt_id"] = json!(attempt);
                row
            })
            .collect();
        lane(&mut value)["active_requests"] = Value::Array(rows);
        let raw = serde_json::to_vec(&value).unwrap();
        assert!(raw.len() <= MAX_RESPONSE_BYTES);
        let out = Capture::from_body(&raw, 200);
        assert!(!out.available);
        assert_eq!(out.requests_seen, Some(MAX_REQUESTS + 1));
        assert!(out.requests_retained <= MAX_REQUESTS);
        assert!(out.requests_dropped.unwrap() > 0);
        assert_eq!(
            out.failure_reason_counts
                .get(&FailureReason::CaptureByteLimit),
            Some(&(out.requests_dropped.unwrap() as u64))
        );
        assert!(serde_json::to_vec(&out).unwrap().len() <= MAX_CAPTURE_BYTES);
    }

    #[test]
    fn mixed_failures_count_retained_null_rows_duplicates_and_each_row_cap_drop() {
        let mut value = fixture(false);
        lane(&mut value)
            .as_object_mut()
            .unwrap()
            .remove("engine_work_progress");
        let mut first = lane(&mut value)["active_requests"][0].clone();
        first["engine_work_state"] = json!(true); // Retained as unavailable/null.
        let mut rows = vec![first.clone(), first]; // Second identity is a duplicate.
        rows.extend((0..MAX_REQUESTS).map(|i| {
            json!({
                "native_request_id":format!("native-{i}"),"attempt_id":format!("attempt-{i}"),
                "state_seq":2,"engine_work_state":null
            })
        }));
        lane(&mut value)["active_requests"] = Value::Array(rows);
        let raw = serde_json::to_vec(&value).unwrap();
        assert!(raw.len() <= MAX_RESPONSE_BYTES);
        let out = Capture::from_body(&raw, 200);
        assert!(!out.available);
        assert_eq!(out.unavailable_reason, Some("missing_progress_diagnostics"));
        assert_eq!(out.requests_seen, Some(MAX_REQUESTS + 2));
        assert_eq!(out.requests_retained, MAX_REQUESTS);
        assert_eq!(out.requests_dropped, Some(2));
        assert_eq!(
            serde_json::to_value(&out.failure_reason_counts).unwrap(),
            json!({
                "missing_progress_diagnostics":1,
                "malformed_request_diagnostics":1,
                "duplicate_request_identity":1,
                "missing_request_diagnostics":MAX_REQUESTS-1,
                "request_row_limit":1
            })
        );
        let encoded = serde_json::to_vec(&out).unwrap();
        assert!(encoded.len() <= MAX_CAPTURE_BYTES);
        assert!(!String::from_utf8(encoded).unwrap().contains("output_ids"));
    }

    #[test]
    fn source_unavailable_counts_survive_a_malformed_retained_record() {
        let mut value = fixture(false);
        let progress = &mut lane(&mut value)["engine_work_progress"];
        progress["counters_available"] = json!(false);
        progress["unavailable_reason"] = json!("unsupported_engine_configuration");
        progress["cumulative"]["completed_batches"] = json!(true);
        let state = &mut lane(&mut value)["active_requests"][0]["engine_work_state"];
        state["available"] = json!(false);
        state["unavailable_reason"] = json!("request_work_metadata_unavailable");
        state["output_token_count"] = json!(true);
        let out = capture(&value);
        assert_eq!(out["requests_retained"], 1);
        assert_eq!(out["requests_dropped"], 0);
        assert_eq!(
            out["failure_reason_counts"],
            json!({
                "malformed_progress_diagnostics":1,"source_progress_unavailable":1,
                "malformed_request_diagnostics":1,"source_request_unavailable":1
            })
        );
    }

    #[test]
    fn constructors_do_not_count_a_provisional_parse_error() {
        assert_eq!(capture(&fixture(false))["failure_reason_counts"], json!({}));
        let malformed = serde_json::to_value(Capture::from_body(b"not JSON", 200)).unwrap();
        assert_eq!(
            malformed["failure_reason_counts"],
            json!({"malformed_response":1})
        );
        let status = serde_json::to_value(Capture::from_body(b"error", 503)).unwrap();
        assert_eq!(
            status["failure_reason_counts"],
            json!({"http_status_not_success":1})
        );
        let transport = serde_json::to_value(Capture::unavailable(
            "native_body_read_timeout",
            Some(200),
            17,
        ))
        .unwrap();
        assert_eq!(
            transport["failure_reason_counts"],
            json!({"native_body_read_timeout":1})
        );
        let other = serde_json::to_value(Capture::unavailable(
            "arbitrary_static_caller_reason",
            None,
            0,
        ))
        .unwrap();
        assert_eq!(
            other["failure_reason_counts"],
            json!({"other_capture_unavailable":1})
        );
    }

    #[test]
    fn bad_strings_and_malformed_rows_are_dropped_without_content() {
        let mut value = fixture(false);
        lane(&mut value)["active_requests"][0]["native_request_id"] =
            json!("x".repeat(MAX_ID_BYTES + 1));
        lane(&mut value)["active_requests"]
            .as_array_mut()
            .unwrap()
            .push(json!(["SECRET_CONTENT"]));
        let out = capture(&value);
        assert_eq!(out["requests_seen"], 2);
        assert_eq!(out["requests_retained"], 0);
        assert_eq!(out["requests_dropped"], 2);
        assert!(!serde_json::to_string(&out)
            .unwrap()
            .contains("SECRET_CONTENT"));
    }

    #[test]
    fn bounded_malformed_inputs_do_not_panic() {
        for body in [
            b"".as_slice(),
            b"null",
            b"[]",
            b"true",
            b"{\"loads\":null}",
            b"{\"loads\":[null]}",
            b"\xff",
        ] {
            let out = Capture::from_body(body, 200);
            assert!(!out.available);
            assert!(out.response_sha256.is_some());
            assert!(serde_json::to_vec(&out).unwrap().len() <= MAX_CAPTURE_BYTES);
        }
    }
}
