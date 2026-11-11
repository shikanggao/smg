//! Opt-in native scheduler telemetry. This is not an admission policy.
//! Monotonic timestamps share the engine's clock, not the gateway's clock.
use serde::{Deserialize, Serialize};

/// Integer work converted to f64 must remain exactly representable.
pub const MAX_EXACT_PROMPT_WORK: u64 = 1 << 53;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeDeadlineTelemetry {
    pub schema_version: u32,
    pub engine_epoch: String,
    pub profile_id: String,
    pub dp_rank: i32,
    pub snapshot_seq: u64,
    pub captured_at_monotonic_secs: f64,
    pub snapshot_age_seconds: f64,
    pub active_inventory_complete: bool,
    pub native_population: usize,
    pub scheduler_semantics: String,
    pub deadline_boundary: String,
    pub deadline_origin: Option<String>,
    pub remaining_work_source: String,
    pub effective_capacity_tokens_per_sec: Option<f64>,
    pub capacity_source: Option<String>,
    pub hard_ttft_deadline_supported: bool,
    pub cancellation_ack_supported: bool,
    pub abort_enforcement_enabled: bool,
    pub abort_max_lag_seconds: Option<f64>,
    /// All ACK events at/after this state sequence remain in the journal.
    /// Initially zero; advances only when an ACK is evicted.
    pub ack_floor_seq: u64,
    pub active_requests: Vec<NativeDeadlineRequest>,
    pub cancellation_acks: Vec<NativeCancellationAck>,
    /// Experimental evidence is independent of the legacy deadline contract.
    /// Invalid optional evidence must never discard the native base payload.
    #[serde(
        default,
        deserialize_with = "optional_joint_capability",
        skip_serializing_if = "Option::is_none"
    )]
    pub joint_work_service_capability: Option<NativeJointWorkServiceCapability>,
    #[serde(
        default,
        deserialize_with = "optional_completed_work",
        skip_serializing_if = "Option::is_none"
    )]
    pub engine_work_progress: Option<NativeCompletedWorkProgress>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeDeadlineRequest {
    /// Null for requests that did not receive the dedicated gateway header.
    pub attempt_id: Option<String>,
    pub native_request_id: Option<String>,
    pub state_seq: u64,
    pub remaining_prefill_work_tokens: Option<f64>,
    pub hard_ttft_deadline_monotonic_secs: Option<f64>,
    pub queue_position: Option<u64>,
    pub state: String,
    pub first_text_emitted: Option<bool>,
    /// Full ordinary-text prompt length; not cached/uncached remaining work.
    #[serde(
        default,
        deserialize_with = "optional_positive_prompt_tokens",
        skip_serializing_if = "Option::is_none"
    )]
    pub original_full_prompt_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub enum NativeJointWorkServiceCapability {
    #[serde(rename = "vllm_original_full_prompt_completed_prefill_v1")]
    VllmOriginalFullPromptCompletedPrefillV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeCompletedWorkProgress {
    pub schema_version: u32,
    pub engine_epoch: String,
    pub profile_id: String,
    pub dp_rank: i32,
    pub source_commit: String,
    pub completion_boundary: String,
    pub snapshot_monotonic_secs: f64,
    pub counters_available: bool,
    pub completion_sequence: u64,
    pub cumulative: NativeCompletedWorkCounters,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeCompletedWorkCounters {
    pub completed_batches: u64,
    pub completed_uncached_prefill_tokens: u64,
    pub completed_generation_context_tokens: u64,
    pub completed_generation_context_attention_tokens: u64,
}

impl NativeDeadlineTelemetry {
    /// Equality of the pre-existing contract only. Optional experimental model
    /// evidence has its own invalidation; it must not change legacy ACK/load
    /// acceptance when absent, malformed, or changed under a duplicate.
    pub fn legacy_snapshot_eq(&self, other: &Self) -> bool {
        self.schema_version == other.schema_version
            && self.engine_epoch == other.engine_epoch
            && self.profile_id == other.profile_id
            && self.dp_rank == other.dp_rank
            && self.snapshot_seq == other.snapshot_seq
            && self.captured_at_monotonic_secs == other.captured_at_monotonic_secs
            && self.snapshot_age_seconds == other.snapshot_age_seconds
            && self.active_inventory_complete == other.active_inventory_complete
            && self.native_population == other.native_population
            && self.scheduler_semantics == other.scheduler_semantics
            && self.deadline_boundary == other.deadline_boundary
            && self.deadline_origin == other.deadline_origin
            && self.remaining_work_source == other.remaining_work_source
            && self.effective_capacity_tokens_per_sec == other.effective_capacity_tokens_per_sec
            && self.capacity_source == other.capacity_source
            && self.hard_ttft_deadline_supported == other.hard_ttft_deadline_supported
            && self.cancellation_ack_supported == other.cancellation_ack_supported
            && self.abort_enforcement_enabled == other.abort_enforcement_enabled
            && self.abort_max_lag_seconds == other.abort_max_lag_seconds
            && self.ack_floor_seq == other.ack_floor_seq
            && self.cancellation_acks == other.cancellation_acks
            && self.active_requests.len() == other.active_requests.len()
            && self
                .active_requests
                .iter()
                .zip(&other.active_requests)
                .all(|(a, b)| {
                    a.attempt_id == b.attempt_id
                        && a.native_request_id == b.native_request_id
                        && a.state_seq == b.state_seq
                        && a.remaining_prefill_work_tokens == b.remaining_prefill_work_tokens
                        && a.hard_ttft_deadline_monotonic_secs
                            == b.hard_ttft_deadline_monotonic_secs
                        && a.queue_position == b.queue_position
                        && a.state == b.state
                        && a.first_text_emitted == b.first_text_emitted
                })
    }
}

fn optional_positive_prompt_tokens<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value
        .and_then(|v| v.as_u64())
        .filter(|n| *n > 0 && *n <= MAX_EXACT_PROMPT_WORK))
}

fn optional_joint_capability<'de, D>(
    deserializer: D,
) -> Result<Option<NativeJointWorkServiceCapability>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value.and_then(|v| serde_json::from_value(v).ok()))
}

fn optional_completed_work<'de, D>(
    deserializer: D,
) -> Result<Option<NativeCompletedWorkProgress>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value.and_then(|v| serde_json::from_value(v).ok()))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeCancellationAck {
    pub attempt_id: String,
    pub native_request_id: Option<String>,
    pub event_seq: u64,
    pub ack_monotonic_secs: f64,
    pub reason: String,
}

/// A malformed optional side channel must not discard otherwise valid raw
/// load measurements or change legacy admission readiness.
pub(crate) fn optional_telemetry<'de, D>(
    deserializer: D,
) -> Result<Option<NativeDeadlineTelemetry>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value.and_then(|value| serde_json::from_value(value).ok()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::WorkerLoadResponse;

    fn base() -> serde_json::Value {
        serde_json::json!({
            "schema_version":1,"engine_epoch":"epoch","profile_id":"profile","dp_rank":0,
            "snapshot_seq":1,"captured_at_monotonic_secs":1.0,"snapshot_age_seconds":0.0,
            "active_inventory_complete":true,"native_population":1,"scheduler_semantics":"continuous_batching",
            "deadline_boundary":"first_nonempty_text","deadline_origin":"engine_http_receive_monotonic",
            "remaining_work_source":"unknown","effective_capacity_tokens_per_sec":null,"capacity_source":null,
            "hard_ttft_deadline_supported":true,"cancellation_ack_supported":true,"abort_enforcement_enabled":true,
            "abort_max_lag_seconds":null,"ack_floor_seq":0,"cancellation_acks":[],
            "active_requests":[{"attempt_id":"a","native_request_id":"n","state_seq":1,
                "remaining_prefill_work_tokens":null,"hard_ttft_deadline_monotonic_secs":10.0,
                "queue_position":null,"state":"waiting","first_text_emitted":false}]
        })
    }

    #[test]
    fn malformed_joint_evidence_is_local_and_does_not_change_legacy_equality() {
        let original: NativeDeadlineTelemetry = serde_json::from_value(base()).unwrap();
        for bad in [
            serde_json::json!(true),
            serde_json::json!({}),
            serde_json::json!("future"),
            serde_json::Value::Null,
        ] {
            let mut value = base();
            value["joint_work_service_capability"] = bad.clone();
            value["engine_work_progress"] = bad.clone();
            value["active_requests"][0]["original_full_prompt_tokens"] = bad;
            let parsed: NativeDeadlineTelemetry = serde_json::from_value(value).unwrap();
            assert!(parsed.joint_work_service_capability.is_none());
            assert!(parsed.engine_work_progress.is_none());
            assert!(parsed.active_requests[0]
                .original_full_prompt_tokens
                .is_none());
            assert!(parsed.legacy_snapshot_eq(&original));
        }
        let mut value = base();
        value["joint_work_service_capability"] =
            serde_json::json!("vllm_original_full_prompt_completed_prefill_v1");
        value["active_requests"][0]["original_full_prompt_tokens"] = serde_json::json!(123);
        let parsed: NativeDeadlineTelemetry = serde_json::from_value(value).unwrap();
        assert!(parsed.legacy_snapshot_eq(&original));
        assert_ne!(parsed, original);
    }

    #[test]
    fn native_full_prompt_count_rejects_lossy_or_noninteger_values_only_locally() {
        for bad in [
            serde_json::json!(0),
            serde_json::json!(-1),
            serde_json::json!(1.5),
            serde_json::json!(true),
            serde_json::json!(MAX_EXACT_PROMPT_WORK + 1),
            serde_json::json!(u64::MAX),
        ] {
            let mut value = base();
            value["active_requests"][0]["original_full_prompt_tokens"] = bad;
            let parsed: NativeDeadlineTelemetry = serde_json::from_value(value).unwrap();
            assert!(parsed.active_requests[0]
                .original_full_prompt_tokens
                .is_none());
        }
        let mut value = base();
        value["active_requests"][0]["original_full_prompt_tokens"] =
            serde_json::json!(MAX_EXACT_PROMPT_WORK);
        let parsed: NativeDeadlineTelemetry = serde_json::from_value(value).unwrap();
        assert_eq!(
            parsed.active_requests[0].original_full_prompt_tokens,
            Some(MAX_EXACT_PROMPT_WORK)
        );
    }

    #[test]
    fn malformed_optional_native_contract_preserves_raw_load() {
        for bad in [
            serde_json::json!("malformed"),
            serde_json::json!({"schema_version":1}),
            serde_json::Value::Null,
        ] {
            let load: WorkerLoadResponse = serde_json::from_value(serde_json::json!({"loads":[{"dp_rank":0,"num_waiting_reqs":9,"num_running_reqs":2,"token_usage":0.4,"deadline_recovery":bad}]})).unwrap();
            assert_eq!(load.loads[0].num_waiting_reqs, 9);
            assert_eq!(load.loads[0].num_running_reqs, 2);
            assert_eq!(load.loads[0].token_usage, 0.4);
            assert!(load.loads[0].deadline_recovery.is_none());
        }
    }

    #[test]
    fn malformed_optional_prefill_provenance_preserves_raw_load() {
        for bad in [
            serde_json::json!("malformed"),
            serde_json::json!([1, 2]),
            serde_json::Value::Null,
        ] {
            let load: WorkerLoadResponse = serde_json::from_value(serde_json::json!({"loads":[{
                "dp_rank":0,"num_waiting_reqs":9,"num_running_reqs":2,"token_usage":0.4,
                "prefill_metric_provenance":bad
            }]}))
            .unwrap();
            assert_eq!(load.loads[0].num_waiting_reqs, 9);
            assert_eq!(load.loads[0].num_running_reqs, 2);
            assert_eq!(load.loads[0].token_usage, 0.4);
            assert!(load.loads[0].prefill_metric_provenance.is_none());
        }
    }
}
