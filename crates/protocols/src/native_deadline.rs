//! Opt-in native scheduler telemetry. This is not an admission policy.
//! Monotonic timestamps share the engine's clock, not the gateway's clock.
use serde::{Deserialize, Serialize};

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
    use crate::worker::WorkerLoadResponse;

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
