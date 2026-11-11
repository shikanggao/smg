//! One bounded, optional vLLM input path; completed poll-window work is not GPU time.
use openai_protocol::{
    native_deadline::{
        NativeCompletedWorkProgress, NativeDeadlineTelemetry, NativeJointWorkServiceCapability,
        MAX_EXACT_PROMPT_WORK,
    },
    worker::WorkerLoadResponse,
};
use serde::Serialize;

const MAX_GAP: f64 = 2.0;
const VLLM_SOURCE: &str = "2a69949bdadf0e8942b7a1619b229cb475beef20";
const COMPLETION_BOUNDARY: &str = "scheduler_update_from_output_after_native_model_future";

#[derive(Clone, Debug)]
struct Point {
    work: NativeCompletedWorkProgress,
    receipt: f64,
    native_sequence: u64,
    native_captured: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct Selection {
    pub method: &'static str,
    pub status: &'static str,
    pub reason: &'static str,
    pub capacity_tokens_per_second: Option<f64>,
    pub engine_interval_start: Option<f64>,
    pub engine_interval_end: Option<f64>,
    pub receipt_interval_seconds: Option<f64>,
    pub completion_sequence_start: Option<u64>,
    pub completion_sequence_end: Option<u64>,
    pub completed_uncached_prefill_tokens: Option<u64>,
    pub native_snapshot_seq: Option<u64>,
    pub native_captured_at_monotonic_secs: Option<f64>,
}
impl Default for Selection {
    fn default() -> Self {
        Self {
            method: "vllm_native_full_prompt_completed_prefill_window_v1",
            status: "unavailable",
            reason: "first_valid_point",
            capacity_tokens_per_second: None,
            engine_interval_start: None,
            engine_interval_end: None,
            receipt_interval_seconds: None,
            completion_sequence_start: None,
            completion_sequence_end: None,
            completed_uncached_prefill_tokens: None,
            native_snapshot_seq: None,
            native_captured_at_monotonic_secs: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct History {
    previous: Option<Point>,
    current: Selection,
    // A changed optional envelope cannot become trustworthy by repeating the
    // same native publication. Only a genuine new publication can reseed.
    blocked_snapshot: Option<(String, String, u64, u64)>,
}

fn snapshot_key(lane: &NativeDeadlineTelemetry) -> (String, String, u64, u64) {
    (
        lane.engine_epoch.clone(),
        lane.profile_id.clone(),
        lane.snapshot_seq,
        lane.captured_at_monotonic_secs.to_bits(),
    )
}

fn point(
    load: Option<&WorkerLoadResponse>,
    accepted: bool,
    receipt: f64,
) -> Result<Point, &'static str> {
    let load = load
        .filter(|_| accepted)
        .ok_or("native_point_unavailable")?;
    if !load.ranks_are_dp_ranks()
        || load.dp_rank_count != 1
        || load.loads.len() != 1
        || load.loads[0].dp_rank != 0
    {
        return Err("unsupported_rank_population");
    }
    if load.loads[0]
        .prefill_metric_provenance
        .as_ref()
        .is_some_and(|p| p.counter_reset)
    {
        return Err("raw_counter_reset");
    }
    let lane = load.loads[0]
        .deadline_recovery
        .as_ref()
        .ok_or("native_point_unavailable")?;
    if lane.joint_work_service_capability
        != Some(NativeJointWorkServiceCapability::VllmOriginalFullPromptCompletedPrefillV1)
    {
        return Err("vllm_joint_capability_absent");
    }
    let p = lane
        .engine_work_progress
        .as_ref()
        .ok_or("completed_work_unavailable")?;
    if p.schema_version != 1
        || !p.counters_available
        || p.dp_rank != 0
        || p.engine_epoch != lane.engine_epoch
        || p.profile_id != lane.profile_id
        || p.dp_rank != lane.dp_rank
        || p.engine_epoch.is_empty()
        || p.profile_id.is_empty()
        || p.source_commit != VLLM_SOURCE
        || p.completion_boundary != COMPLETION_BOUNDARY
        || !receipt.is_finite()
        || !p.snapshot_monotonic_secs.is_finite()
        || !lane.snapshot_age_seconds.is_finite()
        || !(0.0..=30.0).contains(&lane.snapshot_age_seconds)
        || p.snapshot_monotonic_secs != lane.captured_at_monotonic_secs
    {
        return Err("completed_work_contract_invalid");
    }
    Ok(Point {
        work: p.clone(),
        receipt,
        native_sequence: lane.snapshot_seq,
        native_captured: lane.captured_at_monotonic_secs,
    })
}

fn totals(p: &NativeCompletedWorkProgress) -> [u64; 4] {
    let c = p.cumulative;
    [
        c.completed_batches,
        c.completed_uncached_prefill_tokens,
        c.completed_generation_context_tokens,
        c.completed_generation_context_attention_tokens,
    ]
}

impl History {
    pub fn reject_duplicate(&mut self, lane: &NativeDeadlineTelemetry) {
        self.previous = None;
        self.blocked_snapshot = Some(snapshot_key(lane));
        self.current = Selection::default();
        self.current.reason = "changed_optional_native_duplicate";
        self.current.native_snapshot_seq = Some(lane.snapshot_seq);
        self.current.native_captured_at_monotonic_secs = Some(lane.captured_at_monotonic_secs);
    }

    pub fn observe(&mut self, load: Option<&WorkerLoadResponse>, accepted: bool, receipt: f64) {
        self.current = Selection::default();
        let current = match point(load, accepted, receipt) {
            Ok(p) => p,
            Err(reason) => {
                self.previous = None;
                self.current.reason = reason;
                return;
            }
        };
        let lane = load.unwrap().loads[0].deadline_recovery.as_ref().unwrap();
        if self.blocked_snapshot.as_ref() == Some(&snapshot_key(lane)) {
            self.reject_duplicate(lane);
            return;
        }
        self.blocked_snapshot = None;
        if self.previous.as_ref().is_some_and(|old| {
            old.native_sequence == current.native_sequence
                && old.native_captured == current.native_captured
                && old.work.engine_epoch == current.work.engine_epoch
                && old.work.profile_id == current.work.profile_id
                && old.work != current.work
        }) {
            self.reject_duplicate(lane);
            return;
        }
        if let Some(old) = &self.previous {
            let a = &old.work;
            let b = &current.work;
            let dt = b.snapshot_monotonic_secs - a.snapshot_monotonic_secs;
            let receipt_dt = current.receipt - old.receipt;
            let deltas: Option<Vec<_>> = totals(b)
                .into_iter()
                .zip(totals(a))
                .map(|(v, p)| v.checked_sub(p))
                .collect();
            let sequence = b.completion_sequence.checked_sub(a.completion_sequence);
            let reason = if (
                a.engine_epoch.as_str(),
                a.profile_id.as_str(),
                a.dp_rank,
                a.source_commit.as_str(),
                a.completion_boundary.as_str(),
            ) != (
                b.engine_epoch.as_str(),
                b.profile_id.as_str(),
                b.dp_rank,
                b.source_commit.as_str(),
                b.completion_boundary.as_str(),
            ) {
                Some("identity_changed")
            } else if dt <= 0.0 || receipt_dt <= 0.0 {
                Some("nonincreasing_clock")
            } else if dt > MAX_GAP || receipt_dt > MAX_GAP {
                Some("capture_gap")
            } else if deltas.is_none() || sequence.is_none() {
                Some("counter_or_sequence_reversed")
            } else if sequence.unwrap() != deltas.as_ref().unwrap()[0] {
                Some("completion_count_mismatch")
            } else if sequence == Some(0) && deltas.as_ref().unwrap().iter().any(|v| *v > 0) {
                Some("counter_changed_without_completion")
            } else {
                None
            };
            if let Some(reason) = reason {
                self.current.reason = reason;
            } else {
                let prefill = deltas.unwrap()[1];
                self.current.engine_interval_start = Some(a.snapshot_monotonic_secs);
                self.current.engine_interval_end = Some(b.snapshot_monotonic_secs);
                self.current.receipt_interval_seconds = Some(receipt_dt);
                self.current.completion_sequence_start = Some(a.completion_sequence);
                self.current.completion_sequence_end = Some(b.completion_sequence);
                self.current.completed_uncached_prefill_tokens = Some(prefill);
                if prefill > MAX_EXACT_PROMPT_WORK {
                    self.current.reason = "completed_prefill_delta_outside_exact_f64_range";
                } else {
                    let rate = prefill as f64 / dt;
                    if rate.is_finite() && rate > 0.0 {
                        self.current.status = "available";
                        self.current.reason = "positive_completed_prefill_poll_window";
                        self.current.capacity_tokens_per_second = Some(rate);
                    } else {
                        self.current.reason = "zero_or_invalid_completed_prefill_rate";
                    }
                }
            }
        }
        self.current.native_snapshot_seq = Some(current.native_sequence);
        self.current.native_captured_at_monotonic_secs = Some(current.native_captured);
        self.previous = Some(current);
    }

    pub fn selection(&self, lane: &NativeDeadlineTelemetry) -> Option<Selection> {
        if lane.joint_work_service_capability
            != Some(NativeJointWorkServiceCapability::VllmOriginalFullPromptCompletedPrefillV1)
        {
            return None; // Never vacuously activate SG/legacy, including an empty inventory.
        }
        let mut result = self.current.clone();
        if result.native_snapshot_seq != Some(lane.snapshot_seq)
            || result.native_captured_at_monotonic_secs != Some(lane.captured_at_monotonic_secs)
        {
            result.status = "unavailable";
            result.reason = "service_window_native_snapshot_mismatch";
            result.capacity_tokens_per_second = None;
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(seq: u64, time: f64, prefill: u64) -> WorkerLoadResponse {
        let cumulative = serde_json::json!({
            "completed_batches":seq,"completed_uncached_prefill_tokens":prefill,
            "completed_generation_context_tokens":seq,"completed_generation_context_attention_tokens":seq
        });
        let progress = serde_json::json!({
            "schema_version":1,"engine_epoch":"epoch","profile_id":"profile","dp_rank":0,
            "source_commit":VLLM_SOURCE,"completion_boundary":COMPLETION_BOUNDARY,
            "snapshot_monotonic_secs":time,"counters_available":true,"completion_sequence":seq,
            "cumulative":cumulative
        });
        let lane = serde_json::json!({
            "schema_version":1,"engine_epoch":"epoch","profile_id":"profile","dp_rank":0,
            "snapshot_seq":seq,"captured_at_monotonic_secs":time,"snapshot_age_seconds":0.0,
            "active_inventory_complete":true,"native_population":0,"scheduler_semantics":"continuous_batching",
            "deadline_boundary":"first_nonempty_text","deadline_origin":"engine_http_receive_monotonic",
            "remaining_work_source":"unknown","effective_capacity_tokens_per_sec":null,"capacity_source":null,
            "hard_ttft_deadline_supported":true,"cancellation_ack_supported":true,"abort_enforcement_enabled":true,
            "abort_max_lag_seconds":null,"ack_floor_seq":0,"active_requests":[],"cancellation_acks":[],
            "joint_work_service_capability":"vllm_original_full_prompt_completed_prefill_v1",
            "engine_work_progress":progress
        });
        serde_json::from_value(serde_json::json!({
            "dp_rank_count":1,"loads":[{"dp_rank":0,"deadline_recovery":lane}]
        }))
        .unwrap()
    }
    fn lane(load: &WorkerLoadResponse) -> &NativeDeadlineTelemetry {
        load.loads[0].deadline_recovery.as_ref().unwrap()
    }
    fn progress(load: &mut WorkerLoadResponse) -> &mut NativeCompletedWorkProgress {
        load.loads[0]
            .deadline_recovery
            .as_mut()
            .unwrap()
            .engine_work_progress
            .as_mut()
            .unwrap()
    }
    #[test]
    fn positive_window_uses_engine_clock_and_zero_is_not_a_service_estimate() {
        let mut h = History::default();
        let a = load(1, 10.0, 100);
        let b = load(2, 10.5, 300);
        h.observe(Some(&a), true, 100.0);
        h.observe(Some(&b), true, 100.8);
        let result = h.selection(lane(&b)).unwrap();
        assert_eq!(result.capacity_tokens_per_second, Some(400.0));
        assert!((result.receipt_interval_seconds.unwrap() - 0.8).abs() < 1e-12);
        let zero = load(3, 11.0, 300);
        h.observe(Some(&zero), true, 101.0);
        assert_eq!(
            h.selection(lane(&zero)).unwrap().reason,
            "zero_or_invalid_completed_prefill_rate"
        );
    }
    #[test]
    fn negative_point_breaks_history_and_source_clock_caps_are_explicit() {
        for kind in 0..8 {
            let mut h = History::default();
            let a = load(1, 10.0, 100);
            h.observe(Some(&a), true, 100.0);
            let mut bad = load(2, 11.0, 200);
            match kind {
                0 => {
                    bad.loads[0]
                        .deadline_recovery
                        .as_mut()
                        .unwrap()
                        .joint_work_service_capability = None
                }
                1 => progress(&mut bad).snapshot_monotonic_secs += 0.000001,
                2 => progress(&mut bad).source_commit = "other".into(),
                3 => progress(&mut bad).counters_available = false,
                4 => progress(&mut bad).schema_version = 2,
                5 => progress(&mut bad).completion_boundary = "scheduled".into(),
                6 => progress(&mut bad).engine_epoch = "other".into(),
                _ => progress(&mut bad).profile_id = "other".into(),
            }
            h.observe(Some(&bad), true, 101.0);
            let c = load(3, 12.0, 300);
            h.observe(Some(&c), true, 102.0);
            assert_eq!(h.selection(lane(&c)).unwrap().reason, "first_valid_point");
            let d = load(4, 13.0, 400);
            h.observe(Some(&d), true, 103.0);
            assert_eq!(
                h.selection(lane(&d)).unwrap().capacity_tokens_per_second,
                Some(100.0)
            );
        }
    }
    #[test]
    fn invalid_pairs_never_create_a_rate_and_next_interval_uses_current_point() {
        for kind in 0..7 {
            let mut h = History::default();
            let a = load(1, 10.0, 100);
            h.observe(Some(&a), true, 100.0);
            let mut b = load(2, 11.0, 200);
            match kind {
                0 => {
                    progress(&mut b)
                        .cumulative
                        .completed_uncached_prefill_tokens = 99
                }
                1 => progress(&mut b).completion_sequence = 3,
                2 => progress(&mut b).cumulative.completed_batches = 3,
                3 => {
                    progress(&mut b).completion_sequence = 1;
                    progress(&mut b).cumulative.completed_batches = 1;
                }
                4 => {
                    progress(&mut b).snapshot_monotonic_secs = 13.0;
                    b.loads[0]
                        .deadline_recovery
                        .as_mut()
                        .unwrap()
                        .captured_at_monotonic_secs = 13.0;
                }
                5 => {
                    progress(&mut b)
                        .cumulative
                        .completed_uncached_prefill_tokens = MAX_EXACT_PROMPT_WORK + 101
                }
                _ => {
                    progress(&mut b)
                        .cumulative
                        .completed_generation_context_tokens = 0
                }
            }
            h.observe(Some(&b), true, 101.0);
            assert!(h
                .selection(lane(&b))
                .unwrap()
                .capacity_tokens_per_second
                .is_none());
        }
    }
    #[test]
    fn absent_capability_never_activates_even_with_empty_inventory_or_valid_counters() {
        let mut h = History::default();
        let a = load(1, 10.0, 100);
        h.observe(Some(&a), true, 100.0);
        let mut b = load(2, 11.0, 200);
        h.observe(Some(&b), true, 101.0);
        b.loads[0]
            .deadline_recovery
            .as_mut()
            .unwrap()
            .joint_work_service_capability = None;
        assert!(h.selection(lane(&b)).is_none());
        h.observe(Some(&b), false, 101.0);
        assert!(h.previous.is_none());
    }

    #[test]
    fn changed_counter_duplicate_is_never_a_window_endpoint_even_if_repeated() {
        let mut h = History::default();
        h.observe(Some(&load(1, 10.0, 100)), true, 100.0);
        let good = load(2, 11.0, 200);
        h.observe(Some(&good), true, 101.0);
        assert_eq!(
            h.selection(lane(&good)).unwrap().capacity_tokens_per_second,
            Some(100.0)
        );
        let changed = load(2, 11.0, 0);
        h.observe(Some(&changed), true, 101.1);
        h.observe(Some(&changed), true, 101.2);
        assert!(h.previous.is_none());
        assert_eq!(
            h.selection(lane(&changed)).unwrap().reason,
            "changed_optional_native_duplicate"
        );
        let next = load(3, 12.0, 300);
        h.observe(Some(&next), true, 102.0);
        assert_eq!(
            h.selection(lane(&next)).unwrap().reason,
            "first_valid_point"
        );
        let following = load(4, 13.0, 400);
        h.observe(Some(&following), true, 103.0);
        assert_eq!(
            h.selection(lane(&following))
                .unwrap()
                .capacity_tokens_per_second,
            Some(100.0)
        );
    }

    #[test]
    fn identical_duplicate_remains_unready_but_preserves_safe_next_endpoint() {
        let mut h = History::default();
        let a = load(1, 10.0, 100);
        h.observe(Some(&a), true, 100.0);
        h.observe(Some(&a), true, 100.1);
        assert_eq!(h.selection(lane(&a)).unwrap().reason, "nonincreasing_clock");
        let b = load(2, 11.0, 200);
        h.observe(Some(&b), true, 101.0);
        assert_eq!(
            h.selection(lane(&b)).unwrap().capacity_tokens_per_second,
            Some(100.0)
        );
    }
}
