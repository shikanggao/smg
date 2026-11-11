//! SG-only output shadow: existing uncached queue basis with native deadlines.
//! Neither original remaining-work/ACK state nor the C2 training path changes.
use super::*;

const METHOD: &str = "sg_prepared_uncached_work_native_deadline_shadow_v1";
const SG_SOURCE: &str = "c9e5799bd12fd5fefb0fb375ac537a6b7d53892e";

#[derive(Clone, Debug, PartialEq, Serialize)]
struct NativeKey {
    engine_epoch: String,
    profile_id: String,
    dp_rank: i32,
    snapshot_seq: u64,
    captured_at_monotonic_secs: f64,
}
impl NativeKey {
    fn of(lane: &NativeDeadlineTelemetry) -> Self {
        Self {
            engine_epoch: lane.engine_epoch.clone(),
            profile_id: lane.profile_id.clone(),
            dp_rank: lane.dp_rank,
            snapshot_seq: lane.snapshot_seq,
            captured_at_monotonic_secs: lane.captured_at_monotonic_secs,
        }
    }
}

/// Owned at raw publication and carried on the existing bounded native task.
/// This is historical estimated native UNCACHED-token capacity, not completed
/// service, full-original-token/s, cache-hit proof or physical GPU capacity.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct SgRawEvidence {
    calibration_epoch: u64,
    barrier_revision: u64,
    raw_poll: f64,
    raw_expiry: f64,
    source: (u64, u64, String),
    size: basis::Provenance,
    admission_capacity: f64,
    selected_capacity: f64,
    native_waiting: f64,
    raw_queue_estimate: f64,
    capacity: started_capacity::Selection,
}

#[derive(Clone, Debug)]
pub(super) struct Bundle {
    raw: SgRawEvidence,
    native: NativeKey,
    available: f64,
}

#[derive(Clone, Debug, Default)]
pub(super) struct State {
    pub(super) raw_source: Option<started_capacity::SourceKey>,
    revision: u64,
    pub(super) bundle: Option<Bundle>,
    blocked_publication: Option<NativeKey>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SgProjection {
    method: &'static str,
    selected: bool,
    supported: bool,
    available: bool,
    pub total_ttft_seconds: f64,
    capacity_tokens_per_second: f64,
    selected_capacity_basis: &'static str,
    raw_evidence: SgRawEvidence,
    native_tuple: NativeKey,
    original_native_receipt: f64,
    original_native_expiry: f64,
    pair_available_at_gateway_seconds: f64,
    guarded_cutoff_gateway_seconds: f64,
    barrier_revision_at_guard: u64,
    native_waiting_proxy_rows: usize,
    original_prefill_zero_rows: usize,
    waiting_proxy_work_tokens: f64,
    other_retained_work_tokens: f64,
    raw_queue_to_native_waiting_difference_tokens: f64,
    legacy_projection_supported: bool,
    legacy_projection_reason: &'static str,
    legacy_selected_total_seconds: Option<f64>,
    projection: Projection,
    interpretation: &'static str,
}

impl Native {
    pub fn invalidate_sg_raw(&mut self) {
        // A fresh/reset non-SG state has never issued a token in this
        // calibration epoch. Its caller's generation/epoch fence rejects
        // pre-reset tasks; do not mutate unrelated legacy default state.
        if self.sg_raw_basis.raw_source.is_none()
            && self.sg_raw_basis.bundle.is_none()
            && self.sg_raw_basis.revision == 0
        {
            return;
        }
        self.sg_raw_basis.revision = self.sg_raw_basis.revision.saturating_add(1);
        self.sg_raw_basis.bundle = None;
        if let Some(snapshot) = &self.snapshot {
            self.sg_raw_basis.blocked_publication = Some(NativeKey::of(&snapshot.lane));
        }
    }

    pub fn sg_raw_evidence(
        &mut self,
        started: f64,
        max_age: f64,
        calibration_epoch: u64,
        prepared: Option<basis::Provenance>,
        admission_capacity: Option<f64>,
        native_waiting: Option<f64>,
        floor: f64,
    ) -> Option<SgRawEvidence> {
        let proof = (|| {
            let size = prepared?;
            let source_key = self.sg_raw_basis.raw_source?;
            let source = self.capacity_key.clone()?;
            let original = admission_capacity.filter(|v| positive(*v))?;
            let waiting = native_waiting.filter(|v| nonnegative(*v))?;
            if !started.is_finite()
                || !positive(max_age)
                || self.sg_raw_basis.revision == u64::MAX
                || source.2 != "prefill_started_requests"
                || !self.capacity_poll.is_some_and(|(poll, valid, cap)| {
                    poll == started && valid && cap.is_some_and(|c| close(c, original))
                })
                || !matches!(size.kind, basis::Kind::Observed)
                || size.progress_source != "prefill_started_requests"
                || size.counter_epoch != source_key.counter_epoch
                || size.dp_rank != source_key.dp_rank
                || size.dp_rank != 0
                || size.observed_sample_count == 0
                || !positive(size.blended_queue_tokens_per_request)
            {
                return None;
            }
            let capacity = self
                .started_requests
                .selection(started, Some(source_key), floor);
            let current = capacity.same_poll_observed_size.as_ref()?;
            if capacity.status != "available"
                || current.status != "available"
                || current.current_observed_sample_count != Some(size.observed_sample_count)
                || !size
                    .unshrunk_observed_mean
                    .zip(current.current_unshrunk_size_mean)
                    .is_some_and(|(a, b)| positive(a) && positive(b) && close(a, b))
            {
                return None;
            }
            let selected = current.used_capacity.filter(|v| positive(*v))?;
            let queue = waiting * size.blended_queue_tokens_per_request;
            let expiry = started + max_age;
            if !nonnegative(queue) || !expiry.is_finite() {
                return None;
            }
            Some(SgRawEvidence {
                calibration_epoch,
                barrier_revision: self.sg_raw_basis.revision,
                raw_poll: started,
                raw_expiry: expiry,
                source,
                size,
                admission_capacity: original,
                selected_capacity: selected,
                native_waiting: waiting,
                raw_queue_estimate: queue,
                capacity,
            })
        })();
        if proof.is_none() {
            self.invalidate_sg_raw();
        }
        proof
    }

    pub fn commit_sg_raw(
        &mut self,
        evidence: Option<SgRawEvidence>,
        raw_poll: f64,
        calibration_epoch: u64,
        received: f64,
    ) {
        self.sg_raw_basis.bundle = None;
        let accepted = (|| {
            let raw = evidence?;
            let s = self.snapshot.as_ref()?;
            let key = NativeKey::of(&s.lane);
            if !self.valid
                || self.incomplete
                || s.raw_poll != raw_poll
                || raw.raw_poll != raw_poll
                || raw.calibration_epoch != calibration_epoch
                || raw.barrier_revision != self.sg_raw_basis.revision
                || self.capacity_key.as_ref() != Some(&raw.source)
                || !received.is_finite()
                || received < raw.raw_poll
                || received > raw.raw_expiry
                || self.sg_raw_basis.blocked_publication.as_ref() == Some(&key)
                || !key.profile_id.starts_with("sglang-c9e5799:")
                || key.dp_rank != 0
                || s.lane.joint_work_service_capability.is_some()
            {
                return None;
            }
            Some(Bundle {
                raw,
                native: key,
                available: received,
            })
        })();
        if let Some(bundle) = accepted {
            self.sg_raw_basis.blocked_publication = None;
            self.sg_raw_basis.bundle = Some(bundle);
        } else {
            self.invalidate_sg_raw();
        }
    }

    pub(super) fn sg_before_observe(&mut self, load: Option<&WorkerLoadResponse>) {
        let changed = self
            .snapshot
            .as_ref()
            .zip(
                load.and_then(|l| l.loads.first())
                    .and_then(|r| r.deadline_recovery.as_ref()),
            )
            .is_some_and(|(old, new)| {
                NativeKey::of(&old.lane) == NativeKey::of(new)
                    && old
                        .lane
                        .active_requests
                        .iter()
                        .zip(&new.active_requests)
                        .any(|(a, b)| {
                            a.original_prefill_observation != b.original_prefill_observation
                        })
            });
        if changed {
            self.invalidate_sg_raw();
        }
    }

    pub fn select_sg_raw_basis(&self, admission: &mut Admission, calibration_epoch: u64) {
        // Failure preserves the ENTIRE original control/fallback and its
        // training/label-exclusion fields. No direct-readiness or cold bypass.
        admission.sg_raw_basis_deadline = self.sg_projection(admission, calibration_epoch).ok();
    }

    fn sg_projection(
        &self,
        admission: &Admission,
        calibration_epoch: u64,
    ) -> Result<SgProjection, &'static str> {
        let n = admission
            .normalization
            .ok_or("sg_normalization_unavailable")?;
        let snap = admission.training_snapshot;
        let bundle = self
            .sg_raw_basis
            .bundle
            .as_ref()
            .ok_or("sg_complete_pair_unavailable")?;
        let s = self.snapshot.as_ref().ok_or("sg_native_snapshot_missing")?;
        let raw = &bundle.raw;
        let (capacity, capacity_basis) = if admission.projection.selected_raw_fallback.is_none() {
            (
                raw.admission_capacity,
                "paired_guarded_raw_admission_capacity",
            )
        } else {
            let current = admission
                .projection
                .selected_raw_fallback
                .as_ref()
                .and_then(|r| r.capacity_basis.as_ref())
                .ok_or("sg_current_raw_capacity_basis_missing")?;
            if current.method != started_capacity::METHOD
                || current.selected_composition
                    != "recomputed_raw_work_over_same_poll_observed_size"
                || current.capacity.status != "available"
                || current.capacity.raw_poll_started != snap.poll_started
            {
                return Err("sg_current_raw_capacity_basis_unavailable");
            }
            (
                raw.selected_capacity,
                "paired_corrected_same_poll_observed_uncached_size_capacity",
            )
        };
        if !snap.fresh
            || !snap.native_ready
            || !positive(n.z)
            || !admission.projection.gain.is_some_and(positive)
            || !admission
                .projection
                .effective_capacity_tokens_per_second
                .is_some_and(positive)
            || !nonnegative(snap.baseline)
            || !self.valid
            || self.incomplete
            || bundle.native != NativeKey::of(&s.lane)
            || bundle.available > snap.at
            || s.available > snap.at
            || s.received > snap.at
            || snap.at > s.expiry
            || snap.at > raw.raw_expiry
            || snap.at < raw.raw_poll
            || raw.calibration_epoch != calibration_epoch
            || raw.barrier_revision != self.sg_raw_basis.revision
            || self.capacity_key.as_ref() != Some(&raw.source)
            || n.counter_epoch != raw.size.counter_epoch
            || n.dp_rank != raw.size.dp_rank
            || n.progress_source != raw.size.progress_source
            || !self.capacity_poll.is_some_and(|(poll, valid, cap)| {
                poll == snap.poll_started
                    && valid
                    && cap.is_some_and(|c| close(c, n.admission_capacity))
            })
        {
            return Err("sg_pair_or_current_guard_unavailable");
        }
        if s.lane
            .active_requests
            .iter()
            .any(|row| sg_fact(row, s.lane.captured_at_monotonic_secs).is_none())
        {
            return Err("sg_request_fact_identity_or_mode_unavailable");
        }
        let mut private = self.clone();
        private.sg_raw_basis.bundle = None;
        private.capacity_poll = Some((raw.raw_poll, true, Some(raw.admission_capacity)));
        let identity = self.identity();
        let mut waiting = 0;
        let mut stage_zero = 0;
        for row in &mut private.snapshot.as_mut().unwrap().lane.active_requests {
            if row.first_text_emitted == Some(true) {
                continue;
            }
            let known = row.attempt_id.as_ref().and_then(|a| self.admissions.get(a));
            if known.is_some_and(|r| r.first_text_at.is_some_and(|t| t <= snap.at)) {
                continue;
            }
            if row.remaining_prefill_work_tokens.is_some_and(nonnegative) {
                continue;
            }
            if row.state == "waiting" {
                if !sg_fact(row, s.lane.captured_at_monotonic_secs)
                    .is_some_and(ordinary_original_input)
                {
                    return Err("sg_waiting_original_input_or_retraction_unknown");
                }
                let known = known.ok_or("native_predecessor_gateway_order_unknown")?;
                // Preserve existing local prompt-source/value guards before
                // replacing only the source of the waiting-work estimate.
                prompt(known.estimate)?;
                row.remaining_prefill_work_tokens = Some(raw.size.blended_queue_tokens_per_request);
                waiting += 1;
            } else if original_stage_complete(row, s.lane.captured_at_monotonic_secs) {
                if known.is_none_or(|k| {
                    k.at > snap.at
                        || k.identity
                            .as_ref()
                            .is_some_and(|old| Some(old) != identity.as_ref())
                }) {
                    return Err("native_predecessor_gateway_order_unknown");
                }
                row.remaining_prefill_work_tokens = Some(0.0);
                stage_zero += 1;
            }
        }
        let mut projection = Projection::default();
        private.project(
            snap.at,
            raw.raw_poll,
            snap.baseline,
            capacity,
            raw.admission_capacity,
            &mut projection,
        )?;
        let total = projection
            .total_ttft_seconds
            .ok_or("sg_projection_unavailable")?;
        let waiting_work = waiting as f64 * raw.size.blended_queue_tokens_per_request;
        // Counters describe the new source honestly; stage witnesses/queue
        // estimates are not falsely labelled native-known remaining work.
        projection.native_known_remaining_rows -= waiting + stage_zero;
        projection.full_prompt_proxy_rows += waiting;
        Ok(SgProjection { method: METHOD, selected: true, supported: true, available: true, total_ttft_seconds: total,
            capacity_tokens_per_second: capacity, selected_capacity_basis: capacity_basis, raw_evidence: raw.clone(),
            native_tuple: bundle.native.clone(), original_native_receipt: s.received,
            original_native_expiry: s.expiry,
            pair_available_at_gateway_seconds: bundle.available,
            guarded_cutoff_gateway_seconds: snap.at,
            barrier_revision_at_guard: self.sg_raw_basis.revision, native_waiting_proxy_rows: waiting,
            original_prefill_zero_rows: stage_zero, waiting_proxy_work_tokens: waiting_work,
            other_retained_work_tokens: projection.input_work_tokens - waiting_work,
            raw_queue_to_native_waiting_difference_tokens: waiting_work - raw.raw_queue_estimate,
            legacy_projection_supported: admission.projection.supported,
            legacy_projection_reason: admission.projection.reason,
            legacy_selected_total_seconds: admission.projection.total_ttft_seconds, projection,
            interpretation: "SG prepared uncached queue estimate with native deadlines. Original-stage zero is not visible text, GPU idle, future rework or ACK/release. Current local unseen estimates retained. Original fit/reference/label-exclusion state unchanged." })
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9_f64.max(1e-12 * a.abs().max(b.abs()))
}

fn original_stage_complete(
    row: &openai_protocol::native_deadline::NativeDeadlineRequest,
    captured: f64,
) -> bool {
    let Some(p) = sg_fact(row, captured) else {
        return false;
    };
    p.available
        && ordinary_original_input(p)
        && p.abort_pending.is_some()
        && row.first_text_emitted == Some(false)
        && matches!(row.state.as_str(), "decode" | "unknown")
        && p.committed_output_tokens.is_some_and(|n| n > 0)
        && p.original_prompt_tokens
            .zip(p.normal_return_frontier_tokens)
            .is_some_and(|(original, frontier)| {
                original > 0
                    && original <= openai_protocol::native_deadline::MAX_EXACT_PROMPT_WORK
                    && frontier >= original
            })
}

fn ordinary_original_input(
    p: &openai_protocol::native_deadline::SgOriginalPrefillObservation,
) -> bool {
    p.original_alias_verified == Some(true)
        && p.input_embeds_none == Some(true)
        && p.session_none == Some(true)
        && p.multimodal_inputs_none == Some(true)
        && p.lifetime_retraction_count == Some(0)
        && p.is_retracted == Some(false)
        && p.original_prompt_tokens
            .is_some_and(|n| n > 0 && n <= openai_protocol::native_deadline::MAX_EXACT_PROMPT_WORK)
}

fn sg_fact(
    row: &openai_protocol::native_deadline::NativeDeadlineRequest,
    captured: f64,
) -> Option<&openai_protocol::native_deadline::SgOriginalPrefillObservation> {
    let p = row.original_prefill_observation.as_ref()?;
    (p.schema_version == 1
        && p.diagnostic_only
        && p.method == "sg_original_prefill_observation_v1"
        && p.source_commit == SG_SOURCE
        && p.native_request_id == row.native_request_id
        && p.attempt_id == row.attempt_id
        && p.state_seq == row.state_seq
        && p.observed_at_monotonic_secs.is_finite()
        && p.observed_at_monotonic_secs >= 0.0
        && p.observed_at_monotonic_secs <= captured
        && p.supported_mode == Some(true))
    .then_some(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn estimate(tokens: f64) -> pending_work::Estimate {
        pending_work::Estimate {
            tokens: Some(tokens),
            source: "utf8_bytes_div4_proxy",
            text_bytes: Some((tokens * 4.0) as usize),
            uncached_tokens_known: false,
            uncertainty: "fixture",
        }
    }
    fn lane() -> NativeDeadlineTelemetry {
        let mut rows = Vec::new();
        for (id, state, output, abort, frontier) in [
            ("a", "waiting", 0, false, None),
            ("b", "unknown", 1, true, Some(100)),
        ] {
            rows.push(json!({"attempt_id":id,"native_request_id":format!("n-{id}"),"state_seq":1,
                "state":state,"remaining_prefill_work_tokens":null,"hard_ttft_deadline_monotonic_secs":110.0,
                "queue_position":null,"first_text_emitted":false,
                "original_prefill_observation":{"schema_version":1,"diagnostic_only":true,
                    "method":"sg_original_prefill_observation_v1","source_commit":SG_SOURCE,
                    "native_request_id":format!("n-{id}"),"attempt_id":id,"state_seq":1,
                    "observed_at_monotonic_secs":9.9,"available":frontier.is_some(),"supported_mode":true,
                    "original_alias_verified":true,"original_prompt_tokens":100,"normal_return_frontier_tokens":frontier,
                    "committed_output_tokens":output,"lifetime_retraction_count":0,"input_embeds_none":true,
                    "session_none":true,"multimodal_inputs_none":true,"is_retracted":false,"abort_pending":abort}}));
        }
        serde_json::from_value(json!({"schema_version":1,"engine_epoch":"e","profile_id":"sglang-c9e5799:p",
            "dp_rank":0,"snapshot_seq":1,"captured_at_monotonic_secs":10.0,"snapshot_age_seconds":0.0,
            "active_inventory_complete":true,"native_population":2,"scheduler_semantics":"continuous_batching",
            "deadline_boundary":"first_nonempty_text","deadline_origin":"engine_http_receive_monotonic",
            "remaining_work_source":"unknown","effective_capacity_tokens_per_sec":null,"capacity_source":null,
            "hard_ttft_deadline_supported":true,"cancellation_ack_supported":true,"abort_enforcement_enabled":true,
            "abort_max_lag_seconds":null,"ack_floor_seq":0,"active_requests":rows,"cancellation_acks":[]})).unwrap()
    }
    fn size() -> basis::Provenance {
        basis::Provenance {
            kind: basis::Kind::Observed,
            dp_rank: 0,
            counter_epoch: 0,
            progress_source: "prefill_started_requests",
            observed_sample_count: 8,
            unshrunk_observed_mean: Some(50.0),
            configured_prior_tokens: 10.0,
            configured_prior_samples: 8,
            effective_prior_weight: 0.5,
            blended_queue_tokens_per_request: 30.0,
        }
    }
    fn raw(revision: u64, poll: f64) -> SgRawEvidence {
        let load: WorkerLoadResponse = serde_json::from_value(json!({"dp_rank_count":1,"loads":[{
            "dp_rank":0,"native_request_counts_available":true,"native_request_counts_profile":1,
            "timely_prefill_progress_source":"prefill_started_requests","prefill_metric_provenance":{
                "schema_version":1,"counter_epoch":0,"progress_source":"prefill_started_requests",
                "capacity_interval_eligible":true,"counter_reset":false}}]})).unwrap();
        SgRawEvidence {
            calibration_epoch: 1,
            barrier_revision: revision,
            raw_poll: poll,
            raw_expiry: 20.0,
            source: (1, 0, "prefill_started_requests".into()),
            size: size(),
            admission_capacity: 10.0,
            selected_capacity: 10.0,
            native_waiting: 2.0,
            raw_queue_estimate: 60.0,
            capacity: started_capacity::Selection {
                status: "available",
                reason: "complete_current_conversion_basis",
                raw_poll_started: poll,
                source: started_capacity::source_key(&load),
                history_complete: true,
                history_samples: 8,
                previous_conversion_size_mean: Some(50.0),
                lower_quartile_request_start_rate: Some(0.2),
                rebased_learned_capacity: Some(10.0),
                used_capacity: Some(10.0),
                configured_floor: Some(2.0),
                same_poll_observed_size: Some(started_capacity::SamePollObservedSize {
                    method: "same_poll_observed_completed_uncached_size_v1",
                    status: "available",
                    reason: "same_poll_observed_completed_size",
                    current_unshrunk_size_mean: Some(50.0),
                    current_observed_sample_count: Some(8),
                    rebased_learned_capacity: Some(10.0),
                    used_capacity: Some(10.0),
                }),
            },
        }
    }
    fn fixture() -> (Native, Admission) {
        let mut native = Native::default();
        native.register(Some("a"), 0.1, estimate(100.0));
        native.register(Some("b"), 0.2, estimate(100.0));
        native.snapshot = Some(Snapshot {
            lane: lane(),
            received: 1.0,
            available: 1.0,
            started: 0.9,
            raw_poll: 1.0,
            expiry: 20.0,
            last_age: 0.0,
        });
        native.valid = true;
        native.capacity_key = Some((1, 0, "prefill_started_requests".into()));
        native.capacity_poll = Some((1.0, true, Some(10.0)));
        native.commit_sg_raw(Some(raw(0, 1.0)), 1.0, 1, 1.0);
        let snap = decision::Snapshot {
            size_basis: Some(basis::Stamp {
                epoch: 1,
                provenance: size(),
            }),
            at: 2.0,
            profile: 1,
            baseline: 2.0,
            x: 100.0,
            source: "fixture",
            original_queue_extra: 10.0,
            live_queue_extra: Some(10.0),
            scalar_gain: 1.0,
            fresh: true,
            native_ready: true,
            native_waiting: 2.0,
            poll_started: 1.0,
            eligible: true,
            context: decision::Context::default(),
        };
        let predictions = decision::State::default().predictions(snap, snap.at, true);
        let admission = Admission {
            method: super::super::METHOD,
            normalization: Some(Normalization {
                z: 1.0,
                selected_x: 1.0,
                admission_capacity: 10.0,
                capacity_reference: 10.0,
                reference_created_at: 0.0,
                basis_epoch: 1,
                counter_epoch: 0,
                dp_rank: 0,
                progress_source: "prefill_started_requests",
            }),
            normalization_reason: "fixture",
            training_snapshot: snap,
            predictions,
            projection: Projection {
                supported: false,
                reason: "unknown_no_text_decode_work",
                gain: Some(1.0),
                effective_capacity_tokens_per_second: Some(10.0),
                total_ttft_seconds: Some(100.0),
                ..Default::default()
            },
            sg_raw_basis_deadline: None,
        };
        (native, admission)
    }

    #[test]
    fn stage_zero_preserves_abort_owner_and_original_training_and_aggregate() {
        let (n, mut a) = fixture();
        let before = serde_json::to_value(&a).unwrap();
        n.select_sg_raw_basis(&mut a, 1);
        let selected = a.sg_raw_basis_deadline.as_ref().unwrap();
        assert_eq!(selected.total_ttft_seconds, 5.0);
        assert_eq!(selected.waiting_proxy_work_tokens, 30.0);
        assert_eq!(selected.original_prefill_zero_rows, 1);
        assert_eq!(selected.projection.input_work_tokens, 30.0);
        assert_eq!(
            n.snapshot.as_ref().unwrap().lane.active_requests[1].remaining_prefill_work_tokens,
            None
        );
        assert!(n.acks.is_empty());
        a.sg_raw_basis_deadline = None;
        assert_eq!(serde_json::to_value(a).unwrap(), before);
    }

    #[test]
    fn newer_healthy_raw_uses_original_pair_capacity_and_current_unseen_first_text_ack() {
        let (mut n, mut a) = fixture();
        n.register(Some("c"), 1.2, estimate(20.0));
        n.capacity_poll = Some((2.0, true, Some(50.0)));
        a.training_snapshot.poll_started = 2.0;
        a.normalization.as_mut().unwrap().admission_capacity = 50.0;
        assert!(n
            .project(2.0, 2.0, 2.0, 50.0, 50.0, &mut Projection::default())
            .is_err());
        n.select_sg_raw_basis(&mut a, 1);
        let p = a.sg_raw_basis_deadline.as_ref().unwrap();
        assert_eq!(p.total_ttft_seconds, 7.0);
        assert_eq!(p.capacity_tokens_per_second, 10.0);
        assert_eq!(p.original_native_expiry, 20.0);
        assert_eq!(p.projection.unseen_admitted_rows, 1);
        n.first_text(Some("c"), 1.5);
        n.select_sg_raw_basis(&mut a, 1);
        assert_eq!(
            a.sg_raw_basis_deadline.as_ref().unwrap().total_ttft_seconds,
            5.0
        );
        n.local_pre_send_rollback("a");
        n.select_sg_raw_basis(&mut a, 1);
        assert!(a.sg_raw_basis_deadline.is_none());
    }

    #[test]
    fn retraction_alias_frontier_mode_and_missing_proof_do_not_zero_stage() {
        for mode in [
            "retraction",
            "alias",
            "frontier",
            "mode",
            "missing",
            "future",
        ] {
            let (mut n, mut a) = fixture();
            let row = &mut n.snapshot.as_mut().unwrap().lane.active_requests[1];
            if mode == "missing" {
                row.original_prefill_observation = None;
            } else {
                let proof = row.original_prefill_observation.as_mut().unwrap();
                match mode {
                    "retraction" => proof.lifetime_retraction_count = Some(1),
                    "alias" => proof.original_alias_verified = Some(false),
                    "frontier" => proof.normal_return_frontier_tokens = Some(99),
                    "mode" => proof.supported_mode = Some(false),
                    _ => proof.observed_at_monotonic_secs = 11.0,
                }
            }
            n.select_sg_raw_basis(&mut a, 1);
            assert!(a.sg_raw_basis_deadline.is_none(), "{mode}");
        }
    }

    #[test]
    fn negative_epoch_stale_future_and_changed_optional_duplicate_block_restore() {
        let (mut n, mut a) = fixture();
        let token = raw(0, 1.0);
        n.invalidate_sg_raw();
        n.commit_sg_raw(Some(token), 1.0, 1, 1.2);
        assert!(n.sg_raw_basis.bundle.is_none());
        let (mut n, _) = fixture();
        let token = raw(0, 1.0);
        n.commit_sg_raw(Some(token), 1.0, 2, 1.2);
        assert!(n.sg_raw_basis.bundle.is_none());
        let (n, mut a2) = fixture();
        a2.training_snapshot.at = 21.0;
        n.select_sg_raw_basis(&mut a2, 1);
        assert!(a2.sg_raw_basis_deadline.is_none());
        let (n, mut a2) = fixture();
        a2.training_snapshot.at = 0.5;
        n.select_sg_raw_basis(&mut a2, 1);
        assert!(a2.sg_raw_basis_deadline.is_none());
        let (mut n, _) = fixture();
        let mut changed = lane();
        changed.active_requests[1].original_prefill_observation = None;
        let load = WorkerLoadResponse {
            dp_rank_count: 1,
            loads: vec![openai_protocol::worker::SchedulerLoadSnapshot {
                dp_rank: 0,
                deadline_recovery: Some(changed),
                ..Default::default()
            }],
            ..Default::default()
        };
        n.observe(Some(&load), true, 1.1, 1.2, 1.0, 30.0);
        assert!(n.valid);
        assert!(n.sg_raw_basis.bundle.is_none());
        let revision = n.sg_raw_basis.revision;
        n.commit_sg_raw(Some(raw(revision, 1.0)), 1.0, 1, 1.3);
        assert!(n.sg_raw_basis.bundle.is_none());
        // Restoring the original optional value under the same immutable key
        // cannot restore a positive publication after its negative endpoint.
        n.snapshot.as_mut().unwrap().lane = lane();
        let revision = n.sg_raw_basis.revision;
        n.commit_sg_raw(Some(raw(revision, 1.0)), 1.0, 1, 1.4);
        assert!(n.sg_raw_basis.bundle.is_none());
        n.select_sg_raw_basis(&mut a, 1);
        assert!(a.sg_raw_basis_deadline.is_none());
    }

    #[test]
    fn committed_unpaired_new_native_clears_old_pair_and_seen_membership_is_not_reused() {
        let (mut n, mut a) = fixture();
        n.register(Some("c"), 1.1, estimate(20.0));
        let mut next = lane();
        next.snapshot_seq = 2;
        next.captured_at_monotonic_secs = 11.0;
        let mut row = next.active_requests[0].clone();
        row.attempt_id = Some("c".into());
        row.native_request_id = Some("n-c".into());
        row.state_seq = 2;
        row.original_prefill_observation = None;
        next.active_requests.push(row);
        next.native_population = 3;
        let load = WorkerLoadResponse {
            dp_rank_count: 1,
            loads: vec![openai_protocol::worker::SchedulerLoadSnapshot {
                dp_rank: 0,
                deadline_recovery: Some(next),
                ..Default::default()
            }],
            ..Default::default()
        };
        n.observe(Some(&load), true, 1.2, 1.3, 1.0, 30.0);
        assert!(n.valid);
        assert!(n.admissions["c"].seen_native);
        assert!(n.sg_raw_basis.bundle.is_none());
        n.select_sg_raw_basis(&mut a, 1);
        assert!(a.sg_raw_basis_deadline.is_none());
    }

    #[test]
    fn retracted_or_unknown_waiting_work_retains_original_fallback() {
        for (count, retracted) in [
            (Some(1), Some(false)),
            (Some(0), Some(true)),
            (None, Some(false)),
            (Some(0), None),
        ] {
            let (mut n, mut a) = fixture();
            let p = n.snapshot.as_mut().unwrap().lane.active_requests[0]
                .original_prefill_observation
                .as_mut()
                .unwrap();
            p.lifetime_retraction_count = count;
            p.is_retracted = retracted;
            n.select_sg_raw_basis(&mut a, 1);
            assert!(a.sg_raw_basis_deadline.is_none());
        }
    }

    #[test]
    fn negative_and_restored_same_publication_require_a_genuinely_new_native_point() {
        let (mut n, mut a) = fixture();
        let old = raw(0, 1.0);
        n.invalidate_sg_raw();
        n.commit_sg_raw(Some(old), 1.0, 1, 1.2);
        assert!(n.sg_raw_basis.bundle.is_none());
        let mut next = lane();
        next.snapshot_seq = 2;
        next.captured_at_monotonic_secs = 11.0;
        let load = WorkerLoadResponse {
            dp_rank_count: 1,
            loads: vec![openai_protocol::worker::SchedulerLoadSnapshot {
                dp_rank: 0,
                deadline_recovery: Some(next),
                ..Default::default()
            }],
            ..Default::default()
        };
        n.capacity_poll = Some((2.0, true, Some(10.0)));
        let revision = n.sg_raw_basis.revision;
        n.observe(Some(&load), true, 2.0, 2.1, 2.0, 30.0);
        assert!(n.valid);
        n.commit_sg_raw(Some(raw(revision, 2.0)), 2.0, 1, 2.1);
        assert!(n.sg_raw_basis.bundle.is_some());
        a.training_snapshot.at = 3.0;
        a.training_snapshot.poll_started = 2.0;
        n.select_sg_raw_basis(&mut a, 1);
        assert!(a.sg_raw_basis_deadline.is_some());
    }

    #[test]
    fn supported_original_uses_own_admission_capacity_and_fallback_uses_corrected_capacity() {
        let (mut n, mut a) = fixture();
        let mut token = raw(0, 1.0);
        token.selected_capacity = 20.0;
        token.capacity.lower_quartile_request_start_rate = Some(0.4);
        token.capacity.previous_conversion_size_mean = Some(25.0);
        let observed = token.capacity.same_poll_observed_size.as_mut().unwrap();
        observed.rebased_learned_capacity = Some(20.0);
        observed.used_capacity = Some(20.0);
        n.commit_sg_raw(Some(token.clone()), 1.0, 1, 1.0);
        n.select_sg_raw_basis(&mut a, 1);
        assert_eq!(
            a.sg_raw_basis_deadline
                .as_ref()
                .unwrap()
                .capacity_tokens_per_second,
            10.0
        );
        assert_eq!(
            a.sg_raw_basis_deadline
                .as_ref()
                .unwrap()
                .selected_capacity_basis,
            "paired_guarded_raw_admission_capacity"
        );
        a.projection.selected_raw_fallback = Some(RawFallback {
            baseline_seconds: 2.0,
            baseline_source: "provisional",
            raw_extra_seconds: 98.0,
            total_ttft_seconds: 100.0,
            capacity_basis: Some(CapacityBasis {
                method: started_capacity::METHOD,
                selected_composition: "recomputed_raw_work_over_same_poll_observed_size",
                capacity: token.capacity,
                original_raw_extra_seconds: 98.0,
                previous_conversion_raw_extra_seconds: Some(98.0),
                original_capacity: 10.0,
                token_work: 980.0,
                observed_delay_floor: 0.0,
                measured_delay_only: false,
            }),
        });
        n.select_sg_raw_basis(&mut a, 1);
        let selected = a.sg_raw_basis_deadline.as_ref().unwrap();
        assert_eq!(selected.capacity_tokens_per_second, 20.0);
        assert_eq!(selected.total_ttft_seconds, 3.5);
        assert_eq!(
            selected.selected_capacity_basis,
            "paired_corrected_same_poll_observed_uncached_size_capacity"
        );
    }

    #[test]
    fn serialized_sg_optional_fixture_keeps_original_control_and_exact_pair_clocks() {
        // Generate the old control through the real production admission
        // method, rather than hand-authoring its fit/normalization fields.
        let (mut n, _) = fixture();
        n.snapshot.as_mut().unwrap().lane.active_requests.pop();
        n.snapshot.as_mut().unwrap().lane.native_population = 1;
        n.admissions.remove("b");
        let (mut basis, mut pending) = super::super::tests::admissions(2.0, 1, 100.0, 10.0);
        for snapshot in [&mut basis.training_snapshot, &mut basis.prediction_snapshot] {
            snapshot.size_basis = Some(basis::Stamp {
                epoch: 1,
                provenance: size(),
            });
            snapshot.poll_started = 1.0;
            snapshot.native_waiting = 2.0;
        }
        pending.poll_started = 1.0;
        pending.capacity_progress_source = Some("prefill_started_requests".into());
        let mut state = super::super::State {
            native: n,
            ..Default::default()
        };
        state.poll(2.0, Some(7), 1, true, 30.0);
        let mut a = state.admission(&basis, Some(&pending), None);
        assert!(a.projection.supported, "{a:?}");
        assert_eq!(a.projection.total_ttft_seconds, Some(12.0));
        state.native.select_sg_raw_basis(&mut a, 1);
        let value = serde_json::to_value(&a).unwrap();
        let selected = &value["sg_raw_basis_deadline"];
        assert_eq!(selected["method"], METHOD);
        assert_eq!(selected["selected"], true);
        assert_eq!(selected["available"], true);
        assert_eq!(selected["supported"], true);
        assert_eq!(selected["projection"]["supported"], true);
        assert_eq!(value["projection"]["supported"], true);
        assert_eq!(value["projection"]["total_ttft_seconds"], 12.0);
        assert_eq!(
            selected["guarded_cutoff_gateway_seconds"],
            value["training_snapshot"]["at"]
        );
        if let Ok(path) = std::env::var("SMG_SG_E_FIXTURE_PATH") {
            let mut unavailable = a.clone();
            state.native.invalidate_sg_raw();
            state.native.select_sg_raw_basis(&mut unavailable, 1);
            assert!(unavailable.sg_raw_basis_deadline.is_none());
            let unavailable_path = std::path::Path::new(&path)
                .with_file_name("emitted-sg-admission-unavailable-fixture.json");
            std::fs::write(
                unavailable_path,
                serde_json::to_vec_pretty(&serde_json::json!({"admission":unavailable})).unwrap(),
            )
            .unwrap();
            std::fs::write(
                path,
                serde_json::to_vec_pretty(&serde_json::json!({"admission":a})).unwrap(),
            )
            .unwrap();
        }
    }
}
