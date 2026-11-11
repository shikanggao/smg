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

/// Original expiry of an actually issued positive-rate SG raw proof. Zero
/// intervals cannot refresh this anchor, even while the positive window persists.
#[derive(Clone, Copy, Debug, Serialize)]
pub(super) struct PositiveEvidence {
    raw_poll: f64,
    raw_expiry: f64,
    source: started_capacity::SourceKey,
}

/// Owned at raw publication and carried on the existing bounded native task.
/// This is estimated native UNCACHED-token capacity, not completed GPU service.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CapacityEligibility {
    PositiveCurrentInterval,
    CertifiedZeroInterval,
}

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
    capacity_eligibility: CapacityEligibility,
    #[serde(skip_serializing_if = "Option::is_none")]
    positive_evidence_for_zero_interval: Option<PositiveEvidence>,
    capacity: started_capacity::Selection,
}
impl SgRawEvidence {
    pub fn original_expiry(&self) -> f64 {
        self.raw_expiry
    }
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
    positive_evidence: Option<PositiveEvidence>,
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
    current_capacity_eligibility: CapacityEligibility,
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
    pub projection: Projection,
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
            && self.sg_raw_basis.positive_evidence.is_none()
        {
            return;
        }
        self.sg_raw_basis.revision = self.sg_raw_basis.revision.saturating_add(1);
        self.sg_raw_basis.bundle = None;
        self.sg_raw_basis.positive_evidence = None;
        if let Some(snapshot) = &self.snapshot {
            self.sg_raw_basis.blocked_publication = Some(NativeKey::of(&snapshot.lane));
        }
    }

    pub(super) fn sg_zero_positive_evidence(&self, started: f64) -> Option<PositiveEvidence> {
        let evidence = self.sg_raw_basis.positive_evidence?;
        let source = self.sg_raw_basis.raw_source?;
        (started.is_finite()
            && source == evidence.source
            && evidence.raw_poll <= started
            && started < evidence.raw_expiry
            && self
                .started_requests
                .certified_zero_interval(started, Some(source)))
        .then_some(evidence)
    }

    pub(super) fn sg_fence_certified_zero(&mut self) {
        let positive = self.sg_raw_basis.positive_evidence;
        // Preserve baec's zero-poll pair invalidation and late-task fence. Only
        // the existing positive freshness anchor survives, never its bundle.
        self.invalidate_sg_raw();
        if self.sg_raw_basis.revision < u64::MAX {
            self.sg_raw_basis.positive_evidence = positive;
        }
    }

    fn sg_capacity_poll_matches(&self, started: f64, capacity: f64) -> bool {
        self.capacity_poll.is_some_and(|(poll, valid, cap)| {
            poll == started
                && cap.is_some_and(|c| positive(c) && close(c, capacity))
                && (valid || self.sg_zero_positive_evidence(started).is_some())
        })
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
                || !self.sg_capacity_poll_matches(started, original)
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
            let zero_evidence = if self.capacity_poll.is_some_and(|(_, valid, _)| valid) {
                None
            } else {
                Some(self.sg_zero_positive_evidence(started)?)
            };
            let mut expiry =
                zero_evidence.map_or(started + max_age, |e| (started + max_age).min(e.raw_expiry));
            if let Some(old) = self
                .sg_raw_basis
                .positive_evidence
                .filter(|e| e.raw_poll == started && e.source == source_key)
            {
                expiry = expiry.min(old.raw_expiry); // An exact repeat cannot renew its original expiry.
            }
            if !nonnegative(queue) || !expiry.is_finite() || expiry <= started {
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
                capacity_eligibility: if zero_evidence.is_some() {
                    CapacityEligibility::CertifiedZeroInterval
                } else {
                    CapacityEligibility::PositiveCurrentInterval
                },
                positive_evidence_for_zero_interval: zero_evidence,
                capacity,
            })
        })();
        if let Some(raw) = &proof {
            if raw.positive_evidence_for_zero_interval.is_none() {
                self.sg_raw_basis.positive_evidence = Some(PositiveEvidence {
                    raw_poll: raw.raw_poll,
                    raw_expiry: raw.raw_expiry,
                    source: self.sg_raw_basis.raw_source.unwrap(),
                });
            }
        } else {
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
        match self.sg_projection(admission, calibration_epoch) {
            Ok(projection) => {
                admission.sg_raw_basis_deadline = Some(projection);
                admission.sg_raw_basis_rejection_reason = None;
            }
            Err(reason) => {
                admission.sg_raw_basis_deadline = None;
                admission.sg_raw_basis_rejection_reason = Some(reason);
            }
        }
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
            || !self.sg_capacity_poll_matches(snap.poll_started, n.admission_capacity)
        {
            return Err("sg_pair_or_current_guard_unavailable");
        }
        let current_capacity_eligibility = if self.capacity_poll.is_some_and(|(_, valid, _)| valid)
        {
            CapacityEligibility::PositiveCurrentInterval
        } else {
            CapacityEligibility::CertifiedZeroInterval
        };
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
            capacity_tokens_per_second: capacity, selected_capacity_basis: capacity_basis, current_capacity_eligibility, raw_evidence: raw.clone(),
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
    use serde_json::json;

    use super::*;

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
            capacity_eligibility: CapacityEligibility::PositiveCurrentInterval,
            positive_evidence_for_zero_interval: None,
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
            sg_raw_basis_rejection_reason: None,
            paired_native_candidate: None,
        };
        (native, admission)
    }

    fn capacity_frame(
        at: f64,
        previous_at: Option<f64>,
        previous: Option<f64>,
        counter: f64,
        eligible: bool,
        count: u32,
    ) -> WorkerLoadResponse {
        let interval = previous_at.map(|before| at - before);
        let rate = previous
            .zip(interval)
            .map(|(before, dt)| (counter - before) / dt);
        serde_json::from_value(json!({"dp_rank_count":1,"loads":[{
            "dp_rank":0,"native_request_counts_available":true,"native_request_counts_profile":1,
            "timely_prefill_progress_source":"prefill_started_requests",
            "prefill_capacity_learning_active":eligible,"prefill_capacity_sample_count":count,
            "learned_prefill_capacity":(count>=8).then_some(50.0),
            "avg_request_prefill_kv_computed_tokens":50.0,"prefill_size_sample_count":8,
            "prefill_metric_provenance":{"schema_version":1,"counter_epoch":0,
                "progress_source":"prefill_started_requests","counter_reset":false,
                "raw_counter":counter,"previous_raw_counter":previous,"interval_seconds":interval,
                "interval_started_time_seconds":previous_at,"interval_ended_time_seconds":at,"raw_rate":rate,
                "previous_conversion_size_mean":50.0,"current_unshrunk_size_mean":50.0,
                "queue_size_observed_sample_count":8,"capacity_interval_eligible":eligible}}]})).unwrap()
    }

    fn positive_capacity_fixture() -> (Native, Admission) {
        let (mut n, mut a) = fixture();
        for i in 0..=8 {
            let at = f64::from(i);
            let load = capacity_frame(
                at,
                (i > 0).then_some(at - 1.0),
                (i > 0).then_some(at - 1.0),
                at,
                i > 0,
                i,
            );
            n.raw_poll(Some(&load), at, true);
        }
        let positive = n
            .sg_raw_evidence(8.0, 30.0, 1, Some(size()), Some(50.0), Some(2.0), 2.0)
            .unwrap();
        assert!(positive.positive_evidence_for_zero_interval.is_none());
        assert_eq!(n.sg_raw_basis.positive_evidence.unwrap().raw_expiry, 38.0);
        a.normalization.as_mut().unwrap().admission_capacity = 50.0;
        (n, a)
    }

    fn commit_zero(n: &mut Native, at: f64, previous_at: f64) -> Option<SgRawEvidence> {
        let mut load = capacity_frame(at, Some(previous_at), Some(8.0), 8.0, false, 8);
        n.raw_poll(Some(&load), at, true);
        let proof = n.sg_raw_evidence(at, 30.0, 1, Some(size()), Some(50.0), Some(2.0), 2.0);
        let mut lane = n.snapshot.as_ref().unwrap().lane.clone();
        lane.snapshot_seq += 1;
        lane.captured_at_monotonic_secs += 1.0;
        load.loads[0].deadline_recovery = Some(lane);
        n.observe(Some(&load), true, at + 0.01, at + 0.02, at, 30.0);
        n.commit_sg_raw(proof.clone(), at, 1, at + 0.02);
        proof
    }

    #[test]
    fn certified_zero_uses_original_positive_expiry_and_preserves_legacy_control() {
        let (mut n, mut a) = positive_capacity_fixture();
        let raw = commit_zero(&mut n, 9.0, 8.0).unwrap();
        assert_eq!(raw.raw_expiry, 38.0);
        assert_eq!(
            raw.positive_evidence_for_zero_interval.unwrap().raw_poll,
            8.0
        );
        assert!(n.valid);
        a.training_snapshot.at = 9.1;
        a.training_snapshot.poll_started = 9.0;
        let mut old = Projection::default();
        assert_eq!(
            n.project(9.1, 9.0, 2.0, 10.0, 50.0, &mut old),
            Err("request_start_capacity_conversion_missing_or_nonpositive")
        );
        a.projection.reason = "request_start_capacity_conversion_missing_or_nonpositive";
        a.projection.stable_work_fallback_seconds = Some(777.0);
        let before = serde_json::to_value(&a).unwrap();
        n.select_sg_raw_basis(&mut a, 1);
        let selected = a.sg_raw_basis_deadline.as_ref().unwrap();
        assert_eq!(selected.total_ttft_seconds, 2.6);
        assert_eq!(
            selected.current_capacity_eligibility,
            CapacityEligibility::CertifiedZeroInterval
        );
        assert_eq!(
            selected.raw_evidence.capacity_eligibility,
            CapacityEligibility::CertifiedZeroInterval
        );
        assert_eq!(
            selected.legacy_selected_total_seconds,
            before["projection"]["total_ttft_seconds"].as_f64()
        );
        assert_eq!(selected.legacy_selected_total_seconds, Some(100.0));
        assert_eq!(
            selected.legacy_projection_reason,
            "request_start_capacity_conversion_missing_or_nonpositive"
        );
        assert_eq!(selected.raw_evidence.raw_expiry, 38.0);
        let wire = serde_json::to_value(selected).unwrap();
        assert_eq!(
            wire["raw_evidence"]["positive_evidence_for_zero_interval"]["raw_expiry"],
            38.0
        );
        if let Ok(path) = std::env::var("SMG_SG_ZERO_SELECTION_FIXTURE_PATH") {
            std::fs::write(path, serde_json::to_vec_pretty(selected).unwrap()).unwrap();
        }
        a.sg_raw_basis_deadline = None;
        assert_eq!(serde_json::to_value(&a).unwrap(), before);
    }

    #[test]
    fn repeated_zero_never_renews_anchor_and_expires_at_original_boundary() {
        let (mut n, mut a) = positive_capacity_fixture();
        let first = commit_zero(&mut n, 9.0, 8.0).unwrap();
        let later = commit_zero(&mut n, 37.0, 9.0).unwrap();
        assert_eq!(first.raw_expiry, later.raw_expiry);
        assert_eq!(n.sg_raw_basis.positive_evidence.unwrap().raw_poll, 8.0);
        a.training_snapshot.poll_started = 37.0;
        a.training_snapshot.at = 38.0;
        n.select_sg_raw_basis(&mut a, 1);
        assert!(
            a.sg_raw_basis_deadline.is_some(),
            "existing inclusive expiry boundary"
        );
        a.training_snapshot.at = 38.000001;
        n.select_sg_raw_basis(&mut a, 1);
        assert!(a.sg_raw_basis_deadline.is_none());
        assert!(commit_zero(&mut n, 38.0, 37.0).is_none());
        assert!(n.sg_raw_basis.positive_evidence.is_none());
        assert!(n.sg_raw_basis.bundle.is_none());
        assert!(
            n.started_requests
                .selection(38.0, n.sg_raw_basis.raw_source, 2.0)
                .status
                == "available",
            "old positive window still exists but cannot extend freshness"
        );
    }

    #[test]
    fn zero_without_positive_proof_or_after_negative_reset_rebound_is_unavailable() {
        for mode in 0..4 {
            let (mut n, _) = positive_capacity_fixture();
            match mode {
                0 => n.sg_raw_basis.positive_evidence = None,
                1 => n.raw_poll(None, 8.5, false),
                2 => {
                    let mut load = capacity_frame(9.0, Some(8.0), Some(8.0), 8.0, false, 8);
                    load.loads[0]
                        .prefill_metric_provenance
                        .as_mut()
                        .unwrap()
                        .counter_reset = true;
                    n.raw_poll(Some(&load), 9.0, true);
                }
                _ => {
                    n.valid = false;
                    n.reason = "native_remaining_work_rebound";
                    n.invalidate_sg_raw();
                }
            }
            assert!(
                commit_zero(&mut n, 10.0, if mode == 2 { 9.0 } else { 8.0 }).is_none(),
                "mode {mode}"
            );
            assert!(n.sg_raw_basis.bundle.is_none());
        }
    }

    #[test]
    fn malformed_zero_interval_never_borrows_positive_history() {
        for mode in 0..5 {
            let (mut n, _) = positive_capacity_fixture();
            let mut load = capacity_frame(9.0, Some(8.0), Some(8.0), 8.0, false, 8);
            let p = load.loads[0].prefill_metric_provenance.as_mut().unwrap();
            match mode {
                0 => p.raw_rate = None,
                1 => p.raw_rate = Some(f64::NAN),
                2 => p.raw_rate = Some(-1.0),
                3 => p.interval_seconds = Some(0.0),
                _ => p.previous_raw_counter = Some(7.0),
            }
            n.raw_poll(Some(&load), 9.0, true);
            assert!(
                n.sg_raw_evidence(9.0, 30.0, 1, Some(size()), Some(50.0), Some(2.0), 2.0)
                    .is_none(),
                "mode {mode}"
            );
            assert!(n.sg_raw_basis.positive_evidence.is_none());
        }
    }

    #[test]
    fn zero_fences_positive_bundle_and_negative_late_callback_blocks_zero_recovery() {
        let (mut n, mut a) = positive_capacity_fixture();
        let positive = n
            .sg_raw_evidence(8.0, 30.0, 1, Some(size()), Some(50.0), Some(2.0), 2.0)
            .unwrap();
        let mut initial_positive = capacity_frame(8.0, Some(7.0), Some(7.0), 8.0, true, 8);
        let mut positive_lane = n.snapshot.as_ref().unwrap().lane.clone();
        positive_lane.snapshot_seq += 1;
        positive_lane.captured_at_monotonic_secs += 1.0;
        initial_positive.loads[0].deadline_recovery = Some(positive_lane);
        n.observe(Some(&initial_positive), true, 8.01, 8.02, 8.0, 30.0);
        assert!(n.valid);
        n.commit_sg_raw(Some(positive.clone()), 8.0, 1, 8.02);
        assert!(n.sg_raw_basis.bundle.is_some());
        let mut load = capacity_frame(9.0, Some(8.0), Some(8.0), 8.0, false, 8);
        n.raw_poll(Some(&load), 9.0, true);
        let zero = n
            .sg_raw_evidence(9.0, 30.0, 1, Some(size()), Some(50.0), Some(2.0), 2.0)
            .unwrap();
        assert!(n.sg_raw_basis.bundle.is_none());
        a.training_snapshot.at = 9.1;
        a.training_snapshot.poll_started = 9.0;
        n.select_sg_raw_basis(&mut a, 1);
        assert!(
            a.sg_raw_basis_deadline.is_none(),
            "no old bundle before zero callback"
        );
        let mut late_positive = capacity_frame(8.0, Some(7.0), Some(7.0), 8.0, true, 8);
        late_positive.loads[0].deadline_recovery = Some(n.snapshot.as_ref().unwrap().lane.clone());
        n.observe(Some(&late_positive), true, 8.01, 9.12, 8.0, 30.0);
        assert!(
            n.valid,
            "legacy may accept duplicate while SG task revision is fenced"
        );
        n.commit_sg_raw(Some(positive), 8.0, 1, 9.12);
        assert!(
            n.sg_raw_basis.bundle.is_none(),
            "old positive callback fenced"
        );
        let mut lane = n.snapshot.as_ref().unwrap().lane.clone();
        lane.snapshot_seq += 1;
        lane.captured_at_monotonic_secs += 1.0;
        load.loads[0].deadline_recovery = Some(lane);
        n.observe(Some(&load), true, 9.15, 9.2, 9.0, 30.0);
        n.commit_sg_raw(Some(zero), 9.0, 1, 9.2);
        a.training_snapshot.at = 9.3;
        n.select_sg_raw_basis(&mut a, 1);
        assert!(
            a.sg_raw_basis_deadline.is_none(),
            "rejected callback remains a negative barrier"
        );
        assert!(n.sg_raw_basis.positive_evidence.is_none());
    }

    #[test]
    fn actual_native_reset_and_rebound_clear_positive_anchor_and_zero_token() {
        for reset in [false, true] {
            let (mut n, _) = positive_capacity_fixture();
            n.snapshot.as_mut().unwrap().lane.active_requests[0].remaining_prefill_work_tokens =
                Some(10.0);
            let mut load = capacity_frame(9.0, Some(8.0), Some(8.0), 8.0, false, 8);
            n.raw_poll(Some(&load), 9.0, true);
            let token = n
                .sg_raw_evidence(9.0, 30.0, 1, Some(size()), Some(50.0), Some(2.0), 2.0)
                .unwrap();
            let mut lane = n.snapshot.as_ref().unwrap().lane.clone();
            lane.snapshot_seq += 1;
            lane.captured_at_monotonic_secs += 1.0;
            if reset {
                lane.engine_epoch = "new-engine".into();
            } else {
                lane.active_requests[0].remaining_prefill_work_tokens = Some(20.0);
            }
            load.loads[0].deadline_recovery = Some(lane);
            n.observe(Some(&load), true, 9.01, 9.02, 9.0, 30.0);
            assert!(!n.valid);
            assert_eq!(
                n.reason,
                if reset {
                    "native_identity_reset"
                } else {
                    "native_remaining_work_rebound"
                }
            );
            assert!(n.sg_raw_basis.positive_evidence.is_none());
            n.commit_sg_raw(Some(token), 9.0, 1, 9.02);
            assert!(n.sg_raw_basis.bundle.is_none());
        }
    }

    #[test]
    fn positive_duplicate_and_source_transition_cannot_extend_or_borrow_anchor() {
        let (mut n, _) = positive_capacity_fixture();
        let same = n
            .sg_raw_evidence(8.0, 60.0, 1, Some(size()), Some(50.0), Some(2.0), 2.0)
            .unwrap();
        assert_eq!(same.raw_expiry, 38.0);
        assert_eq!(n.sg_raw_basis.positive_evidence.unwrap().raw_expiry, 38.0);
        let mut load = capacity_frame(9.0, Some(8.0), Some(8.0), 8.0, false, 8);
        load.loads[0].native_request_counts_profile = Some(2);
        n.raw_poll(Some(&load), 9.0, true);
        assert!(n
            .sg_raw_evidence(9.0, 30.0, 1, Some(size()), Some(50.0), Some(2.0), 2.0)
            .is_none());
        assert!(n.sg_raw_basis.positive_evidence.is_none());
    }

    #[test]
    fn conflicting_duplicate_zero_cannot_restore_bounded_positive_anchor() {
        let (mut n, _) = positive_capacity_fixture();
        commit_zero(&mut n, 9.0, 8.0).unwrap();
        let mut load = capacity_frame(9.0, Some(8.0), Some(8.0), 8.0, false, 8);
        load.loads[0]
            .prefill_metric_provenance
            .as_mut()
            .unwrap()
            .interval_seconds = Some(0.5);
        n.raw_poll(Some(&load), 9.0, true);
        assert!(n
            .sg_raw_evidence(9.0, 30.0, 1, Some(size()), Some(50.0), Some(2.0), 2.0)
            .is_none());
        assert!(n.sg_raw_basis.positive_evidence.is_none());
        let restored = capacity_frame(9.0, Some(8.0), Some(8.0), 8.0, false, 8);
        n.raw_poll(Some(&restored), 9.0, true);
        assert!(n
            .sg_raw_evidence(9.0, 30.0, 1, Some(size()), Some(50.0), Some(2.0), 2.0)
            .is_none());
        assert!(n.sg_raw_basis.bundle.is_none());
    }

    #[test]
    fn positive_raw_after_zero_pair_keeps_zero_attribution_until_positive_callback() {
        let (mut n, mut a) = positive_capacity_fixture();
        commit_zero(&mut n, 9.0, 8.0).unwrap();
        let mut positive = capacity_frame(10.0, Some(9.0), Some(8.0), 9.0, true, 9);
        n.raw_poll(Some(&positive), 10.0, true);
        let token = n
            .sg_raw_evidence(10.0, 30.0, 1, Some(size()), Some(50.0), Some(2.0), 2.0)
            .unwrap();
        assert_eq!(n.sg_raw_basis.positive_evidence.unwrap().raw_expiry, 40.0);
        a.training_snapshot.at = 10.1;
        a.training_snapshot.poll_started = 10.0;
        n.select_sg_raw_basis(&mut a, 1);
        let old_pair = a.sg_raw_basis_deadline.as_ref().unwrap();
        assert_eq!(
            old_pair.raw_evidence.capacity_eligibility,
            CapacityEligibility::CertifiedZeroInterval
        );
        assert_eq!(
            old_pair.current_capacity_eligibility,
            CapacityEligibility::PositiveCurrentInterval
        );
        // The raw donor kind alone does not certify the current zero guard.
        assert_eq!(
            old_pair.raw_evidence.raw_expiry, 38.0,
            "new positive evidence cannot renew old zero proof"
        );
        assert_eq!(
            old_pair.legacy_selected_total_seconds,
            a.projection.total_ttft_seconds
        );
        let mut lane = n.snapshot.as_ref().unwrap().lane.clone();
        lane.snapshot_seq += 1;
        lane.captured_at_monotonic_secs += 1.0;
        positive.loads[0].deadline_recovery = Some(lane);
        n.observe(Some(&positive), true, 10.11, 10.2, 10.0, 30.0);
        n.commit_sg_raw(Some(token), 10.0, 1, 10.2);
        a.training_snapshot.at = 10.3;
        n.select_sg_raw_basis(&mut a, 1);
        let current = a.sg_raw_basis_deadline.as_ref().unwrap();
        assert_eq!(
            current.raw_evidence.capacity_eligibility,
            CapacityEligibility::PositiveCurrentInterval
        );
        assert_eq!(
            current.total_ttft_seconds, 2.6,
            "unchanged SG formula after positive pair"
        );
        assert_ne!(
            current.total_ttft_seconds,
            current.legacy_selected_total_seconds.unwrap(),
            "positive prior-baec arm is SG, not original C2 fallback"
        );
    }

    #[test]
    fn positive_duplicate_after_zero_is_not_a_complete_baec_selector_reference() {
        let (mut n, mut a) = positive_capacity_fixture();
        commit_zero(&mut n, 9.0, 8.0).unwrap();
        let zero_key = NativeKey::of(&n.snapshot.as_ref().unwrap().lane);
        let mut positive = capacity_frame(10.0, Some(9.0), Some(8.0), 9.0, true, 9);
        n.raw_poll(Some(&positive), 10.0, true);
        let token = n
            .sg_raw_evidence(10.0, 30.0, 1, Some(size()), Some(50.0), Some(2.0), 2.0)
            .unwrap();
        let mut duplicate = n.snapshot.as_ref().unwrap().lane.clone();
        duplicate.snapshot_age_seconds = 1.0;
        positive.loads[0].deadline_recovery = Some(duplicate);
        n.observe(Some(&positive), true, 10.1, 10.2, 10.0, 30.0);
        assert_eq!(NativeKey::of(&n.snapshot.as_ref().unwrap().lane), zero_key);
        n.commit_sg_raw(Some(token), 10.0, 1, 10.2);
        a.training_snapshot.poll_started = 10.0;
        a.training_snapshot.at = 10.3;
        n.select_sg_raw_basis(&mut a, 1);
        let selected = a.sg_raw_basis_deadline.unwrap();
        assert_eq!(
            selected.raw_evidence.capacity_eligibility,
            CapacityEligibility::PositiveCurrentInterval
        );
        assert_eq!(
            selected.current_capacity_eligibility,
            CapacityEligibility::PositiveCurrentInterval
        );
        assert_eq!(selected.total_ttft_seconds, 2.6);
        // baec's rejected zero callback blocks zero_key. Positive enums certify
        // unchanged arithmetic, not that prior private selector would accept it.
        assert_ne!(
            selected.total_ttft_seconds,
            selected.legacy_selected_total_seconds.unwrap()
        );
    }

    #[test]
    fn stage_zero_preserves_abort_owner_and_original_training_and_aggregate() {
        let (n, mut a) = fixture();
        let before = serde_json::to_value(&a).unwrap();
        n.select_sg_raw_basis(&mut a, 1);
        let selected = a.sg_raw_basis_deadline.as_ref().unwrap();
        assert_eq!(selected.total_ttft_seconds, 5.0);
        assert_eq!(
            selected.raw_evidence.capacity_eligibility,
            CapacityEligibility::PositiveCurrentInterval
        );
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
