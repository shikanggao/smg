//! Paired calibration shadow with explicit queue-size basis epochs.
//! A missing provenance record is not evidence of a prior/observed transition.
use openai_protocol::worker::WorkerLoadResponse;
use serde::Serialize;

use super::decision;

pub(super) const METHOD: &str = "size_basis_consistent_calibration_shadow_two_arm_v2";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Kind {
    PriorOnly,
    Observed,
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::{PrefillMetricProvenance, SchedulerLoadSnapshot};

    use super::*;

    fn p(kind: Kind, count: u64) -> Provenance {
        Provenance {
            kind,
            dp_rank: 0,
            counter_epoch: 1,
            progress_source: "prefill_started_requests",
            observed_sample_count: count,
            unshrunk_observed_mean: (count > 0).then_some(2000.0),
            configured_prior_tokens: 1024.0,
            configured_prior_samples: 32,
            effective_prior_weight: 32.0 / (32.0 + count as f64),
            blended_queue_tokens_per_request: (1024.0 * 32.0 + 2000.0 * count as f64)
                / (32.0 + count as f64),
        }
    }

    fn snapshot(state: &State, provenance: Option<Provenance>) -> decision::Snapshot {
        decision::Snapshot {
            size_basis: state.stamp(provenance),
            at: 1.0,
            profile: 7,
            baseline: 1.0,
            x: 100.0,
            source: "original_n_fallback",
            original_queue_extra: 100.0,
            live_queue_extra: None,
            scalar_gain: 4.0,
            fresh: true,
            native_ready: true,
            native_waiting: 1.0,
            poll_started: 0.0,
            eligible: true,
            context: decision::Context {
                complete: true,
                ..Default::default()
            },
        }
    }

    fn label(snapshot: decision::Snapshot, id: u64) -> decision::Label {
        decision::Label {
            id,
            at: 101.0,
            observed_at: 101.0,
            elapsed: 100.0,
            exact: true,
            first_label_at: 101.0,
            snapshot,
        }
    }

    #[test]
    fn delayed_prior_labels_cannot_enter_verified_observed_basis_epoch() {
        let mut state = State::default();
        let prior = p(Kind::PriorOnly, 0);
        state.poll(0.0, Some(7), Some(prior), true, 5.0);
        let admission = snapshot(&state, Some(prior));
        assert!(state.register(Some(1), 1.0, Some(admission)));
        let observed = p(Kind::Observed, 23);
        state.poll(2.0, Some(7), Some(observed), true, 5.0);
        assert_eq!(state.epoch, 1);
        assert_eq!(state.reason, "prior_only_to_observed");
        assert!(state.label(&label(admission, 1), true).is_none());
        assert_eq!(state.diagnostics()["state"]["rows"], 0);
        assert_eq!(
            admission
                .size_basis
                .unwrap()
                .provenance
                .observed_sample_count,
            0
        );
    }

    #[test]
    fn same_basis_labels_keep_admission_provenance_as_counts_mature() {
        let mut state = State::default();
        let observed = p(Kind::Observed, 1);
        state.poll(0.0, Some(7), Some(observed), true, 5.0);
        let admission = snapshot(&state, Some(observed));
        state.register(Some(2), 1.0, Some(admission));
        state.poll(2.0, Some(7), Some(p(Kind::Observed, 32)), true, 5.0);
        let retained = state.label(&label(admission, 2), true).unwrap();
        assert_eq!(state.epoch, 0);
        assert_eq!(
            retained
                .snapshot
                .size_basis
                .unwrap()
                .provenance
                .observed_sample_count,
            1
        );
        assert_eq!(state.diagnostics()["state"]["rows"], 1);
        state.poll(102.0, Some(7), Some(p(Kind::Observed, 64)), true, 5.0);
        assert_eq!(state.diagnostics()["state"]["rows"], 1);
    }

    #[test]
    fn missing_provenance_never_invents_transition_and_preserves_control_fallback() {
        let mut state = State::default();
        state.poll(0.0, Some(7), None, true, 5.0);
        let s = snapshot(&state, None);
        let original = decision::Admission {
            snapshot: s,
            predictions: decision::State::default().predictions(s, 1.0, false),
        };
        let candidate = state.admission(&original, &original, "control", 1.0, false);
        assert!(candidate.paired_fallback_to_original);
        assert_eq!(
            serde_json::to_value(&candidate.predictions).unwrap(),
            serde_json::to_value(&original.predictions).unwrap()
        );
        assert!(candidate.predictions.iter().all(|p| !p.available));
        state.poll(2.0, Some(7), Some(p(Kind::Observed, 1)), true, 5.0);
        assert_eq!(state.epoch, 0);
    }

    #[test]
    fn known_untrained_basis_uses_available_raw_fallback_without_changing_control() {
        let mut state = State::default();
        let observed = p(Kind::Observed, 1);
        state.poll(0.0, Some(7), Some(observed), true, 5.0);
        let s = snapshot(&state, Some(observed));
        let original = decision::Admission {
            snapshot: s,
            predictions: decision::State::default().predictions(s, 1.0, true),
        };
        let mut selected = original.clone();
        selected.snapshot.x = 80.0;
        selected.predictions = decision::State::default().predictions(selected.snapshot, 1.0, true);
        let candidate = state.admission(&original, &selected, "selected_v3", 1.0, true);
        assert!(!candidate.paired_fallback_to_original);
        assert_eq!(candidate.training_snapshot.x, 100.0);
        assert_eq!(candidate.prediction_snapshot.x, 80.0);
        assert_eq!(candidate.paired_control_predictions[1].total, 321.0);
        assert_eq!(candidate.predictions[1].total, 81.0);
        assert!(candidate.predictions[1].available);
        assert!(candidate.predictions[1].scalar_fallback);
        assert_eq!(selected.predictions[1].total, 321.0);
    }

    #[test]
    fn explicit_sample_presence_not_a_mean_equal_to_prior_defines_basis() {
        let mut load = WorkerLoadResponse {
            dp_rank_count: 1,
            loads: vec![SchedulerLoadSnapshot {
                num_waiting_uncached_tokens_available: Some(false),
                prefill_capacity_learning_active: Some(false),
                avg_request_prefill_kv_computed_tokens: Some(1024.0),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(provenance(&load, 1024.0, 32).is_none());
        load.loads[0].prefill_metric_provenance = Some(PrefillMetricProvenance {
            schema_version: 1,
            queue_size_observed_sample_count: Some(0),
            progress_source: Some("prefill_started_requests".into()),
            ..Default::default()
        });
        assert_eq!(provenance(&load, 1024.0, 32).unwrap().kind, Kind::PriorOnly);
        load.loads[0]
            .prefill_metric_provenance
            .as_mut()
            .unwrap()
            .queue_size_observed_sample_count = Some(2);
        load.loads[0].prefill_size_sample_count = Some(2);
        assert_eq!(provenance(&load, 1024.0, 32).unwrap().kind, Kind::Observed);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub(super) struct Provenance {
    pub kind: Kind,
    pub dp_rank: i32,
    pub counter_epoch: u64,
    pub progress_source: &'static str,
    pub observed_sample_count: u64,
    pub unshrunk_observed_mean: Option<f64>,
    pub configured_prior_tokens: f64,
    pub configured_prior_samples: u32,
    pub effective_prior_weight: f64,
    pub blended_queue_tokens_per_request: f64,
}

pub(super) fn provenance(
    load: &WorkerLoadResponse,
    prior_tokens: f64,
    prior_samples: u32,
) -> Option<Provenance> {
    if !load.ranks_are_dp_ranks() || load.dp_rank_count != 1 || load.loads.len() != 1 {
        return None;
    }
    let rank = &load.loads[0];
    let metrics = rank.prefill_metric_provenance.as_ref()?;
    if rank.num_waiting_uncached_tokens_available != Some(false)
        || rank.prefill_capacity_learning_active.is_none()
        || metrics.schema_version != 1
        || !prior_tokens.is_finite()
        || prior_tokens <= 0.0
    {
        return None;
    }
    let progress_source = match metrics.progress_source.as_deref()? {
        "local_compute_tokens" => "local_compute_tokens",
        "prefill_started_requests" => "prefill_started_requests",
        _ => return None,
    };
    let count = metrics.queue_size_observed_sample_count?;
    if count != rank.prefill_size_sample_count.unwrap_or(0) {
        return None;
    }
    let mean = rank
        .avg_request_prefill_kv_computed_tokens
        .or(rank.median_request_prefill_kv_computed_tokens)
        .filter(|v| v.is_finite() && *v >= 0.0);
    let (kind, prior_weight, blended) = if count == 0 {
        (Kind::PriorOnly, 1.0, prior_tokens)
    } else {
        let weight = f64::from(prior_samples) / (f64::from(prior_samples) + count as f64);
        (
            Kind::Observed,
            weight,
            (prior_tokens * f64::from(prior_samples) + mean? * count as f64)
                / (f64::from(prior_samples) + count as f64),
        )
    };
    Some(Provenance {
        kind,
        dp_rank: rank.dp_rank,
        counter_epoch: metrics.counter_epoch,
        progress_source,
        observed_sample_count: count,
        unshrunk_observed_mean: mean,
        configured_prior_tokens: prior_tokens,
        configured_prior_samples: prior_samples,
        effective_prior_weight: prior_weight,
        blended_queue_tokens_per_request: blended,
    })
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(super) struct Stamp {
    pub epoch: u64,
    pub provenance: Provenance,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct Admission {
    pub method: &'static str,
    pub basis_epoch: u64,
    pub basis_reason: &'static str,
    pub paired_fallback_to_original: bool,
    pub training_snapshot: decision::Snapshot,
    pub prediction_snapshot: decision::Snapshot,
    pub paired_control_method: &'static str,
    pub paired_control_predictions: [decision::Prediction; 2],
    pub predictions: [decision::Prediction; 2],
}

#[derive(Debug, Default)]
pub(super) struct State {
    inner: decision::State,
    pub epoch: u64,
    profile: Option<u64>,
    known: Option<Provenance>,
    usable: bool,
    exhausted: bool,
    pub reason: &'static str,
}

impl State {
    fn reset(&mut self, reason: &'static str) {
        self.inner = decision::State::default();
        if let Some(next) = self.epoch.checked_add(1) {
            self.epoch = next;
            self.reason = reason;
        } else {
            self.exhausted = true;
            self.reason = "basis_epoch_exhausted";
        }
    }

    pub fn poll(
        &mut self,
        now: f64,
        profile: Option<u64>,
        provenance: Option<Provenance>,
        fresh: bool,
        max_gap: f64,
    ) {
        if self.profile != profile {
            if self.profile.is_some() {
                self.reset("basis_profile_changed");
            }
            self.profile = profile;
            self.known = None;
        }
        self.usable = fresh && provenance.is_some() && !self.exhausted;
        if let Some(new) = provenance.filter(|_| self.usable) {
            if let Some(old) = self.known {
                if old.counter_epoch != new.counter_epoch
                    || old.progress_source != new.progress_source
                    || old.dp_rank != new.dp_rank
                {
                    self.reset("basis_counter_or_source_changed");
                } else if old.kind != new.kind {
                    self.reset(if old.kind == Kind::PriorOnly {
                        "prior_only_to_observed"
                    } else {
                        "observed_to_prior_only"
                    });
                } else {
                    self.reason = "basis_unchanged";
                }
            } else {
                self.reason = "basis_first_verified";
            }
            self.known = Some(new);
        } else {
            self.reason = "basis_provenance_missing_or_stale";
        }
        self.inner
            .poll(now, profile, self.usable && !self.exhausted, max_gap);
    }

    pub fn stamp(&self, provenance: Option<Provenance>) -> Option<Stamp> {
        let provenance = provenance.filter(|_| self.usable && !self.exhausted)?;
        (Some(provenance) == self.known).then_some(Stamp {
            epoch: self.epoch,
            provenance,
        })
    }

    pub fn register(
        &mut self,
        id: Option<u64>,
        started: f64,
        snapshot: Option<decision::Snapshot>,
    ) -> bool {
        let Some(mut snapshot) = snapshot.filter(|s| self.matches(*s)) else {
            return false;
        };
        snapshot.scalar_gain = 1.0;
        self.inner.register(id, started, Some(snapshot))
    }

    fn matches(&self, snapshot: decision::Snapshot) -> bool {
        !self.exhausted
            && snapshot.size_basis.is_some_and(|stamp| {
                stamp.epoch == self.epoch
                    && self.known.is_some_and(|p| p.kind == stamp.provenance.kind)
            })
            && self.profile == Some(snapshot.profile)
    }

    pub fn label(&mut self, original: &decision::Label, terminal: bool) -> Option<decision::Label> {
        if !self.matches(original.snapshot) {
            return None;
        }
        self.inner.label(
            original.id,
            original.observed_at,
            original.at,
            original.exact,
            terminal,
            Some(original.elapsed),
        )
    }

    pub fn terminal_unknown(&mut self, id: Option<u64>) {
        self.inner.terminal_unknown(id);
    }

    pub fn admission(
        &self,
        original: &decision::Admission,
        control: &decision::Admission,
        control_method: &'static str,
        now: f64,
        available: bool,
    ) -> Admission {
        let paired_fallback_to_original = !self.usable || !self.matches(original.snapshot);
        let mut snapshot = control.snapshot;
        snapshot.scalar_gain = 1.0;
        Admission {
            method: METHOD,
            basis_epoch: self.epoch,
            basis_reason: if paired_fallback_to_original {
                "basis_unavailable_preserve_original"
            } else {
                "verified_size_basis_raw_or_independent_fit"
            },
            paired_fallback_to_original,
            training_snapshot: original.snapshot,
            prediction_snapshot: if paired_fallback_to_original {
                control.snapshot
            } else {
                snapshot
            },
            paired_control_method: control_method,
            paired_control_predictions: control.predictions.clone(),
            predictions: if paired_fallback_to_original {
                control.predictions.clone()
            } else {
                self.inner.predictions(snapshot, now, available)
            },
        }
    }

    pub fn diagnostics(&self) -> serde_json::Value {
        serde_json::json!({"basis_epoch":self.epoch,"basis":self.known,"usable":self.usable,
            "reason":self.reason,"state":self.inner.diagnostics()})
    }
}
