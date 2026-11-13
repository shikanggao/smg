//! Private C2 denominator candidate. Request starts remain a work proxy, not
//! completed engine service. No monitor, training or native guard reads this state.
use std::collections::VecDeque;

use openai_protocol::worker::WorkerLoadResponse;
use serde::Serialize;

const WINDOW: usize = 32;
const MIN_SAMPLES: usize = 8;
pub const METHOD: &str = "started_requests_same_poll_observed_conversion_basis_v2";

fn nonnegative(v: f64) -> bool {
    v.is_finite() && v >= 0.0
}
fn positive(v: f64) -> bool {
    v.is_finite() && v > 0.0
}
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9_f64.max(1e-12 * a.abs().max(b.abs()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct SourceKey {
    profile: u64,
    pub counter_epoch: u64,
    pub dp_rank: i32,
}

pub fn source_key(load: &WorkerLoadResponse) -> Option<SourceKey> {
    if !load.ranks_are_dp_ranks() || load.dp_rank_count != 1 || load.loads.len() != 1 {
        return None;
    }
    let r = &load.loads[0];
    let p = r.prefill_metric_provenance.as_ref()?;
    (r.dp_rank == 0
        && r.native_request_counts_available == Some(true)
        && r.timely_prefill_progress_source.as_deref() == Some("prefill_started_requests")
        && p.schema_version == 1
        && p.progress_source.as_deref() == Some("prefill_started_requests"))
    .then_some(SourceKey {
        profile: r.native_request_counts_profile?,
        counter_epoch: p.counter_epoch,
        dp_rank: r.dp_rank,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Anchor {
    counter: f64,
    ended: f64,
}
#[derive(Clone, Copy, Debug)]
struct Sample {
    raw_rate: f64,
    original_size: f64,
}
#[derive(Clone, Copy, Debug, PartialEq)]
struct Poll {
    started: f64,
    key: SourceKey,
    anchor: Anchor,
    previous: Option<f64>,
    began: Option<f64>,
    interval: Option<f64>,
    raw_rate: Option<f64>,
    size: Option<f64>,
    eligible: bool,
    count: usize,
    learned: Option<f64>,
    reset: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ObservedSize {
    mean: Option<f64>,
    count: Option<u64>,
    reason: &'static str,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct SamePollObservedSize {
    pub method: &'static str,
    pub status: &'static str,
    pub reason: &'static str,
    pub current_unshrunk_size_mean: Option<f64>,
    pub current_observed_sample_count: Option<u64>,
    pub rebased_learned_capacity: Option<f64>,
    pub used_capacity: Option<f64>,
}
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Selection {
    pub status: &'static str,
    pub reason: &'static str,
    pub raw_poll_started: f64,
    pub source: Option<SourceKey>,
    pub history_complete: bool,
    pub history_samples: usize,
    pub previous_conversion_size_mean: Option<f64>,
    pub lower_quartile_request_start_rate: Option<f64>,
    /// Previous-conversion (1d) reference, not the optional same-poll product.
    pub rebased_learned_capacity: Option<f64>,
    /// Previous-conversion (1d) divisor. The actual selected divisor is the
    /// nested same_poll_observed_size.used_capacity only when CapacityBasis's
    /// selected_composition names that new branch.
    pub used_capacity: Option<f64>,
    pub configured_floor: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub same_poll_observed_size: Option<SamePollObservedSize>,
}

#[derive(Clone, Debug)]
pub struct History {
    key: Option<SourceKey>,
    anchor: Option<Anchor>,
    last_started: Option<f64>,
    latest: Option<Poll>,
    latest_observed_size: Option<ObservedSize>,
    samples: VecDeque<Sample>,
    complete: bool,
    reason: &'static str,
}
impl Default for History {
    fn default() -> Self {
        Self {
            key: None,
            anchor: None,
            last_started: None,
            latest: None,
            latest_observed_size: None,
            samples: VecDeque::new(),
            complete: false,
            reason: "history_unobserved",
        }
    }
}
impl History {
    fn observed_size(load: Option<&WorkerLoadResponse>) -> ObservedSize {
        let mut evidence = ObservedSize {
            mean: None,
            count: None,
            reason: "current_observed_size_unavailable",
        };
        let Some(load) = load.filter(|l| source_key(l).is_some()) else {
            return evidence;
        };
        let r = &load.loads[0];
        let p = r.prefill_metric_provenance.as_ref().unwrap();
        if p.counter_reset {
            evidence.reason = "current_observed_size_counter_reset";
            return evidence;
        }
        evidence.mean = p.current_unshrunk_size_mean.filter(|v| positive(*v));
        evidence.count = p.queue_size_observed_sample_count.filter(|n| *n > 0);
        if evidence.count.is_none() || evidence.count != r.prefill_size_sample_count {
            evidence.reason = "current_observed_size_count_unavailable_or_mismatch";
        } else if !evidence.mean.is_some_and(|mean| {
            r.avg_request_prefill_kv_computed_tokens
                .or(r.median_request_prefill_kv_computed_tokens)
                .is_some_and(|value| positive(value) && close(mean, value))
        }) {
            evidence.reason = "current_observed_size_mean_unavailable_or_mismatch";
        } else {
            evidence.reason = "same_poll_observed_completed_size";
        }
        evidence
    }

    fn unknown(&mut self, reason: &'static str) {
        self.samples.clear();
        self.complete = false;
        self.reason = reason;
    }
    fn poll(load: &WorkerLoadResponse, started: f64) -> Option<Poll> {
        let key = source_key(load)?;
        let r = &load.loads[0];
        let p = r.prefill_metric_provenance.as_ref()?;
        let count = usize::try_from(r.prefill_capacity_sample_count?).ok()?;
        if !started.is_finite()
            || count > WINDOW
            || r.prefill_capacity_learning_active != Some(p.capacity_interval_eligible)
        {
            return None;
        }
        let anchor = Anchor {
            counter: p.raw_counter.filter(|v| nonnegative(*v))?,
            ended: p.interval_ended_time_seconds.filter(|v| v.is_finite())?,
        };
        Some(Poll {
            started,
            key,
            anchor,
            previous: p.previous_raw_counter,
            began: p.interval_started_time_seconds,
            interval: p.interval_seconds,
            raw_rate: p.raw_rate,
            size: p.previous_conversion_size_mean,
            eligible: p.capacity_interval_eligible,
            count,
            learned: r.learned_prefill_capacity,
            reset: p.counter_reset,
        })
    }
    pub fn observe(&mut self, load: Option<&WorkerLoadResponse>, started: f64, success: bool) {
        if !started.is_finite() {
            self.unknown("invalid_poll_clock");
            self.anchor = None;
            self.latest = None;
            self.latest_observed_size = None;
            return;
        }
        let poll = success
            .then(|| load.and_then(|l| Self::poll(l, started)))
            .flatten();
        if self.last_started.is_some_and(|old| started <= old) {
            if poll.is_some() && poll == self.latest {
                if self.latest_observed_size != Some(Self::observed_size(load)) {
                    self.latest_observed_size = Some(ObservedSize {
                        mean: None,
                        count: None,
                        reason: "duplicate_current_observed_size_changed",
                    });
                }
                return; // Exact repeat of the same already observed interval.
            }
            self.unknown("duplicate_conflicting_or_out_of_order_poll");
            self.anchor = None;
            self.latest = None;
            self.latest_observed_size = None;
            return;
        }
        self.last_started = Some(started);
        let Some(poll) = poll else {
            self.unknown("source_poll_unavailable");
            self.anchor = None;
            self.latest = None;
            self.latest_observed_size = None;
            return;
        };
        if self.key.is_some_and(|key| key != poll.key) {
            self.unknown("source_identity_changed");
            self.anchor = None;
        }
        self.key = Some(poll.key);
        self.latest = Some(poll);
        self.latest_observed_size = Some(Self::observed_size(load));
        let initial = poll.previous.is_none()
            && poll.began.is_none()
            && poll.interval.is_none()
            && poll.raw_rate.is_none();
        let rate = if initial && poll.count == 0 && self.anchor.is_none() {
            None
        } else {
            let (Some(previous), Some(began), Some(interval)) =
                (poll.previous, poll.began, poll.interval)
            else {
                self.unknown("interval_witness_unavailable");
                self.anchor = Some(poll.anchor);
                return;
            };
            if !nonnegative(previous)
                || !began.is_finite()
                || !positive(interval)
                || !positive(poll.anchor.ended - began)
                || !close(poll.anchor.ended - began, interval)
            {
                self.unknown("interval_or_raw_rate_inconsistent");
                self.anchor = Some(poll.anchor);
                return;
            }
            if self
                .anchor
                .is_some_and(|old| old.counter != previous || old.ended != began)
            {
                self.unknown("counter_or_time_adjacency_gap");
            }
            if poll.reset {
                if poll.raw_rate.is_some() {
                    self.unknown("reset_rate_witness_inconsistent");
                    self.anchor = Some(poll.anchor);
                    return;
                }
                None
            } else {
                let rate = (poll.anchor.counter - previous) / interval;
                if !nonnegative(rate)
                    || !poll
                        .raw_rate
                        .is_some_and(|v| nonnegative(v) && close(rate, v))
                {
                    self.unknown("interval_or_raw_rate_inconsistent");
                    self.anchor = Some(poll.anchor);
                    return;
                }
                Some(rate)
            }
        };
        self.anchor = Some(poll.anchor);
        // A published empty source window proves the unknown prefix is gone.
        // Broader recovery epoch/reset flags alone do not prove a capacity clear.
        if poll.count == 0 {
            if poll.eligible || poll.learned.is_some() {
                self.unknown("invalid_empty_source_window");
            } else {
                self.samples.clear();
                self.complete = true;
                self.reason = "insufficient_samples";
            }
            self.anchor = Some(poll.anchor);
            return;
        }
        if poll.reset {
            self.unknown("counter_reset_window_not_proven_empty");
            self.anchor = Some(poll.anchor);
            return;
        }
        let Some(rate) = rate else {
            self.unknown("interval_witness_unavailable");
            return;
        };
        if poll.eligible {
            let Some(size) = poll.size.filter(|v| positive(*v)) else {
                self.unknown("eligible_original_conversion_unavailable");
                return;
            };
            if !positive(rate) || !positive(rate * size) {
                self.unknown("eligible_original_rate_nonpositive");
                return;
            }
            self.samples.push_back(Sample {
                raw_rate: rate,
                original_size: size,
            });
            if self.samples.len() > WINDOW {
                self.samples.pop_front();
            }
        }
        if self.complete && self.samples.len() != poll.count {
            self.unknown("source_sample_count_mismatch");
        }
        if self.samples.len() == WINDOW && poll.count == WINDOW {
            self.complete = true;
        }
        if self.complete && self.samples.len() >= MIN_SAMPLES {
            let mut original: Vec<_> = self
                .samples
                .iter()
                .map(|s| s.raw_rate * s.original_size)
                .collect();
            original.sort_by(f64::total_cmp);
            let old = original[(original.len() - 1) / 4];
            if !poll.learned.is_some_and(|v| positive(v) && close(v, old)) {
                self.unknown("original_capacity_recurrence_mismatch");
                return;
            }
            self.reason = "complete_original_window";
        } else if self.complete {
            if poll.learned.is_some() {
                self.unknown("premature_source_capacity");
            } else {
                self.reason = "insufficient_samples";
            }
        }
    }
    /// The current interval was fully checked by observe, but adds no sample.
    /// This says nothing about GPU idleness or the age of the positive window.
    pub fn certified_zero_interval(&self, started: f64, key: Option<SourceKey>) -> bool {
        self.complete
            && self.reason == "complete_original_window"
            && self.samples.len() >= MIN_SAMPLES
            && self.latest.is_some_and(|p| {
                p.started == started
                    && Some(p.key) == key
                    && !p.reset
                    && !p.eligible
                    && p.raw_rate == Some(0.0)
                    && p.previous == Some(p.anchor.counter)
                    && p.size.is_some_and(positive)
                    && p.learned.is_some_and(positive)
            })
    }

    pub fn selection(&self, started: f64, key: Option<SourceKey>, floor: f64) -> Selection {
        let mut result = Selection {
            status: "unavailable",
            reason: self.reason,
            raw_poll_started: started,
            source: key,
            history_complete: false,
            history_samples: 0,
            previous_conversion_size_mean: None,
            lower_quartile_request_start_rate: None,
            rebased_learned_capacity: None,
            used_capacity: None,
            configured_floor: positive(floor).then_some(floor),
            same_poll_observed_size: None,
        };
        if !self
            .latest
            .is_some_and(|p| p.started == started && Some(p.key) == key)
        {
            result.reason = "captured_source_or_poll_mismatch";
            return result;
        }
        result.history_complete = self.complete;
        result.history_samples = self.samples.len();
        result.previous_conversion_size_mean = self.latest.and_then(|p| p.size);
        if !self.complete
            || self.samples.len() < MIN_SAMPLES
            || self.reason != "complete_original_window"
        {
            return result;
        }
        let Some(size) = result
            .previous_conversion_size_mean
            .filter(|v| positive(*v))
        else {
            result.reason = "current_prior_conversion_unavailable";
            return result;
        };
        let mut rates: Vec<_> = self.samples.iter().map(|s| s.raw_rate).collect();
        rates.sort_by(f64::total_cmp);
        let quantile = rates[(rates.len() - 1) / 4];
        let learned = quantile * size;
        if !positive(learned) || !positive(floor) {
            result.reason = "rebased_capacity_or_configured_floor_invalid";
            return result;
        }
        result.status = "available";
        result.reason = "complete_current_conversion_basis";
        result.lower_quartile_request_start_rate = Some(quantile);
        result.rebased_learned_capacity = Some(learned);
        result.used_capacity = Some(learned.max(floor));
        // The previous-conversion selection above remains the eligibility gate
        // and exact 1d reference. Current optional size metadata never changes
        // rate membership, original-size recurrence, or old fallback readiness.
        if let Some(current) = self.latest_observed_size {
            let mut selected = SamePollObservedSize {
                method: "same_poll_observed_completed_uncached_size_v1",
                status: "unavailable",
                reason: current.reason,
                current_unshrunk_size_mean: current.mean,
                current_observed_sample_count: current.count,
                rebased_learned_capacity: None,
                used_capacity: None,
            };
            if current.reason == "same_poll_observed_completed_size" {
                let capacity = quantile * current.mean.unwrap();
                if positive(capacity) {
                    selected.status = "available";
                    selected.rebased_learned_capacity = Some(capacity);
                    selected.used_capacity = Some(capacity.max(floor));
                } else {
                    selected.reason = "same_poll_rebased_capacity_invalid";
                }
            }
            result.same_poll_observed_size = Some(selected);
        }
        result
    }
}

#[cfg(test)]
mod tests;
