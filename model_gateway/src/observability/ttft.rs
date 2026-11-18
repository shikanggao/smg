//! Admission-time prediction accounting. Independent of learning epochs: a
//! worker reset must not erase the outcome of an already admitted attempt.
//! Request IDs belong in traces, never metric labels. One handle is shared by
//! placement and stream observation; the last owner resolves an unfinished
//! attempt as unknown. Poll-based lower bounds never become exact latencies.
use std::{sync::Arc, time::Instant};

use parking_lot::Mutex;

pub(crate) const ERROR_BUCKETS: &[f64] = &[0.0, 1.0, 5.0, 10.0, 30.0, 60.0, 100.0, 300.0];
pub(crate) const ERROR_METRIC: &str = "smg_ttft_overestimate_seconds";

pub(crate) fn describe() {
    metrics::describe_counter!(
        "smg_ttft_predictions_total",
        "Admission attempts by prediction availability"
    );
    metrics::describe_counter!(
        "smg_ttft_outcomes_total",
        "One evaluation outcome per admission attempt"
    );
    metrics::describe_counter!(
        "smg_ttft_classifications_total",
        "Observable prediction/TTFT classifications; rejected counterfactuals are excluded"
    );
    metrics::describe_histogram!(
        ERROR_METRIC,
        "Positive prediction error on exact first-text outcomes only"
    );
    metrics::describe_counter!(
        "smg_ttft_invalid_evidence_used_total",
        "Selected prediction validity violations detected at the decision boundary"
    );
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Mode {
    Shadow,
    Enforce,
}
impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Shadow => "shadow",
            Self::Enforce => "enforce",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Outcome {
    FirstText,
    /// The stream remained observed without text until this timestamp.
    Interrupted,
    Rejected,
    /// Observation was lost; elapsed wall time alone is not a lower bound.
    Unknown,
}

pub(crate) use crate::worker::ttft_model::prediction::SelectedPrediction as Prediction;

#[derive(Debug)]
struct Snapshot {
    prediction: Option<f64>,
    threshold: Option<f64>,
    engine: &'static str,
}

#[derive(Debug, Default)]
struct State {
    snapshot: Option<Snapshot>,
    resolved: bool,
}

#[derive(Debug)]
struct Inner {
    model: Arc<str>,
    mode: Mode,
    started: Instant,
    state: Mutex<State>,
}

/// Cloning the handle does not count another attempt or outcome.
#[derive(Clone, Debug)]
pub(crate) struct Observation(Arc<Inner>);

impl Observation {
    pub(crate) fn new(model: &str, mode: Mode, started: Instant) -> Self {
        Self(Arc::new(Inner {
            model: Arc::from(model),
            mode,
            started,
            state: Mutex::new(State::default()),
        }))
    }

    /// Freeze exactly once before dispatch. A missing forecast still counts.
    pub(crate) fn freeze(
        &self,
        prediction: Option<Prediction>,
        threshold: Option<f64>,
        engine: &'static str,
    ) {
        self.0
            .freeze(&mut self.0.state.lock(), prediction, threshold, engine);
    }

    pub(crate) fn resolve(&self, outcome: Outcome, at: Instant) {
        self.0.resolve(outcome, at);
    }
}

impl Inner {
    fn freeze(
        &self,
        state: &mut State,
        prediction: Option<Prediction>,
        threshold: Option<f64>,
        engine: &'static str,
    ) {
        if state.snapshot.is_some() {
            return;
        }
        let threshold = threshold.filter(|v| v.is_finite() && *v > 0.0);
        let prediction = prediction.and_then(|p| {
            let violation = if !p.total.is_finite() || p.total < 0.0 {
                Some("invalid_prediction")
            } else if !p.evidence_valid {
                Some("invalid_evidence")
            } else if threshold.is_none() {
                Some("invalid_threshold")
            } else { None };
            if let Some(reason) = violation {
                metrics::counter!("smg_ttft_invalid_evidence_used_total", "model" => self.model.clone(), "engine" => engine, "mode" => self.mode.name(), "reason" => reason).increment(1);
                None
            } else { Some(p.total) }
        });
        metrics::counter!("smg_ttft_predictions_total", "model" => self.model.clone(), "engine" => engine, "mode" => self.mode.name(), "availability" => if prediction.is_some() { "usable" } else { "unavailable" }).increment(1);
        state.snapshot = Some(Snapshot {
            prediction,
            threshold,
            engine,
        });
    }

    fn resolve(&self, outcome: Outcome, at: Instant) {
        let mut state = self.state.lock();
        if state.resolved {
            return;
        }
        self.freeze(&mut state, None, None, "unknown");
        state.resolved = true;
        let Some(s) = &state.snapshot else {
            return;
        };
        let elapsed = at
            .checked_duration_since(self.started)
            .map(|v| v.as_secs_f64());
        let (kind, actual_late, exact) = match (outcome, elapsed) {
            (Outcome::FirstText, Some(t)) => ("exact", s.threshold.map(|b| t >= b), Some(t)),
            (Outcome::Interrupted, Some(t)) if s.threshold.is_some_and(|b| t >= b) => {
                ("late_censored", Some(true), None)
            }
            (Outcome::Interrupted, Some(_)) => ("early_interrupted", None, None),
            (Outcome::Rejected, _) => ("rejected", None, None),
            _ => ("unknown", None, None),
        };
        metrics::counter!("smg_ttft_outcomes_total", "model" => self.model.clone(), "engine" => s.engine, "mode" => self.mode.name(), "outcome" => kind).increment(1);
        if let (Some(predicted), Some(threshold), Some(late)) =
            (s.prediction, s.threshold, actual_late)
        {
            let result = match (predicted >= threshold, late) {
                (true, true) => "tp",
                (true, false) => "fp",
                (false, true) => "fn",
                (false, false) => "tn",
            };
            metrics::counter!("smg_ttft_classifications_total", "model" => self.model.clone(), "engine" => s.engine, "mode" => self.mode.name(), "result" => result).increment(1);
        }
        if let (Some(predicted), Some(actual)) = (s.prediction, exact) {
            metrics::histogram!(ERROR_METRIC, "model" => self.model.clone(), "engine" => s.engine, "mode" => self.mode.name()).record((predicted - actual).max(0.0));
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.resolve(Outcome::Unknown, Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use metrics_exporter_prometheus::{Matcher, PrometheusBuilder};

    use super::*;

    fn capture(f: impl FnOnce()) -> String {
        let recorder = PrometheusBuilder::new()
            .set_buckets_for_metric(Matcher::Full(ERROR_METRIC.into()), ERROR_BUCKETS)
            .unwrap()
            .build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, f);
        handle.render()
    }
    fn observed(start: Instant, predicted: Option<f64>) -> Observation {
        let o = Observation::new("model", Mode::Shadow, start);
        o.freeze(
            predicted.map(|total| Prediction {
                total,
                evidence_valid: true,
            }),
            Some(300.0),
            "vllm",
        );
        o
    }
    fn counter(text: &str, metric: &str, label: &str, value: &str) -> f64 {
        text.lines()
            .filter(|l| {
                l.starts_with(&format!("{metric}{{")) && l.contains(&format!("{label}=\"{value}\""))
            })
            .map(|l| l.rsplit_once(' ').unwrap().1.parse::<f64>().unwrap())
            .sum()
    }

    #[test]
    fn classifications_use_saved_prediction_and_total_deadline_boundary() {
        let s = capture(|| {
            let now = Instant::now();
            for (prediction, actual) in [
                (350.0, 200.0),
                (200.0, 300.0),
                (300.0, 300.0),
                (299.0, 299.0),
            ] {
                let o = observed(now, Some(prediction));
                o.freeze(
                    Some(Prediction {
                        total: 0.0,
                        evidence_valid: true,
                    }),
                    Some(1.0),
                    "sglang",
                );
                o.resolve(Outcome::FirstText, now + Duration::from_secs_f64(actual));
                o.resolve(Outcome::Interrupted, now + Duration::from_secs(400));
            }
        });
        for result in ["fp", "fn", "tp", "tn"] {
            assert_eq!(
                counter(&s, "smg_ttft_classifications_total", "result", result),
                1.0
            );
        }
        assert_eq!(
            counter(&s, "smg_ttft_outcomes_total", "outcome", "exact"),
            4.0
        );
        assert_eq!(
            counter(&s, "smg_ttft_overestimate_seconds_bucket", "le", "100"),
            3.0
        );
        assert!(!s.contains("engine=\"sglang\""));
    }

    #[test]
    fn censored_and_rejected_outcomes_never_create_exact_errors() {
        let s = capture(|| {
            let now = Instant::now();
            for (outcome, seconds) in [
                (Outcome::Interrupted, 299),
                (Outcome::Interrupted, 300),
                (Outcome::Rejected, 400),
                (Outcome::Unknown, 400),
            ] {
                observed(now, Some(400.0)).resolve(outcome, now + Duration::from_secs(seconds));
            }
        });
        for outcome in ["early_interrupted", "late_censored", "rejected", "unknown"] {
            assert_eq!(
                counter(&s, "smg_ttft_outcomes_total", "outcome", outcome),
                1.0
            );
        }
        assert_eq!(
            counter(&s, "smg_ttft_classifications_total", "result", "tp"),
            1.0
        );
        assert!(!s.contains(ERROR_METRIC));
    }

    #[test]
    fn missing_predictions_and_unattached_observers_remain_in_coverage() {
        let s = capture(|| {
            let now = Instant::now();
            observed(now, None).resolve(Outcome::FirstText, now + Duration::from_secs(20));
            let o = Observation::new("model", Mode::Enforce, now);
            let copy = o.clone();
            drop(o);
            drop(copy);
        });
        assert_eq!(
            counter(
                &s,
                "smg_ttft_predictions_total",
                "availability",
                "unavailable"
            ),
            2.0
        );
        assert_eq!(
            counter(&s, "smg_ttft_outcomes_total", "outcome", "unknown"),
            1.0
        );
        assert!(!s.contains("smg_ttft_classifications_total"));
    }

    #[test]
    fn selected_invalid_evidence_is_detected_and_not_scored() {
        let s = capture(|| {
            let now = Instant::now();
            let o = Observation::new("model", Mode::Shadow, now);
            o.freeze(
                Some(Prediction {
                    total: 10.0,
                    evidence_valid: false,
                }),
                Some(300.0),
                "sglang",
            );
            o.resolve(Outcome::FirstText, now + Duration::from_secs(10));
        });
        assert_eq!(
            counter(
                &s,
                "smg_ttft_invalid_evidence_used_total",
                "reason",
                "invalid_evidence"
            ),
            1.0
        );
        assert_eq!(
            counter(
                &s,
                "smg_ttft_predictions_total",
                "availability",
                "unavailable"
            ),
            1.0
        );
        assert!(!s.contains(ERROR_METRIC));
    }

    #[test]
    fn concurrent_resolution_is_exactly_once() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let now = Instant::now();
        let o = metrics::with_local_recorder(&recorder, || observed(now, Some(10.0)));
        std::thread::scope(|scope| {
            for _ in 0..16 {
                let o = o.clone();
                let recorder = &recorder;
                scope.spawn(move || {
                    metrics::with_local_recorder(recorder, || {
                        o.resolve(Outcome::FirstText, now + Duration::from_secs(20))
                    })
                });
            }
        });
        metrics::with_local_recorder(&recorder, || drop(o));
        let s = handle.render();
        assert_eq!(
            counter(&s, "smg_ttft_outcomes_total", "outcome", "exact"),
            1.0
        );
        assert_eq!(
            counter(&s, "smg_ttft_classifications_total", "result", "tn"),
            1.0
        );
        assert_eq!(
            counter(&s, "smg_ttft_outcomes_total", "outcome", "unknown"),
            0.0
        );
    }
}
