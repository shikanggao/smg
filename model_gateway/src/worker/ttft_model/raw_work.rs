//! Immutable queue-work basis, retained across native publication gaps.
use serde::Serialize;
fn nonnegative(v: f64) -> bool {
    v.is_finite() && v >= 0.0
}
fn positive(v: f64) -> bool {
    v.is_finite() && v > 0.0
}
/// Frozen with CalibrationForecast, before a later poll can replace Entry.sample.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct FrozenRawWork {
    pub poll_started: f64,
    pub source: Option<&'static str>,
    pub key: Option<super::started_capacity::SourceKey>,
    pub token_work: f64,
    pub original_capacity: f64,
    pub observed_delay_floor: f64,
    pub measured_delay_only: bool,
}
impl FrozenRawWork {
    pub fn extra(self, capacity: f64) -> Option<f64> {
        if !nonnegative(self.token_work)
            || !positive(capacity)
            || !nonnegative(self.observed_delay_floor)
        {
            return None;
        }
        let extra = if self.measured_delay_only {
            self.observed_delay_floor
        } else {
            (self.token_work / capacity).max(self.observed_delay_floor)
        };
        nonnegative(extra).then_some(extra)
    }
}
