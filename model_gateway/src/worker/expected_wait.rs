//! Expected-wait arithmetic shared by routing and admission.
//!
//! Callers own signal validation and missing-data fallbacks. This module only
//! evaluates the common score once those inputs have been resolved.

/// Default KV-pressure time penalty, in seconds.
pub const DEFAULT_KV_PRESSURE_WEIGHT: f64 = 0.15;
/// Token estimate for a request without a known token count.
pub const DEFAULT_MEAN_PREFILL_TOKENS: u32 = 1024;
/// Fallback aggregate throughput, in tokens per second.
pub const DEFAULT_THROUGHPUT: f64 = 2000.0;

/// Fraction of newly dispatched work used by calibrated admission.
pub const DEFAULT_DISPATCH_BLOCKING_FACTOR: f64 = 0.05;

/// Prepared expected-wait score for one worker load snapshot.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExpectedWait {
    queued_tokens: f64,
    throughput: f64,
    base_overhead: f64,
    dispatch_blocking_factor: f64,
}

impl ExpectedWait {
    pub(crate) fn token_work(self, dispatched_tokens: u64) -> f64 {
        self.queued_tokens + self.dispatch_blocking_factor * dispatched_tokens as f64
    }

    pub(crate) fn queued_work(self) -> f64 {
        self.queued_tokens
    }

    pub(crate) fn with_base_overhead(mut self, seconds: f64) -> Self {
        self.base_overhead = seconds;
        self
    }

    pub(crate) fn calibrated(
        queued_tokens: f64,
        throughput: f64,
        base_overhead: f64,
        dispatch_blocking_factor: f64,
    ) -> Self {
        Self {
            queued_tokens,
            throughput,
            base_overhead,
            dispatch_blocking_factor,
        }
    }

    pub(crate) fn new(queued_tokens: f64, throughput: f64, usage: f64, weight: f64) -> Self {
        let k = usage.clamp(0.0, 0.999);
        Self {
            queued_tokens,
            throughput,
            base_overhead: weight * k / (1.0 - k),
            dispatch_blocking_factor: 1.0,
        }
    }

    pub(crate) fn seconds(self, dispatched_tokens: u64) -> f64 {
        self.base_overhead + self.token_work(dispatched_tokens) / self.throughput
    }
}

#[cfg(test)]
mod tests {
    use super::ExpectedWait;

    #[test]
    fn calibrated_wait_applies_only_base_queue_and_dispatch_terms() {
        let wait = ExpectedWait::calibrated(8_000.0, 5_000.0, 0.05, 0.4);
        assert!((wait.seconds(2_000) - 1.81).abs() < 1e-9);
    }

    #[test]
    fn combines_queued_dispatched_and_kv_wait() {
        let wait = ExpectedWait::new(800.0, 500.0, 0.5, 0.2);

        assert!((wait.seconds(200) - 2.2).abs() < f64::EPSILON);
    }

    #[test]
    fn clamps_kv_usage_to_formula_bounds() {
        let below_zero = ExpectedWait::new(0.0, 100.0, -1.0, 0.2).seconds(0);
        let at_upper_bound = ExpectedWait::new(0.0, 100.0, 0.999, 0.2).seconds(0);
        let above_one = ExpectedWait::new(0.0, 100.0, 2.0, 0.2).seconds(0);

        assert_eq!(below_zero, 0.0);
        assert!(at_upper_bound.is_finite());
        assert_eq!(above_one, at_upper_bound);
    }
}
