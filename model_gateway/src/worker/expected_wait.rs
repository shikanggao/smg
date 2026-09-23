//! Expected-wait arithmetic shared by routing and admission.
//!
//! Callers own signal validity and missing-data policy. Preparing a sample
//! resolves the KV term once; dispatch updates only add token work.

/// Default KV-pressure time penalty, in seconds.
pub const DEFAULT_KV_PRESSURE_WEIGHT: f64 = 0.15;
/// Token estimate for a request without a known token count.
pub const DEFAULT_MEAN_PREFILL_TOKENS: u32 = 1024;
/// Fallback aggregate generation throughput, in tokens per second.
pub const DEFAULT_THROUGHPUT: f64 = 2000.0;
/// Fraction of already-dispatched prompt work that blocks a new request.
pub const DEFAULT_DISPATCH_BLOCKING_FACTOR: f64 = 0.05;
/// Maximum KV-pressure contribution to estimated-wait admission, in seconds.
pub const DEFAULT_MAX_KV_PENALTY_SECS: f64 = 5.0;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ExpectedWait {
    queued_tokens: f64,
    throughput: f64,
    base_overhead: f64,
    queue_work_correction: f64,
    dispatch_blocking_factor: f64,
    kv_wait: f64,
}

impl ExpectedWait {
    pub(crate) fn new(queued_tokens: f64, throughput: f64, usage: f64, weight: f64) -> Self {
        let k = usage.clamp(0.0, 0.999);
        Self {
            queued_tokens,
            throughput,
            base_overhead: 0.0,
            queue_work_correction: 1.0,
            dispatch_blocking_factor: 1.0,
            kv_wait: weight * k / (1.0 - k),
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the calibrated equation is clearer when each named coefficient stays explicit"
    )]
    pub(crate) fn calibrated(
        queued_tokens: f64,
        throughput: f64,
        usage: f64,
        base_overhead: f64,
        queue_work_correction: f64,
        dispatch_blocking_factor: f64,
        kv_pressure_threshold: f64,
        kv_pressure_weight: f64,
    ) -> Self {
        let k = usage.clamp(0.0, 0.999);
        Self {
            queued_tokens,
            throughput,
            base_overhead,
            queue_work_correction,
            dispatch_blocking_factor,
            kv_wait: kv_pressure_weight * (k - kv_pressure_threshold).max(0.0) / (1.0 - k),
        }
    }

    pub(crate) fn seconds(self, dispatched_tokens: u64) -> f64 {
        self.base_overhead
            + self.queue_work_correction
                * (self.queued_tokens + self.dispatch_blocking_factor * dispatched_tokens as f64)
                / self.throughput
            + self.kv_wait
    }

    /// Limit the KV-pressure contribution without changing queue accounting.
    pub(crate) fn with_kv_wait_cap(mut self, max_secs: f64) -> Self {
        self.kv_wait = self.kv_wait.min(max_secs);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::ExpectedWait;

    #[test]
    fn calibrated_wait_applies_base_queue_and_dispatch_terms_with_kv_disabled() {
        let wait = ExpectedWait::calibrated(8_000.0, 5_000.0, 0.99, 0.05, 0.7, 0.4, 0.8, 0.0);

        assert!((wait.seconds(2_000) - 1.282).abs() < 1e-9);
    }
}
