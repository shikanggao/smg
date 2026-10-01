//! Token-work expected-wait arithmetic used by admission and routing.
//!
//! Callers own signal validity and missing-data policy. Preparing a sample
//! contains only token work and calibrated overhead; dispatch updates add work.

/// Default KV-pressure time penalty, in seconds.
pub const DEFAULT_KV_PRESSURE_WEIGHT: f64 = 0.15;
/// Token estimate for a request without a known token count.
pub const DEFAULT_MEAN_PREFILL_TOKENS: u32 = 1024;
/// Fallback aggregate generation throughput, in tokens per second.
pub const DEFAULT_THROUGHPUT: f64 = 2000.0;
/// Fraction of already-dispatched prompt work that blocks a new request.
pub const DEFAULT_DISPATCH_BLOCKING_FACTOR: f64 = 0.05;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ExpectedWait {
    queued_tokens: f64,
    throughput: f64,
    base_overhead: f64,
    queue_work_correction: f64,
    dispatch_blocking_factor: f64,
}

impl ExpectedWait {
    pub(crate) fn calibrated(
        queued_tokens: f64,
        throughput: f64,
        base_overhead: f64,
        queue_work_correction: f64,
        dispatch_blocking_factor: f64,
    ) -> Self {
        Self {
            queued_tokens,
            throughput,
            base_overhead,
            queue_work_correction,
            dispatch_blocking_factor,
        }
    }

    pub(crate) fn seconds(self, dispatched_tokens: u64) -> f64 {
        self.base_overhead
            + self.queue_work_correction
                * (self.queued_tokens + self.dispatch_blocking_factor * dispatched_tokens as f64)
                / self.throughput
    }
}

#[cfg(test)]
mod tests {
    use super::ExpectedWait;

    #[test]
    fn calibrated_wait_applies_only_base_queue_and_dispatch_terms() {
        let wait = ExpectedWait::calibrated(8_000.0, 5_000.0, 0.05, 0.7, 0.4);

        assert!((wait.seconds(2_000) - 1.282).abs() < 1e-9);
    }
}
