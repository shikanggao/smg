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
    kv_wait: f64,
}

impl ExpectedWait {
    pub(crate) fn new(queued_tokens: f64, throughput: f64, usage: f64, weight: f64) -> Self {
        let k = usage.clamp(0.0, 0.999);
        Self {
            queued_tokens,
            throughput,
            kv_wait: weight * k / (1.0 - k),
        }
    }

    pub(crate) fn seconds(self, dispatched_tokens: u64) -> f64 {
        self.seconds_with_dispatch_factor(dispatched_tokens, 1.0)
    }

    /// Limit the KV-pressure contribution without changing queue accounting.
    pub(crate) fn with_kv_wait_cap(mut self, max_secs: f64) -> Self {
        self.kv_wait = self.kv_wait.min(max_secs);
        self
    }

    /// Discount prompt work already dispatched into a continuously batched engine.
    pub(crate) fn seconds_with_dispatch_factor(
        self,
        dispatched_tokens: u64,
        dispatch_blocking_factor: f64,
    ) -> f64 {
        (self.queued_tokens + dispatched_tokens as f64 * dispatch_blocking_factor) / self.throughput
            + self.kv_wait
    }
}
