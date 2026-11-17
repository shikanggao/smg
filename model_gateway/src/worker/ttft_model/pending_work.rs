//! Causal pending-size shadow ablation. Full prompt size is NOT uncached work.
//! The local pre-first-text population is not a native waiting partition. We
//! replace only a coverage-weighted fraction of the historical per-request proxy;
//! unseen/unknown work retains the original proxy and exact native tokens win.
//! No prompt contents, tokenizer calls, fitted state, or admission policy live here.
use std::{collections::HashMap, time::Instant};

use openai_protocol::common::PromptWorkHint;
use serde::Serialize;

use super::decision;

pub const METHOD: &str = "pending_request_work_capacity_source_gate_two_arm_v4";
const MAX_PENDING: usize = 8192;
const MAX_TEXT_BYTES: usize = 16 * 1024 * 1024;
const MAX_BODY_TOKENS: usize = 4 * 1024 * 1024;
const MAX_AGE_SECONDS: f64 = 600.0;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Estimate {
    pub tokens: Option<f64>,
    pub source: &'static str,
    pub text_bytes: Option<usize>,
    pub uncached_tokens_known: bool,
    pub uncertainty: &'static str,
}
impl Default for Estimate {
    fn default() -> Self {
        Self::from_hint(PromptWorkHint::Unknown)
    }
}
impl Estimate {
    pub fn from_hint(hint: PromptWorkHint) -> Self {
        let (tokens, source, text_bytes, uncertainty) = match hint {
            PromptWorkHint::TextBytes(n) if n > 0 && n <= MAX_TEXT_BYTES => {
                (Some(n as f64 / 4.0), "utf8_bytes_div4_proxy", Some(n),
                    "utf8_bytes_div4_heuristic; tokenizer_chat_template_cache_and_remaining_work_unknown")
            }
            PromptWorkHint::BodyTokenIds(n) if n > 0 && n <= MAX_BODY_TOKENS => {
                (Some(n as f64), "body_token_ids_full_prompt", None,
                    "full_prompt_length_exact; cache_residency_and_remaining_work_unknown")
            }
            _ => (None, "unknown_or_unsupported", None,
                "unsupported_input_or_size_bound; historical_proxy_fallback"),
        };
        Self {
            tokens,
            source,
            text_bytes,
            uncached_tokens_known: false,
            uncertainty,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Handle {
    pub generation: u64,
    pub id: u64,
    pub tracked: bool,
}
#[derive(Debug, Default)]
pub struct Inventory {
    generation: u64,
    rows: HashMap<u64, (Instant, Estimate)>,
    overflow: usize,
}
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Summary {
    pub generation: u64,
    pub captured_at: f64,
    pub pending: usize,
    pub known: usize,
    pub unknown: usize,
    pub expired: usize,
    pub overflow: usize,
    pub token_sum: f64,
    pub text_proxy_count: usize,
    pub body_token_count: usize,
}
impl Inventory {
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn register(&mut self, id: u64, at: Instant, estimate: Estimate) -> Handle {
        let tracked = self.rows.len() < MAX_PENDING;
        if tracked {
            self.rows.insert(id, (at, estimate));
        } else {
            self.overflow = self.overflow.saturating_add(1);
        }
        Handle {
            generation: self.generation,
            id,
            tracked,
        }
    }
    pub fn remove(&mut self, handle: Handle) {
        if self.generation != handle.generation {
            return;
        }
        if handle.tracked {
            self.rows.remove(&handle.id);
        } else {
            // The observer owns this handle and consumes it once with Option::take.
            self.overflow = self.overflow.saturating_sub(1);
        }
    }
    pub fn reset(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.rows.clear();
        self.overflow = 0;
    }
    pub fn summary(&self, at: Instant) -> Summary {
        let mut summary = Summary {
            generation: self.generation,
            captured_at: super::diagnostic_time(at),
            pending: self.rows.len(),
            overflow: self.overflow,
            ..Summary::default()
        };
        for (started, estimate) in self.rows.values() {
            if at.saturating_duration_since(*started).as_secs_f64() > MAX_AGE_SECONDS {
                summary.expired += 1;
            } else if let Some(tokens) = estimate.tokens {
                summary.known += 1;
                summary.token_sum += tokens;
                summary.text_proxy_count += usize::from(estimate.source == "utf8_bytes_div4_proxy");
                summary.body_token_count +=
                    usize::from(estimate.source == "body_token_ids_full_prompt");
            } else {
                summary.unknown += 1;
            }
        }
        summary
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Queue {
    pub summary: Summary,
    pub original_tokens: f64,
    pub candidate_tokens: f64,
    pub native_waiting: f64,
    pub native_running: f64,
    pub coverage: f64,
    pub pending_mean: Option<f64>,
    pub historical_mean: Option<f64>,
    // A computable proxy, before selecting a numerator from capacity provenance.
    pub applied: bool,
    pub reason: &'static str,
}
impl Queue {
    pub fn new(
        summary: Summary,
        original: f64,
        waiting: f64,
        running: f64,
        historical: Option<f64>,
        eligible: bool,
        native_tokens: bool,
    ) -> Self {
        let mut q = Self {
            summary,
            original_tokens: original,
            candidate_tokens: original,
            native_waiting: waiting,
            native_running: running,
            coverage: 0.0,
            pending_mean: None,
            historical_mean: historical,
            applied: false,
            reason: "unsupported_native_counts_or_rank",
        };
        if native_tokens {
            q.reason = "authoritative_native_tokens";
        } else if !eligible
            || !waiting.is_finite()
            || !running.is_finite()
            || waiting < 0.0
            || running < 0.0
        {
            // Keep original arithmetic, including unsupported native paths.
        } else if summary.overflow > 0 {
            q.reason = "inventory_overflow";
        } else if waiting == 0.0 {
            q.reason = "native_queue_empty";
        } else if let Some(historical) = historical.filter(|v| v.is_finite() && *v > 0.0) {
            if summary.known > 0 && summary.token_sum.is_finite() {
                let mean = summary.token_sum / summary.known as f64;
                // Denominator includes native decoders; local post-first-text work
                // is absent. This is conservative coverage, NOT queue membership.
                let coverage = (summary.known as f64 / (waiting + running)).min(1.0);
                let tokens = waiting * (coverage * mean + (1.0 - coverage) * historical);
                if tokens.is_finite() {
                    q.candidate_tokens = tokens;
                    q.coverage = coverage;
                    q.pending_mean = Some(mean);
                    q.applied = true;
                    q.reason = "pending_full_prompt_proxy_mix";
                }
            } else {
                q.reason = "no_known_pending_sizes";
            }
        } else {
            q.reason = "historical_proxy_unavailable";
        }
        q
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Admission {
    pub method: &'static str,
    pub current_inventory_generation: u64,
    pub queue: Queue,
    pub capacity_progress_source: Option<String>,
    pub selected_pending_applied: bool,
    pub source_gate_reason: &'static str,
    pub poll_started: f64,
    pub at: f64,
    pub debt_seconds: f64,
    pub capacity: f64,
    pub recovery_fraction: f64,
    pub recovery_cap_seconds: Option<f64>,
    pub original_x: f64,
    pub control_x: f64,
    pub candidate_x: f64,
    pub original_raw_total: f64,
    pub control_raw_total: f64,
    pub candidate_raw_total: f64,
    pub original_predictions: [decision::Prediction; 2],
    pub control_predictions: [decision::Prediction; 2],
    pub predictions: [decision::Prediction; 2],
    pub available: bool,
}

/// Swap queue work while preserving the absolute native drain horizon and debt.
/// Fallback paths reproduce original X exactly (including native authoritative zero).
pub fn candidate_x(
    q: Queue,
    capacity: f64,
    debt: f64,
    original_x: f64,
    recovery_cap: Option<f64>,
) -> (f64, f64) {
    let original_queue = q.original_tokens / capacity;
    let fraction = if original_queue > 0.0 {
        ((original_x - debt) / original_queue).clamp(0.0, 1.0)
    } else {
        1.0
    };
    if !q.applied {
        return (original_x, fraction);
    }
    // Retain the absolute v2 arithmetic as the same-state control. Multiplying
    // a larger replacement queue by the fraction would inflate a valid drain cap.
    let queue = q.candidate_tokens / capacity;
    let x = recovery_cap
        .filter(|cap| cap.is_finite() && *cap >= 0.0)
        .map_or(queue, |cap| queue.min(cap))
        + debt;
    if x.is_finite() && x >= 0.0 {
        (x, fraction)
    } else {
        (original_x, fraction)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn pending_work_size_shift_partial_coverage_and_no_double_count() {
        let mut inventory = Inventory::default();
        let now = Instant::now();
        inventory.register(1, now, Estimate::from_hint(PromptWorkHint::TextBytes(4000)));
        inventory.register(
            2,
            now,
            Estimate::from_hint(PromptWorkHint::TextBytes(12000)),
        );
        inventory.register(3, now, Estimate::default());
        let q = Queue::new(
            inventory.summary(now),
            400.0,
            4.0,
            1.0,
            Some(100.0),
            true,
            false,
        );
        assert_eq!(q.coverage, 0.4);
        assert_eq!(q.pending_mean, Some(2000.0));
        assert_eq!(q.candidate_tokens, 3440.0); // 4*(.4*2000+.6*100), not local+native.
        let (x, r) = candidate_x(q, 100.0, 1.0, 3.0, Some(2.0));
        assert_eq!(r, 0.5);
        assert_eq!(x, 3.0);
    }
    #[test]
    fn pending_work_native_zero_unknown_and_unsupported_keep_original() {
        let summary = Summary {
            known: 1,
            token_sum: 100000.0,
            ..Summary::default()
        };
        for q in [
            Queue::new(summary, 0.0, 2.0, 1.0, None, true, true),
            Queue::new(summary, 200.0, 2.0, 1.0, None, true, true),
            Queue::new(summary, 123.0, 2.0, 1.0, Some(61.5), false, false),
            Queue::new(Summary::default(), 123.0, 2.0, 1.0, Some(61.5), true, false),
        ] {
            assert!(!q.applied);
            assert_eq!(q.candidate_tokens, q.original_tokens);
            assert_eq!(candidate_x(q, 100.0, 0.1, 1.33, Some(0.0)).0, 1.33);
        }
    }
    #[test]
    fn pending_work_absolute_cap_preserves_debt_and_can_bind_only_the_new_queue() {
        let mut q = Queue::new(
            Summary {
                known: 1,
                token_sum: 600.0,
                ..Summary::default()
            },
            400.0,
            1.0,
            0.0,
            Some(400.0),
            true,
            false,
        );
        // A 200s drain horizon is not rescaled to 300s by the larger queue.
        assert_eq!(candidate_x(q, 1.0, 5.0, 205.0, Some(200.0)).0, 205.0);
        // The old queue did not hit this cap, but the new queue does.
        assert_eq!(candidate_x(q, 1.0, 5.0, 405.0, Some(500.0)).0, 505.0);
        q.candidate_tokens = 100.0;
        assert_eq!(candidate_x(q, 1.0, 5.0, 205.0, Some(200.0)).0, 105.0);
        assert_eq!(candidate_x(q, 1.0, 5.0, 205.0, Some(0.0)).0, 5.0);
        // No eligible drain evidence grants no discount, even with an old fraction.
        for cap in [None, Some(f64::NAN), Some(f64::INFINITY), Some(-1.0)] {
            assert_eq!(candidate_x(q, 1.0, 5.0, 205.0, cap).0, 105.0);
        }
    }
    #[test]
    fn pending_work_inventory_first_text_cancel_reset_expiry_and_overflow() {
        let now = Instant::now();
        let mut i = Inventory::default();
        let h = i.register(
            1,
            now,
            Estimate::from_hint(PromptWorkHint::BodyTokenIds(40)),
        );
        let frozen = i.summary(now);
        i.remove(h);
        i.remove(h);
        assert_eq!(frozen.known, 1);
        assert_eq!(i.summary(now).pending, 0);
        let old = i.register(2, now, Estimate::default());
        i.reset();
        for id in 0..MAX_PENDING {
            i.register(id as u64, now, Estimate::default());
        }
        let overflow = i.register(99999, now, Estimate::default());
        assert_eq!(i.summary(now).pending, MAX_PENDING);
        assert_eq!(i.summary(now).overflow, 1);
        i.remove(old);
        assert_eq!(i.summary(now).pending, MAX_PENDING);
        assert_eq!(
            i.summary(now + Duration::from_secs(601)).expired,
            MAX_PENDING
        );
        assert!(!Queue::new(i.summary(now), 100.0, 1.0, 0.0, Some(100.0), true, false).applied);
        i.remove(overflow);
        assert_eq!(i.summary(now).overflow, 0);
    }
    #[test]
    fn pending_work_byte_proxy_is_not_tokenization_and_has_bounds() {
        let t = "你好😀";
        let estimate = Estimate::from_hint(PromptWorkHint::TextBytes(t.len()));
        assert_eq!(estimate.tokens, Some(2.5));
        assert!(estimate.uncertainty.contains("utf8_bytes_div4_heuristic"));
        assert!(!estimate.uncached_tokens_known);
        for hint in [
            PromptWorkHint::TextBytes(0),
            PromptWorkHint::TextBytes(usize::MAX),
            PromptWorkHint::BodyTokenIds(usize::MAX),
            PromptWorkHint::Unknown,
        ] {
            assert!(Estimate::from_hint(hint).tokens.is_none());
        }
    }
    #[test]
    fn pending_work_protocol_provenance_rejects_partial_and_multiplied_inputs() {
        use openai_protocol::{
            chat::ChatCompletionRequest, common::GenerationRequest, completion::CompletionRequest,
            generate::GenerateRequest,
        };
        use serde_json::json;
        let chat = |body| {
            serde_json::from_value::<ChatCompletionRequest>(body)
                .unwrap()
                .prompt_work_hint()
        };
        assert_eq!(
            Estimate::from_hint(chat(
                json!({"model":"test","messages":[{"role":"user","content":"你好😀"}]})
            ))
            .tokens,
            Some(2.5)
        );
        for body in [
            json!({"model":"test","n":2,"messages":[{"role":"user","content":"hello"}]}),
            json!({"model":"test","messages":[{"role":"user","content":[{"type":"text","text":"hello"},{"type":"image_url","image_url":{"url":"https://example.invalid/image"}}]}]}),
            json!({"model":"test","tools":[],"messages":[{"role":"user","content":"hello"}]}),
            json!({"model":"test","messages":[{"role":"assistant","content":"hello","tool_calls":[]}]}),
        ] {
            assert!(Estimate::from_hint(chat(body)).tokens.is_none());
        }
        for body in [
            json!({"model":"test","prompt":["one","two"]}),
            json!({"model":"test","prompt":"one","n":2}),
            json!({"model":"test","prompt":"one","best_of":2}),
            json!({"model":"test","prompt":"one","suffix":"two"}),
        ] {
            let request: CompletionRequest = serde_json::from_value(body).unwrap();
            assert!(Estimate::from_hint(request.prompt_work_hint())
                .tokens
                .is_none());
        }
        let body: GenerateRequest = serde_json::from_value(json!({"input_ids":[1,2,3]})).unwrap();
        let estimate = Estimate::from_hint(body.prompt_work_hint());
        assert_eq!(estimate.tokens, Some(3.0));
        assert!(estimate.uncertainty.contains("full_prompt_length_exact"));
        assert!(!estimate.uncertainty.contains("tokenization"));
        assert!(!estimate.uncached_tokens_known);
        for body in [
            json!({"input_ids":[[1,2],[3,4]]}),
            json!({"input_ids":[1,-2]}),
            json!({"input_ids":[1,2],"sampling_params":{"n":2}}),
            json!({"input_ids":[1,2],"image_data":"image"}),
        ] {
            let request: GenerateRequest = serde_json::from_value(body).unwrap();
            assert!(Estimate::from_hint(request.prompt_work_hint())
                .tokens
                .is_none());
        }
        // Routing hints are deliberately absent from the body-only trait API.
    }
}
