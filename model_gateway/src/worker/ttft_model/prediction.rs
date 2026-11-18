/// The single admission-time TTFT prediction. Invalid evidence must never veto a request.
#[derive(Clone, Copy, Debug)]
pub struct SelectedPrediction {
    pub total: f64,
    pub evidence_valid: bool,
}
impl SelectedPrediction {
    pub fn with_router_elapsed(mut self, elapsed: f64) -> Self {
        self.evidence_valid &= elapsed.is_finite() && elapsed >= 0.0;
        self.total += elapsed;
        self
    }

    pub fn exceeds(self, threshold: Option<f64>) -> bool {
        self.evidence_valid
            && self.total.is_finite()
            && self.total >= 0.0
            && threshold.is_some_and(|t| t.is_finite() && t > 0.0 && self.total >= t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inclusive_deadline_requires_valid_finite_evidence() {
        let good = SelectedPrediction {
            total: 300.0,
            evidence_valid: true,
        };
        assert!(good.exceeds(Some(300.0)));
        assert!(good.with_router_elapsed(2.0).exceeds(Some(302.0)));
        assert!(!good.with_router_elapsed(-1.0).exceeds(Some(1.0)));
        assert!(!good.exceeds(Some(301.0)));
        for total in [f64::NAN, f64::INFINITY, -1.0] {
            assert!(!SelectedPrediction { total, ..good }.exceeds(Some(300.0)));
        }
        assert!(!SelectedPrediction {
            evidence_valid: false,
            ..good
        }
        .exceeds(Some(300.0)));
        for threshold in [None, Some(0.0), Some(f64::NAN), Some(f64::INFINITY)] {
            assert!(!good.exceeds(threshold));
        }
    }
}
