//! Request lifecycle and native release accounting, independent of admission policy.
pub mod deadline_recovery;
pub mod feature;
pub mod native_deadline;

/// A shared monotonic diagnostic clock; never compared with engine wall clocks.
pub fn diagnostic_time(now: std::time::Instant) -> f64 {
    static ORIGIN: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let origin = *ORIGIN.get_or_init(|| now);
    now.checked_duration_since(origin).map_or_else(
        || -origin.duration_since(now).as_secs_f64(),
        |elapsed| elapsed.as_secs_f64(),
    )
}
