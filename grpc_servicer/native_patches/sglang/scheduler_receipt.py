"""Bounded observations of actual c9e5799 scheduler choices; never predictor input."""
import math
import time

SOURCE = "c9e5799bd12fd5fefb0fb375ac537a6b7d53892e"
METHOD = "sg_scheduler_decision_receipt_v1"
MAX_EXACT = (1 << 53) - 1
MAX_BYTES = 4096
CHOICES = ("prefill", "decode", "idle", "unsupported")
STOP_CODES = (
    "not_evaluated", "full_or_empty_no_chunk", "no_request_slots", "test_retract",
    "lora_limit", "running_full_no_preemption", "preemption_refused",
    "waiting_queue_exhausted", "empty_prefill_selection", "adder_no_token", "adder_other",
)
VALUE_KEYS = (
    "running_bs", "running_full", "allocatable_slots", "rem_total_tokens",
    "cur_rem_tokens", "stored_rem_input_tokens", "stored_rem_chunk_tokens",
)


def _number(value):
    return (type(value) is int and abs(value) <= MAX_EXACT
            or type(value) is float and math.isfinite(value) and abs(value) <= MAX_EXACT)


def _identity_string(value):
    if not isinstance(value, str) or not 0 < len(value) <= 256:
        return None
    try:
        return value if len(value.encode()) <= 256 and not any(ord(c) < 32 or ord(c) == 127 for c in value) else None
    except UnicodeError:
        return None


class SchedulerDecisionReceipt:
    def __init__(self, engine_epoch, profile_id, dp_rank):
        self.identity = dict(engine_epoch=_identity_string(engine_epoch), profile_id=_identity_string(profile_id),
                             dp_rank=0 if type(dp_rank) is int and dp_rank == 0 else None)
        self.identity_valid = all(value is not None for value in self.identity.values())
        self.decision_seq = self.exceptions = self.diagnostic_errors = 0
        self.choice_counts = {key: 0 for key in CHOICES}
        self.stop_counts = {key: 0 for key in STOP_CODES}
        self.pending = self.latest = None
        self.overflow = False

    def begin(self):
        if self.pending is not None:
            self.error("reentrant_scheduler_observation")
        self.pending = dict(prefill_stop_code="not_evaluated", last_adder_result=None,
                            adder_attempt_seq=0, prefetch_skips=0, evaluated_values={},
                            diagnostic_error=None)

    def error(self, reason):
        if self.diagnostic_errors >= MAX_EXACT:
            self.overflow = True
        else:
            self.diagnostic_errors += 1
        for target in (self.pending, self.latest):
            if target is not None:
                target["diagnostic_error"] = reason

    def values(self, site, **values):
        if self.pending is None:
            return
        for key, value in values.items():
            if key not in VALUE_KEYS or not (value is None or _number(value) or key == "running_full" and type(value) is bool):
                self.error("invalid_evaluated_value")
                continue
            self.pending["evaluated_values"][key] = dict(value=value, site=site,
                                                        adder_attempt_seq=self.pending["adder_attempt_seq"])

    def stop(self, code, only_if_unset=False):
        if self.pending is None:
            return
        if code not in STOP_CODES:
            self.error("unknown_stop_code")
        elif not only_if_unset or self.pending["prefill_stop_code"] == "not_evaluated":
            self.pending["prefill_stop_code"] = code

    def adder_begin(self):
        if self.pending is None:
            return
        if self.pending["adder_attempt_seq"] >= MAX_EXACT:
            self.error("attempt_counter_range")
            return
        self.pending["adder_attempt_seq"] += 1
        # Never label a previous add attempt's allocator reading as current.
        for key in ("rem_total_tokens", "cur_rem_tokens"):
            self.pending["evaluated_values"].pop(key, None)

    def adder_result(self, result, code):
        if self.pending is None:
            return
        name = getattr(result, "name", None)
        if name not in ("CONTINUE", "NO_TOKEN", "OTHER"):
            self.error("unknown_adder_result")
            return
        self.pending["last_adder_result"] = name
        if name != "CONTINUE":
            self.stop(code)

    def prefetch_skip(self):
        if self.pending is not None:
            if self.pending["prefetch_skips"] < MAX_EXACT:
                self.pending["prefetch_skips"] += 1
            else:
                self.error("prefetch_counter_range")

    def finish(self, batch):
        pending = self.pending
        if pending is None:
            return
        mode = getattr(getattr(batch, "forward_mode", None), "name", None) if batch is not None else "IDLE"
        choice = {"EXTEND": "prefill", "DECODE": "decode", "IDLE": "idle"}.get(mode, "unsupported")
        observed = time.monotonic()
        if not _number(observed) or observed < 0:
            self.error("invalid_observation_clock")
            observed = None
        overflow = self.overflow or self.decision_seq >= MAX_EXACT
        sequence = self.decision_seq if overflow else self.decision_seq + 1
        counts, stops = dict(self.choice_counts), dict(self.stop_counts)
        if not overflow:
            counts[choice] += 1
            stops[pending["prefill_stop_code"]] += 1
        latest = dict(pending, decision_seq=sequence, selected_mode=choice,
                      selected_forward_mode=mode if mode in ("EXTEND", "DECODE", "IDLE", "MIXED") else "UNSUPPORTED",
                      observed_at_monotonic_secs=observed, batch_sequence=None,
                      submission_observed_at_monotonic_secs=None)
        self.decision_seq, self.choice_counts, self.stop_counts, self.latest, self.pending, self.overflow = (
            sequence, counts, stops, latest, None, overflow)
        if batch is not None and not self.overflow:
            try:
                batch._smg_scheduler_decision_seq = self.decision_seq
            except Exception:
                self.error("batch_link_tag_unavailable")

    def raised(self):
        if self.exceptions >= MAX_EXACT:
            self.overflow = True
        else:
            self.exceptions += 1
        self.latest = dict(decision_seq=self.decision_seq, selected_mode=None,
                           selected_forward_mode=None, prefill_stop_code=None, last_adder_result=None,
                           adder_attempt_seq=0, prefetch_skips=0, evaluated_values={},
                           diagnostic_error="scheduler_choice_raised", observed_at_monotonic_secs=time.monotonic(),
                           batch_sequence=None, submission_observed_at_monotonic_secs=None)
        self.pending = None

    def link_submission(self, batch, work):
        if self.latest is not None and work is not None and getattr(batch, "_smg_scheduler_decision_seq", None) == self.latest["decision_seq"]:
            observed = self.latest.get("observed_at_monotonic_secs")
            if (type(work.sequence) is int and 0 < work.sequence <= MAX_EXACT
                    and _number(work.scheduled_at) and _number(observed)
                    and 0 <= observed <= work.scheduled_at):
                self.latest["batch_sequence"] = work.sequence
                self.latest["submission_observed_at_monotonic_secs"] = work.scheduled_at
            else:
                self.error("invalid_submission_link")

    def snapshot(self, native_snapshot_time=None):
        import json
        if self.latest is not None and native_snapshot_time is not None:
            observed = self.latest.get("observed_at_monotonic_secs")
            submitted = self.latest.get("submission_observed_at_monotonic_secs")
            if (not _number(native_snapshot_time) or not _number(observed)
                    or not 0 <= observed <= native_snapshot_time
                    or submitted is not None and not observed <= submitted <= native_snapshot_time):
                self.error("invalid_snapshot_clock_binding")
        value = dict(schema_version=1, diagnostic_only=True, method=METHOD, source_commit=SOURCE,
                     **self.identity, available=self.identity_valid and self.latest is not None and not self.overflow,
                     unavailable_reason=None, counters_available=self.identity_valid and not self.overflow,
                     decision_seq=self.decision_seq, choice_counts=dict(self.choice_counts),
                     prefill_stop_counts=dict(self.stop_counts), scheduler_exceptions=self.exceptions,
                     diagnostic_error_count=self.diagnostic_errors,
                     counter_scope="observed scheduler returns; interval complete only if error count unchanged",
                     values_scope="last evaluated or assigned at named site, not a binding constraint or snapshot gauge",
                     decision_in_progress=self.pending is not None)
        if self.latest is not None:
            value.update(self.latest)
        if not self.identity_valid or self.overflow:
            value["available"] = value["counters_available"] = False
            value["unavailable_reason"] = "identity_or_counter_range"
        elif self.latest is None:
            value["unavailable_reason"] = "no_completed_scheduler_choice"
        elif self.latest.get("diagnostic_error"):
            value["available"] = False
            value["unavailable_reason"] = self.latest["diagnostic_error"]
        if len(json.dumps(value, separators=(",", ":")).encode()) > MAX_BYTES:
            return dict(schema_version=1, diagnostic_only=True, method=METHOD, source_commit=SOURCE,
                        **self.identity, available=False, counters_available=False,
                        unavailable_reason="receipt_size_limit")
        return value


def _recorder(owner):
    return getattr(owner, "_smg_scheduler_receipt", None)


def observer_failed(owner, abandon=False):
    try:
        recorder = _recorder(owner)
        if recorder is not None:
            recorder.error("observer_failure")
            if abandon:
                recorder.pending = None
    except Exception:
        pass


def note(owner, code=None, only_if_unset=False, **values):
    try:
        recorder = _recorder(owner)
        if recorder is not None:
            if code is not None:
                recorder.stop(code, only_if_unset)
            if values:
                recorder.values(code or "prefill_full_gate", **values)
    except Exception:
        observer_failed(owner)


def bind_adder(scheduler, adder):
    try:
        recorder = _recorder(scheduler)
        if recorder is None:
            return
        adder._smg_scheduler_receipt = recorder
        stored_budgets(adder, "adder_init")
    except Exception:
        observer_failed(scheduler)


def stored_budgets(adder, site="budget_update"):
    try:
        recorder = _recorder(adder)
        if recorder is not None:
            recorder.values(site, stored_rem_input_tokens=adder.rem_input_tokens,
                            stored_rem_chunk_tokens=adder.rem_chunk_tokens)
    except Exception:
        observer_failed(adder)


def adder_begin(adder):
    try:
        recorder = _recorder(adder)
        if recorder is not None:
            recorder.adder_begin()
            stored_budgets(adder, "adder_attempt")
    except Exception:
        observer_failed(adder)


def stop_result(adder, result, code):
    try:
        recorder = _recorder(adder)
        if recorder is not None:
            recorder.adder_result(result, code)
            stored_budgets(adder, "adder_return")
    except Exception:
        observer_failed(adder)
    return result


def budget_value(adder, name, value):
    try:
        recorder = _recorder(adder)
        if recorder is not None:
            recorder.values("allocator_property_return", **{name: value})
    except Exception:
        observer_failed(adder)
    return value


def slots_value(scheduler, value, running_bs):
    try:
        recorder = _recorder(scheduler)
        if recorder is not None:
            recorder.values("get_num_allocatable_reqs_return", allocatable_slots=value, running_bs=running_bs)
    except Exception:
        observer_failed(scheduler)
    return value


def prefetch_skip(scheduler):
    try:
        recorder = _recorder(scheduler)
        if recorder is not None:
            recorder.prefetch_skip()
    except Exception:
        observer_failed(scheduler)


def install_adder(cls):
    """Observe original returns only; never call an allocator/getter again."""
    import functools
    if cls.__dict__.get("_smg_receipt_hooks", False):
        return
    for name in ("rem_total_tokens", "cur_rem_tokens"):
        descriptor = cls.__dict__[name]
        def getter(self, _get=descriptor.fget, _name=name):
            return budget_value(self, _name, _get(self))
        setattr(cls, name, property(getter, descriptor.fset, descriptor.fdel, descriptor.__doc__))
    for name in ("add_one_req", "add_chunked_req"):
        original = cls.__dict__[name]
        def wrapped(self, *args, _original=original, _name=name, **kwargs):
            adder_begin(self)
            result = _original(self, *args, **kwargs)
            stored_budgets(self, "adder_return")
            if _name == "add_one_req":
                stop_result(self, result, "adder_no_token" if getattr(result, "name", None) == "NO_TOKEN" else "adder_other")
            return result
        setattr(cls, name, functools.wraps(original)(wrapped))
    cls._smg_receipt_hooks = True
