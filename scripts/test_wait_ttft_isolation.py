"""Ingress boundary and negative probe regressions; no cluster calls."""
import copy
import errno
import json
import unittest
from unittest.mock import patch

try:
    from scripts.wait_ttft_isolation import (
        DRIVER_APP, PROBE_APP, RUN_LABEL, render_isolation,
        validate_admitted_policy, validate_policy_context, validate_probe_result, validate_router_binding)
except ModuleNotFoundError:
    from wait_ttft_isolation import (
        DRIVER_APP, PROBE_APP, RUN_LABEL, render_isolation,
        validate_admitted_policy, validate_policy_context, validate_probe_result, validate_router_binding)


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.router = dict(metadata=dict(namespace="default", name="owned-router", uid="router-uid",
            labels={"ome.io/inferenceservice": "test-service", "component": "router", "pod-template-hash": "abc123"}),
            status=dict(phase="Running", podIP="10.0.0.8", conditions=[dict(type="Ready", status="True")]))
        self.image = "registry.example/driver@sha256:" + "a"*64
        self.rendered = render_isolation(self.router, [self.router], "owned-run-a", self.image, "cpu-node")

    def test_exact_router_selector_and_only_owned_driver_tcp_ports(self):
        policy, probe = self.rendered["policy"], self.rendered["probe"]
        self.assertEqual(policy["metadata"]["namespace"], "default")
        self.assertEqual(policy["spec"], dict(
            podSelector=dict(matchLabels=self.router["metadata"]["labels"]), policyTypes=["Ingress"],
            ingress=[{"from": [{"podSelector": {"matchLabels": {
                "app.kubernetes.io/name": DRIVER_APP, RUN_LABEL: "owned-run-a"}}}],
                "ports": [{"protocol": "TCP", "port": 8080}, {"protocol": "TCP", "port": 29000}]}]))
        self.assertNotEqual(probe["metadata"]["labels"]["app.kubernetes.io/name"], DRIVER_APP)
        self.assertEqual(probe["metadata"]["labels"]["app.kubernetes.io/name"], PROBE_APP)
        self.assertNotIn("egress", policy["spec"])

    def test_shared_selector_rejects_even_failed_or_terminating_pod(self):
        other = copy.deepcopy(self.router)
        other["metadata"].update(name="old-router", uid="old-uid", deletionTimestamp="2026-01-01T00:00:00Z")
        other["status"]["phase"] = "Failed"
        with self.assertRaisesRegex(ValueError, "exactly one"):
            render_isolation(self.router, [self.router, other], "run-a", self.image, "cpu-node")

    def test_missing_wrong_uid_namespace_unready_and_missing_label_rejected(self):
        for change in ("namespace", "uid", "ready", "label", "terminating", "ip"):
            router = copy.deepcopy(self.router)
            if change == "namespace": router["metadata"]["namespace"] = "other"
            if change == "uid": router["metadata"]["uid"] = ""
            if change == "ready": router["status"]["conditions"] = []
            if change == "label": del router["metadata"]["labels"]["pod-template-hash"]
            if change == "terminating": router["metadata"]["deletionTimestamp"] = "now"
            if change == "ip": router["status"]["podIP"] = "hostname"
            with self.subTest(change=change), self.assertRaises(ValueError):
                render_isolation(router, [router], "run-a", self.image, "cpu-node")
        with self.assertRaises(ValueError):
            render_isolation(self.router, [], "run-a", self.image, "cpu-node")

    def test_router_replacement_or_ip_change_before_apply_fails(self):
        binding = self.rendered["binding"]
        self.assertTrue(validate_router_binding(binding, self.router, {"items": [self.router]})["ok"])
        for field in ("uid", "podIP"):
            changed = copy.deepcopy(self.router)
            (changed["metadata"] if field == "uid" else changed["status"])[field] = (
                "replacement-uid" if field == "uid" else "10.0.0.9")
            with self.assertRaisesRegex(ValueError, "binding changed"):
                validate_router_binding(binding, changed, [changed])

    def test_probe_is_pinned_bounded_and_has_no_service_account_or_privilege(self):
        spec = self.rendered["probe"]["spec"]
        self.assertEqual(spec["nodeName"], "cpu-node")
        self.assertEqual(spec["activeDeadlineSeconds"], 120)
        self.assertFalse(spec["automountServiceAccountToken"])
        self.assertTrue(spec["securityContext"]["runAsNonRoot"])
        container = spec["containers"][0]
        self.assertEqual(container["image"], self.image)
        self.assertEqual(container["resources"]["limits"], dict(cpu="100m", memory="128Mi"))
        self.assertEqual(container["securityContext"], dict(allowPrivilegeEscalation=False,
            readOnlyRootFilesystem=True, capabilities=dict(drop=["ALL"])))
        self.assertNotIn("/chat/completions", container["command"][2])
        self.assertIn("/health", container["command"][2])
        with self.assertRaises(ValueError):
            render_isolation(self.router, [self.router], "run-a", "driver:tag", "cpu-node")

    def test_admission_tamper_or_missing_uid_rejected(self):
        expected = self.rendered["policy"]
        admitted = copy.deepcopy(expected)
        admitted["metadata"]["uid"] = "policy-uid"
        self.assertEqual(validate_admitted_policy(admitted, expected)["uid"], "policy-uid")
        mutations = [lambda p: p["spec"]["ingress"][0]["ports"].append(dict(protocol="UDP", port=8080)),
                     lambda p: p["spec"]["ingress"][0].update({"from": []}),
                     lambda p: p["spec"].update(podSelector={}),
                     lambda p: p["metadata"].update(namespace="other"),
                     lambda p: p["metadata"].pop("uid")]
        for mutate in mutations:
            changed = copy.deepcopy(admitted)
            mutate(changed)
            with self.assertRaises(ValueError): validate_admitted_policy(changed, expected)

    def test_existing_policy_matching_router_rejected_including_expressions(self):
        selectors = [{}, {"matchLabels": {"component": "router"}},
            {"matchExpressions": [{"key": "component", "operator": "In", "values": ["router"]}]},
            {"matchExpressions": [{"key": "component", "operator": "NotIn", "values": ["engine"]}]},
            {"matchExpressions": [{"key": "component", "operator": "Exists"}]},
            {"matchExpressions": [{"key": "other", "operator": "DoesNotExist"}]}]
        for selector in selectors:
            policy = dict(metadata=dict(namespace="default", name="existing", uid="existing-uid"),
                          spec=dict(podSelector=selector))
            with self.subTest(selector=selector), self.assertRaisesRegex(ValueError, "selects target router"):
                validate_policy_context([policy], self.router)

    def test_nonselecting_policy_allowed_but_unsupported_selector_fails_closed(self):
        policy = dict(metadata=dict(namespace="default", name="engine-only", uid="existing-uid"),
                      spec=dict(podSelector=dict(matchLabels=dict(component="engine"))))
        self.assertTrue(validate_policy_context({"items": [policy]}, self.router)["ok"])
        for selector in [{"unknown": {}}, {"matchExpressions": [{"key": "component", "operator": "Other"}]},
                         {"matchExpressions": [{"key": "component", "operator": "Exists", "values": ["x"]}]}]:
            changed = copy.deepcopy(policy)
            changed["spec"]["podSelector"] = selector
            with self.subTest(selector=selector), self.assertRaises(ValueError):
                validate_policy_context([changed], self.router)

    def result(self):
        return dict(schema_version=1,
            target={k: self.rendered["binding"][k] for k in ("router_uid", "host", "port", "path", "url")},
            attempted=True, attempt_count=1, method="GET", timeout_seconds=2, status="DENIED",
            failure_kind="timeout", error_type="TimeoutError", elapsed_seconds=2.001,
            started_at="2026-10-19T00:00:00+00:00", finished_at="2026-10-19T00:00:02.001+00:00")

    def test_only_target_bound_actual_denial_accepted(self):
        result = self.result()
        self.assertTrue(validate_probe_result(json.dumps(result), self.rendered["binding"])["ok"])
        result.update(failure_kind="connection_failure", errno=errno.ECONNREFUSED)
        self.assertTrue(validate_probe_result(result, self.rendered["binding"])["ok"])
        for key, value in [("attempted", False), ("attempt_count", 0), ("status", "REACHABLE"),
                           ("failure_kind", "other"), ("elapsed_seconds", float("nan")),
                           ("timeout_seconds", 30), ("finished_at", None)]:
            changed = self.result()
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_probe_result(changed, self.rendered["binding"])
        changed = self.result()
        changed["target"]["port"] = 29000
        with self.assertRaises(ValueError): validate_probe_result(changed, self.rendered["binding"])

    def test_generated_probe_executes_one_attempt_and_reachable_is_failure(self):
        class Response:
            def sendall(self, request):
                assert request.startswith(b"GET /health HTTP/1.0\r\n")
            def close(self): pass
        for error, status, exit_code in [(None, "REACHABLE", 1), (TimeoutError(), "DENIED", 0),
                (ConnectionRefusedError(errno.ECONNREFUSED, "denied"), "DENIED", 0),
                (RuntimeError("local failure"), "ERROR", 2)]:
            with self.subTest(status=status), patch("socket.create_connection") as opened, patch("builtins.print") as printed:
                opened.return_value = Response()
                opened.side_effect = error
                with self.assertRaises(SystemExit) as caught:
                    exec(self.rendered["probe"]["spec"]["containers"][0]["command"][2], {})
                self.assertEqual(caught.exception.code, exit_code)
                opened.assert_called_once_with((self.rendered["binding"]["host"], 8080), timeout=2)
                receipt = json.loads(printed.call_args.args[0])
                self.assertEqual(receipt["status"], status)
                if status == "DENIED":
                    self.assertTrue(validate_probe_result(receipt, self.rendered["binding"])["ok"])
                else:
                    with self.assertRaises(ValueError): validate_probe_result(receipt, self.rendered["binding"])


if __name__ == "__main__":
    unittest.main()
