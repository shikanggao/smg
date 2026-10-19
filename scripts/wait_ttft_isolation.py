"""Pure, fail-closed manifests for one owned DEV router ingress boundary.

The caller must supply an unfiltered namespace Pod inventory (including Failed
Pods), revalidate the binding immediately before applying the policy, save the
admitted objects/UIDs, and collect probe logs through the host coordinator.
"""
from __future__ import annotations

import datetime
import hashlib
import ipaddress
import json
import math
import re


ROUTER_LABELS = ("ome.io/inferenceservice", "component", "pod-template-hash")
DRIVER_APP = "smg-production-driver"
PROBE_APP = "smg-ttft-negative-probe"
RUN_LABEL = "codex.oracle.com/run-id"
LIMITATIONS = [
    "Ingress NetworkPolicies are additive; another policy can grant wider access.",
    "The negative probe checks one ordinary Pod source; node/kubelet traffic may bypass this boundary.",
    "The policy does not cover engine/direct access or traffic bypassing this router.",
    "Pod labels are not an authentication boundary against an actor allowed to create matching Pods.",
]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _pod_binding(router_pod, all_pods):
    meta, status = router_pod.get("metadata", {}), router_pod.get("status", {})
    labels = meta.get("labels", {})
    _require(meta.get("namespace") == "default", "Router must be in namespace default")
    _require(bool(meta.get("name")) and bool(meta.get("uid")), "Router name and current UID are required")
    _require(not meta.get("deletionTimestamp"), "Router is terminating")
    _require(status.get("phase") == "Running" and any(
        c.get("type") == "Ready" and c.get("status") == "True"
        for c in status.get("conditions", [])), "Router must be Running and Ready")
    _require(all(isinstance(labels.get(key), str) and labels[key] for key in ROUTER_LABELS)
             and labels["component"] == "router", "Required existing router selector labels are missing")
    selector = {key: labels[key] for key in ROUTER_LABELS}
    pods = all_pods.get("items", []) if isinstance(all_pods, dict) else all_pods
    _require(isinstance(pods, list), "Expected unfiltered namespace Pod inventory")
    matches = [pod for pod in pods if pod.get("metadata", {}).get("namespace") == "default"
               and all(pod.get("metadata", {}).get("labels", {}).get(k) == v for k, v in selector.items())]
    _require(len(matches) == 1 and matches[0].get("metadata", {}).get("uid") == meta["uid"],
             "Router selector must match exactly one Pod and its current UID, including Failed Pods")
    _require(matches[0].get("metadata", {}).get("name") == meta["name"], "Router inventory name mismatch")
    try:
        ip = str(ipaddress.ip_address(status["podIP"]))
    except (KeyError, ValueError, TypeError):
        raise ValueError("Router must have a valid current Pod IP") from None
    _require(matches[0].get("status", {}).get("podIP") == status["podIP"], "Router inventory IP mismatch")
    host = f"[{ip}]" if ":" in ip else ip
    return dict(namespace="default", router_name=meta["name"], router_uid=meta["uid"],
                router_ip=ip, selector=selector, host=ip, port=8080,
                path="/health", url=f"http://{host}:8080/health")


def validate_router_binding(binding, router_pod, all_pods):
    """Recheck exact UID/IP/selector against the latest full namespace inventory."""
    current = _pod_binding(router_pod, all_pods)
    _require(current == binding, "Router binding changed before isolation policy application")
    return dict(ok=True, router_uid=current["router_uid"], router_ip=current["router_ip"])


def validate_policy_context(existing_policies, router_pod):
    """Reject an existing selecting policy: ingress permissions are a union."""
    _require(router_pod.get("metadata", {}).get("namespace") == "default", "Router must be in namespace default")
    labels = router_pod.get("metadata", {}).get("labels", {})
    policies = existing_policies.get("items") if isinstance(existing_policies, dict) else existing_policies
    _require(isinstance(policies, list), "Expected full namespace NetworkPolicy inventory")
    checked = []
    for policy in policies:
        meta = policy.get("metadata", {})
        _require(isinstance(meta.get("namespace"), str) and bool(meta["namespace"]),
                 "Existing policy namespace is required")
        if meta.get("namespace") != "default":
            continue
        selector = policy.get("spec", {}).get("podSelector")
        _require(isinstance(selector, dict) and not (selector.keys() - {"matchLabels", "matchExpressions"}),
                 "Existing policy has an unsupported Pod selector")
        required, expressions = selector.get("matchLabels", {}), selector.get("matchExpressions", [])
        _require(isinstance(required, dict) and all(isinstance(k, str) and k and isinstance(v, str)
                 for k, v in required.items()) and isinstance(expressions, list), "Invalid existing Pod selector")
        matches = all(labels.get(k) == v for k, v in required.items())
        for expression in expressions:
            _require(isinstance(expression, dict) and not (expression.keys() - {"key", "operator", "values"}),
                     "Unsupported existing selector expression")
            key, op, values = expression.get("key"), expression.get("operator"), expression.get("values", [])
            _require(isinstance(key, str) and key and op in {"In", "NotIn", "Exists", "DoesNotExist"}
                     and isinstance(values, list) and all(isinstance(v, str) for v in values),
                     "Unsupported existing selector expression")
            _require(bool(values) if op in {"In", "NotIn"} else not values, "Invalid selector expression values")
            match = {"In": key in labels and labels[key] in values,
                     "NotIn": key not in labels or labels[key] not in values,
                     "Exists": key in labels, "DoesNotExist": key not in labels}[op]
            matches = matches and match
        _require(not matches, f"Existing NetworkPolicy selects target router: {meta.get('name', '<unnamed>')}")
        checked.append(dict(name=meta.get("name"), uid=meta.get("uid")))
    return dict(ok=True, checked_nonselecting_policies=checked)


def _probe_code(binding):
    target = {key: binding[key] for key in ("router_uid", "host", "port", "path", "url")}
    return '''import datetime, errno, json, socket, time
target = json.loads(%r)
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
begin = time.monotonic()
result = dict(schema_version=1, target=target, attempted=True, attempt_count=1,
              method="GET", timeout_seconds=2, started_at=started)
exit_code = 2
try:
    connection = socket.create_connection((target["host"], target["port"]), timeout=2)
except Exception as error:
    code = getattr(error, "errno", None)
    if isinstance(error, (TimeoutError, socket.timeout)):
        result.update(status="DENIED", failure_kind="timeout", error_type=type(error).__name__)
        exit_code = 0
    elif isinstance(error, OSError) and code in {
            errno.ECONNREFUSED, errno.ECONNRESET, errno.ECONNABORTED,
            errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ETIMEDOUT}:
        result.update(status="DENIED", failure_kind="connection_failure",
                      error_type=type(error).__name__, errno=code)
        exit_code = 0
    else:
        result.update(status="ERROR", error_type=type(error).__name__)
else:
    # An established connection is a failed negative probe even if HTTP stalls.
    result.update(status="REACHABLE")
    exit_code = 1
    try:
        request = "GET " + target["path"] + " HTTP/1.0\\r\\nHost: " + target["host"] + "\\r\\nConnection: close\\r\\n\\r\\n"
        connection.sendall(request.encode("ascii"))
    except OSError:
        pass
    finally:
        connection.close()
result.update(finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              elapsed_seconds=time.monotonic()-begin)
print(json.dumps(result, sort_keys=True), flush=True)
raise SystemExit(exit_code)
''' % json.dumps(target, sort_keys=True)


def render_isolation(router_pod, all_pods, run_id, driver_image, cpu_node):
    """Render only an owned ingress policy and bounded, non-inference probe Pod."""
    _require(isinstance(run_id, str) and re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?", run_id),
             "Run ID must be a DNS label of at most 63 characters")
    _require(isinstance(driver_image, str) and re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", driver_image),
             "Probe must use the digest-pinned CPU driver image")
    _require(isinstance(cpu_node, str) and bool(cpu_node.strip()), "Pinned CPU node is required")
    binding = _pod_binding(router_pod, all_pods)
    suffix = f"{run_id[:30]}-{hashlib.sha256(run_id.encode()).hexdigest()[:10]}"
    owner = {"app.kubernetes.io/name": "smg-ttft-isolation", RUN_LABEL: run_id}
    policy = dict(apiVersion="networking.k8s.io/v1", kind="NetworkPolicy",
        metadata=dict(name=f"ttft-isolation-{suffix}", namespace="default", labels=owner),
        spec=dict(podSelector=dict(matchLabels=binding["selector"]), policyTypes=["Ingress"],
            ingress=[{"from": [{"podSelector": {"matchLabels": {
                "app.kubernetes.io/name": DRIVER_APP, RUN_LABEL: run_id}}}],
                "ports": [{"protocol": "TCP", "port": port} for port in (8080, 29000)]}]))
    probe = dict(apiVersion="v1", kind="Pod", metadata=dict(
        name=f"ttft-deny-probe-{suffix}", namespace="default",
        labels={"app.kubernetes.io/name": PROBE_APP, RUN_LABEL: run_id}), spec=dict(
        nodeName=cpu_node, restartPolicy="Never", activeDeadlineSeconds=120,
        terminationGracePeriodSeconds=1, automountServiceAccountToken=False,
        securityContext=dict(runAsNonRoot=True, runAsUser=65532, runAsGroup=65532,
                             seccompProfile=dict(type="RuntimeDefault")),
        containers=[dict(name="probe", image=driver_image, imagePullPolicy="IfNotPresent",
            command=["python3", "-c", _probe_code(binding)],
            resources=dict(requests=dict(cpu="50m", memory="64Mi"), limits=dict(cpu="100m", memory="128Mi")),
            securityContext=dict(allowPrivilegeEscalation=False, readOnlyRootFilesystem=True,
                                 capabilities=dict(drop=["ALL"])))]))
    return dict(policy=policy, probe=probe, binding=binding, limitations=list(LIMITATIONS))


def validate_admitted_policy(admitted, expected):
    """Reject any admitted spec change or ownership/namespace/name mismatch."""
    _require(admitted.get("apiVersion") == "networking.k8s.io/v1" and admitted.get("kind") == "NetworkPolicy",
             "Expected admitted networking.k8s.io/v1 NetworkPolicy")
    meta, want = admitted.get("metadata", {}), expected.get("metadata", {})
    _require(all(meta.get(key) == want.get(key) for key in ("name", "namespace", "labels")),
             "Admitted policy identity or owned labels changed")
    _require(isinstance(meta.get("uid"), str) and bool(meta["uid"]), "Admitted policy UID is required for cleanup")
    _require(admitted.get("spec") == expected.get("spec"), "Admitted policy spec differs from rendered policy")
    return dict(ok=True, namespace=meta["namespace"], name=meta["name"], uid=meta["uid"])


def validate_probe_result(result, binding):
    """Accept only a completed denial from one health attempt at the bound router."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (ValueError, TypeError):
            raise ValueError("Expected one probe JSON object from owned Pod logs") from None
    _require(isinstance(result, dict), "Expected probe JSON object")
    target = {key: binding[key] for key in ("router_uid", "host", "port", "path", "url")}
    _require(result.get("schema_version") == 1 and result.get("target") == target,
             "Probe result does not match the exact bound router health target")
    _require(result.get("attempted") is True and result.get("attempt_count") == 1
             and result.get("method") == "GET" and result.get("timeout_seconds") == 2,
             "Probe must record exactly one actual health attempt with a two-second timeout")
    _require(result.get("status") == "DENIED" and result.get("failure_kind") in {"timeout", "connection_failure"},
             "Negative probe reached router or did not prove a connection denial")
    _require("http_status" not in result, "Probe received an HTTP response and reached the router")
    if result["failure_kind"] == "timeout":
        _require(result.get("error_type") in {"TimeoutError", "timeout"}, "Probe did not record a timeout")
    if result["failure_kind"] == "connection_failure":
        import errno
        _require(result.get("errno") in {errno.ECONNREFUSED, errno.ECONNRESET, errno.ECONNABORTED,
                 errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ETIMEDOUT}, "Probe did not record a connection failure")
    elapsed = result.get("elapsed_seconds")
    _require(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed <= 120,
             "Probe elapsed time is absent or outside its bounded deadline")
    try:
        start = datetime.datetime.fromisoformat(result["started_at"])
        finish = datetime.datetime.fromisoformat(result["finished_at"])
        _require(start.utcoffset() == datetime.timedelta(0) and finish.utcoffset() == datetime.timedelta(0)
                 and 0 <= (finish-start).total_seconds() <= 120, "Probe timestamps are not a bounded UTC attempt")
    except (KeyError, TypeError, ValueError):
        raise ValueError("Probe timestamps must record a completed, bounded UTC attempt") from None
    return dict(ok=True, status="DENIED", target=target, elapsed_seconds=elapsed,
                failure_kind=result["failure_kind"])
