"""Secret-reference-only serving auth and non-inference access controls.

This module never accepts, generates, or returns the real runtime credential.
The host coordinator owns in-memory Secret creation and exact-UID cleanup.
Kubernetes expands the placeholder in process argv; do not capture process
argv/environment or serialize the live Rust RouterConfig (Serialize is not
redacted). This boundary does not protect direct engine access.
"""

from __future__ import annotations

import copy
import json
import re
from urllib.parse import urlsplit

KEY_ENV = "SMG_EXPERIMENT_INGRESS_KEY"
SECRET_KEY = "token"
CONTROL_NAMES = ("health", "missing", "wrong", "correct")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def render_ingress_auth(run_id, secret_name):
    """Return only tenant CLI placeholders and identical Secret env references."""
    _require(
        isinstance(run_id, str) and re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?", run_id),
        "Run ID must be a DNS label",
    )
    _require(
        isinstance(secret_name, str)
        and len(secret_name) <= 253
        and all(
            re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?", part)
            for part in secret_name.split(".")
        ),
        "Secret name must be a Kubernetes DNS subdomain",
    )
    env = dict(
        name=KEY_ENV,
        valueFrom=dict(secretKeyRef=dict(name=secret_name, key=SECRET_KEY, optional=False)),
    )
    return dict(
        router_args=["--tenant-api-key", f"{run_id}:$({KEY_ENV})"],
        router_env=[copy.deepcopy(env)],
        driver_env=[copy.deepcopy(env)],
    )


def _validate_env(env, expected):
    _require(isinstance(env, list), "Expected container env list")
    matching = [entry for entry in env if isinstance(entry, dict) and entry.get("name") == KEY_ENV]
    _require(
        matching == expected,
        "Ingress credential env must be one exact Secret reference without a value",
    )


def validate_router_auth(args, env, rendered):
    """Validate admitted placeholder args/env; never echo a mismatched value."""
    _require(
        isinstance(args, list) and all(isinstance(arg, str) for arg in args),
        "Expected router argument list",
    )
    _validate_env(env, rendered["router_env"])
    _require(
        not any(arg == "--api-key" or arg.startswith("--api-key=") for arg in args),
        "Shared API key must not be used for this serving-path boundary",
    )
    positions = [
        i
        for i, arg in enumerate(args)
        if arg == "--tenant-api-key" or arg.startswith("--tenant-api-key=")
    ]
    _require(
        len(positions) == 1 and args[positions[0] : positions[0] + 2] == rendered["router_args"],
        "Router must have exactly the owned tenant-key placeholder argument",
    )
    return dict(ok=True, mechanism="tenant-api-key", credential_storage="Secret reference only")


def validate_driver_auth(env, rendered):
    """Validate that the driver reads the same owned Secret reference."""
    _validate_env(env, rendered["driver_env"])
    return dict(ok=True, credential_storage="Secret reference only")


def render_auth_probe(base_url):
    """Python program for exec in the owned driver; no token enters the command.

    It makes one health GET and three deliberately incomplete JSON requests.
    Any unexpected response, missing env, or local/transport error fails closed.
    Response bodies, exception text, headers, and credentials are never printed.
    """
    _require(isinstance(base_url, str), "Expected router base URL")
    try:
        parsed = urlsplit(base_url)
        valid = (
            parsed.scheme == "http"
            and bool(parsed.hostname)
            and parsed.port == 8080
            and not parsed.username
            and not parsed.password
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
            and not any(c.isspace() for c in base_url)
        )
    except ValueError:
        valid = False
    _require(
        valid,
        "Probe target must be a plain router http URL on port 8080 without credentials or query",
    )
    return """import json, os, secrets, urllib.error, urllib.request
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None

def main():
    key = os.environ.get({!r})
    statuses = {{name: None for name in ("health", "missing", "wrong", "correct")}}
    checks = {{name: False for name in statuses}}
    if not key or len(key) < 32 or any(c.isspace() or ord(c) < 33 or ord(c) > 126 for c in key):
        return dict(schema_version=1, status="FAIL", statuses=statuses, checks=checks)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({{}}), NoRedirect())
    base_url = {!r}
    def request_status(path, token=None, post=False):
        headers = {{"Content-Type": "application/json"}} if post else {{}}
        if token is not None:
            headers["Authorization"] = "Bearer " + token
        try:
            request = urllib.request.Request(base_url + path, data=b"{{}}" if post else None,
                                             headers=headers, method="POST" if post else "GET")
            with opener.open(request, timeout=2) as response:
                return response.status
        except urllib.error.HTTPError as error:
            code = error.code
            error.close()
            return code
        except Exception:
            return None
    wrong = secrets.token_urlsafe(48)
    while wrong == key:
        wrong = secrets.token_urlsafe(48)
    statuses["health"] = request_status("/health")
    statuses["missing"] = request_status("/v1/chat/completions", post=True)
    statuses["wrong"] = request_status("/v1/chat/completions", token=wrong, post=True)
    statuses["correct"] = request_status("/v1/chat/completions", token=key, post=True)
    checks = dict(health=statuses["health"] == 200, missing=statuses["missing"] == 401,
                  wrong=statuses["wrong"] == 401, correct=statuses["correct"] in (400, 422))
    return dict(schema_version=1, status="PASS" if all(checks.values()) else "FAIL",
                statuses=statuses, checks=checks)
try:
    result = main()
except Exception:
    result = dict(schema_version=1, status="FAIL",
                  statuses={{name: None for name in ("health", "missing", "wrong", "correct")}},
                  checks={{name: False for name in ("health", "missing", "wrong", "correct")}})
print(json.dumps(result, sort_keys=True), flush=True)
raise SystemExit(0 if result["status"] == "PASS" else 1)
""".format(KEY_ENV, base_url.rstrip("/"))


def validate_auth_probe_result(result):
    """Accept only exact, reproducible status checks with no extra log fields."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (ValueError, TypeError):
            raise ValueError("Expected one access-control JSON result") from None
    _require(
        isinstance(result, dict)
        and set(result) == {"schema_version", "status", "statuses", "checks"},
        "Probe result must contain only schema, status codes, and checks",
    )
    statuses, checks = result.get("statuses"), result.get("checks")
    _require(
        isinstance(statuses, dict)
        and set(statuses) == set(CONTROL_NAMES)
        and isinstance(checks, dict)
        and set(checks) == set(CONTROL_NAMES),
        "Missing access controls",
    )
    _require(
        all(type(value) is int for value in statuses.values()),
        "Every access control must have an HTTP status",
    )
    expected = dict(
        health=statuses["health"] == 200,
        missing=statuses["missing"] == 401,
        wrong=statuses["wrong"] == 401,
        correct=statuses["correct"] in (400, 422),
    )
    _require(
        result.get("schema_version") == 1
        and result.get("status") == "PASS"
        and all(value is True for value in checks.values())
        and checks == expected
        and all(expected.values()),
        "Access controls failed: require health 200, missing/wrong 401, correct validation 400/422",
    )
    return dict(ok=True, statuses=statuses, checks=checks)
