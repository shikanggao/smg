"""Secret-reference and generated non-inference control regressions."""

import copy
import io
import json
import unittest
import urllib.error
from contextlib import redirect_stdout
from unittest.mock import patch

try:
    from scripts.wait_ttft_ingress_auth import (
        KEY_ENV,
        render_auth_probe,
        render_ingress_auth,
        validate_auth_probe_result,
        validate_driver_auth,
        validate_router_auth,
    )
except ModuleNotFoundError:
    from wait_ttft_ingress_auth import (
        KEY_ENV,
        render_auth_probe,
        render_ingress_auth,
        validate_auth_probe_result,
        validate_driver_auth,
        validate_router_auth,
    )


class IngressAuthTests(unittest.TestCase):
    FIXTURE_KEY = "fixture_only_no_real_credential_" + "x" * 40

    def setUp(self):
        self.rendered = render_ingress_auth("owned-run", "owned-ingress-secret")
        self.args = ["--host", "0.0.0.0"] + self.rendered["router_args"]
        self.env = [dict(name="UNRELATED", value="preserved")] + self.rendered["router_env"]
        self.program = render_auth_probe("http://10.0.0.8:8080")

    def test_render_contains_only_owned_secret_refs_and_placeholder(self):
        self.assertEqual(
            self.rendered["router_args"],
            ["--tenant-api-key", "owned-run:$(SMG_EXPERIMENT_INGRESS_KEY)"],
        )
        ref = dict(
            name=KEY_ENV,
            valueFrom=dict(
                secretKeyRef=dict(name="owned-ingress-secret", key="token", optional=False)
            ),
        )
        self.assertEqual(self.rendered["router_env"], [ref])
        self.assertEqual(self.rendered["driver_env"], [ref])
        self.assertNotIn(self.FIXTURE_KEY, json.dumps(self.rendered) + self.program)
        self.assertTrue(validate_router_auth(self.args, self.env, self.rendered)["ok"])
        self.assertTrue(validate_driver_auth(self.env, self.rendered)["ok"])

    def test_literal_key_wrong_secret_duplicate_env_and_extra_credentials_fail_without_echo(self):
        variations = []
        literal = copy.deepcopy(self.env)
        literal[-1] = dict(name=KEY_ENV, value=self.FIXTURE_KEY)
        variations.append((self.args, literal))
        wrong = copy.deepcopy(self.env)
        wrong[-1]["valueFrom"]["secretKeyRef"]["name"] = "different-secret"
        variations.extend(
            [
                (self.args, wrong),
                (self.args, self.env + self.env[-1:]),
                (self.args + ["--api-key", self.FIXTURE_KEY], self.env),
                (self.args + ["--tenant-api-key", "other:" + self.FIXTURE_KEY], self.env),
                (["--tenant-api-key=owned-run:" + self.FIXTURE_KEY], self.env),
            ]
        )
        for args, env in variations:
            with self.assertRaises(ValueError) as caught:
                validate_router_auth(args, env, self.rendered)
            self.assertNotIn(self.FIXTURE_KEY, str(caught.exception))

    def run_program(self, codes, key=True, error=None):
        requests, out = [], io.StringIO()

        class Response:
            def __init__(self, status):
                self.status = status

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def opened(request, timeout):
            requests.append(request)
            self.assertEqual(timeout, 2)
            if error is not None:
                raise error
            code = codes[len(requests) - 1]
            if code >= 400:
                raise urllib.error.HTTPError(request.full_url, code, "ignored", None, None)
            return Response(code)

        env = {KEY_ENV: self.FIXTURE_KEY} if key else {}
        with (
            patch.dict("os.environ", env, clear=True),
            patch("urllib.request.OpenerDirector.open", side_effect=opened),
            redirect_stdout(out),
        ):
            with self.assertRaises(SystemExit) as caught:
                exec(self.program, {})
        self.assertNotIn(self.FIXTURE_KEY, out.getvalue())
        return json.loads(out.getvalue()), requests, caught.exception.code

    def test_generated_program_requires_all_four_non_inference_controls(self):
        for valid_status in (400, 422):
            result, requests, code = self.run_program([200, 401, 401, valid_status])
            self.assertEqual(code, 0)
            self.assertTrue(validate_auth_probe_result(result)["ok"])
            self.assertEqual(len(requests), 4)
            self.assertEqual(requests[0].get_method(), "GET")
            self.assertIsNone(requests[0].data)
            for request in requests[1:]:
                self.assertEqual(request.get_method(), "POST")
                self.assertEqual(request.full_url, "http://10.0.0.8:8080/v1/chat/completions")
                self.assertEqual(request.data, b"{}")
            self.assertIsNone(requests[1].get_header("Authorization"))
            self.assertNotEqual(
                requests[2].get_header("Authorization"), "Bearer " + self.FIXTURE_KEY
            )
            self.assertEqual(requests[3].get_header("Authorization"), "Bearer " + self.FIXTURE_KEY)

    def test_missing_env_transport_error_or_wrong_auth_accepted_fails_without_secret_output(self):
        result, requests, code = self.run_program([], key=False)
        self.assertEqual((len(requests), code), (0, 1))
        with self.assertRaises(ValueError):
            validate_auth_probe_result(result)
        for statuses in (
            [200, 400, 401, 422],
            [200, 401, 422, 422],
            [200, 401, 401, 200],
            [503, 401, 401, 422],
        ):
            result, _, code = self.run_program(statuses)
            self.assertEqual(code, 1)
            with self.assertRaises(ValueError):
                validate_auth_probe_result(result)
        result, _, code = self.run_program([], error=RuntimeError(self.FIXTURE_KEY))
        self.assertEqual(code, 1)
        with self.assertRaises(ValueError):
            validate_auth_probe_result(result)

    def test_tampered_checks_or_extra_payload_rejected(self):
        result, _, _ = self.run_program([200, 401, 401, 422])
        for change in (
            dict(statuses=dict(health=200, missing=401, wrong=400, correct=422)),
            dict(headers={"Authorization": "fixture"}),
            dict(checks={}),
            dict(status="FAIL"),
        ):
            with self.assertRaises(ValueError):
                validate_auth_probe_result(result | change)

    def test_probe_target_and_owned_names_are_bound(self):
        for url in (
            "https://10.0.0.8:8080",
            "http://10.0.0.8:80",
            "http://user:pass@10.0.0.8:8080",
            "http://10.0.0.8:8080/v1",
            "http://10.0.0.8:8080?token=fixture",
        ):
            with self.assertRaises(ValueError):
                render_auth_probe(url)
        for run, secret in (("invalid:tenant", "owned"), ("owned", "Invalid_Name"), ("owned", "")):
            with self.assertRaises(ValueError):
                render_ingress_auth(run, secret)


if __name__ == "__main__":
    unittest.main()
