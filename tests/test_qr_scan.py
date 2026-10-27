"""Focused API tests for QR decoding; the test database and images are disposable."""

from __future__ import annotations

import atexit
import asyncio
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import unittest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from PIL import Image


_test_root = Path(tempfile.mkdtemp(prefix="paysim-qr-test-"))
_database = (_test_root / "qr-test.sqlite3").resolve()
_protected = {
    (Path(__file__).resolve().parents[1] / "backend" / "data" / "demo.sqlite3").resolve(),
    (Path(__file__).resolve().parents[1] / "backend" / "api-check.sqlite3").resolve(),
}
if _database in _protected:
    raise RuntimeError("Refusing to use a protected project database for QR tests.")
os.environ["PAYSIM_DB_PATH"] = str(_database)
atexit.register(shutil.rmtree, _test_root, ignore_errors=True)

from backend.main import app, lifespan  # noqa: E402


def make_qr(path: Path, payload: str) -> Path:
    try:
        import zxingcpp
    except ImportError:
        raise unittest.SkipTest("zxing-cpp is required to create QR fixtures")
    barcode = zxingcpp.create_barcode(payload, zxingcpp.BarcodeFormat.QRCode)
    Image.fromarray(barcode.to_image(scale=5)).save(path)
    return path


class QRScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def post_image(self, path: Path):
        with path.open("rb") as image_file:
            return self.client.post(
                "/api/url/scan-qr",
                files={"image": (path.name, image_file, "image/png")},
            )

    def transaction_payload(self):
        return {
            "step": 500,
            "amount": 25000.0,
            "oldbalanceOrg": 30000.0,
            "newbalanceOrig": 5000.0,
            "oldbalanceDest": 1000.0,
            "newbalanceDest": 26000.0,
            "type": "CASH_OUT",
        }

    def authenticated_headers(self):
        username = f"gnn-{uuid.uuid4().hex[:12]}"
        registered = self.client.post(
            "/api/auth/register",
            json={"username": username, "password": "test-password-123"},
        )
        self.assertEqual(registered.status_code, 201, registered.text)
        return {"Authorization": f"Bearer {registered.json()['access_token']}"}

    def test_startup_keeps_xgboost_when_optional_gnn_bundle_is_missing(self):
        async def verify_lifespan():
            isolated_app = FastAPI()
            xgb_model = SimpleNamespace(n_features_in_=11)
            encoder = SimpleNamespace(categories_=[[
                "CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"
            ]])
            with (
                patch("backend.main.persistence.initialize"),
                patch("backend.main.joblib.load", side_effect=[xgb_model, encoder]),
                patch(
                    "backend.main.gnn_inference.load_node_only_bundle",
                    side_effect=FileNotFoundError("test bundle missing"),
                ),
            ):
                async with lifespan(isolated_app):
                    self.assertIs(isolated_app.state.model, xgb_model)
                    self.assertEqual(isolated_app.state.transaction_types, encoder.categories_[0])
                    self.assertIsNone(isolated_app.state.gnn_bundle)
                    self.assertIn("test bundle missing", isolated_app.state.gnn_load_error)

        asyncio.run(verify_lifespan())

    def test_gnn_assessment_requires_authentication_and_saves_separate_history_kind(self):
        payload = self.transaction_payload()
        unauthorized = self.client.post("/api/predict/gnn", json=payload)
        self.assertEqual(unauthorized.status_code, 401)

        headers = self.authenticated_headers()
        response = self.client.post("/api/predict/gnn", json=payload, headers=headers)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["model"], "experimental-node-feature-only-gnn-v2")
        self.assertEqual(body["model_label"], "Experimental node-feature-only GNN")
        self.assertIn(body["decision"], {"FRAUD", "NOT FRAUD"})
        self.assertIsInstance(body["raw_score"], float)
        self.assertEqual(body["threshold"], 0.9023056626319885)
        self.assertIn("not calibrated", body["score_note"])
        self.assertIn("history_id", body)

        history = self.client.get("/api/history", headers=headers)
        self.assertEqual(history.status_code, 200, history.text)
        assessment = history.json()["assessments"][0]
        self.assertEqual(assessment["kind"], "gnn_transaction")
        self.assertEqual(
            assessment["component_version"],
            "experimental-node-feature-only-gnn-v2",
        )

    def test_missing_gnn_bundle_does_not_disable_xgboost_endpoint(self):
        headers = self.authenticated_headers()
        previous_bundle = self.client_context.app.state.gnn_bundle
        previous_error = self.client_context.app.state.gnn_load_error
        try:
            self.client_context.app.state.gnn_bundle = None
            self.client_context.app.state.gnn_load_error = "test GNN artifact unavailable"
            gnn_response = self.client.post(
                "/api/predict/gnn", json=self.transaction_payload(), headers=headers
            )
            self.assertEqual(gnn_response.status_code, 503)
            self.assertIn("test GNN artifact unavailable", gnn_response.json()["detail"])

            xgb_response = self.client.post(
                "/api/predict", json=self.transaction_payload(), headers=headers
            )
            self.assertEqual(xgb_response.status_code, 200, xgb_response.text)
            self.assertEqual(xgb_response.json()["threshold"], 0.95)
            self.assertIn("fraud_probability", xgb_response.json())
        finally:
            self.client_context.app.state.gnn_bundle = previous_bundle
            self.client_context.app.state.gnn_load_error = previous_error

    def test_gnn_inference_error_is_isolated_and_reported_as_unavailable(self):
        headers = self.authenticated_headers()
        with patch(
            "backend.main.gnn_inference.predict_node_only",
            side_effect=RuntimeError("inference failed"),
        ):
            response = self.client.post(
                "/api/predict/gnn", json=self.transaction_payload(), headers=headers
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("Experimental GNN assessment failed", response.json()["detail"])

        xgb_response = self.client.post(
            "/api/predict", json=self.transaction_payload(), headers=headers
        )
        self.assertEqual(xgb_response.status_code, 200, xgb_response.text)

    def test_invalid_image_returns_readable_validation_error(self):
        image_path = _test_root / "invalid.png"
        image_path.write_bytes(b"this is not an image")
        response = self.post_image(image_path)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json(), {"detail": "The upload is not a supported, readable image."})

    def test_valid_url_qr_is_decoded_and_checked_locally(self):
        url = "http://bit.ly/login?verify=payment"
        image_path = make_qr(_test_root / "url.png", url)
        response = self.post_image(image_path)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["decoded_content"], url)
        self.assertEqual(body["content_type"], "url")
        self.assertEqual(body["detected_format"], "website_url")
        self.assertEqual(body["url_analysis"]["submitted_url"], url)
        self.assertGreaterEqual(
            {item["code"] for item in body["url_analysis"]["indicators"]},
            {"non_https", "shortener", "sensitive_terms"},
        )
        self.assertEqual(body["risk_assessment"]["status"], "assessed")
        self.assertEqual(body["risk_assessment"]["level"], "review")
        self.assertGreaterEqual(len(body["risk_assessment"]["warning_signs"]), 3)
        self.assertIn("not opened", body["message"])

    def test_normal_https_qr_is_website_format_without_claiming_safety(self):
        url = "https://example.com/account"
        image_path = make_qr(_test_root / "https.png", url)
        response = self.post_image(image_path)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["decoded_content"], url)
        self.assertEqual(body["detected_format"], "website_url")
        self.assertEqual(body["risk_assessment"]["status"], "assessed")
        self.assertEqual(body["risk_assessment"]["level"], "no-listed-indicators")
        self.assertIn("does not establish that a URL is safe", body["risk_assessment"]["explanation"])

    def test_upi_payment_qr_is_not_scored_as_a_website_or_called_fraud(self):
        uri = "upi://pay?pa=alice%40upi&pn=Alice%20Store&am=125.50&cu=INR"
        image_path = make_qr(_test_root / "upi.png", uri)
        response = self.post_image(image_path)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["decoded_content"], uri)
        self.assertEqual(body["detected_format"], "upi_payment_uri")
        self.assertEqual(body["risk_assessment"]["status"], "not_assessed")
        self.assertNotIn("url_analysis", body)
        self.assertNotIn("fraud", body["risk_assessment"]["label"].lower())
        self.assertIn("No payment was initiated", body["message"])
        self.assertEqual(
            [(field["key"], field["value"]) for field in body["upi_fields"]],
            [("pa", "alice@upi"), ("pn", "Alice Store"), ("am", "125.50"), ("cu", "INR")],
        )
        self.assertTrue(all("unverified" in field["label"] for field in body["upi_fields"][:2]))

    def test_plain_text_qr_is_returned_without_url_analysis(self):
        text = "Payment instructions: reference 4821"
        image_path = make_qr(_test_root / "text.png", text)
        response = self.post_image(image_path)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["decoded_content"], text)
        self.assertEqual(body["content_type"], "text")
        self.assertEqual(body["detected_format"], "plain_text")
        self.assertEqual(body["risk_assessment"]["status"], "not_assessed")
        self.assertNotIn("url_analysis", body)
        self.assertIn("not a website URL", body["message"])

    def test_valid_image_without_qr_returns_readable_decode_error(self):
        image_path = _test_root / "no-qr.png"
        Image.new("RGB", (256, 256), color="white").save(image_path)
        response = self.post_image(image_path)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json(), {"detail": "No readable QR code was found in the image."})
