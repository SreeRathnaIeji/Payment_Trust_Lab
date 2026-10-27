"""Inference API for the trained chronological PaySim XGBoost model."""

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Literal
import re
import sqlite3
import math

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from PIL import Image, UnidentifiedImageError

from backend import persistence
from backend import gnn_inference
from backend.url_analysis import (
    URL_RULES_VERSION,
    analyze_url,
    classify_qr_content,
    extract_upi_payment_fields,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = (
    PROJECT_ROOT
    / "Experiments"
    / "artifacts"
    / "chronological"
    / "xgboost"
)
MODEL_PATH = ARTIFACT_DIR / "xgboost_main_features.joblib"
ENCODER_PATH = ARTIFACT_DIR / "type_encoder.joblib"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
THRESHOLD = 0.95
FLOAT32_MAX = float(np.finfo(np.float32).max)
FLOAT32_MAX_INT = int(FLOAT32_MAX)
NUMERIC_FEATURES = (
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)
MAX_QR_IMAGE_BYTES = 5 * 1024 * 1024
MAX_QR_IMAGE_PIXELS = 20_000_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    persistence.initialize()
    app.state.model = None
    app.state.encoder = None
    app.state.load_error = None
    app.state.gnn_bundle = None
    app.state.gnn_load_error = None
    try:
        app.state.model = joblib.load(MODEL_PATH)
        app.state.encoder = joblib.load(ENCODER_PATH)
        categories = [str(value) for value in app.state.encoder.categories_[0]]
        if app.state.model.n_features_in_ != len(NUMERIC_FEATURES) + len(categories):
            raise ValueError("Saved model feature count does not match the encoder schema.")
        app.state.transaction_types = categories
    except Exception as exc:
        app.state.model = None
        app.state.encoder = None
        app.state.load_error = f"Could not load model artifacts: {exc}"
        app.state.transaction_types = []
    try:
        app.state.gnn_bundle = gnn_inference.load_node_only_bundle()
    except Exception as exc:
        app.state.gnn_bundle = None
        app.state.gnn_load_error = f"Could not load experimental node-only GNN artifacts: {exc}"
    yield


app = FastAPI(
    title="PaySim Fraud Detection API",
    description="Inference-only research prototype using a trained PaySim model.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
bearer = HTTPBearer(auto_error=False)


def _json_safe_validation_detail(value):
    """Keep echoed non-finite input values from breaking FastAPI's 422 response."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe_validation_detail(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_validation_detail(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request, exc: RequestValidationError
):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": _json_safe_validation_detail(exc.errors())},
    )


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=1, le=FLOAT32_MAX_INT, description="PaySim time step")
    amount: float = Field(ge=0, le=FLOAT32_MAX, allow_inf_nan=False)
    oldbalanceOrg: float = Field(ge=0, le=FLOAT32_MAX, allow_inf_nan=False)
    newbalanceOrig: float = Field(ge=0, le=FLOAT32_MAX, allow_inf_nan=False)
    oldbalanceDest: float = Field(ge=0, le=FLOAT32_MAX, allow_inf_nan=False)
    newbalanceDest: float = Field(ge=0, le=FLOAT32_MAX, allow_inf_nan=False)
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    # Optional PaySim-style identifiers are saved only for this user's local graph view.
    nameOrig: str | None = Field(default=None, min_length=1, max_length=128)
    nameDest: str | None = Field(default=None, min_length=1, max_length=128)


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)


class URLRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(min_length=3, max_length=2048)


class GraphRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_account: str = Field(min_length=1, max_length=128)
    target_account: str = Field(min_length=1, max_length=128)


def optional_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> int | None:
    if credentials is None:
        return None
    user_id = persistence.verify_token(credentials.credentials)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Session is invalid or expired.")
    return user_id


def require_user(user_id: int | None = Depends(optional_user_id)) -> int:
    if user_id is None:
        raise HTTPException(status_code=401, detail="Sign in to use this account feature.")
    return user_id


def require_artifacts(request: Request) -> None:
    if request.app.state.model is None or request.app.state.encoder is None:
        raise HTTPException(
            status_code=503,
            detail=request.app.state.load_error or "Model artifacts are unavailable.",
        )


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health(request: Request):
    if request.app.state.model is None:
        return {
            "status": "unavailable",
            "model_loaded": False,
            "error": request.app.state.load_error,
        }
    return {"status": "ok", "model_loaded": True, "threshold": THRESHOLD}


@app.post("/api/predict")
def predict(transaction: Transaction, request: Request, user_id: int | None = Depends(optional_user_id)):
    require_artifacts(request)
    numeric = np.asarray(
        [[getattr(transaction, feature) for feature in NUMERIC_FEATURES]],
        dtype=np.float32,
    )
    categorical_input = pd.DataFrame({"type": [transaction.type]})
    categorical = request.app.state.encoder.transform(categorical_input).astype(np.float32)
    model_input = np.concatenate((numeric, categorical), axis=1).astype(
        np.float32, copy=False
    )
    if model_input.shape[1] != request.app.state.model.n_features_in_:
        raise HTTPException(status_code=500, detail="Prepared input does not match the saved model.")

    probability = float(request.app.state.model.predict_proba(model_input)[0, 1])
    is_fraud = probability >= THRESHOLD
    response = {
        "fraud_probability": probability,
        "decision": "Potential fraud" if is_fraud else "Likely legitimate",
        "is_fraud": is_fraud,
        "threshold": THRESHOLD,
        "explanation": (
            "This is a model prediction from a research prototype trained and evaluated "
            "on synthetic PaySim data. It is not a guarantee or a real-world banking decision."
        ),
    }
    if user_id is not None:
        response["history_id"] = persistence.save_assessment(
            user_id,
            kind="transaction",
            inputs=transaction.model_dump(),
            score=probability,
            threshold=THRESHOLD,
            decision=response["decision"],
            component_version="paysim-xgb-main-chron-v1",
        )
    return response


@app.post("/api/predict/gnn")
def predict_gnn(
    transaction: Transaction,
    request: Request,
    user_id: int = Depends(require_user),
):
    bundle = request.app.state.gnn_bundle
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=request.app.state.gnn_load_error
            or "The experimental node-only GNN is unavailable.",
        )
    try:
        result = gnn_inference.predict_node_only(transaction, bundle)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Experimental GNN assessment failed: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Experimental GNN assessment failed; the XGBoost assessment remains available.",
        ) from exc

    decision = result["decision"]
    response = {
        "model": "experimental-node-feature-only-gnn-v2",
        "model_label": "Experimental node-feature-only GNN",
        "decision": decision,
        "is_fraud": bool(result["is_fraud"]),
        "raw_score": float(result["raw_score"]),
        "threshold": float(result["threshold"]),
        "score_note": "Raw experimental model output; not calibrated as a real-world fraud probability.",
        "explanation": (
            "This experimental node-feature-only model uses transaction fields only. "
            "It does not use account graph history and is not the primary assessment model."
        ),
    }
    response["history_id"] = persistence.save_assessment(
        user_id,
        kind="gnn_transaction",
        inputs=transaction.model_dump(),
        score=response["raw_score"],
        threshold=response["threshold"],
        decision=decision,
        component_version="experimental-node-feature-only-gnn-v2",
    )
    return response


@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
def register(credentials: Credentials):
    try:
        user_id = persistence.create_user(credentials.username, credentials.password)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="That username is already registered.") from exc
    return {"access_token": persistence.issue_token(user_id), "token_type": "bearer", "username": credentials.username.strip()}


@app.post("/api/auth/login")
def login(credentials: Credentials):
    user_id = persistence.authenticate(credentials.username, credentials.password)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Username or password is incorrect.")
    return {"access_token": persistence.issue_token(user_id), "token_type": "bearer", "username": persistence.username_for(user_id)}


@app.get("/api/auth/me")
def current_user(user_id: int = Depends(require_user)):
    return {"username": persistence.username_for(user_id)}


@app.post("/api/auth/logout")
def logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    user_id: int = Depends(require_user),
):
    # require_user validates the same bearer token before this handler runs.
    persistence.revoke_token(credentials.credentials)
    return {"status": "signed_out"}


@app.get("/api/history")
def history(user_id: int = Depends(require_user)):
    return {"assessments": persistence.list_assessments(user_id)}


@app.post("/api/graph/analyze")
def analyze_account_relationship(data: GraphRequest, user_id: int = Depends(require_user)):
    summary = persistence.relationship_summary(user_id, data.source_account, data.target_account)
    return {
        "source_account": data.source_account,
        "target_account": data.target_account,
        "history_scope": "only the signed-in user's saved assessments",
        "metrics": summary,
        "explanation": "This is a small local relationship summary, not a trained graph model and not a PaySim-wide account lookup. It does not affect transaction predictions.",
    }


@app.post("/api/url/check")
def check_url(data: URLRequest, user_id: int | None = Depends(optional_user_id)):
    try:
        result = analyze_url(data.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if user_id is not None:
        history_id = persistence.save_assessment(
            user_id,
            kind="url",
            inputs={"url": result["submitted_url"]},
            score=None,
            threshold=None,
            decision=str(result["risk_level"]),
            component_version=URL_RULES_VERSION,
        )
        result["history_id"] = history_id
    return result


@app.post("/api/url/scan-qr")
async def scan_qr_image(
    image: UploadFile = File(...),
    user_id: int | None = Depends(optional_user_id),
):
    """Decode one QR image and locally inspect URL payloads; never visit them."""
    content = await image.read(MAX_QR_IMAGE_BYTES + 1)
    if len(content) > MAX_QR_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large. Maximum size is 5 MB.")
    if not content:
        raise HTTPException(status_code=422, detail="The uploaded image is empty.")

    try:
        with Image.open(BytesIO(content)) as probe:
            if probe.width * probe.height > MAX_QR_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions are too large to scan.")
            probe.verify()
        with Image.open(BytesIO(content)) as source:
            source.load()
            qr_image = source.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=422, detail="The upload is not a supported, readable image.") from exc

    try:
        import zxingcpp
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="QR decoding is unavailable because the zxing-cpp package is not installed.",
        ) from exc

    try:
        barcodes = zxingcpp.read_barcodes(
            qr_image, formats=zxingcpp.BarcodeFormat.QRCode
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail="The image could not be decoded as a QR code.") from exc
    if not barcodes:
        raise HTTPException(status_code=422, detail="No readable QR code was found in the image.")
    decoded_content = str(barcodes[0].text).strip()
    if not decoded_content:
        raise HTTPException(status_code=422, detail="The QR code did not contain readable text.")

    detected_format = classify_qr_content(decoded_content)
    result: dict[str, object] = {
        "decoded_content": decoded_content,
        "detected_format": detected_format,
    }
    if detected_format == "website_url":
        result["content_type"] = "url"
        try:
            analysis = analyze_url(decoded_content)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"The QR contains a website URL that could not be checked: {exc}",
            ) from exc
        result["url_analysis"] = analysis
        risk_labels = {
            "high": "High risk indicators found",
            "review": "Warning signs to review",
            "no-listed-indicators": "No listed warning signs detected",
        }
        result["risk_assessment"] = {
            "status": "assessed",
            "level": analysis["risk_level"],
            "label": risk_labels[str(analysis["risk_level"])],
            "warning_signs": analysis["indicators"],
            "explanation": analysis["explanation"],
        }
        result["message"] = (
            "Website URL decoded and checked locally. It was not opened. "
            "No listed warning signs does not guarantee safety."
        )
        if user_id is not None:
            result["url_analysis"]["history_id"] = persistence.save_assessment(
                user_id,
                kind="url",
                inputs={"url": analysis["submitted_url"]},
                score=None,
                threshold=None,
                decision=str(analysis["risk_level"]),
                component_version=URL_RULES_VERSION,
            )
        return result

    if detected_format == "upi_payment_uri":
        try:
            fields = extract_upi_payment_fields(decoded_content)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result["content_type"] = "upi_payment_uri"
        result["upi_fields"] = fields
        result["risk_assessment"] = {
            "status": "not_assessed",
            "level": "not_assessed",
            "label": "UPI payment request recognized; payment legitimacy not assessed",
            "warning_signs": [],
            "explanation": (
                "A UPI URI alone does not establish fraud or verify the recipient. "
                "Payee fields are unverified text supplied by the QR. No payment was initiated."
            ),
        }
        result["message"] = "UPI payment URI decoded for inspection only. No payment was initiated."
        return result

    if detected_format == "other_uri":
        result["content_type"] = "uri"
        result["risk_assessment"] = {
            "status": "not_assessed",
            "level": "not_assessed",
            "label": "Non-website URI; URL warning rules not applied",
            "warning_signs": [],
            "explanation": "This payload is not a supported website URL or UPI payment URI.",
        }
        result["message"] = "The URI was decoded but not opened or assessed."
        return result

    result["content_type"] = "text"
    result["risk_assessment"] = {
        "status": "not_assessed",
        "level": "not_assessed",
        "label": "Plain text; website URL warning rules not applied",
        "warning_signs": [],
        "explanation": "Plain text is not assessed as a website URL and no safety conclusion is made.",
    }
    result["message"] = "This QR code contains plain text, not a website URL. No URL warning check was run."
    return result
