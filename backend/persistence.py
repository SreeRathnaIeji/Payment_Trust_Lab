"""Local SQLite accounts and per-user assessment history for the demo app."""

from __future__ import annotations

import json
import hashlib
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bcrypt
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("PAYSIM_DB_PATH", PROJECT_ROOT / "backend" / "data" / "demo.sqlite3"))
TOKEN_MAX_AGE = 60 * 60 * 24 * 30
DUMMY_HASH = bcrypt.hashpw(hashlib.sha256(b"not-a-user-password").digest(), bcrypt.gensalt(rounds=12))


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize() -> None:
    with connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash BLOB NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                kind TEXT NOT NULL,
                created_at TEXT NOT NULL,
                input_json TEXT NOT NULL,
                score REAL,
                threshold REAL,
                decision TEXT NOT NULL,
                component_version TEXT NOT NULL,
                source_account TEXT,
                target_account TEXT
            );
            CREATE TABLE IF NOT EXISTS revoked_tokens (
                token_hash TEXT PRIMARY KEY,
                revoked_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_assessments_user_created
                ON assessments(user_id, created_at DESC, id DESC);
            """
        )
        db.execute(
            "INSERT OR IGNORE INTO app_settings(setting_key, setting_value) VALUES('token_secret', ?)",
            (secrets.token_urlsafe(48),),
        )


def _serializer() -> URLSafeTimedSerializer:
    with connect() as db:
        row = db.execute(
            "SELECT setting_value FROM app_settings WHERE setting_key='token_secret'"
        ).fetchone()
    if row is None:
        raise RuntimeError("Account storage has not been initialized.")
    return URLSafeTimedSerializer(row[0], salt="paysim-demo-session-v1")


def create_user(username: str, password: str) -> int:
    normalized = username.strip()
    password_hash = bcrypt.hashpw(hashlib.sha256(password.encode("utf-8")).digest(), bcrypt.gensalt(rounds=12))
    with connect() as db:
        cursor = db.execute(
            "INSERT INTO users(username, password_hash, created_at) VALUES(?, ?, ?)",
            (normalized, password_hash, datetime.now(timezone.utc).isoformat()),
        )
        return int(cursor.lastrowid)


def authenticate(username: str, password: str) -> int | None:
    with connect() as db:
        user = db.execute(
            "SELECT id, password_hash FROM users WHERE username = ? COLLATE NOCASE",
            (username.strip(),),
        ).fetchone()
    if user is None:
        bcrypt.checkpw(hashlib.sha256(password.encode("utf-8")).digest(), DUMMY_HASH)
        return None
    try:
        digest = hashlib.sha256(password.encode("utf-8")).digest()
        return int(user["id"]) if bcrypt.checkpw(digest, user["password_hash"]) else None
    except ValueError:
        return None


def issue_token(user_id: int) -> str:
    # The nonce ensures distinct sessions even when a user signs in twice in
    # the same timestamp interval used by URLSafeTimedSerializer.
    return _serializer().dumps({"uid": user_id, "nonce": secrets.token_urlsafe(24)})


def revoke_token(token: str) -> None:
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with connect() as db:
        db.execute(
            "INSERT OR IGNORE INTO revoked_tokens(token_hash, revoked_at) VALUES(?, ?)",
            (token_hash, datetime.now(timezone.utc).isoformat()),
        )


def verify_token(token: str) -> int | None:
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with connect() as db:
        revoked = db.execute(
            "SELECT 1 FROM revoked_tokens WHERE token_hash = ?", (token_hash,)
        ).fetchone()
    if revoked:
        return None
    try:
        payload = _serializer().loads(token, max_age=TOKEN_MAX_AGE)
        user_id = int(payload["uid"])
    except (BadSignature, SignatureExpired, KeyError, TypeError, ValueError):
        return None
    with connect() as db:
        row = db.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
    return user_id if row else None


def username_for(user_id: int) -> str | None:
    with connect() as db:
        row = db.execute("SELECT username FROM users WHERE id=?", (user_id,)).fetchone()
    return str(row[0]) if row else None


def save_assessment(
    user_id: int,
    *,
    kind: str,
    inputs: dict[str, Any],
    score: float | None,
    threshold: float | None,
    decision: str,
    component_version: str,
) -> int:
    created_at = datetime.now(timezone.utc).isoformat()
    source = inputs.get("nameOrig")
    target = inputs.get("nameDest")
    with connect() as db:
        cursor = db.execute(
            """INSERT INTO assessments
               (user_id, kind, created_at, input_json, score, threshold, decision,
                component_version, source_account, target_account)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                kind,
                created_at,
                json.dumps(inputs, separators=(",", ":")),
                score,
                threshold,
                decision,
                component_version,
                source,
                target,
            ),
        )
        return int(cursor.lastrowid)


def list_assessments(user_id: int, limit: int = 100) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            """SELECT id, kind, created_at, input_json, score, threshold, decision,
                      component_version
               FROM assessments WHERE user_id=? ORDER BY id DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "kind": row["kind"],
            "created_at": row["created_at"],
            "inputs": json.loads(row["input_json"]),
            "score": row["score"],
            "threshold": row["threshold"],
            "decision": row["decision"],
            "component_version": row["component_version"],
        }
        for row in rows
    ]


def relationship_summary(user_id: int, source: str, target: str) -> dict[str, int]:
    """Build a tiny directed graph from this user's prior saved assessments only."""
    with connect() as db:
        row = db.execute(
            """SELECT
                 SUM(CASE WHEN source_account=? THEN 1 ELSE 0 END) AS source_outgoing,
                 SUM(CASE WHEN target_account=? THEN 1 ELSE 0 END) AS target_incoming,
                 SUM(CASE WHEN source_account=? AND target_account=? THEN 1 ELSE 0 END) AS repeated_pair,
                 COUNT(DISTINCT CASE WHEN source_account=? THEN target_account
                                     WHEN target_account=? THEN source_account END) AS shared_neighbors
               FROM assessments WHERE user_id=? AND kind='transaction'""",
            (source, target, source, target, source, source, user_id),
        ).fetchone()
    return {key: int(row[key] or 0) for key in ("source_outgoing", "target_incoming", "repeated_pair", "shared_neighbors")}
