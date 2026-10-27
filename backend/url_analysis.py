"""Non-network URL indicator checks. This module never visits submitted URLs."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import parse_qsl, urlsplit


URL_RULES_VERSION = "local-url-indicators-v1"
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "is.gd", "rb.gy", "ow.ly"}
SENSITIVE_TERMS = {"login", "verify", "password", "wallet", "payment", "bank", "urgent"}
UPI_FIELDS = {
    "pa": "Payee VPA (unverified)",
    "pn": "Payee name supplied (unverified)",
    "mc": "Merchant code supplied",
    "tid": "Transaction ID supplied",
    "tr": "Transaction reference supplied",
    "tn": "Transaction note supplied",
    "am": "Amount supplied",
    "cu": "Currency supplied",
    "url": "Merchant URL supplied",
    "mode": "Mode supplied",
    "orgid": "Organization ID supplied",
    "sign": "Signature supplied",
    "purpose": "Purpose supplied",
    "category": "Category supplied",
    "validitystart": "Validity start supplied",
    "validityend": "Validity end supplied",
    "block": "Block flag supplied",
}


def classify_qr_content(value: str) -> str:
    """Classify a decoded QR payload without treating every URI as a website."""
    candidate = value.strip()
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return "other_uri" if "://" in candidate else "plain_text"
    scheme = parsed.scheme.lower()
    if scheme == "upi":
        return "upi_payment_uri" if parsed.netloc.lower() == "pay" else "other_uri"
    if scheme in {"http", "https"}:
        return "website_url"
    if candidate.lower().startswith("www."):
        return "website_url"
    if re.fullmatch(
        r"(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?::\d{1,5})?(?:[/?#][^\s]*)?",
        candidate,
    ):
        return "website_url"
    if scheme:
        return "other_uri"
    return "plain_text"


def extract_upi_payment_fields(value: str) -> list[dict[str, str]]:
    """Extract only allowlisted key/value pairs actually present in a UPI URI."""
    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() != "upi" or parsed.netloc.lower() != "pay":
        raise ValueError("Unsupported UPI URI; expected a `upi://pay` request.")
    fields = []
    for key, field_value in parse_qsl(parsed.query, keep_blank_values=True):
        label = UPI_FIELDS.get(key.lower())
        if label is not None:
            fields.append({"key": key, "label": label, "value": field_value})
    return fields


def analyze_url(raw_url: str) -> dict[str, object]:
    value = raw_url.strip()
    candidate = value if "://" in value else f"https://{value}"
    try:
        parsed = urlsplit(candidate)
        host = (parsed.hostname or "").rstrip(".").lower()
        if not host or any(character.isspace() for character in host):
            raise ValueError("A valid hostname is required.")
        _ = parsed.port  # Access validates malformed/out-of-range ports.
    except ValueError as exc:
        raise ValueError(f"Invalid URL: {exc}") from exc

    indicators: list[dict[str, str]] = []

    def add(code: str, message: str, severity: str = "review") -> None:
        indicators.append({"code": code, "message": message, "severity": severity})

    if parsed.scheme.lower() != "https":
        add("non_https", "The URL does not use HTTPS.")
    if parsed.username is not None or parsed.password is not None:
        add("userinfo", "The URL contains embedded user information.", "high")
    try:
        ipaddress.ip_address(host.strip("[]"))
        add("ip_host", "The hostname is a literal IP address.", "high")
    except ValueError:
        pass
    if host.startswith("xn--") or ".xn--" in host:
        add("punycode", "The hostname includes an internationalized punycode label.")
    if host in SHORTENERS:
        add("shortener", "The hostname belongs to a known URL-shortening service.")
    if parsed.port not in (None, 80, 443):
        add("unusual_port", "The URL uses a non-standard port.")
    terms = sorted(term for term in SENSITIVE_TERMS if term in f"{host}{parsed.path}{parsed.query}".lower())
    if terms:
        add("sensitive_terms", "The URL contains words often used in account or payment prompts: " + ", ".join(terms) + ".")

    level = "high" if any(item["severity"] == "high" for item in indicators) else "review" if indicators else "no-listed-indicators"
    return {
        "submitted_url": value,
        "hostname": host,
        "risk_level": level,
        "indicators": indicators,
        "component_version": URL_RULES_VERSION,
        "explanation": (
            "This local rule check does not visit the URL and is not a safety guarantee. "
            "No listed indicator does not establish that a URL is safe."
        ),
    }
