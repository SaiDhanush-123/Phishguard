from __future__ import annotations

import ipaddress
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from urllib.parse import SplitResult, urlsplit

import tldextract

from .config import CONFIG


_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_HOST_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$", re.IGNORECASE)


@lru_cache(maxsize=1)
def _extractor() -> tldextract.TLDExtract:
    # Empty suffix_list_urls guarantees that inference never performs a network fetch.
    return tldextract.TLDExtract(
        suffix_list_urls=(),
        include_psl_private_domains=False,
        extra_suffixes=CONFIG.get("public_suffix_extra", []),
    )


@dataclass(frozen=True)
class NormalizedURL:
    original: str
    parse_value: str
    valid: bool
    error: str | None
    supplied_scheme: bool
    scheme: str
    unicode_hostname: str
    ascii_hostname: str
    registrable_domain: str
    domain_label: str
    suffix: str
    subdomain: str
    path: str
    query: str
    fragment: str
    port: int | None
    is_ip: bool
    has_punycode: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _invalid(original: str, parse_value: str, error: str, supplied_scheme: bool = False) -> NormalizedURL:
    return NormalizedURL(
        original=original,
        parse_value=parse_value,
        valid=False,
        error=error,
        supplied_scheme=supplied_scheme,
        scheme="",
        unicode_hostname="",
        ascii_hostname="",
        registrable_domain="",
        domain_label="",
        suffix="",
        subdomain="",
        path="",
        query="",
        fragment="",
        port=None,
        is_ip=False,
        has_punycode=False,
    )


def _validate_hostname(hostname: str) -> str | None:
    if len(hostname) > 253:
        return "hostname exceeds 253 characters"
    for label in hostname.split("."):
        if not label:
            return "hostname contains an empty label"
        if len(label) > 63:
            return "hostname label exceeds 63 characters"
        if not _HOST_LABEL_RE.fullmatch(label):
            return "hostname contains invalid characters"
    return None


def _parts(value: str) -> tuple[SplitResult | None, str | None]:
    try:
        return urlsplit(value), None
    except ValueError as exc:
        return None, f"malformed URL: {exc}"


def normalize_url(value: object) -> NormalizedURL:
    if value is None:
        return _invalid("", "", "input is empty")
    original = str(value).strip()
    if not original or original.lower() == "nan":
        return _invalid(original, original, "input is empty")

    supplied_scheme = bool(_SCHEME_RE.match(original))
    if original.startswith("//"):
        parse_value = f"http:{original}"
    elif supplied_scheme:
        parse_value = original
    else:
        parse_value = f"http://{original}"

    parsed, parse_error = _parts(parse_value)
    if parse_error or parsed is None:
        return _invalid(original, parse_value, parse_error or "malformed URL", supplied_scheme)
    if parsed.scheme.lower() not in {"http", "https"}:
        return _invalid(original, parse_value, "only HTTP and HTTPS URLs are supported", supplied_scheme)
    try:
        hostname = (parsed.hostname or "").strip().rstrip(".").lower()
        port = parsed.port
    except ValueError as exc:
        return _invalid(original, parse_value, f"invalid hostname or port: {exc}", supplied_scheme)
    if not hostname:
        return _invalid(original, parse_value, "URL does not contain a hostname", supplied_scheme)

    try:
        ascii_hostname = hostname.encode("idna").decode("ascii").lower()
        unicode_hostname = ascii_hostname.encode("ascii").decode("idna")
    except UnicodeError as exc:
        return _invalid(original, parse_value, f"invalid internationalized hostname: {exc}", supplied_scheme)

    try:
        ipaddress.ip_address(ascii_hostname.strip("[]"))
        is_ip = True
    except ValueError:
        is_ip = False

    if is_ip:
        registrable = ascii_hostname
        domain_label = ascii_hostname
        suffix = ""
        subdomain = ""
    else:
        host_error = _validate_hostname(ascii_hostname)
        if host_error:
            return _invalid(original, parse_value, host_error, supplied_scheme)
        extracted = _extractor()(ascii_hostname)
        if not extracted.domain or not extracted.suffix:
            return _invalid(
                original,
                parse_value,
                "hostname has no recognized public suffix",
                supplied_scheme,
            )
        domain_label = extracted.domain.lower()
        suffix = extracted.suffix.lower()
        subdomain = extracted.subdomain.lower()
        registrable = f"{domain_label}.{suffix}"

    return NormalizedURL(
        original=original,
        parse_value=parse_value,
        valid=True,
        error=None,
        supplied_scheme=supplied_scheme,
        scheme=parsed.scheme.lower(),
        unicode_hostname=unicode_hostname,
        ascii_hostname=ascii_hostname,
        registrable_domain=registrable,
        domain_label=domain_label,
        suffix=suffix,
        subdomain=subdomain,
        path=parsed.path or "",
        query=parsed.query or "",
        fragment=parsed.fragment or "",
        port=port,
        is_ip=is_ip,
        has_punycode=any(label.startswith("xn--") for label in ascii_hostname.split(".")),
    )
