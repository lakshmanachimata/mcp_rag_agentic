"""Detect whether the app is being accessed via localhost."""

from __future__ import annotations

import streamlit as st

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _parse_hostname(host_header: str) -> str:
    return host_header.split(":")[0].lower().strip("[]")


def is_localhost_request() -> bool:
    """Return True when the browser URL host is localhost."""
    try:
        headers = st.context.headers
        for key in ("Host", "host", "X-Forwarded-Host"):
            try:
                host = headers[key].strip()
            except KeyError:
                continue
            if host:
                return _parse_hostname(host) in _LOCAL_HOSTS
    except Exception:
        pass

    # Show settings when the host cannot be determined (e.g. first script run).
    return True
