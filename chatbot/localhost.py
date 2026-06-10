"""Detect whether the app is being accessed via localhost."""

from __future__ import annotations

import urllib.parse

import streamlit as st
import streamlit.components.v1 as components

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_REMOTE_HOST_SUFFIXES = (
    ".devtunnels.ms",
    ".ngrok.io",
    ".ngrok-free.app",
    ".trycloudflare.com",
    ".loca.lt",
    ".streamlit.app",
)
_HEADER_KEYS = ("Host", "host", "X-Forwarded-Host", "Origin", "Referer")


def _parse_hostname(host_header: str) -> str:
    return host_header.split(":")[0].lower().strip("[]")


def _hostname_from_url(value: str) -> str:
    try:
        parsed = urllib.parse.urlparse(value if "://" in value else f"https://{value}")
        return (parsed.hostname or "").lower().strip("[]")
    except Exception:
        return ""


def _is_localhost_hostname(hostname: str) -> bool:
    return hostname in _LOCAL_HOSTS


def _is_remote_tunnel_hostname(hostname: str) -> bool:
    return any(hostname.endswith(suffix) for suffix in _REMOTE_HOST_SUFFIXES)


def _hostnames_from_headers() -> list[str]:
    hostnames: list[str] = []
    try:
        headers = st.context.headers
        for key in _HEADER_KEYS:
            try:
                value = headers[key].strip()
            except KeyError:
                continue
            if not value:
                continue
            if key.lower() in ("origin", "referer"):
                hostname = _hostname_from_url(value)
            else:
                hostname = _parse_hostname(value)
            if hostname:
                hostnames.append(hostname)
    except Exception:
        pass
    return hostnames


def _sync_client_hostname() -> None:
    """Read browser hostname via query param (set once by injected JS)."""
    if "client_is_localhost" in st.session_state:
        return

    if "local" in st.query_params:
        st.session_state.client_is_localhost = st.query_params["local"] == "1"
        return

    components.html(
        """
        <script>
        (function () {
            const host = window.location.hostname;
            const isLocal =
                host === "localhost" || host === "127.0.0.1" || host === "[::1]";
            const url = new URL(window.location.href);
            if (!url.searchParams.has("local")) {
                url.searchParams.set("local", isLocal ? "1" : "0");
                window.location.replace(url.toString());
            }
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def is_localhost_request() -> bool:
    """Return True only when the browser URL host is localhost."""
    if "client_is_localhost" in st.session_state:
        return bool(st.session_state.client_is_localhost)

    hostnames = _hostnames_from_headers()
    if hostnames:
        for hostname in hostnames:
            if _is_remote_tunnel_hostname(hostname) or not _is_localhost_hostname(hostname):
                st.session_state.client_is_localhost = False
                return False
        st.session_state.client_is_localhost = True
        return True

    _sync_client_hostname()
    return bool(st.session_state.get("client_is_localhost", False))
