"""HTTP plumbing for providers that speak JSON over HTTP.

Kept apart from any one provider for two reasons: Gemini will want the same
thing in Phase 5, and a provider's parsing logic can then be tested against a
fake transport instead of a patched private method.

Only the standard library is used. The payloads are small JSON documents, so
``urllib`` is enough and rolex stays installable with no runtime dependencies.
"""

import http.client
import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any, Protocol

from rolex.core.errors import ProviderError, UnreachableError

#: Failures that happen while reading the response rather than while
#: connecting. urllib only wraps connection-phase problems in URLError, so
#: these arrive bare and would otherwise escape as unhandled exceptions —
#: breaking is_available()'s promise never to raise.
#: OSError covers TimeoutError and ConnectionError; URLError, which also
#: subclasses OSError, is always caught before these.
_READ_FAILURES = (http.client.HTTPException, OSError)


class Transport(Protocol):
    """The HTTP operations a provider needs."""

    def get_json(self, url: str, *, timeout: float) -> dict[str, Any]:
        """GET `url` and parse the JSON object it returns."""
        ...

    def post_json(self, url: str, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        """POST `body` as JSON and parse the JSON object that comes back."""
        ...

    def post_stream(
        self, url: str, body: dict[str, Any], *, timeout: float
    ) -> Iterator[dict[str, Any]]:
        """POST `body` and yield one parsed object per line of NDJSON."""
        ...


class UrllibTransport:
    """The real one, over :mod:`urllib`."""

    def get_json(self, url: str, *, timeout: float) -> dict[str, Any]:
        return self._json(urllib.request.Request(url), timeout=timeout)

    def post_json(self, url: str, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        return self._json(_request(url, body), timeout=timeout)

    def post_stream(
        self, url: str, body: dict[str, Any], *, timeout: float
    ) -> Iterator[dict[str, Any]]:
        try:
            with urllib.request.urlopen(_request(url, body), timeout=timeout) as response:
                for raw in response:
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ProviderError(f"bad JSON in stream from {url}: {line!r}") from exc
        except urllib.error.HTTPError as exc:
            raise _http_error(url, exc) from exc
        except urllib.error.URLError as exc:
            raise UnreachableError(f"could not reach {url} ({exc.reason})") from exc
        except _READ_FAILURES as exc:
            raise UnreachableError(f"lost the connection to {url} ({exc})") from exc

    def _json(self, request: urllib.request.Request, *, timeout: float) -> dict[str, Any]:
        url = request.full_url
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise _http_error(url, exc) from exc
        except urllib.error.URLError as exc:
            raise UnreachableError(f"could not reach {url} ({exc.reason})") from exc
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{url} sent something that was not JSON") from exc
        except _READ_FAILURES as exc:
            raise UnreachableError(f"lost the connection to {url} ({exc})") from exc

        if not isinstance(payload, dict):
            raise ProviderError(f"{url} returned {type(payload).__name__}, expected an object")
        return payload


def _request(url: str, body: dict[str, Any]) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def _http_error(url: str, exc: urllib.error.HTTPError) -> ProviderError:
    detail = exc.read().decode("utf-8", errors="replace").strip()
    return ProviderError(f"{url} returned {exc.code}: {detail}")
