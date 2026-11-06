"""The HTTP transport, against a local socket.

A real server is started on a free port, so what is exercised is urllib's
actual behaviour rather than a guess about it. Nothing leaves the machine.
"""

import http.server
import json
import threading
from collections.abc import Callable, Iterator

import pytest

from rolex.core.errors import ProviderError, UnreachableError
from rolex.llm.transport import UrllibTransport

Handler = Callable[[http.server.BaseHTTPRequestHandler], None]


@pytest.fixture
def serve() -> Iterator[Callable[[Handler], str]]:
    """Start a one-off HTTP server running `handler`; yields its base URL."""
    servers: list[http.server.HTTPServer] = []

    def start(handler: Handler) -> str:
        class Once(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                handler(self)

            def do_POST(self) -> None:
                handler(self)

            def log_message(self, *_: object) -> None:
                return None

        server = http.server.HTTPServer(("127.0.0.1", 0), Once)
        servers.append(server)
        # The default poll interval makes every shutdown take half a second,
        # which across this file is most of the suite's runtime.
        threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()
        host, port = server.server_address[:2]
        # server_address is typed as a union; on an AF_INET socket it is
        # (str, int), and the str() keeps mypy happy without a cast.
        return f"http://{host!s}:{port}"

    yield start

    for server in servers:
        server.shutdown()


def _send(request: http.server.BaseHTTPRequestHandler, status: int, body: bytes) -> None:
    request.send_response(status)
    request.send_header("Content-Length", str(len(body)))
    request.end_headers()
    request.wfile.write(body)


def test_get_json_parses_an_object(serve: Callable[[Handler], str]) -> None:
    url = serve(lambda r: _send(r, 200, json.dumps({"models": []}).encode()))
    assert UrllibTransport().get_json(f"{url}/api/tags", timeout=5) == {"models": []}


def test_post_json_sends_the_body(serve: Callable[[Handler], str]) -> None:
    seen: list[dict[str, object]] = []

    def handler(request: http.server.BaseHTTPRequestHandler) -> None:
        length = int(request.headers["Content-Length"])
        seen.append(json.loads(request.rfile.read(length)))
        _send(request, 200, b'{"ok": true}')

    url = serve(handler)
    assert UrllibTransport().post_json(f"{url}/api/chat", {"model": "m"}, timeout=5) == {"ok": True}
    assert seen == [{"model": "m"}]


def test_post_stream_yields_one_object_per_line(serve: Callable[[Handler], str]) -> None:
    body = b'{"a": 1}\n\n{"a": 2}\n'
    url = serve(lambda r: _send(r, 200, body))
    chunks = list(UrllibTransport().post_stream(f"{url}/x", {}, timeout=5))
    assert chunks == [{"a": 1}, {"a": 2}]


def test_http_error_includes_the_status_and_body(serve: Callable[[Handler], str]) -> None:
    url = serve(lambda r: _send(r, 404, b"no such model"))
    with pytest.raises(ProviderError, match="404: no such model"):
        UrllibTransport().get_json(f"{url}/api/tags", timeout=5)


def test_non_json_body_is_an_error(serve: Callable[[Handler], str]) -> None:
    url = serve(lambda r: _send(r, 200, b"<html>not json</html>"))
    with pytest.raises(ProviderError, match="not JSON"):
        UrllibTransport().get_json(f"{url}/x", timeout=5)


def test_json_array_is_rejected(serve: Callable[[Handler], str]) -> None:
    url = serve(lambda r: _send(r, 200, b"[1, 2]"))
    with pytest.raises(ProviderError, match="expected an object"):
        UrllibTransport().get_json(f"{url}/x", timeout=5)


def test_nothing_listening_is_unreachable() -> None:
    # Port 1 on loopback: reserved, and nothing may bind it.
    with pytest.raises(UnreachableError, match="could not reach"):
        UrllibTransport().get_json("http://127.0.0.1:1/x", timeout=1)


def test_a_slow_response_is_unreachable_not_a_crash(serve: Callable[[Handler], str]) -> None:
    """is_available() promises never to raise, so a stalled read must not."""

    def handler(request: http.server.BaseHTTPRequestHandler) -> None:
        request.send_response(200)
        request.send_header("Content-Length", "100")
        request.end_headers()
        request.wfile.write(b"{")  # then never finishes

    url = serve(handler)
    with pytest.raises(UnreachableError):
        UrllibTransport().get_json(f"{url}/x", timeout=0.5)


def test_a_dropped_stream_never_escapes_as_a_raw_oserror(
    serve: Callable[[Handler], str],
) -> None:
    """A server that dies mid-answer must not crash the REPL.

    Whether urllib notices the truncation at all depends on the platform, so
    this does not demand an error — only that if one comes, it is one of ours
    and chat's `except ProviderError` will catch it.
    """

    def handler(request: http.server.BaseHTTPRequestHandler) -> None:
        request.send_response(200)
        request.send_header("Content-Length", "1000")
        request.end_headers()
        request.wfile.write(b'{"a": 1}\n')
        request.wfile.flush()
        request.close_connection = True

    url = serve(handler)
    try:
        chunks = list(UrllibTransport().post_stream(f"{url}/x", {}, timeout=0.5))
    except ProviderError:
        return
    assert chunks == [{"a": 1}]


def test_bad_json_mid_stream_is_reported(serve: Callable[[Handler], str]) -> None:
    url = serve(lambda r: _send(r, 200, b'{"a": 1}\nnot json\n'))
    stream = UrllibTransport().post_stream(f"{url}/x", {}, timeout=5)
    assert next(stream) == {"a": 1}
    with pytest.raises(ProviderError, match="bad JSON"):
        next(stream)
