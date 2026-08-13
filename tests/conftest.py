"""Shared test fixtures."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


@pytest.fixture
def font_dir(tmp_path: Path) -> Path:
    """A directory for synthetic fonts, created lazily by the builders."""
    return tmp_path / "fonts"


@pytest.fixture
def http_assets() -> Iterator[Callable[[dict[str, bytes]], str]]:
    """Serve a dict of ``name -> bytes`` over HTTP and return the base URL.

    A real socket rather than a mocked ``urlopen``: the fetch's contract is about
    what ends up on disk after a partial or wrong response, and that is exactly
    what a stubbed transport would paper over. Bound to 127.0.0.1 on an ephemeral
    port, so the suite stays offline and parallel-safe.
    """
    servers: list[ThreadingHTTPServer] = []

    def serve(files: dict[str, bytes]) -> str:
        class Handler(BaseHTTPRequestHandler):
            # Named as http.server dispatches it, hence the capitals.
            def do_GET(self) -> None:
                body = files.get(self.path.lstrip("/"))
                if body is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                """Silence the default stderr access log, which pytest would capture."""

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        host, port = server.server_address[:2]
        return f"http://{host}:{port}"

    yield serve

    for server in servers:
        server.shutdown()
        server.server_close()
