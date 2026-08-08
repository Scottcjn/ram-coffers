"""Persistent, multiplexed connections to console nodes.

Three properties, each of which exists because of something the fleet does:

**Persistent.** A token touches every shelf and every layer touches several
consoles, so a forward pass makes hundreds of requests. Opening a TCP connection
per request would add a handshake to each; at a quarter-millisecond RTT that is
most of the time budget. Connections are opened once and kept.

**Multiplexed.** Replies are matched to requests by ``request_id``, so the
coordinator can have the whole shelf's requests in flight simultaneously and
collect them as they land. A reader thread per connection owns the socket's read
side; callers wait on their own event. This is the part that makes a wide shelf
faster than a narrow one — without it, fan-out would serialise.

**Nagle off.** Every message here is a small, complete, latency-critical unit.
``TCP_NODELAY`` is not an optimisation in this design, it is a correctness
requirement for the latency budget: with it left on, a 40 ms delayed-ACK
interaction can dominate a forward pass.

Failures are translated into :mod:`gen9_cluster.errors` types with the node's id
attached, so a coordinator can decide about retries without inspecting socket
errnos.
"""

from __future__ import annotations

import errno
import socket
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

from .errors import (ConnectError, Gen9Error, ProtocolError, TimeoutError_)
from .protocol import HEADER_SIZE, Frame, MsgType, decode_error

#: How long to wait for a socket to connect.
DEFAULT_CONNECT_TIMEOUT = 5.0
#: How long to wait for a reply before giving up on it.
DEFAULT_REQUEST_TIMEOUT = 30.0


@dataclass
class _Pending:
    event: threading.Event = field(default_factory=threading.Event)
    frame: Optional[Frame] = None
    error: Optional[BaseException] = None


class NodeConnection:
    """One persistent connection to one console.

    Thread-safe: any number of threads may call :meth:`request` concurrently,
    which is exactly what a shelf coordinator does when it fans a token's top-k
    out to the consoles holding those experts.
    """

    def __init__(self, unit_id: str, host: str, port: int, *,
                 connect_timeout: float = DEFAULT_CONNECT_TIMEOUT):
        self.host = host
        self.port = port
        self.unit_id = unit_id or f"{host}:{port}"
        self.connect_timeout = connect_timeout
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()          # guards writes and _next_id
        self._pending: Dict[int, _Pending] = {}
        self._pending_lock = threading.Lock()
        self._next_id = 1
        self._reader: Optional[threading.Thread] = None
        self._closed = False

    # -- lifecycle --------------------------------------------------------
    def connect(self) -> None:
        if self._sock is not None:
            return
        try:
            sock = socket.create_connection((self.host, self.port),
                                            timeout=self.connect_timeout)
        except OSError as exc:
            raise ConnectError(f"cannot reach {self.host}:{self.port}: {exc}",
                               unit_id=self.unit_id) from exc
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(None)
        self._sock = sock
        self._closed = False
        self._reader = threading.Thread(target=self._read_loop,
                                        name=f"g9xc-{self.unit_id}",
                                        daemon=True)
        self._reader.start()

    def close(self) -> None:
        self._closed = True
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        self._fail_all(ConnectError("connection closed", unit_id=self.unit_id))

    def __enter__(self) -> "NodeConnection":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @property
    def connected(self) -> bool:
        return self._sock is not None and not self._closed

    # -- request/response -------------------------------------------------
    def request(self, frame: Frame, *,
                timeout: float = DEFAULT_REQUEST_TIMEOUT) -> Frame:
        """Send ``frame`` and wait for the reply carrying the same id."""
        if self._sock is None:
            self.connect()
        pending = _Pending()
        with self._lock:
            request_id = self._next_request_id()
            frame.request_id = request_id
            with self._pending_lock:
                self._pending[request_id] = pending
            try:
                self._sock.sendall(frame.encode())    # type: ignore[union-attr]
            except OSError as exc:
                with self._pending_lock:
                    self._pending.pop(request_id, None)
                self.close()
                raise ConnectError(f"send failed: {exc}",
                                   unit_id=self.unit_id) from exc

        if not pending.event.wait(timeout):
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise TimeoutError_(
                f"no reply to {frame.msg_type.name} within {timeout:g}s",
                unit_id=self.unit_id, layer=frame.layer)
        if pending.error is not None:
            raise pending.error
        assert pending.frame is not None
        reply = pending.frame
        if reply.msg_type is MsgType.ERROR:
            raise Gen9Error(decode_error(reply.payload), unit_id=self.unit_id,
                            layer=reply.layer)
        return reply

    def _next_request_id(self) -> int:
        """Ids never repeat within a connection and never reach zero.

        Zero is reserved for one-way messages, so a late reply to an abandoned
        request can never be mistaken for one.
        """
        request_id = self._next_id
        self._next_id = ((self._next_id + 1) & 0xFFFFFFFF) or 1
        return request_id

    def send_oneway(self, frame: Frame) -> None:
        """Fire and forget: SHUTDOWN, and telemetry the sender does not read."""
        if self._sock is None:
            self.connect()
        with self._lock:
            frame.request_id = 0
            try:
                self._sock.sendall(frame.encode())    # type: ignore[union-attr]
            except OSError as exc:
                self.close()
                raise ConnectError(f"send failed: {exc}",
                                   unit_id=self.unit_id) from exc

    # -- reader thread ----------------------------------------------------
    def _read_loop(self) -> None:
        sock = self._sock
        try:
            while sock is not None and not self._closed:
                head = _recv_exactly(sock, HEADER_SIZE)
                if head is None:
                    raise ConnectError("node closed the connection",
                                       unit_id=self.unit_id)
                try:
                    frame, length = Frame.decode_header(head)
                except ValueError as exc:
                    raise ProtocolError(str(exc), unit_id=self.unit_id) from exc
                if length:
                    body = _recv_exactly(sock, length)
                    if body is None:
                        raise ConnectError("truncated payload",
                                           unit_id=self.unit_id)
                    frame.payload = body
                with self._pending_lock:
                    pending = self._pending.pop(frame.request_id, None)
                if pending is None:
                    continue        # a reply to something we stopped waiting on
                pending.frame = frame
                pending.event.set()
        except (ConnectError, ProtocolError) as exc:
            self._fail_all(exc)
        except OSError as exc:
            if not self._closed and exc.errno not in (errno.EBADF,):
                self._fail_all(ConnectError(f"read failed: {exc}",
                                            unit_id=self.unit_id))
            else:
                self._fail_all(ConnectError("connection closed",
                                            unit_id=self.unit_id))

    def _fail_all(self, exc: BaseException) -> None:
        with self._pending_lock:
            pending, self._pending = self._pending, {}
        for item in pending.values():
            item.error = exc
            item.event.set()


def _recv_exactly(sock: socket.socket, count: int) -> Optional[bytes]:
    """Read exactly ``count`` bytes, or None at a clean end of stream."""
    chunks = []
    remaining = count
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class ConnectionPool:
    """One connection per node, created on demand and reused.

    A shelf coordinator holds one of these for its members. Reconnection is the
    caller's decision, not the pool's: a node that dropped may have dropped for
    a reason the coordinator wants to act on (drain it, fail its shards over)
    rather than reconnect into a crash loop.
    """

    def __init__(self, *, connect_timeout: float = DEFAULT_CONNECT_TIMEOUT):
        self._connections: Dict[str, NodeConnection] = {}
        self._lock = threading.Lock()
        self.connect_timeout = connect_timeout

    def get(self, unit_id: str, host: str, port: int) -> NodeConnection:
        with self._lock:
            conn = self._connections.get(unit_id)
            if conn is not None and conn.connected:
                return conn
            conn = NodeConnection(unit_id, host, port,
                                  connect_timeout=self.connect_timeout)
            self._connections[unit_id] = conn
        conn.connect()
        return conn

    def drop(self, unit_id: str) -> None:
        with self._lock:
            conn = self._connections.pop(unit_id, None)
        if conn is not None:
            conn.close()

    def close(self) -> None:
        with self._lock:
            connections = list(self._connections.values())
            self._connections.clear()
        for conn in connections:
            conn.close()

    def __enter__(self) -> "ConnectionPool":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
