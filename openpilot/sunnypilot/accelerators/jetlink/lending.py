"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Who may do endpoint IO on the gadget, while one process owns it throughout.

The comma is the USB device: the link exists only while some process holds ep0
with the UDC bound. Two processes used to take turns at that, and every change
of owner was an unplug and a replug as the Jetson saw it, a fresh libusb open
and a fresh server session. jetlinkd holds ep0 for as long as the link is
enabled now, so none of that happens.

What still has to change hands is the right to read the endpoint files.
FunctionFS keeps a queued read queued until something completes it, so a second
reader would sit behind the first and take its reply. This is the handshake for
that: modeld borrows for the length of a drive, jetlinkd stays off the
endpoints while it does, and the connection is the lease, so a modeld that is
killed returns it by dying. There is no "give it back" message: the socket
closing is the only signal, because it is the only one a killed process sends.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path

from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.accelerators.jetlink import helpers

SOCKET = Path('/dev/shm/jetlink-lend.sock')
# how long a borrower waits for jetlinkd to put the gadget down. It only has
# something to put down if it was mid-provision at ignition, and then it is one
# re-enumeration; the usual answer is immediate
BORROW_TIMEOUT = 8.0
# a stuck write is already 15 s old by the time this is asked for
BOUNCE_TIMEOUT = 10.0
RETRY = 0.25
POLL = 0.5


def _send(conn: socket.socket, msg: dict) -> None:
  conn.sendall(json.dumps(msg).encode() + b'\n')


def _recv_line(conn: socket.socket, buf: bytearray, deadline: float) -> dict | None:
  """One json message off the socket, or None if the deadline passes first.

  Both ends speak newline-delimited json over a stream, so a message can arrive
  in pieces or with the next one behind it, and the recv timeout is a poll
  rather than a refusal: the lender answers one borrower at a time and can be a
  moment late while it lets go of the one before.

  The peer going away raises, because that is the one thing neither end may
  read as "nothing yet": for the lender it is the whole lease ending.
  """
  while time.monotonic() < deadline:
    if b'\n' in buf:
      line, _, rest = bytes(buf).partition(b'\n')
      buf[:] = rest
      return json.loads(line)
    try:
      chunk = conn.recv(4096)
    except TimeoutError:
      continue
    if not chunk:
      raise ConnectionResetError('the peer closed the link socket')
    buf.extend(chunk)
  return None


class Loan:
  """The right to do endpoint IO on a gadget jetlinkd owns.

  Held for the length of a drive: modeld is stopped at every ignition-off and
  SIGKILLed if it lingers, so the socket closing is how the link is handed
  back, and a modeld that crashed hands it back the same way.
  """

  def __init__(self, conn: socket.socket, buf: bytearray, mount: str, udc: str):
    self.conn = conn
    self.mount = mount
    self.udc = udc
    self._buf = buf
    self._lock = threading.Lock()
    self._closed = False

  @property
  def closed(self) -> bool:
    return self._closed

  def bounce(self) -> bool:
    """Ask the owner to take the gadget down and put it back up.

    The only thing that dequeues a FunctionFS write nobody is reading is the
    unbind, and the unbind belongs to whoever holds ep0. Called from the write
    watchdog on a link that is already 15 s stuck, so the re-enumeration it
    costs is not the expensive part.
    """
    with self._lock:
      if self._closed:
        return False
      try:
        _send(self.conn, {'op': 'bounce'})
        reply = _recv_line(self.conn, self._buf, time.monotonic() + BOUNCE_TIMEOUT)
      except OSError:
        cloudlog.exception("jetlink: could not ask for a gadget bounce")
        return False
      return bool(reply and reply.get('ok'))

  def close(self) -> None:
    with self._lock:
      self._closed = True
      try:
        self.conn.close()
      except OSError:
        pass


def borrow(name: str = 'modeld', timeout: float = BORROW_TIMEOUT, path: Path = SOCKET) -> Loan | None:
  """Ask jetlinkd for the endpoints, or None if there is nobody to ask.

  None is the ordinary answer on a device where the link was only just turned
  on, or whose daemon died: the caller opens the gadget itself, as it always
  did, so a drive never loses the large model to a daemon fault.
  """
  deadline = time.monotonic() + timeout
  try:
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(POLL)
    conn.connect(str(path))
  except OSError:
    return None   # no jetlinkd listening; the caller owns the gadget itself
  buf = bytearray()
  try:
    while time.monotonic() < deadline:
      _send(conn, {'op': 'borrow', 'name': name})
      reply = _recv_line(conn, buf, deadline)
      if reply is None:
        break   # out of time
      if reply.get('ok'):
        cloudlog.warning("jetlink: borrowed the gadget from jetlinkd (udc %s)", reply.get('udc'))
        return Loan(conn, buf, str(reply['mount']), str(reply['udc']))
      if not reply.get('retry'):
        cloudlog.warning("jetlink: jetlinkd would not lend the gadget (%s)", reply.get('detail'))
        break
      time.sleep(RETRY)
  except (OSError, ValueError, KeyError):
    cloudlog.exception("jetlink: could not borrow the gadget")
  conn.close()
  return None

class Lender:
  """jetlinkd's side: one borrower at a time, for as long as it stays connected.

  `lendable` says whether the gadget is in the state a borrower can take over
  from, bound with no endpoint file open here; while it is not, a borrow is
  answered "retry" and the daemon's own loop puts it there.
  """

  def __init__(self, lendable: Callable[[], bool], bounce: Callable[[], bool],
               path: Path = SOCKET):
    self._lendable = lendable
    self._bounce = bounce
    self.path = path
    self.borrower = ''
    self._sock: socket.socket | None = None
    self._thread: threading.Thread | None = None
    self._stop = threading.Event()
    self._lent = threading.Event()

  @property
  def listening(self) -> bool:
    """Can anybody ask us for the endpoints? If not, holding ep0 only keeps
    the borrower out; see Jetlinkd.step."""
    return self._sock is not None

  @property
  def lent(self) -> bool:
    """Is somebody using the endpoints? True from the first ask, not the first
    successful one: the daemon has to get off them before it can say yes."""
    return self._lent.is_set()

  def start(self) -> bool:
    if self._thread is not None:
      return True
    try:
      self._clear_stale()
      sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
      sock.bind(str(self.path))
      sock.listen(1)
      sock.settimeout(POLL)
    except OSError:
      # a read-only /dev/shm, or a path somebody else owns. modeld opens the
      # gadget itself when nobody answers, so this is not fatal
      cloudlog.exception("jetlink: could not listen on %s", self.path)
      return False
    self._sock = sock
    self._thread = threading.Thread(target=self._serve, name='jetlink_lend', daemon=True)
    self._thread.start()
    return True

  def stop(self) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(2.0)
      self._thread = None
    if self._sock is not None:
      self._sock.close()
      self._sock = None
    try:
      self.path.unlink(missing_ok=True)
    except OSError:
      pass

  def _clear_stale(self) -> None:
    """A socket file a dead daemon left behind. Proven dead by a connect that
    is refused, so a second jetlinkd cannot take the link from a live one."""
    if not self.path.exists():
      return
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
      probe.settimeout(0.5)
      probe.connect(str(self.path))
    except OSError:
      os.unlink(self.path)
    finally:
      probe.close()

  def _serve(self) -> None:
    while not self._stop.is_set():
      try:
        conn, _ = self._sock.accept()
      except (TimeoutError, OSError):
        continue
      try:
        self._handle(conn)
      except Exception:
        cloudlog.exception("jetlink: the borrower's connection failed")
      finally:
        conn.close()
        if self._lent.is_set():
          cloudlog.warning("jetlink: %s handed the gadget back", self.borrower or 'the borrower')
        self._lent.clear()
        self.borrower = ''

  def _handle(self, conn: socket.socket) -> None:
    conn.settimeout(POLL)
    buf = bytearray()
    while not self._stop.is_set():
      try:
        msg = _recv_line(conn, buf, time.monotonic() + POLL)
      except (OSError, ValueError):
        return   # the borrower exited, was killed mid-drive, or is not one
      # None is only the poll coming round again; the lease ends on EOF, which
      # is what a modeld that manager stopped or SIGKILLed sends
      if msg is not None:
        self._answer(conn, msg)

  def _answer(self, conn: socket.socket, msg: dict) -> None:
    op = msg.get('op')
    if op == 'borrow':
      first = not self._lent.is_set()
      self.borrower = str(msg.get('name') or 'a borrower')
      self._lent.set()
      udc = helpers.bound_udc()
      if not (udc and self._lendable()):
        # the daemon is mid-exchange, or has not bound yet. It sees `lent` on
        # its next cycle and puts the endpoints down for us
        _send(conn, {'ok': False, 'retry': True, 'detail': 'the gadget is still in use here'})
        return
      if first:
        cloudlog.warning("jetlink: lending the gadget to %s, udc %s", self.borrower, udc)
      _send(conn, {'ok': True, 'udc': udc, 'mount': str(helpers.FFS_MOUNT)})
    elif op == 'bounce':
      cloudlog.warning("jetlink: %s asked for a gadget bounce", self.borrower)
      _send(conn, {'ok': bool(self._bounce())})
    else:
      _send(conn, {'ok': False, 'detail': f'unknown op {op!r}'})
