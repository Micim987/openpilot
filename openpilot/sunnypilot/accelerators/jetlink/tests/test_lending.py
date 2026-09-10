"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Handing the endpoints over without handing the gadget over.

jetlinkd holds ep0 and the UDC bind for as long as the link is enabled, so a
drive starting or ending is no longer an unplug the Jetson has to recover from.
What still changes hands is the right to read the endpoint files, and this is
the handshake for it.
"""

import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from openpilot.sunnypilot.accelerators.jetlink import lending


class LendingTest(unittest.TestCase):
  def setUp(self):
    # a short path: an AF_UNIX address is about 100 bytes and a pytest tmp_path
    # spends most of that before the filename
    self.dir = Path(tempfile.mkdtemp(dir='/tmp'))
    self.addCleanup(shutil.rmtree, self.dir, True)
    self.addCleanup(lending.release)
    self.path = self.dir / 's'
    self.free = True          # the daemon has nothing open on the endpoints
    self.bounced = 0
    p = mock.patch.object(lending, 'RETRY', 0.01)
    self.addCleanup(p.stop)
    p.start()
    p = mock.patch.object(lending.helpers, 'bound_udc', side_effect=lambda: self.udc)
    self.addCleanup(p.stop)
    self.udc = 'udc0'
    p.start()

  def lender(self) -> lending.Lender:
    lender = lending.Lender(lambda: self.free, self.bounce, path=self.path)
    assert lender.start()
    self.addCleanup(lender.stop)
    return lender

  def bounce(self) -> bool:
    self.bounced += 1
    return True

  @staticmethod
  def until(predicate, timeout=3.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
      if predicate():
        return True
      time.sleep(0.01)
    return False


class Borrowing(LendingTest):
  def test_a_borrower_is_told_where_the_gadget_is(self):
    lender = self.lender()
    loan = lending.borrow(path=self.path)
    assert loan is not None
    assert loan.udc == 'udc0'
    assert loan.mount == str(lending.helpers.FFS_MOUNT)
    assert lender.lent and lender.borrower == 'modeld'

  def test_nobody_listening_is_not_an_error(self):
    # the link was only just turned on, or the daemon died. The caller opens
    # the gadget itself, as it always did
    assert lending.borrow(path=self.path, timeout=0.1) is None

  def test_the_daemon_is_told_to_get_off_the_endpoints_before_it_says_yes(self):
    # a borrow that lands while the daemon is mid-exchange: it hears about it
    # on the first ask, and answers once it has put the endpoints down
    self.free = False
    lender = self.lender()
    got = []
    import threading
    t = threading.Thread(target=lambda: got.append(lending.borrow(path=self.path, timeout=3.0)), daemon=True)
    t.start()
    assert self.until(lambda: lender.lent), 'the daemon was never told to let go'
    assert not got, 'lent the endpoints while they were still in use'
    self.free = True
    t.join(3.0)
    assert got and got[0] is not None

  def test_a_borrow_nobody_can_answer_gives_up_and_says_so(self):
    self.free = False
    lender = self.lender()
    assert lending.borrow(path=self.path, timeout=0.2) is None
    assert lender.lent, 'the daemon must still know somebody wants it'

  def test_the_connection_is_the_lease(self):
    # modeld is stopped at every ignition-off and SIGKILLed if it lingers;
    # dying is how it hands the link back
    lender = self.lender()
    loan = lending.borrow(path=self.path)
    assert lender.lent
    loan.close()
    assert self.until(lambda: not lender.lent), 'the link never came back'

  def test_one_loan_per_process(self):
    # a join that fails and tries again wants the link it already has
    self.lender()
    first = lending.borrow(path=self.path)
    assert lending.borrow(path=self.path) is first

  def test_a_closed_loan_is_replaced_not_reused(self):
    self.lender()
    first = lending.borrow(path=self.path)
    first.close()
    second = lending.borrow(path=self.path)
    assert second is not None and second is not first

  def test_a_gadget_that_is_not_bound_yet_is_waited_for(self):
    self.udc = None
    lender = self.lender()
    assert lending.borrow(path=self.path, timeout=0.2) is None
    assert lender.lent
    self.udc = 'udc0'
    assert lending.borrow(path=self.path, timeout=1.0) is not None


class Bouncing(LendingTest):
  def test_a_stuck_write_reaches_the_owner(self):
    # unbinding is the only thing that dequeues a FunctionFS write nobody is
    # reading, and the unbind belongs to whoever holds ep0
    self.lender()
    loan = lending.borrow(path=self.path)
    assert loan.bounce() is True
    assert self.bounced == 1

  def test_a_bounce_after_the_loan_is_over_is_not_an_error(self):
    self.lender()
    loan = lending.borrow(path=self.path)
    loan.close()
    assert loan.bounce() is False
    assert self.bounced == 0


class StaleSockets(LendingTest):
  def test_a_socket_a_dead_daemon_left_is_cleared(self):
    self.path.write_text('')          # anything at the address stops bind()
    lender = self.lender()
    assert lending.borrow(path=self.path) is not None
    assert lender.lent

  def test_a_live_daemon_keeps_its_socket(self):
    # two jetlinkds is a misconfiguration, and the second must not take the
    # gadget away from the one that owns it
    first = self.lender()
    second = lending.Lender(lambda: self.free, self.bounce, path=self.path)
    self.addCleanup(second.stop)
    assert second.start() is False
    assert lending.borrow(path=self.path) is not None
    assert first.lent and not second.lent


if __name__ == '__main__':
  unittest.main()
