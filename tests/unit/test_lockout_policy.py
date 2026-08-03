"""The lockout backoff curve, in isolation.

`lockout_delay` is pure so the policy can be pinned without a database, a clock
or an Argon2 hash per case — the integration tests then only have to prove the
counter is persisted, not that the arithmetic is right.
"""

from __future__ import annotations

from graphrag.accounts import lockout_delay

BASE = 60
MAX = 3600
THRESHOLD = 10


def _delay(failures: int, threshold: int = THRESHOLD) -> int:
    return lockout_delay(failures, threshold, BASE, MAX)


def test_below_the_threshold_there_is_no_lock():
    """A typo run must not lock a real user out; the per-IP limit is what caps
    volume, and the lock only exists for the slow distributed case."""
    assert all(_delay(n) == 0 for n in range(THRESHOLD))


def test_the_first_lock_is_the_base_delay():
    assert _delay(THRESHOLD) == BASE


def test_each_further_failure_doubles_the_wait():
    assert _delay(THRESHOLD + 1) == 2 * BASE
    assert _delay(THRESHOLD + 2) == 4 * BASE
    assert _delay(THRESHOLD + 3) == 8 * BASE


def test_the_wait_is_capped():
    assert _delay(THRESHOLD + 6) == MAX
    assert _delay(THRESHOLD + 50) == MAX


def test_a_relentless_attacker_cannot_overflow_the_exponent():
    """`2 ** (failures - threshold)` is unbounded if the exponent isn't clamped,
    and a script hammering a locked account drives `failures` arbitrarily high.
    Computing a million-bit integer per login attempt would be the DoS."""
    assert _delay(10_000_000) == MAX


def test_threshold_zero_disables_lockout():
    """The documented off switch. It has to hold at every failure count, not
    just low ones."""
    assert _delay(1, threshold=0) == 0
    assert _delay(10_000, threshold=0) == 0


def test_the_cap_holds_when_it_is_lower_than_the_base():
    """A misconfiguration (max < base) must clamp rather than invert."""
    assert lockout_delay(THRESHOLD, THRESHOLD, 600, 60) == 60
