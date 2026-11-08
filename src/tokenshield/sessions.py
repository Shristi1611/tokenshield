"""Per-session token ledger: a live circuit breaker for runaway conversations.

A request is refused once its session's running total is strictly above the budget. The request
that crosses the line still completes, so one request can overshoot. Sessions live in memory
only, and the oldest are forgotten once max_sessions is exceeded.
"""

from __future__ import annotations

import math
import re
import threading
from collections import OrderedDict

_SESSION_RE = re.compile(r"[A-Za-z0-9_\-]{1,64}")


def valid_session(value: str) -> bool:
    return _SESSION_RE.fullmatch(value) is not None


class SessionLedger:
    def __init__(self, budget: float | None = None, max_sessions: int = 1000) -> None:
        if budget is not None and (math.isnan(budget) or budget < 0):
            raise ValueError("budget must not be negative")
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        self.budget = budget
        self._max_sessions = max_sessions
        self._used: OrderedDict[str, int] = OrderedDict()
        self._lock = threading.Lock()

    def used(self, session: str) -> int:
        with self._lock:
            return self._used.get(session, 0)

    def over_budget(self, session: str) -> bool:
        return self.budget is not None and self.used(session) > self.budget

    def add(self, session: str, tokens: int) -> int:
        if tokens < 0:
            raise ValueError("tokens must not be negative")
        with self._lock:
            total = self._used.pop(session, 0) + tokens
            self._used[session] = total
            while len(self._used) > self._max_sessions:
                self._used.popitem(last=False)
            return total