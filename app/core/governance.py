from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Callable

from app.core.config import settings


class GovernanceError(RuntimeError):
	pass


class RateLimitExceeded(GovernanceError):
	pass


class CircuitBreakerOpen(GovernanceError):
	pass


class ExecutionTimeout(GovernanceError):
	pass


@dataclass
class CircuitBreakerState:
	name: str
	state: str = "closed"
	consecutive_failures: int = 0
	opened_at: float | None = None
	last_error: str | None = None


class SlidingWindowRateLimiter:
	def __init__(self):
		self._lock = threading.RLock()
		self._windows: dict[str, deque[float]] = {}

	def acquire(self, key: str, *, limit: int, window_seconds: float):
		now = time.monotonic()
		with self._lock:
			bucket = self._windows.setdefault(key, deque())
			while bucket and now - bucket[0] > window_seconds:
				bucket.popleft()
			if len(bucket) >= limit:
				raise RateLimitExceeded(
					f"rate limit exceeded for {key}: {len(bucket)}/{limit} within {window_seconds:.0f}s"
				)
			bucket.append(now)


class CircuitBreaker:
	def __init__(self, *, name: str, failure_threshold: int, recovery_seconds: float):
		self._lock = threading.RLock()
		self._threshold = max(1, failure_threshold)
		self._recovery_seconds = max(1.0, recovery_seconds)
		self._state = CircuitBreakerState(name=name)

	def before_call(self):
		with self._lock:
			if self._state.state != "open":
				return
			if self._state.opened_at is None:
				self._state.state = "half_open"
				return
			if time.monotonic() - self._state.opened_at >= self._recovery_seconds:
				self._state.state = "half_open"
				return
			raise CircuitBreakerOpen(
				f"circuit breaker open for {self._state.name}; last_error={self._state.last_error or 'unknown'}"
			)

	def on_success(self):
		with self._lock:
			self._state.state = "closed"
			self._state.consecutive_failures = 0
			self._state.opened_at = None
			self._state.last_error = None

	def on_failure(self, error: Exception):
		with self._lock:
			self._state.consecutive_failures += 1
			self._state.last_error = str(error)
			if self._state.consecutive_failures >= self._threshold:
				self._state.state = "open"
				self._state.opened_at = time.monotonic()
			elif self._state.state == "half_open":
				self._state.state = "open"
				self._state.opened_at = time.monotonic()

	def snapshot(self) -> dict[str, Any]:
		with self._lock:
			return {
				"name": self._state.name,
				"state": self._state.state,
				"consecutive_failures": self._state.consecutive_failures,
				"last_error": self._state.last_error,
				"opened_at": self._state.opened_at,
			}


class GovernanceManager:
	def __init__(self):
		self._lock = threading.RLock()
		self._rate_limiter = SlidingWindowRateLimiter()
		self._circuit_breakers: dict[str, CircuitBreaker] = {}

	def _get_breaker(self, name: str, *, failure_threshold: int, recovery_seconds: float) -> CircuitBreaker:
		with self._lock:
			breaker = self._circuit_breakers.get(name)
			if breaker is None:
				breaker = CircuitBreaker(
					name=name,
					failure_threshold=failure_threshold,
					recovery_seconds=recovery_seconds,
				)
				self._circuit_breakers[name] = breaker
			return breaker

	def enforce_request_rate_limit(self, key: str):
		self._rate_limiter.acquire(
			f"request:{key}",
			limit=settings.request_rate_limit_max_requests,
			window_seconds=settings.request_rate_limit_window_seconds,
		)

	def execute_llm(self, name: str, func: Callable[[], Any]) -> Any:
		self._rate_limiter.acquire(
			f"llm:{name}",
			limit=settings.llm_rate_limit_max_requests,
			window_seconds=settings.llm_rate_limit_window_seconds,
		)
		breaker = self._get_breaker(
			f"llm:{name}",
			failure_threshold=settings.llm_circuit_failure_threshold,
			recovery_seconds=settings.llm_circuit_recovery_seconds,
		)
		return self._execute_with_breaker(
			breaker=breaker,
			func=func,
			timeout_seconds=settings.llm_timeout_seconds,
			timeout_label=f"llm:{name}",
		)

	def execute_tool(self, name: str, func: Callable[[], Any]) -> Any:
		self._rate_limiter.acquire(
			f"tool:{name}",
			limit=settings.tool_rate_limit_max_requests,
			window_seconds=settings.tool_rate_limit_window_seconds,
		)
		breaker = self._get_breaker(
			f"tool:{name}",
			failure_threshold=settings.tool_circuit_failure_threshold,
			recovery_seconds=settings.tool_circuit_recovery_seconds,
		)
		return self._execute_with_breaker(
			breaker=breaker,
			func=func,
			timeout_seconds=settings.tool_timeout_seconds,
			timeout_label=f"tool:{name}",
		)

	def _execute_with_breaker(
		self,
		*,
		breaker: CircuitBreaker,
		func: Callable[[], Any],
		timeout_seconds: float,
		timeout_label: str,
	) -> Any:
		breaker.before_call()
		try:
			with ThreadPoolExecutor(max_workers=1) as executor:
				future = executor.submit(func)
				try:
					result = future.result(timeout=timeout_seconds)
				except FutureTimeoutError as exc:
					future.cancel()
					raise ExecutionTimeout(
						f"{timeout_label} timed out after {timeout_seconds:.1f}s"
					) from exc
			breaker.on_success()
			return result
		except GovernanceError as exc:
			breaker.on_failure(exc)
			raise
		except Exception as exc:  # noqa: BLE001
			breaker.on_failure(exc)
			raise

	def snapshot(self) -> dict[str, Any]:
		with self._lock:
			circuits = {
				name: breaker.snapshot()
				for name, breaker in sorted(self._circuit_breakers.items())
			}
		return {
			"request_limit": {
				"window_seconds": settings.request_rate_limit_window_seconds,
				"max_requests": settings.request_rate_limit_max_requests,
			},
			"llm": {
				"timeout_seconds": settings.llm_timeout_seconds,
				"rate_limit_window_seconds": settings.llm_rate_limit_window_seconds,
				"rate_limit_max_requests": settings.llm_rate_limit_max_requests,
				"circuit_failure_threshold": settings.llm_circuit_failure_threshold,
				"circuit_recovery_seconds": settings.llm_circuit_recovery_seconds,
			},
			"tool": {
				"timeout_seconds": settings.tool_timeout_seconds,
				"rate_limit_window_seconds": settings.tool_rate_limit_window_seconds,
				"rate_limit_max_requests": settings.tool_rate_limit_max_requests,
				"circuit_failure_threshold": settings.tool_circuit_failure_threshold,
				"circuit_recovery_seconds": settings.tool_circuit_recovery_seconds,
			},
			"circuits": circuits,
		}


governance_manager = GovernanceManager()


