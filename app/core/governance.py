from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextvars import copy_context
from dataclasses import dataclass
from math import ceil
from typing import Any, Callable, Protocol

from app.core.config import settings

logger = logging.getLogger(__name__)

try:
	from redis import Redis
except ImportError:  # pragma: no cover
	Redis = None


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


class GovernanceBackend(Protocol):
	backend_name: str

	def acquire_window(self, key: str, *, limit: int, window_seconds: float): ...

	def get_circuit(self, name: str) -> CircuitBreakerState: ...

	def save_circuit(self, state: CircuitBreakerState): ...

	def list_circuits(self) -> dict[str, dict[str, Any]]: ...


class MemoryGovernanceBackend:
	backend_name = "memory"

	def __init__(self):
		self._lock = threading.RLock()
		self._windows: dict[str, deque[float]] = {}
		self._circuits: dict[str, CircuitBreakerState] = {}

	def acquire_window(self, key: str, *, limit: int, window_seconds: float):
		now = time.monotonic()
		with self._lock:
			bucket = self._windows.setdefault(key, deque())
			while bucket and now - bucket[0] > window_seconds:
				bucket.popleft()
			if len(bucket) >= limit:
				raise RateLimitExceeded(f"rate limit exceeded for {key}: {len(bucket)}/{limit} within {window_seconds:.0f}s")
			bucket.append(now)

	def get_circuit(self, name: str) -> CircuitBreakerState:
		with self._lock:
			return self._circuits.get(name, CircuitBreakerState(name=name))

	def save_circuit(self, state: CircuitBreakerState):
		with self._lock:
			self._circuits[state.name] = state

	def list_circuits(self) -> dict[str, dict[str, Any]]:
		with self._lock:
			return {
				name: {
					"name": state.name,
					"state": state.state,
					"consecutive_failures": state.consecutive_failures,
					"opened_at": state.opened_at,
					"last_error": state.last_error,
				}
				for name, state in sorted(self._circuits.items())
			}


class RedisGovernanceBackend:
	backend_name = "redis"

	def __init__(self, redis_url: str, key_prefix: str):
		if Redis is None:
			raise RuntimeError("redis package is not installed")
		self._client = Redis.from_url(redis_url, decode_responses=True)
		self._prefix = key_prefix
		self._circuit_names_key = self._key("circuits")

	def _key(self, name: str) -> str:
		return f"{self._prefix}:{name}"

	def acquire_window(self, key: str, *, limit: int, window_seconds: float):
		redis_key = self._key(f"rate:{key}")
		current = int(self._client.incr(redis_key))
		if current == 1:
			self._client.expire(redis_key, ceil(window_seconds))
		if current > limit:
			raise RateLimitExceeded(f"rate limit exceeded for {key}: {current}/{limit} within {window_seconds:.0f}s")

	def get_circuit(self, name: str) -> CircuitBreakerState:
		data = self._client.hgetall(self._key(f"circuit:{name}"))
		if not data:
			return CircuitBreakerState(name=name)
		opened_at = data.get("opened_at")
		return CircuitBreakerState(
			name=name,
			state=data.get("state", "closed"),
			consecutive_failures=int(data.get("consecutive_failures", 0)),
			opened_at=float(opened_at) if opened_at not in (None, "", "None") else None,
			last_error=data.get("last_error") or None,
		)

	def save_circuit(self, state: CircuitBreakerState):
		self._client.hset(
			self._key(f"circuit:{state.name}"),
			mapping={
				"state": state.state,
				"consecutive_failures": state.consecutive_failures,
				"opened_at": "" if state.opened_at is None else state.opened_at,
				"last_error": state.last_error or "",
			},
		)
		self._client.sadd(self._circuit_names_key, state.name)

	def list_circuits(self) -> dict[str, dict[str, Any]]:
		names = sorted(self._client.smembers(self._circuit_names_key))
		return {
			name: {
				"name": state.name,
				"state": state.state,
				"consecutive_failures": state.consecutive_failures,
				"opened_at": state.opened_at,
				"last_error": state.last_error,
			}
			for name in names
			if (state := self.get_circuit(name))
		}


class GovernanceManager:
	def __init__(self):
		self._backend = self._init_backend()

	def _init_backend(self) -> GovernanceBackend:
		if settings.governance_backend.lower() != "redis":
			return MemoryGovernanceBackend()
		if not settings.redis_url:
			logger.warning("governance backend configured as redis but redis_url is empty; falling back to memory")
			return MemoryGovernanceBackend()
		try:
			backend = RedisGovernanceBackend(settings.redis_url, settings.redis_key_prefix)
			backend._client.ping()
			return backend
		except Exception as exc:  # noqa: BLE001
			logger.warning("failed to initialize redis governance backend, falling back to memory: %s", exc)
			return MemoryGovernanceBackend()

	def enforce_request_rate_limit(self, key: str):
		self._backend.acquire_window(
			f"request:{key}",
			limit=settings.request_rate_limit_max_requests,
			window_seconds=settings.request_rate_limit_window_seconds,
		)

	def execute_llm(self, name: str, func: Callable[[], Any]) -> Any:
		self._backend.acquire_window(
			f"llm:{name}",
			limit=settings.llm_rate_limit_max_requests,
			window_seconds=settings.llm_rate_limit_window_seconds,
		)
		return self._execute_with_breaker(
			circuit_name=f"llm:{name}",
			func=func,
			timeout_seconds=settings.llm_timeout_seconds,
			timeout_label=f"llm:{name}",
			failure_threshold=settings.llm_circuit_failure_threshold,
			recovery_seconds=settings.llm_circuit_recovery_seconds,
		)

	def execute_tool(self, name: str, func: Callable[[], Any]) -> Any:
		self._backend.acquire_window(
			f"tool:{name}",
			limit=settings.tool_rate_limit_max_requests,
			window_seconds=settings.tool_rate_limit_window_seconds,
		)
		return self._execute_with_breaker(
			circuit_name=f"tool:{name}",
			func=func,
			timeout_seconds=settings.tool_timeout_seconds,
			timeout_label=f"tool:{name}",
			failure_threshold=settings.tool_circuit_failure_threshold,
			recovery_seconds=settings.tool_circuit_recovery_seconds,
		)

	def _before_call(self, state: CircuitBreakerState, *, recovery_seconds: float):
		if state.state != "open":
			return state
		if state.opened_at is None or time.monotonic() - state.opened_at >= max(1.0, recovery_seconds):
			state.state = "half_open"
			self._backend.save_circuit(state)
			return state
		raise CircuitBreakerOpen(f"circuit breaker open for {state.name}; last_error={state.last_error or 'unknown'}")

	def _execute_with_breaker(
		self,
		*,
		circuit_name: str,
		func: Callable[[], Any],
		timeout_seconds: float,
		timeout_label: str,
		failure_threshold: int,
		recovery_seconds: float,
	) -> Any:
		state = self._backend.get_circuit(circuit_name)
		self._before_call(state, recovery_seconds=recovery_seconds)
		try:
			captured_context = copy_context()
			with ThreadPoolExecutor(max_workers=1) as executor:
				future = executor.submit(captured_context.run, func)
				try:
					result = future.result(timeout=timeout_seconds)
				except FutureTimeoutError as exc:
					future.cancel()
					raise ExecutionTimeout(f"{timeout_label} timed out after {timeout_seconds:.1f}s") from exc
			self._backend.save_circuit(CircuitBreakerState(name=circuit_name))
			return result
		except GovernanceError as exc:
			self._record_failure(state, exc, failure_threshold=failure_threshold)
			raise
		except Exception as exc:  # noqa: BLE001
			self._record_failure(state, exc, failure_threshold=failure_threshold)
			raise

	def _record_failure(self, state: CircuitBreakerState, error: Exception, *, failure_threshold: int):
		state.consecutive_failures += 1
		state.last_error = str(error)
		if state.consecutive_failures >= max(1, failure_threshold) or state.state == "half_open":
			state.state = "open"
			state.opened_at = time.monotonic()
		self._backend.save_circuit(state)

	def snapshot(self) -> dict[str, Any]:
		return {
			"backend": self._backend.backend_name,
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
			"circuits": self._backend.list_circuits(),
		}


governance_manager = GovernanceManager()


