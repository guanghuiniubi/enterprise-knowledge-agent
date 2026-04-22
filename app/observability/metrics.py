from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.core.governance import governance_manager


@dataclass(frozen=True)
class AlertRecord:
    name: str
    status: str
    severity: str
    message: str
    value: float | int | str | None = None
    threshold: float | int | str | None = None
    sample_size: int = 0


class MetricsStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._counters: dict[str, float] = defaultdict(float)
        self._series: dict[str, deque[tuple[float, float]]] = defaultdict(deque)

    def reset(self):
        with self._lock:
            self._counters = defaultdict(float)
            self._series = defaultdict(deque)

    def _metric_key(self, name: str, labels: dict[str, Any] | None = None) -> str:
        if not labels:
            return name
        suffix = ",".join(f"{key}={labels[key]}" for key in sorted(labels))
        return f"{name}|{suffix}"

    def increment(self, name: str, value: float = 1.0, labels: dict[str, Any] | None = None):
        key = self._metric_key(name, labels)
        with self._lock:
            self._counters[key] += value
            self._append_series_locked(key, value)

    def observe(self, name: str, value: float, labels: dict[str, Any] | None = None):
        key = self._metric_key(name, labels)
        with self._lock:
            self._append_series_locked(key, value)

    def _append_series_locked(self, key: str, value: float):
        now = time.time()
        bucket = self._series[key]
        bucket.append((now, float(value)))
        self._trim_locked(bucket, now)

    def _trim_locked(self, bucket: deque[tuple[float, float]], now: float):
        while bucket and now - bucket[0][0] > settings.observability_window_seconds:
            bucket.popleft()
        while len(bucket) > settings.observability_max_samples:
            bucket.popleft()

    def window_values(self, name: str, labels: dict[str, Any] | None = None) -> list[float]:
        key = self._metric_key(name, labels)
        now = time.time()
        with self._lock:
            bucket = self._series[key]
            self._trim_locked(bucket, now)
            return [value for _, value in bucket]

    def counter_value(self, name: str, labels: dict[str, Any] | None = None) -> float:
        key = self._metric_key(name, labels)
        with self._lock:
            return self._counters.get(key, 0.0)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            observations: dict[str, dict[str, float | int]] = {}
            for key, bucket in self._series.items():
                values = [value for _, value in bucket]
                if not values:
                    continue
                ordered = sorted(values)
                observations[key] = {
                    "count": len(values),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                    "avg": round(sum(values) / len(values), 4),
                    "p50": round(self._percentile(ordered, 0.50), 4),
                    "p95": round(self._percentile(ordered, 0.95), 4),
                }
            return {
                "window_seconds": settings.observability_window_seconds,
                "max_samples": settings.observability_max_samples,
                "counters": counters,
                "observations": observations,
            }

    def _percentile(self, ordered_values: list[float], ratio: float) -> float:
        if not ordered_values:
            return 0.0
        index = max(0, min(len(ordered_values) - 1, math.ceil(len(ordered_values) * ratio) - 1))
        return ordered_values[index]


class ObservabilityManager:
    def __init__(self):
        self.metrics = MetricsStore()

    def reset(self):
        self.metrics.reset()

    def _rate(self, numerator: float, denominator: float) -> float:
        return round(numerator / max(1.0, denominator), 4)

    def _metric_p95(self, name: str, labels: dict[str, Any] | None = None) -> float:
        values = self.metrics.window_values(name, labels=labels)
        if not values:
            return 0.0
        return round(self.metrics._percentile(sorted(values), 0.95), 4)

    def _matching_counters(self, prefix: str) -> dict[str, float]:
        snapshot = self.metrics.snapshot()
        return {
            key: float(value)
            for key, value in snapshot["counters"].items()
            if key == prefix or key.startswith(f"{prefix}|")
        }

    def _matching_observations(self, prefix: str) -> dict[str, dict[str, float | int]]:
        snapshot = self.metrics.snapshot()
        return {
            key: value
            for key, value in snapshot["observations"].items()
            if key == prefix or key.startswith(f"{prefix}|")
        }

    def record_http_request(self, *, method: str, path: str, status_code: int, latency_ms: float):
        labels = {"method": method, "path": path, "status": str(status_code)}
        self.metrics.increment("http_requests_total")
        self.metrics.increment("http_requests_total", labels=labels)
        self.metrics.observe("http_request_latency_ms", latency_ms)
        self.metrics.observe("http_request_latency_ms", latency_ms, labels={"path": path})
        if status_code >= 500:
            self.metrics.increment("http_5xx_total")
            self.metrics.increment("http_5xx_total", labels={"path": path})
        elif status_code >= 400:
            self.metrics.increment("http_4xx_total")
            self.metrics.increment("http_4xx_total", labels={"path": path})

    def record_chat_request(self, *, route: str, latency_ms: float, fallback: bool, step_count: int, tool_calls: int, tool_failures: int):
        self.metrics.increment("chat_requests_total")
        self.metrics.increment("chat_requests_by_route_total", labels={"route": route})
        self.metrics.observe("chat_latency_ms", latency_ms)
        self.metrics.observe("chat_latency_ms", latency_ms, labels={"route": route})
        self.metrics.observe("agent_step_count", float(step_count))
        self.metrics.observe("tool_call_count", float(tool_calls))
        if fallback:
            self.metrics.increment("chat_fallback_total")
        if tool_calls:
            self.metrics.increment("tool_calls_total", value=tool_calls)
        if tool_failures:
            self.metrics.increment("tool_failures_total", value=tool_failures)

    def record_stream_request(self, *, route: str, latency_ms: float, fallback: bool, step_count: int, tool_calls: int, tool_failures: int):
        self.metrics.increment("stream_requests_total")
        self.metrics.increment("stream_requests_by_route_total", labels={"route": route})
        self.record_chat_request(
            route=route,
            latency_ms=latency_ms,
            fallback=fallback,
            step_count=step_count,
            tool_calls=tool_calls,
            tool_failures=tool_failures,
        )

    def record_security_block(self):
        self.metrics.increment("security_blocks_total")

    def record_prompt_injection_check(self, *, hit: bool, blocked: bool):
        self.metrics.increment("prompt_injection_checks_total")
        if hit:
            self.metrics.increment("prompt_injection_hits_total")
        if blocked:
            self.metrics.increment("prompt_injection_blocked_total")

    def record_acl_check(self, *, allowed: bool, stage: str, visibility: str = "unknown", reason: str = "unknown"):
        labels = {"stage": stage, "visibility": visibility, "reason": reason}
        self.metrics.increment("acl_checks_total")
        self.metrics.increment("acl_checks_total", labels=labels)
        if not allowed:
            self.metrics.increment("acl_denies_total")
            self.metrics.increment("acl_denies_total", labels=labels)

    def record_output_filter_hit(self):
        self.metrics.increment("output_filter_hits_total")

    def record_chat_error(self, kind: str):
        self.metrics.increment("chat_errors_total")
        self.metrics.increment("chat_errors_by_type_total", labels={"type": kind})

    def record_retrieval(self, *, source: str, latency_ms: float, candidate_count: int, accessible_count: int, result_count: int):
        labels = {"source": source}
        self.metrics.increment("retrieval_requests_total")
        self.metrics.increment("retrieval_requests_total", labels=labels)
        self.metrics.observe("retrieval_latency_ms", latency_ms)
        self.metrics.observe("retrieval_latency_ms", latency_ms, labels=labels)
        self.metrics.observe("retrieval_candidates_count", float(candidate_count))
        self.metrics.observe("retrieval_accessible_count", float(accessible_count))
        self.metrics.observe("retrieval_results_count", float(result_count))
        if result_count == 0:
            self.metrics.increment("retrieval_empty_total")
            self.metrics.increment("retrieval_empty_total", labels=labels)

    def record_rerank(self, *, strategy: str, latency_ms: float, input_count: int, output_count: int):
        labels = {"strategy": strategy}
        self.metrics.increment("rerank_requests_total")
        self.metrics.increment("rerank_requests_total", labels=labels)
        self.metrics.observe("rerank_latency_ms", latency_ms)
        self.metrics.observe("rerank_latency_ms", latency_ms, labels=labels)
        self.metrics.observe("rerank_input_count", float(input_count))
        self.metrics.observe("rerank_output_count", float(output_count))

    def record_llm_call(self, *, operation: str, latency_ms: float, success: bool, error_kind: str | None = None):
        labels = {"operation": operation}
        self.metrics.increment("llm_calls_total")
        self.metrics.increment("llm_calls_total", labels=labels)
        self.metrics.observe("llm_latency_ms", latency_ms)
        self.metrics.observe("llm_latency_ms", latency_ms, labels=labels)
        if not success:
            self.metrics.increment("llm_failures_total")
            self.metrics.increment("llm_failures_total", labels={**labels, "error_kind": error_kind or "unknown"})

    def record_tool_call(self, *, name: str, latency_ms: float, success: bool, error_kind: str | None = None):
        labels = {"tool": name}
        self.metrics.increment("tool_calls_runtime_total")
        self.metrics.increment("tool_calls_runtime_total", labels=labels)
        self.metrics.observe("tool_latency_ms", latency_ms)
        self.metrics.observe("tool_latency_ms", latency_ms, labels=labels)
        if not success:
            self.metrics.increment("tool_failures_runtime_total")
            self.metrics.increment("tool_failures_runtime_total", labels={**labels, "error_kind": error_kind or "unknown"})

    def record_evaluation_run(self, *, case_count: int, accuracy: float, fallback_rate: float):
        self.metrics.increment("evaluation_runs_total")
        self.metrics.observe("evaluation_case_count", float(case_count))
        self.metrics.observe("evaluation_accuracy", accuracy)
        self.metrics.observe("evaluation_fallback_rate", fallback_rate)

    def derived_snapshot(self) -> dict[str, Any]:
        chat_requests = self.metrics.counter_value("chat_requests_total")
        tool_calls = self.metrics.counter_value("tool_calls_total")
        prompt_checks = self.metrics.counter_value("prompt_injection_checks_total")
        acl_checks = self.metrics.counter_value("acl_checks_total")

        return {
            "fallback_rate": self._rate(
                self.metrics.counter_value("chat_fallback_total"),
                chat_requests,
            ),
            "prompt_injection_hit_rate": self._rate(
                self.metrics.counter_value("prompt_injection_hits_total"),
                prompt_checks,
            ),
            "prompt_injection_block_rate": self._rate(
                self.metrics.counter_value("prompt_injection_blocked_total"),
                prompt_checks,
            ),
            "acl_deny_rate": self._rate(
                self.metrics.counter_value("acl_denies_total"),
                acl_checks,
            ),
            "tool_failure_rate": self._rate(
                self.metrics.counter_value("tool_failures_total"),
                tool_calls,
            ),
            "chat_p95_latency_ms": self._metric_p95("chat_latency_ms"),
            "http_p95_latency_ms": self._metric_p95("http_request_latency_ms"),
            "llm_p95_latency_ms": self._metric_p95("llm_latency_ms"),
            "retrieval_p95_latency_ms": self._metric_p95("retrieval_latency_ms"),
            "rerank_p95_latency_ms": self._metric_p95("rerank_latency_ms"),
            "tool_p95_latency_ms": self._metric_p95("tool_latency_ms"),
        }

    def runtime_snapshot(self) -> dict[str, Any]:
        raw = self.metrics.snapshot()
        return {
            **raw,
            "derived": self.derived_snapshot(),
            "alerts": self.alert_snapshot(),
        }

    def dashboard_snapshot(self) -> dict[str, Any]:
        derived = self.derived_snapshot()
        alert_snapshot = self.alert_snapshot()
        governance = governance_manager.snapshot()
        return {
            "status": alert_snapshot["status"],
            "generated_at": round(time.time(), 3),
            "kpis": [
                {
                    "key": "fallback_rate",
                    "label": "Fallback Rate",
                    "value": derived["fallback_rate"],
                    "unit": "ratio",
                    "description": "Agent 触发降级回答的请求占比。",
                },
                {
                    "key": "prompt_injection_hit_rate",
                    "label": "Prompt Injection Hit Rate",
                    "value": derived["prompt_injection_hit_rate"],
                    "unit": "ratio",
                    "description": "命中 prompt injection 模式的输入占比。",
                },
                {
                    "key": "acl_deny_rate",
                    "label": "ACL Deny Rate",
                    "value": derived["acl_deny_rate"],
                    "unit": "ratio",
                    "description": "知识访问控制拒绝的校验占比。",
                },
                {
                    "key": "tool_failure_rate",
                    "label": "Tool Failure Rate",
                    "value": derived["tool_failure_rate"],
                    "unit": "ratio",
                    "description": "工具调用失败占比。",
                },
                {
                    "key": "chat_p95_latency_ms",
                    "label": "Chat P95 Latency",
                    "value": derived["chat_p95_latency_ms"],
                    "unit": "ms",
                    "description": "最近窗口内 chat 请求 p95 延迟。",
                },
                {
                    "key": "retrieval_p95_latency_ms",
                    "label": "Retrieval P95 Latency",
                    "value": derived["retrieval_p95_latency_ms"],
                    "unit": "ms",
                    "description": "检索链路最近窗口的 p95 延迟。",
                },
            ],
            "derived": derived,
            "alerts": alert_snapshot,
            "governance": governance,
            "counters": {
                "requests": self._matching_counters("chat_requests_total"),
                "errors": self._matching_counters("chat_errors_total") | self._matching_counters("chat_errors_by_type_total"),
                "security": self._matching_counters("prompt_injection_checks_total")
                | self._matching_counters("prompt_injection_hits_total")
                | self._matching_counters("prompt_injection_blocked_total")
                | self._matching_counters("security_blocks_total")
                | self._matching_counters("acl_checks_total")
                | self._matching_counters("acl_denies_total")
                | self._matching_counters("output_filter_hits_total"),
                "tooling": self._matching_counters("tool_calls_total")
                | self._matching_counters("tool_failures_total")
                | self._matching_counters("tool_calls_runtime_total")
                | self._matching_counters("tool_failures_runtime_total"),
            },
            "observations": {
                "latency": self._matching_observations("chat_latency_ms")
                | self._matching_observations("http_request_latency_ms")
                | self._matching_observations("llm_latency_ms")
                | self._matching_observations("retrieval_latency_ms")
                | self._matching_observations("rerank_latency_ms")
                | self._matching_observations("tool_latency_ms"),
                "retrieval": self._matching_observations("retrieval_candidates_count")
                | self._matching_observations("retrieval_accessible_count")
                | self._matching_observations("retrieval_results_count"),
            },
        }

    def alert_snapshot(self) -> dict[str, Any]:
        alerts = self._evaluate_alerts()
        firing = [alert for alert in alerts if alert.status == "firing"]
        return {
            "status": "firing" if firing else "ok",
            "alert_count": len(firing),
            "alerts": [alert.__dict__ for alert in alerts],
        }

    def _evaluate_alerts(self) -> list[AlertRecord]:
        alerts: list[AlertRecord] = []
        latency = self.metrics.window_values("chat_latency_ms")
        if len(latency) >= settings.alert_min_samples:
            p95 = self.metrics._percentile(sorted(latency), 0.95)
            alerts.append(self._threshold_alert(
                name="chat_latency_p95_high",
                severity="warning",
                value=round(p95, 4),
                threshold=settings.alert_latency_p95_threshold_ms,
                sample_size=len(latency),
                comparator=lambda value, threshold: value > threshold,
                message_template="chat p95 latency {value}ms exceeds threshold {threshold}ms",
            ))

        requests = self.metrics.counter_value("chat_requests_total")
        if requests >= settings.alert_min_samples:
            fallback_rate = self.metrics.counter_value("chat_fallback_total") / max(1.0, requests)
            alerts.append(self._threshold_alert(
                name="chat_fallback_rate_high",
                severity="warning",
                value=round(fallback_rate, 4),
                threshold=settings.alert_fallback_rate_threshold,
                sample_size=int(requests),
                comparator=lambda value, threshold: value > threshold,
                message_template="fallback rate {value} exceeds threshold {threshold}",
            ))
            security_rate = self.metrics.counter_value("security_blocks_total") / max(1.0, requests)
            alerts.append(self._threshold_alert(
                name="security_block_rate_high",
                severity="warning",
                value=round(security_rate, 4),
                threshold=settings.alert_security_block_rate_threshold,
                sample_size=int(requests),
                comparator=lambda value, threshold: value > threshold,
                message_template="security block rate {value} exceeds threshold {threshold}",
            ))

        prompt_checks = self.metrics.counter_value("prompt_injection_checks_total")
        if prompt_checks >= settings.alert_min_samples:
            prompt_injection_hit_rate = self.metrics.counter_value("prompt_injection_hits_total") / max(1.0, prompt_checks)
            alerts.append(self._threshold_alert(
                name="prompt_injection_hit_rate_high",
                severity="warning",
                value=round(prompt_injection_hit_rate, 4),
                threshold=settings.alert_prompt_injection_hit_rate_threshold,
                sample_size=int(prompt_checks),
                comparator=lambda value, threshold: value > threshold,
                message_template="prompt injection hit rate {value} exceeds threshold {threshold}",
            ))

        acl_checks = self.metrics.counter_value("acl_checks_total")
        if acl_checks >= settings.alert_min_samples:
            acl_deny_rate = self.metrics.counter_value("acl_denies_total") / max(1.0, acl_checks)
            alerts.append(self._threshold_alert(
                name="acl_deny_rate_high",
                severity="warning",
                value=round(acl_deny_rate, 4),
                threshold=settings.alert_acl_deny_rate_threshold,
                sample_size=int(acl_checks),
                comparator=lambda value, threshold: value > threshold,
                message_template="acl deny rate {value} exceeds threshold {threshold}",
            ))

        tool_calls = self.metrics.counter_value("tool_calls_total")
        if tool_calls >= settings.alert_min_samples:
            tool_failure_rate = self.metrics.counter_value("tool_failures_total") / max(1.0, tool_calls)
            alerts.append(self._threshold_alert(
                name="tool_failure_rate_high",
                severity="critical",
                value=round(tool_failure_rate, 4),
                threshold=settings.alert_tool_failure_rate_threshold,
                sample_size=int(tool_calls),
                comparator=lambda value, threshold: value > threshold,
                message_template="tool failure rate {value} exceeds threshold {threshold}",
            ))

        open_circuits = [
            payload["name"]
            for payload in governance_manager.snapshot().get("circuits", {}).values()
            if payload.get("state") == "open"
        ]
        alerts.append(AlertRecord(
            name="open_circuit_breakers",
            status="firing" if open_circuits else "ok",
            severity="critical",
            message="open circuit breakers detected" if open_circuits else "no open circuit breakers",
            value=",".join(open_circuits) if open_circuits else 0,
            threshold=0,
            sample_size=len(open_circuits),
        ))
        return alerts

    def _threshold_alert(
            self,
            *,
            name: str,
            severity: str,
            value: float,
            threshold: float,
            sample_size: int,
            comparator,
            message_template: str,
    ) -> AlertRecord:
        firing = comparator(value, threshold)
        return AlertRecord(
            name=name,
            status="firing" if firing else "ok",
            severity=severity,
            message=message_template.format(value=value, threshold=threshold),
            value=value,
            threshold=threshold,
            sample_size=sample_size,
        )


observability_manager = ObservabilityManager()
