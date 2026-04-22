from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings


@dataclass(frozen=True)
class AccessContext:
    user_id: str
    roles: tuple[str, ...] = field(default_factory=tuple)
    departments: tuple[str, ...] = field(default_factory=tuple)
    clearance_level: int = 0

    @classmethod
    def from_payload(
            cls,
            *,
            user_id: str,
            roles: list[str] | tuple[str, ...] | None = None,
            departments: list[str] | tuple[str, ...] | None = None,
            clearance_level: int | None = None,
    ) -> "AccessContext":
        return cls(
            user_id=user_id,
            roles=tuple(sorted({item.strip() for item in (roles or []) if item and item.strip()})),
            departments=tuple(sorted({item.strip() for item in (departments or []) if item and item.strip()})),
            clearance_level=max(0, int(clearance_level or 0)),
        )


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    visibility: str
    reason: str


class KnowledgeAccessController:
    def _extract_access_policy(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        metadata = metadata or {}
        access = metadata.get("access") or {}
        if not isinstance(access, dict):
            access = {}
        return {
            "visibility": str(access.get("visibility") or metadata.get("visibility") or settings.knowledge_default_visibility).lower(),
            "allowed_users": list(access.get("allowed_users") or metadata.get("allowed_users") or []),
            "allowed_roles": list(access.get("allowed_roles") or metadata.get("allowed_roles") or []),
            "allowed_departments": list(access.get("allowed_departments") or metadata.get("allowed_departments") or []),
            "min_clearance": int(access.get("min_clearance") or metadata.get("min_clearance") or settings.knowledge_default_min_clearance),
        }

    def evaluate(self, metadata: dict[str, Any] | None, access_context: AccessContext | None) -> AccessDecision:
        policy = self._extract_access_policy(metadata)
        visibility = policy["visibility"]

        if visibility == "public":
            return AccessDecision(allowed=True, visibility=visibility, reason="public")

        if access_context is None:
            return AccessDecision(allowed=False, visibility=visibility, reason="missing_access_context")

        if access_context.clearance_level < policy["min_clearance"]:
            return AccessDecision(allowed=False, visibility=visibility, reason="insufficient_clearance")

        allowed_users = {str(item) for item in policy["allowed_users"] if str(item).strip()}
        if allowed_users and access_context.user_id not in allowed_users:
            return AccessDecision(allowed=False, visibility=visibility, reason="user_not_allowed")

        allowed_roles = {str(item) for item in policy["allowed_roles"] if str(item).strip()}
        if allowed_roles and not (allowed_roles & set(access_context.roles)):
            return AccessDecision(allowed=False, visibility=visibility, reason="role_not_allowed")

        allowed_departments = {str(item) for item in policy["allowed_departments"] if str(item).strip()}
        if allowed_departments and not (allowed_departments & set(access_context.departments)):
            return AccessDecision(allowed=False, visibility=visibility, reason="department_not_allowed")

        if visibility in {"internal", "restricted", "confidential", "private"}:
            return AccessDecision(allowed=True, visibility=visibility, reason="allowed")

        return AccessDecision(allowed=False, visibility=visibility, reason="invalid_visibility")

    def can_access(self, metadata: dict[str, Any] | None, access_context: AccessContext | None) -> bool:
        return self.evaluate(metadata, access_context).allowed

    def filter_rows(self, rows: list[dict], access_context: AccessContext | None) -> list[dict]:
        from app.observability.metrics import observability_manager

        result = []
        for row in rows:
            decision = self.evaluate(row.get("metadata", {}), access_context)
            observability_manager.record_acl_check(
                allowed=decision.allowed,
                stage="vector_search",
                visibility=decision.visibility,
                reason=decision.reason,
            )
            if decision.allowed:
                result.append(row)
        return result

    def filter_documents(self, rows: list[Any], access_context: AccessContext | None) -> list[Any]:
        from app.observability.metrics import observability_manager

        result = []
        for row in rows:
            metadata = getattr(row, "metadata_json", {}) or {}
            decision = self.evaluate(metadata, access_context)
            observability_manager.record_acl_check(
                allowed=decision.allowed,
                stage="list_topics",
                visibility=decision.visibility,
                reason=decision.reason,
            )
            if decision.allowed:
                result.append(row)
        return result


knowledge_access_controller = KnowledgeAccessController()
