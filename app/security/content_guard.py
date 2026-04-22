from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.config import settings
from app.security.redaction import redactor


PROMPT_INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"ignore\s+all\s+previous\s+instructions",
        r"ignore\s+the\s+system\s+prompt",
        r"reveal\s+(the\s+)?system\s+prompt",
        r"developer\s+message",
        r"tool\s+schema",
        r"忽略(之前|以上|所有).{0,10}(指令|规则|提示)",
        r"输出(系统|开发者).{0,10}(提示词|prompt)",
        r"你现在不是.{0,10}(助手|agent)",
        r"jailbreak",
    ]
]

PROMPT_LEAKAGE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"system\s+prompt",
        r"developer\s+message",
        r"tool\s+schema",
        r"内部提示词",
        r"系统提示词",
    ]
]


@dataclass
class GuardDecision:
    blocked: bool = False
    sanitized_text: str = ""
    reasons: list[str] = field(default_factory=list)


class ContentGuard:
    def inspect_user_input(self, text: str) -> GuardDecision:
        reasons = [pattern.pattern for pattern in PROMPT_INJECTION_PATTERNS if pattern.search(text or "")]
        sanitized = re.sub(r"<\s*/?\s*(system|assistant|developer)\s*>", "", text or "", flags=re.IGNORECASE)
        blocked = bool(reasons) and settings.security_prompt_injection_mode == "block"
        return GuardDecision(blocked=blocked, sanitized_text=sanitized.strip(), reasons=reasons)

    def filter_output(self, text: str) -> GuardDecision:
        sanitized = redactor.redact_text(text or "") if settings.output_filter_redact_secrets else (text or "")
        reasons: list[str] = []
        if any(pattern.search(sanitized) for pattern in PROMPT_LEAKAGE_PATTERNS):
            reasons.append("prompt_leakage")
            sanitized = "抱歉，我不能暴露系统内部提示词、工具协议或安全策略细节。"
        return GuardDecision(blocked=False, sanitized_text=sanitized, reasons=reasons)


content_guard = ContentGuard()

