from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class PromptVersion:
	name: str
	version: str
	template: str
	description: str = ""
	metadata: dict[str, Any] = field(default_factory=dict)


class PromptRegistry:
	def __init__(self):
		self._lock = RLock()
		self._prompts: dict[str, dict[str, PromptVersion]] = {}
		self._active_versions: dict[str, str] = {}
		self._bootstrap_defaults()

	def register(
		self,
		*,
		name: str,
		version: str,
		template: str,
		description: str = "",
		metadata: dict[str, Any] | None = None,
		make_active: bool = False,
	) -> PromptVersion:
		prompt = PromptVersion(
			name=name,
			version=version,
			template=template.strip(),
			description=description,
			metadata=metadata or {},
		)
		with self._lock:
			self._prompts.setdefault(name, {})[version] = prompt
			if make_active or name not in self._active_versions:
				self._active_versions[name] = version
		return prompt

	def activate(self, name: str, version: str) -> PromptVersion:
		prompt = self.get(name=name, version=version)
		with self._lock:
			self._active_versions[name] = version
		return prompt

	def get(self, *, name: str, version: str | None = None) -> PromptVersion:
		with self._lock:
			versions = self._prompts.get(name)
			if not versions:
				raise KeyError(f"unknown prompt: {name}")
			resolved_version = version or self._active_versions.get(name)
			if not resolved_version or resolved_version not in versions:
				raise KeyError(f"unknown prompt version: {name}@{resolved_version}")
			return versions[resolved_version]

	def render(self, name: str, version: str | None = None, **kwargs: Any) -> str:
		prompt = self.get(name=name, version=version)
		return prompt.template.format(**kwargs)

	def active_versions(self) -> dict[str, str]:
		with self._lock:
			return dict(self._active_versions)

	def list_prompts(self) -> list[dict[str, Any]]:
		with self._lock:
			payload: list[dict[str, Any]] = []
			for name in sorted(self._prompts):
				versions = self._prompts[name]
				payload.append({
					"name": name,
					"active_version": self._active_versions.get(name),
					"versions": [
						{
							"version": version,
							"description": prompt.description,
							"metadata": prompt.metadata,
							"template": prompt.template,
						}
						for version, prompt in sorted(versions.items())
					],
				})
			return payload

	def _bootstrap_defaults(self):
		self.register(
			name="agent_system",
			version="v1",
			description="基础 Agent 系统提示词，强调工具优先和澄清。",
			template="""
你是一个用于面试准备的知识 Agent。
你的目标不是一次性瞎答，而是优先调用工具查知识，再结合观察结果给出结构化回答。

工作规则：
1. 优先使用工具收集信息，尤其是 search_knowledge 和 read_topic。
2. 当用户问题太泛时，不要乱答，直接输出：CLARIFICATION: <你的追问>
3. 当信息已经足够时，直接给出最终答案，不要再输出 JSON。
4. 如果要调用工具，可以附带一句简短 planning note，说明你下一步想做什么。
5. 最终答案要尽量包含：核心结论、回答思路、易错点、可追问。
6. 如果引用了某个主题，请把主题名自然带进答案里。
7. 如果治理层提示当前工具或模型繁忙，请优先利用已拿到的观察结果继续回答，必要时明确说明降级。
""",
			metadata={"owner": "agent", "category": "system"},
			make_active=True,
		)
		self.register(
			name="agent_user",
			version="v1",
			description="Agent 的用户侧输入模板。",
			template="""
历史会话：
{context}

当前问题：{question}

请在需要时自主调用工具。
""",
			metadata={"owner": "agent", "category": "user"},
			make_active=True,
		)
		self.register(
			name="fallback_summary_system",
			version="v1",
			description="Agent 降级总结提示词。",
			template="你是面试知识点总结助手。请输出核心结论、回答思路、易错点、可追问。",
			metadata={"owner": "agent", "category": "fallback"},
			make_active=True,
		)
		self.register(
			name="fallback_summary_user",
			version="v1",
			description="Agent 降级总结的用户模板。",
			template="""
历史会话：
{context}

用户问题：{question}

Agent 已收集资料：
{observations}
""",
			metadata={"owner": "agent", "category": "fallback"},
			make_active=True,
		)
		self.register(
			name="route_classifier_system",
			version="v1",
			description="意图路由分类提示词。",
			template="""
你是企业知识问答系统的意图分类器。
请根据用户问题和上下文，将问题分类到以下类别之一：
1. knowledge_qa: 企业制度、知识文档、FAQ类问题
2. ticket_query: 工单状态、处理进度查询
3. org_query: 组织架构、部门负责人、联系人查询
4. workflow_query: 审批流程、流程节点、办理步骤查询
5. clarification: 用户信息不足，必须先追问再继续

如果问题缺少关键参数，例如用户问“我的工单怎么样了”但没有工单号，或者“审批到哪一步了”但没有业务单号，可以返回 clarification。

输出必须是JSON，格式如下：{"route":"knowledge_qa","reason":"...","missing_slots":["ticket_id"]}
""",
			metadata={"owner": "router", "category": "system"},
			make_active=True,
		)


prompt_registry = PromptRegistry()


