from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import yaml

from app.core.config import settings


@dataclass(frozen=True)
class PromptVersion:
	name: str
	version: str
	template: str
	description: str = ""
	metadata: dict[str, Any] = field(default_factory=dict)


class PromptRegistry:
	def __init__(self, store_path: str | None = None):
		self._lock = RLock()
		self._prompts: dict[str, dict[str, PromptVersion]] = {}
		self._active_versions: dict[str, str] = {}
		self._store_path = Path(store_path).resolve() if store_path else None
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
		persist: bool = True,
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
			if persist:
				self._persist_locked()
		return prompt

	def activate(self, name: str, version: str) -> PromptVersion:
		prompt = self.get(name=name, version=version)
		with self._lock:
			self._active_versions[name] = version
			self._persist_locked()
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

	def reload(self):
		with self._lock:
			self._prompts = {}
			self._active_versions = {}
			self._bootstrap_defaults()

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
		if self._store_path and self._store_path.is_file():
			self._load_from_yaml(self._store_path)
			return

		default_path = Path(__file__).with_name("prompts.yaml")
		self._load_from_yaml(default_path)
		self._persist_locked()

	def _load_from_yaml(self, path: Path):
		payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
		active_versions = payload.get("active_versions") or {}
		prompts = payload.get("prompts") or {}

		for name, versions in prompts.items():
			for version, config in (versions or {}).items():
				prompt = PromptVersion(
					name=name,
					version=version,
					template=str((config or {}).get("template") or "").strip(),
					description=str((config or {}).get("description") or ""),
					metadata=dict((config or {}).get("metadata") or {}),
				)
				self._prompts.setdefault(name, {})[version] = prompt

		for name, version in active_versions.items():
			if name in self._prompts and version in self._prompts[name]:
				self._active_versions[name] = version

		for name, versions in self._prompts.items():
			if name not in self._active_versions and versions:
				self._active_versions[name] = sorted(versions)[0]

	def _persist_locked(self):
		if not self._store_path:
			return
		self._store_path.parent.mkdir(parents=True, exist_ok=True)
		payload = {
			"active_versions": self._active_versions,
			"prompts": {
				name: {
					version: {
						"description": prompt.description,
						"metadata": prompt.metadata,
						"template": prompt.template,
					}
					for version, prompt in sorted(versions.items())
				}
				for name, versions in sorted(self._prompts.items())
			},
		}
		self._store_path.write_text(
			yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
			encoding="utf-8",
		)


prompt_registry = PromptRegistry(settings.prompt_registry_path)


