from pathlib import Path

import yaml

from app.core.config import settings


class MarkdownLoader:
    def _split_front_matter(self, text: str) -> tuple[dict, str]:
        if not text.startswith("---\n"):
            return {}, text
        parts = text.split("\n---\n", 1)
        if len(parts) != 2:
            return {}, text
        front_matter = yaml.safe_load(parts[0][4:]) or {}
        if not isinstance(front_matter, dict):
            front_matter = {}
        return front_matter, parts[1]

    def load_directory(self, base_path: str) -> list[dict]:
        base = Path(base_path)
        docs = []

        for path in sorted(base.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            front_matter, content = self._split_front_matter(text)
            access = front_matter.get("access") or {
                "visibility": settings.knowledge_default_visibility,
                "min_clearance": settings.knowledge_default_min_clearance,
            }
            docs.append({
                "source_path": str(path),
                "file_name": path.name,
                "title": front_matter.get("title") or path.stem,
                "content": content,
                "metadata": {
                    **{key: value for key, value in front_matter.items() if key != "title"},
                    "source_file": path.name,
                    "source_path": str(path),
                    "doc_type": "markdown",
                    "language": front_matter.get("language", "zh"),
                    "access": access,
                }
            })

        return docs


markdown_loader = MarkdownLoader()
