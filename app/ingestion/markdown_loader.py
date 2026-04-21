from pathlib import Path


class MarkdownLoader:
    def load_directory(self, base_path: str) -> list[dict]:
        base = Path(base_path)
        docs = []

        for path in sorted(base.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            docs.append({
                "source_path": str(path),
                "file_name": path.name,
                "title": path.stem,
                "content": text,
                "metadata": {
                    "source_file": path.name,
                    "source_path": str(path),
                    "doc_type": "markdown",
                    "language": "zh"
                }
            })

        return docs


markdown_loader = MarkdownLoader()
