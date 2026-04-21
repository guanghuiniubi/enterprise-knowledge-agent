import re


class MarkdownChunker:
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def split(self, text: str, chunk_size: int = 500, overlap: int = 80) -> list[dict]:
        sections = self._split_by_headers(text)
        chunks = []

        for section in sections:
            section_text = section["content"].strip()
            if not section_text:
                continue

            start = 0
            index = 0
            while start < len(section_text):
                end = min(len(section_text), start + chunk_size)
                chunk_text = section_text[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "header_path": section["header_path"],
                        "content": chunk_text,
                        "token_count": len(chunk_text),
                        "chunk_local_index": index
                    })
                if end >= len(section_text):
                    break
                start = max(0, end - overlap)
                index += 1

        return chunks

    def _split_by_headers(self, text: str) -> list[dict]:
        matches = list(self.HEADER_PATTERN.finditer(text))
        if not matches:
            return [{"header_path": None, "content": text}]

        sections = []
        for i, match in enumerate(matches):
            header = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end]
            sections.append({
                "header_path": header,
                "content": content
            })

        return sections


markdown_chunker = MarkdownChunker()
