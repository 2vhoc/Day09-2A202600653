from __future__ import annotations

import re


HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$")


def parse_policy_markdown(markdown_text: str) -> list[dict]:
    chunks: list[dict] = []
    current_h2: str | None = None
    current_h3: str | None = None
    current_content: list[str] = []

    def flush_chunk() -> None:
        if not current_h2 or not current_h3:
            return

        content = "\n".join(current_content).strip()
        if not content:
            return

        rendered_text = "\n\n".join(
            [
                f"## {current_h2}",
                f"### {current_h3}",
                content,
            ]
        )
        chunks.append(
            {
                "section_h2": current_h2,
                "section_h3": current_h3,
                "citation": f"{current_h2} > {current_h3}",
                "content": content,
                "rendered_text": rendered_text,
            }
        )

    for line in markdown_text.splitlines():
        heading_match = HEADING_RE.match(line)
        if heading_match:
            marker, title = heading_match.groups()
            title = title.strip()

            if marker == "##":
                flush_chunk()
                current_h2 = title
                current_h3 = None
                current_content = []
                continue

            if marker == "###":
                flush_chunk()
                current_h3 = title
                current_content = []
                continue

        if current_h2 and current_h3:
            current_content.append(line)

    flush_chunk()
    return chunks
