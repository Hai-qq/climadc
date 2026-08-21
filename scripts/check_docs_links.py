from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _markdown_files() -> list[Path]:
    roots = [ROOT / "README.md", ROOT / "README.zh-CN.md"]
    roots.extend((ROOT / "docs").rglob("*.md"))
    return sorted(path for path in roots if "superpowers" not in path.parts)


def main() -> int:
    failures: list[str] = []
    for document in _markdown_files():
        text = document.read_text(encoding="utf-8")
        for match in LINK.finditer(text):
            raw = match.group(1).strip().strip("<>")
            if not raw or raw.startswith(("#", "http://", "https://", "mailto:")):
                continue
            path_text = unquote(raw.split("#", 1)[0])
            if not path_text:
                continue
            candidate = (document.parent / path_text).resolve()
            try:
                candidate.relative_to(ROOT)
            except ValueError:
                failures.append(f"{document.relative_to(ROOT)}: link escapes repository: {raw}")
                continue
            if not candidate.exists():
                failures.append(f"{document.relative_to(ROOT)}: missing local target: {raw}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
