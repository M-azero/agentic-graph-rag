"""JSON / JSONL loader. Records become "key: value" sentence blocks.

Raw JSON embeds badly: braces, quotes and commas take most of the token budget
and carry none of the meaning, and a chunk that starts mid-object tells a reader
nothing about what they are looking at. So a record is flattened the way
`CSVLoader` flattens a row — `user.name: Ada; user.role: admin` — with nested
paths joined by dots, so the structure survives in the label instead of in
punctuation.

A top-level array is treated as a table of records; any other document is one
record. `.jsonl`/`.ndjson` are read a line at a time, which is the only way to
read the truncated exports that format is usually produced by.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from graphrag.core.errors import IngestionError
from graphrag.core.logging import get_logger
from graphrag.core.types import Document
from graphrag.ingestion.loaders.base import Loader

log = get_logger(__name__)

_MAX_RECORDS = 5000  # matches CSVLoader: a data dump is not one document
_LINE_FORMATS = (".jsonl", ".ndjson")


def _flatten(value: Any, prefix: str = "") -> Iterator[tuple[str, str]]:
    """Yield `(dotted path, scalar)` for every leaf under `value`.

    Empty strings and nulls are dropped rather than rendered as `key: None` —
    a sparse record would otherwise spend most of its chunk on absent fields.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _flatten(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _flatten(item, f"{prefix}[{i}]" if prefix else f"[{i}]")
    elif value is not None and value != "":
        yield prefix or "value", str(value)


class JSONLoader(Loader):
    suffixes = (".json", *_LINE_FORMATS)

    def load(self, path: Path) -> Document:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() in _LINE_FORMATS:
            records, skipped = self._read_lines(text)
        else:
            records, skipped = self._read_document(text, path), 0

        blocks = [
            block
            for block in ("; ".join(f"{k}: {v}" for k, v in _flatten(r))
                          for r in records[:_MAX_RECORDS])
            if block
        ]
        if skipped:
            log.warning("json_lines_skipped", source=path.name, lines=skipped)
        return Document(
            source=str(path),
            content="\n\n".join(blocks),
            metadata={"type": "json", "records": len(blocks), "skipped": skipped},
        )

    @staticmethod
    def _read_lines(text: str) -> tuple[list[Any], int]:
        """One record per line. A bad line is counted, not fatal.

        The whole point of the format is that it streams, so a file cut off
        mid-write is normal and losing the 4,999 good records to the last
        broken one would be the wrong trade. The count reaches the log and the
        document's metadata so the loss is never silent.
        """
        records: list[Any] = []
        skipped = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
        return records, skipped

    @staticmethod
    def _read_document(text: str, path: Path) -> list[Any]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise IngestionError(f"{path.name} is not valid JSON: {exc}") from exc
        return data if isinstance(data, list) else [data]
