"""SQLite state: every user table's rows, in a canonical order."""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Iterable, Optional

from .base import Snapshot


class SqliteObserver:
    """Snapshots a SQLite file as table to sorted rows.

    Rows are sorted because insertion order is not an effect, and they are
    kept as a list rather than a set so that a duplicated INSERT shows up as
    two identical rows, which is the signal the audit is looking for.

    The file is opened read only, through a URI, so snapshotting cannot
    itself create the database or take a write lock on a server that is still
    running. A missing file snapshots as absent rather than raising, because
    a tool creating the database *is* the effect under test.
    """

    def __init__(
        self,
        path: str,
        *,
        name: str = "sqlite",
        include_schema: bool = True,
        tables: Optional[Iterable[str]] = None,
        max_rows: int = 20000,
    ) -> None:
        self.path = os.path.abspath(path)
        self.name = name
        self.include_schema = include_schema
        self.tables = tuple(tables) if tables else None
        self.max_rows = max_rows

    def snapshot(self) -> Snapshot:
        if not os.path.exists(self.path):
            return {"__present__": False}
        uri = f"file:{self.path}?mode=ro"
        try:
            connection = sqlite3.connect(uri, uri=True, timeout=5)
        except sqlite3.Error as exc:
            return {"__present__": True, "__error__": str(exc)}
        try:
            return self._read(connection)
        except sqlite3.Error as exc:
            return {"__present__": True, "__error__": str(exc)}
        finally:
            connection.close()

    def _read(self, connection: sqlite3.Connection) -> Snapshot:
        state: Snapshot = {"__present__": True}
        rows = connection.execute(
            "select name, sql from sqlite_master where type = 'table' "
            "and name not like 'sqlite_%' order by name"
        ).fetchall()
        for table, create_sql in rows:
            if self.tables is not None and table not in self.tables:
                continue
            entry: dict[str, Any] = {}
            if self.include_schema:
                entry["schema"] = " ".join((create_sql or "").split())
            quoted = table.replace('"', '""')
            fetched = connection.execute(
                f'select * from "{quoted}" limit {self.max_rows + 1}'
            ).fetchall()
            if len(fetched) > self.max_rows:
                entry["rows"] = f"over {self.max_rows}"
                entry["count"] = len(fetched)
            else:
                entry["rows"] = sorted(
                    [list(row) for row in fetched],
                    key=lambda row: [(value is None, str(value)) for value in row],
                )
            state[table] = entry
        return state
