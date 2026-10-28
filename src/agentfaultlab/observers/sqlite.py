"""SQLite state: every user table's rows, in a canonical order."""

from __future__ import annotations

import glob
import os
import sqlite3
from typing import Any, Iterable, Optional

from .base import Snapshot


class SqliteObserver:
    """Snapshots a SQLite file as table to sorted rows.

    Rows are sorted because insertion order is not an effect, and they are
    kept as a list rather than a set so that a duplicated INSERT shows up as
    two identical rows, which is the signal the audit is looking for.

    A table's shape comes from `PRAGMA table_info`, not from the text of its
    CREATE statement. The text is not canonical: `create table t (a text)` and
    `CREATE TABLE t (a TEXT)` describe the same table, and an agent asked to
    create a table writes whichever it likes. Comparing the raw SQL made
    identical outcomes read as different states, so the schema is compared as
    a column list instead.

    `ignore_columns` drops columns that carry no effect but change every run:
    a row id, a created_at, a session id. Without it, a server that stamps
    every row with a timestamp makes S1 and S2 differ from every faulted end
    state, so every verdict would read DIVERGED and nothing would be learned.
    Dropping an autoincrement id is safe for the question being asked, because
    rows are compared as a list: a duplicated INSERT still shows up as the
    same row twice. Whatever is dropped is recorded in the snapshot, so a
    result can never quietly depend on an observer that was looking away.

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
        ignore_columns: Optional[Iterable[str]] = None,
        max_rows: int = 20000,
    ) -> None:
        self.path = os.path.abspath(path)
        self.name = name
        self.include_schema = include_schema
        self.tables = tuple(tables) if tables else None
        self.ignore_columns = tuple(
            entry.lower() for entry in (ignore_columns or ())
        )
        self.max_rows = max_rows

    def resolve(self) -> Optional[str]:
        """The database file, resolving a glob in the configured path.

        Some servers keep their store under a directory named for the machine
        or the account that created it. Writing that name into a spec would
        make the spec work on one computer only, which defeats the point of a
        spec, and would record the operator's account name in a public file.
        A single glob segment keeps the spec portable.

        Ambiguity is refused rather than guessed: if a glob matches more than
        one database the snapshot says so and the trial is not silently
        pointed at whichever one sorted first.
        """
        if not any(ch in self.path for ch in "*?["):
            return self.path if os.path.exists(self.path) else None
        matches = sorted(glob.glob(self.path))
        if len(matches) == 1:
            return matches[0]
        self._ambiguous = matches
        return None

    def _ignored(self, table: str, column: str) -> bool:
        """True for a column named either bare or as `table.column`."""
        lowered = column.lower()
        qualified = f"{table.lower()}.{lowered}"
        return lowered in self.ignore_columns or qualified in self.ignore_columns

    def snapshot(self) -> Snapshot:
        self._ambiguous: list[str] = []
        resolved = self.resolve()
        if resolved is None:
            if self._ambiguous:
                # Recorded in the snapshot, so two baselines that each matched
                # a different number of stores compare unequal and the row
                # comes out UNPROVEN instead of quietly wrong.
                return {
                    "__present__": False,
                    "__ambiguous__": [os.path.basename(m) for m in self._ambiguous],
                }
            return {"__present__": False}
        uri = f"file:{resolved}?mode=ro"
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
            "select name from sqlite_master where type = 'table' "
            "and name not like 'sqlite_%' order by name"
        ).fetchall()
        for (table,) in rows:
            if self.tables is not None and table not in self.tables:
                continue
            entry: dict[str, Any] = {}
            columns = self._columns(connection, table)
            kept = [
                index
                for index, column in enumerate(columns)
                if not self._ignored(table, column[0])
            ]
            dropped = [
                column[0] for column in columns if self._ignored(table, column[0])
            ]
            if dropped:
                entry["ignored_columns"] = dropped
            if self.include_schema:
                entry["columns"] = [columns[index] for index in kept]
            quoted = table.replace('"', '""')
            fetched = connection.execute(
                f'select * from "{quoted}" limit {self.max_rows + 1}'
            ).fetchall()
            if len(fetched) > self.max_rows:
                entry["rows"] = f"over {self.max_rows}"
                entry["count"] = len(fetched)
            else:
                entry["rows"] = sorted(
                    [[row[index] for index in kept] for row in fetched],
                    key=lambda row: [(value is None, str(value)) for value in row],
                )
            state[table] = entry
        return state

    @staticmethod
    def _columns(connection: sqlite3.Connection, table: str) -> list[list[Any]]:
        """A table's shape, canonically: name, declared type, not null, key."""
        quoted = table.replace('"', '""')
        info = connection.execute(f'pragma table_info("{quoted}")').fetchall()
        # cid, name, type, notnull, default, pk
        return [
            [row[1], str(row[2] or "").upper(), int(row[3]), int(row[5])]
            for row in info
        ]
