import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class SQLiteStateStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parse_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS doc_status (
                    doc_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS triples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    page_idx INTEGER,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    UNIQUE(doc_id, source_id, subject, predicate, object)
                )
                """
            )

    def list_doc_statuses(self, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT doc_id, payload
            FROM doc_status
            ORDER BY rowid DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        items: List[Dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload"])
            payload["doc_id"] = row["doc_id"]
            items.append(payload)
        return items

    def get_parse_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT payload FROM parse_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    def upsert_parse_cache(self, cache_key: str, payload: Dict[str, Any]) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO parse_cache(cache_key, payload)
                VALUES(?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET payload=excluded.payload
                """,
                (cache_key, json.dumps(payload, ensure_ascii=False)),
            )

    def get_doc_status(self, doc_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT payload FROM doc_status WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    def upsert_doc_status(self, doc_id: str, payload: Dict[str, Any]) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO doc_status(doc_id, payload)
                VALUES(?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET payload=excluded.payload
                """,
                (doc_id, json.dumps(payload, ensure_ascii=False)),
            )

    def add_triples(
        self,
        doc_id: str,
        source_id: str,
        source_type: str,
        page_idx: Optional[int],
        triples: Iterable[Dict[str, str]],
    ) -> None:
        with self._lock, self.conn:
            for triple in triples:
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO triples(
                        doc_id, source_id, source_type, page_idx,
                        subject, predicate, object
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        source_id,
                        source_type,
                        page_idx,
                        str(triple.get("subject", "")).strip(),
                        str(triple.get("predicate", "")).strip(),
                        str(triple.get("object", "")).strip(),
                    ),
                )

    def search_triples(
        self,
        query: str,
        doc_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", query) if len(t) > 1]
        if not tokens:
            return []

        where = []
        params: List[Any] = []

        if doc_id:
            where.append("doc_id = ?")
            params.append(doc_id)

        token_conditions = []
        for token in tokens:
            token_conditions.append(
                "(lower(subject) LIKE ? OR lower(predicate) LIKE ? OR lower(object) LIKE ?)"
            )
            params.extend([f"%{token}%", f"%{token}%", f"%{token}%"])

        where.append("(" + " OR ".join(token_conditions) + ")")
        sql = f"""
            SELECT doc_id, source_id, source_type, page_idx, subject, predicate, object
            FROM triples
            WHERE {' AND '.join(where)}
            ORDER BY id DESC
            LIMIT ?
        """
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]