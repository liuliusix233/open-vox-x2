"""todo_goals.py — Global Todo & Active Goals store on vox_unified.db.

Provides persistent cross-session todos and active user goals (with optional destination
field ready for Phase 3 AR navigation) plus native OpenAI Tool Calling schemas.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

TZ_SHANGHAI = timezone(timedelta(hours=8))

TODO_GOAL_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS todos (
    todo_id     TEXT PRIMARY KEY,
    client_id   TEXT NOT NULL DEFAULT 'voxx2',
    title       TEXT NOT NULL,
    note        TEXT NOT NULL DEFAULT '',
    priority    TEXT NOT NULL DEFAULT 'normal' CHECK(priority IN ('low','normal','high')),
    status      TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','done','cancelled')),
    due_ts      REAL,
    created_ts  REAL NOT NULL,
    updated_ts  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_todos_client_status ON todos(client_id, status, due_ts);

CREATE TABLE IF NOT EXISTS active_goals (
    goal_id     TEXT PRIMARY KEY,
    client_id   TEXT NOT NULL DEFAULT 'voxx2',
    title       TEXT NOT NULL,
    destination TEXT NOT NULL DEFAULT '',
    status_note TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','completed','cancelled')),
    created_ts  REAL NOT NULL,
    updated_ts  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_goals_client_status ON active_goals(client_id, status, updated_ts DESC);
"""

TODO_GOAL_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "manage_todo",
            "description": "管理用户的全局待办事项（新增、列出、标记完成或取消）。随口提到待会儿要做的事或明确设待办时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "list", "done", "cancel"],
                        "description": "操作类型",
                    },
                    "title": {"type": "string", "description": "待办标题（add 时必填；done/cancel 可按标题模糊匹配）"},
                    "due_iso": {"type": "string", "description": "可选截止北京时间，格式 YYYY-MM-DD HH:MM"},
                    "priority": {"type": "string", "enum": ["low", "normal", "high"], "description": "优先级，默认 normal"},
                    "note": {"type": "string", "description": "可选补充说明"},
                    "todo_id": {"type": "string", "description": "可选精确待办 ID（用于 done/cancel）"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "manage_goal",
            "description": "管理用户当前正在进行的活跃目标/行程意图（如准备去图书馆、出门买东西），并记录目的地供后续导航或自动完成。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["set", "update", "complete", "cancel", "list"],
                        "description": "操作类型",
                    },
                    "title": {"type": "string", "description": "目标简述，如 去图书馆、回家、买咖啡"},
                    "destination": {"type": "string", "description": "可选具体地点名（用于后续自动导航）"},
                    "status_note": {"type": "string", "description": "当前进展备注，如 正在穿外套、已进馆"},
                    "goal_id": {"type": "string", "description": "可选目标 ID"},
                },
                "required": ["action"],
            },
        },
    },
]


def _parse_iso_bj(val: str | None) -> float | None:
    if not val:
        return None
    s = val.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=TZ_SHANGHAI).timestamp()
        except ValueError:
            continue
    return None


def _fmt_ts(ts: float | None) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts, tz=TZ_SHANGHAI).strftime("%m-%d %H:%M")


class TodoGoalStore:
    """Thread-safe global todo and active goal store sharing vox_unified.db."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(TODO_GOAL_SCHEMA_SQL)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- Todos ----

    def add_todo(
        self,
        client_id: str,
        title: str,
        due_iso: str | None = None,
        priority: str = "normal",
        note: str = "",
    ) -> dict[str, Any]:
        now = time.time()
        tid = f"td-{uuid.uuid4().hex[:6]}"
        due_ts = _parse_iso_bj(due_iso)
        prio = priority if priority in ("low", "normal", "high") else "normal"
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO todos(todo_id, client_id, title, note, priority, status, due_ts, created_ts, updated_ts)
                VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?)
                """,
                (tid, client_id, title.strip(), note.strip(), prio, due_ts, now, now),
            )
            self._conn.commit()
        return {"ok": True, "todo_id": tid, "title": title.strip(), "due": _fmt_ts(due_ts), "priority": prio}

    def list_todos(self, client_id: str, include_done: bool = False, limit: int = 12) -> list[dict[str, Any]]:
        with self._lock:
            sql = """
                SELECT todo_id, title, note, priority, status, due_ts
                FROM todos WHERE client_id = ?
            """
            params: list[Any] = [client_id]
            if not include_done:
                sql += " AND status = 'open'"
            sql += " ORDER BY CASE WHEN due_ts IS NULL THEN 1 ELSE 0 END, due_ts ASC, updated_ts DESC LIMIT ?"
            params.append(limit)
            rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "todo_id": r["todo_id"],
                "title": r["title"],
                "note": r["note"],
                "priority": r["priority"],
                "status": r["status"],
                "due": _fmt_ts(r["due_ts"]),
                "due_ts": r["due_ts"],
            }
            for r in rows
        ]

    def update_todo_status(
        self, client_id: str, status: str, todo_id: str = "", title: str = ""
    ) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            if todo_id:
                cur = self._conn.execute(
                    "UPDATE todos SET status=?, updated_ts=? WHERE client_id=? AND todo_id=?",
                    (status, now, client_id, todo_id.strip()),
                )
            elif title.strip():
                cur = self._conn.execute(
                    """
                    UPDATE todos SET status=?, updated_ts=?
                    WHERE todo_id = (
                        SELECT todo_id FROM todos
                        WHERE client_id=? AND status='open' AND title LIKE ?
                        ORDER BY updated_ts DESC LIMIT 1
                    )
                    """,
                    (status, now, client_id, f"%{title.strip()}%"),
                )
            else:
                return {"ok": False, "reason": "需要 todo_id 或 title"}
            self._conn.commit()
            return {"ok": cur.rowcount > 0, "updated": cur.rowcount, "status": status}

    def get_due_todos(self, client_id: str, within_min: float = 20.0) -> list[dict[str, Any]]:
        cutoff = time.time() + within_min * 60.0
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT todo_id, title, priority, due_ts FROM todos
                WHERE client_id=? AND status='open' AND due_ts IS NOT NULL AND due_ts <= ?
                ORDER BY due_ts ASC LIMIT 5
                """,
                (client_id, cutoff),
            ).fetchall()
        return [
            {"todo_id": r["todo_id"], "title": r["title"], "priority": r["priority"], "due": _fmt_ts(r["due_ts"])}
            for r in rows
        ]

    # ---- Active Goals ----

    def set_or_update_goal(
        self,
        client_id: str,
        title: str,
        destination: str = "",
        status_note: str = "",
        goal_id: str = "",
    ) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            if goal_id:
                cur = self._conn.execute(
                    """
                    UPDATE active_goals
                    SET title=COALESCE(NULLIF(?,''), title),
                        destination=COALESCE(NULLIF(?,''), destination),
                        status_note=COALESCE(NULLIF(?,''), status_note),
                        updated_ts=?
                    WHERE client_id=? AND goal_id=?
                    """,
                    (title.strip(), destination.strip(), status_note.strip(), now, client_id, goal_id.strip()),
                )
                self._conn.commit()
                if cur.rowcount > 0:
                    return {"ok": True, "goal_id": goal_id.strip(), "updated": True}

            gid = f"gl-{uuid.uuid4().hex[:6]}"
            self._conn.execute(
                """
                INSERT INTO active_goals(goal_id, client_id, title, destination, status_note, status, created_ts, updated_ts)
                VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
                """,
                (gid, client_id, title.strip(), destination.strip(), status_note.strip(), now, now),
            )
            self._conn.commit()
        return {"ok": True, "goal_id": gid, "title": title.strip(), "destination": destination.strip()}

    def list_goals(self, client_id: str, limit: int = 6) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT goal_id, title, destination, status_note, updated_ts
                FROM active_goals WHERE client_id=? AND status='active'
                ORDER BY updated_ts DESC LIMIT ?
                """,
                (client_id, limit),
            ).fetchall()
        return [
            {
                "goal_id": r["goal_id"],
                "title": r["title"],
                "destination": r["destination"],
                "status_note": r["status_note"],
                "updated": _fmt_ts(r["updated_ts"]),
            }
            for r in rows
        ]

    def close_goal(self, client_id: str, status: str, goal_id: str = "", title: str = "") -> dict[str, Any]:
        now = time.time()
        with self._lock:
            if goal_id:
                cur = self._conn.execute(
                    "UPDATE active_goals SET status=?, updated_ts=? WHERE client_id=? AND goal_id=?",
                    (status, now, client_id, goal_id.strip()),
                )
            elif title.strip():
                cur = self._conn.execute(
                    """
                    UPDATE active_goals SET status=?, updated_ts=?
                    WHERE goal_id = (
                        SELECT goal_id FROM active_goals
                        WHERE client_id=? AND status='active' AND title LIKE ?
                        ORDER BY updated_ts DESC LIMIT 1
                    )
                    """,
                    (status, now, client_id, f"%{title.strip()}%"),
                )
            else:
                return {"ok": False, "reason": "需要 goal_id 或 title"}
            self._conn.commit()
            return {"ok": cur.rowcount > 0, "updated": cur.rowcount, "status": status}

    # ---- Dynamic context formatting & tool dispatch ----

    def format_context_block(self, client_id: str) -> str:
        goals = self.list_goals(client_id, limit=3)
        todos = self.list_todos(client_id, include_done=False, limit=5)
        if not goals and not todos:
            return ""
        lines: list[str] = []
        if goals:
            lines.append("[ACTIVE_GOALS]")
            for g in goals:
                dst = f" -> 目的地:{g['destination']}" if g["destination"] else ""
                note = f" ({g['status_note']})" if g["status_note"] else ""
                lines.append(f"- [{g['goal_id']}] {g['title']}{dst}{note}")
        if todos:
            lines.append("[OPEN_TODOS]")
            for t in todos:
                due = f" 截止:{t['due']}" if t["due"] else ""
                prio = " [高优]" if t["priority"] == "high" else ""
                lines.append(f"- [{t['todo_id']}] {t['title']}{due}{prio}")
        return "\n".join(lines)

    def exec_tool(self, client_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "manage_todo":
            act = str(args.get("action", "list")).lower()
            if act == "add":
                title = str(args.get("title", "")).strip()
                if not title:
                    return {"ok": False, "reason": "缺少待办标题"}
                return self.add_todo(
                    client_id=client_id,
                    title=title,
                    due_iso=args.get("due_iso"),
                    priority=str(args.get("priority", "normal")),
                    note=str(args.get("note", "")),
                )
            if act == "list":
                items = self.list_todos(client_id)
                return {"ok": True, "count": len(items), "todos": items}
            if act in ("done", "cancel"):
                st = "done" if act == "done" else "cancelled"
                return self.update_todo_status(
                    client_id=client_id,
                    status=st,
                    todo_id=str(args.get("todo_id", "")),
                    title=str(args.get("title", "")),
                )
            return {"ok": False, "reason": f"unknown action: {act}"}

        if name == "manage_goal":
            act = str(args.get("action", "list")).lower()
            if act in ("set", "update"):
                title = str(args.get("title", "")).strip()
                if not title and not args.get("goal_id"):
                    return {"ok": False, "reason": "缺少目标标题"}
                return self.set_or_update_goal(
                    client_id=client_id,
                    title=title,
                    destination=str(args.get("destination", "")),
                    status_note=str(args.get("status_note", "")),
                    goal_id=str(args.get("goal_id", "")),
                )
            if act == "list":
                items = self.list_goals(client_id)
                return {"ok": True, "count": len(items), "goals": items}
            if act in ("complete", "cancel"):
                st = "completed" if act == "complete" else "cancelled"
                return self.close_goal(
                    client_id=client_id,
                    status=st,
                    goal_id=str(args.get("goal_id", "")),
                    title=str(args.get("title", "")),
                )
            return {"ok": False, "reason": f"unknown action: {act}"}

        return {"ok": False, "reason": f"unknown tool: {name}"}
