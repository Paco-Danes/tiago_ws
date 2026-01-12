import json
import os
import re
import sqlite3
from typing import List, Sequence, Tuple, Optional
from pydantic import BaseModel

import rospy

from .speech import tiago_say

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    OpenAI = None
    _HAS_OPENAI = False

_client = None


def _get_openai_client() -> Optional["OpenAI"]:
    """
    Lazy singleton OpenAI client.
    Uses OPENAI_API_KEY from env (standard SDK behavior).
    """
    global _client
    if not _HAS_OPENAI:
        return None
    if _client is not None:
        return _client

    if not os.environ.get("OPENAI_API_KEY"):
        rospy.logwarn("OPENAI_API_KEY not set; supervisor SQL will fall back to a canned query.")
        return None

    try:
        _client = OpenAI()
        return _client
    except Exception as e:
        rospy.logwarn("Failed to init OpenAI client; supervisor SQL will fall back. err=%s", e)
        return None


_SQL_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sql": {
            "type": "string",
            "description": "A single SQLite SELECT query answering the supervisor question."
        }
    },
    "required": ["sql"],
}


_FORBIDDEN_SQL_TOKENS = (
    "insert", "update", "delete", "drop", "alter", "create", "replace",
    "pragma", "attach", "detach", "vacuum", "reindex"
)


def _fallback_message_query(msg: str) -> str:
    # Always safe: SELECT with a literal string.
    safe = msg.replace("'", "''")
    return f"SELECT '{safe}' AS message LIMIT 1"


def _sanitize_and_validate_sql(sql: str) -> str:
    """
    Ensure:
      - single statement
      - SELECT-only (read-only)
      - no sqlite_master access
      - no forbidden tokens
      - no multi-statement
      - reasonably bounded output
    """
    if not sql:
        raise ValueError("Empty SQL")

    s = sql.strip()

    # Remove trailing semicolons; disallow internal semicolons (multi-statement).
    s = s.rstrip(";").strip()
    if ";" in s:
        raise ValueError("Multi-statement SQL is not allowed")

    # Must start with SELECT
    if not re.match(r"(?is)^\s*select\b", s):
        raise ValueError("Only SELECT queries are allowed")

    # Block dangerous tokens even if it starts with SELECT.
    for tok in _FORBIDDEN_SQL_TOKENS:
        if re.search(rf"(?is)\b{re.escape(tok)}\b", s):
            raise ValueError(f"Forbidden token in SQL: {tok}")

    # Block sqlite internal tables
    if re.search(r"(?is)\bsqlite_master\b", s):
        raise ValueError("Access to sqlite_master is not allowed")

    # Bound output if model forgets LIMIT (don’t override if already present)
    if not re.search(r"(?is)\blimit\b", s):
        s = f"{s} LIMIT 50"

    # Extra guard: keep it from being huge
    if len(s) > 2000:
        raise ValueError("SQL too long")

    return s


def _run_select(conn: sqlite3.Connection, sql: str, max_rows: int = 50) -> Tuple[List[str], List[Tuple]]:
    cur = conn.execute(sql)
    cols: List[str] = []
    if cur.description:
        cols = [d[0] for d in cur.description]
    rows = cur.fetchmany(max_rows + 1)
    if len(rows) > max_rows:
        rows = rows[:max_rows]
    return cols, rows


def _print_table(cols: Sequence[str], rows: Sequence[Sequence]) -> None:
    if not cols:
        print("(No columns returned)")
        return
    if not rows:
        print("(No rows)")
        return

    # compute widths
    str_rows = [[("" if v is None else str(v)) for v in r] for r in rows]
    widths = [len(c) for c in cols]
    for r in str_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(r: Sequence[str]) -> str:
        return " | ".join(r[i].ljust(widths[i]) for i in range(len(cols)))

    header = fmt_row(list(cols))
    sep = "-+-".join("-" * w for w in widths)

    print(header)
    print(sep)
    for r in str_rows:
        print(fmt_row(r))


def nl_question_to_sql(question: str, patient_name: str) -> str:
    """
    Uses OpenAI SDK `responses.parse()` with a Pydantic schema.
    This avoids manual JSON parsing and avoids the missing text.format.name issue.
    """
    client = _get_openai_client()
    if client is None:
        return _fallback_message_query("LLM disabled (no OPENAI_API_KEY).")

    model = rospy.get_param("~supervisor_sql_model", "gpt-5-mini")

    class SQLQuery(BaseModel):
        sql: str

    developer_instructions = (
        "You are a strict SQL generator for a healthcare robot.\n"
        "Return ONLY a single SQLite SELECT query.\n"
        "\n"
        "Database: SQLite.\n"
        "You have exactly one table:\n"
        "med_admin(id INTEGER, patient TEXT, timestamp_utc TEXT, outcome TEXT, attempts INTEGER, notes TEXT, second_strategy TEXT)\n"
        "patient: name of the patient \n"
        "timestamp_utc: ISO8601 UTC timestamp of administration attempt\n"
        "outcome: one of 'taken', 'refused', 'aborted_no_person_found', 'aborted_no_recognition'\n"
        "attempts: number of attempts made (1 or 2)\n"
        "second_strategy: one of 'joke', 'concern', or NULL\n"
        "\n"
        "Rules:\n"
        "1) Output must be a SINGLE SQLite SELECT statement.\n"
        "2) Never write INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/PRAGMA/ATTACH/DETACH.\n"
        "3) Prefer filtering by patient when the question is about the current patient.\n"
        f"   Current patient name: {patient_name}\n"
        "4) timestamp_utc is ISO8601 UTC text (e.g., 2026-01-11T10:20:30+00:00).\n"
        "   Use ORDER BY timestamp_utc DESC for most-recent.\n"
        "5) Always include a LIMIT (<= 50).\n"
        "\n"
        "If the question cannot be answered from this table, output:\n"
        "SELECT 'I cannot answer that from the medication administration log.' AS message LIMIT 1\n"
    )

    user_prompt = (
        "Supervisor question (convert to SQL):\n"
        f"{question.strip()}"
    )

    try:
        resp = client.responses.parse(
            model=model,
            input=[
                {"role": "developer", "content": developer_instructions},
                {"role": "user", "content": user_prompt},
            ],
            text_format=SQLQuery,
        )

        parsed = resp.output_parsed
        sql = str(parsed.sql).strip()

        sql = _sanitize_and_validate_sql(sql)
        return sql

    except Exception as e:
        rospy.logwarn("Supervisor SQL generation failed; err=%s", e)
        return _fallback_message_query("Sorry — I couldn't generate a query for that request.")


def supervisor_interaction(conn: sqlite3.Connection, patient_name: str) -> None:
    """
    Called after face recognition returns 'supervisor'.
    Asks for a question, converts NL -> SQL, runs the query, prints result.
    """
    tiago_say("Do you have any questions about the patient?")

    while not rospy.is_shutdown():
        q = input("\nSupervisor question (ENTER to stop): ").strip()
        if not q:
            tiago_say("Okay. If you need anything else, I'm here.")
            return

        sql = nl_question_to_sql(q, patient_name=patient_name)
        print("\nSQL> " + sql)

        try:
            cols, rows = _run_select(conn, sql, max_rows=50)
            _print_table(cols, rows)
        except Exception as e:
            rospy.logwarn("DB query failed; sql=%s err=%s", sql, e)
            tiago_say("Sorry — I couldn't run that query.")
            print(f"(DB error: {e})")
