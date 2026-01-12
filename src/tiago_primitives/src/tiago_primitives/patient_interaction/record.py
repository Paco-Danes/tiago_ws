import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Literal
import rospy

SecondStrategy = Optional[Literal["joke", "concern"]]


def _ensure_column(conn: sqlite3.Connection, table: str, col_name: str, col_def_sql: str) -> None:
    """
    Adds a column if missing (simple migration).
    """
    cur = conn.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cur.fetchall()]  # row[1] = name
    if col_name not in cols:
        rospy.loginfo("DB migration: adding column %s to %s", col_name, table)
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def_sql}")
        conn.commit()


def ensure_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS med_admin (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL,
            outcome TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            notes TEXT,
            second_strategy TEXT
        )
        """
    )
    conn.commit()

    # If the table already existed from older runs, ensure the new column exists.
    _ensure_column(conn, "med_admin", "second_strategy", "TEXT")

    return conn


def log_admin(
    conn: sqlite3.Connection,
    patient: str,
    outcome: str,
    attempts: int,
    notes: str = "",
    second_strategy: SecondStrategy = None,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO med_admin (patient, timestamp_utc, outcome, attempts, notes, second_strategy)
        VALUES (?,?,?,?,?,?)
        """,
        (patient, ts, outcome, int(attempts), notes, second_strategy),
    )
    conn.commit()
    rospy.loginfo(
        "Saved record -> patient=%s outcome=%s attempts=%d second_strategy=%s",
        patient, outcome, attempts, str(second_strategy),
    )


def get_best_second_strategy(conn: sqlite3.Connection, patient: str) -> str:
    """
    Returns "joke" or "concern" based on this patient's historical success rate
    *when that strategy was used as the 2nd attempt*.

    Success definition:
      - row.second_strategy = X
      - outcome = 'taken'
      - attempts = 2
    Denominator: all rows where second_strategy = X (i.e. patient reached attempt 2).

    If no history, defaults to 'joke'.
    """
    sql = """
    SELECT
      second_strategy,
      SUM(CASE WHEN outcome='taken' AND attempts=2 THEN 1 ELSE 0 END) AS successes,
      COUNT(*) AS trials
    FROM med_admin
    WHERE patient = ?
      AND second_strategy IN ('joke','concern')
    GROUP BY second_strategy
    """
    rows = conn.execute(sql, (patient,)).fetchall()

    # default when no data
    if not rows:
        return "joke"

    stats = {r[0]: (int(r[1]), int(r[2])) for r in rows}  # strategy -> (successes, trials)

    def rate(strategy: str) -> float:
        s, t = stats.get(strategy, (0, 0))
        return (s / t) if t > 0 else 0.0

    rj = rate("joke")
    rc = rate("concern")

    if rj > rc:
        return "joke"
    if rc > rj:
        return "concern"

    # tie-breaker: pick the one with more trials (more evidence); otherwise default to joke
    tj = stats.get("joke", (0, 0))[1]
    tc = stats.get("concern", (0, 0))[1]
    if tc > tj:
        return "concern"
    return "joke"
