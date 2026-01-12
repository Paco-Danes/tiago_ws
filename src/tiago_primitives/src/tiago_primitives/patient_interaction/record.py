import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Literal, Dict, Tuple
import rospy

SecondStrategy = Optional[Literal["joke", "concern"]]

# -----------------------------
# DB setup + generic admin log
# -----------------------------

def ensure_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)

    # Existing table (kept as-is)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS med_admin (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL,
            outcome TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            notes TEXT
        )
        """
    )

    # New table: per-patient bandit stats for attempt-2 choice
    # n = number of times arm used as attempt 2
    # s = number of successes (pill taken after attempt 2)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bandit_arm_stats (
            patient TEXT NOT NULL,
            arm TEXT NOT NULL,
            n INTEGER NOT NULL DEFAULT 0,
            s INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (patient, arm)
        )
        """
    )

    conn.commit()
    return conn


def log_admin(conn: sqlite3.Connection, patient: str, outcome: str, attempts: int, notes: str = "") -> None:
    ts = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO med_admin (patient, timestamp_utc, outcome, attempts, notes) VALUES (?,?,?,?,?)",
        (patient, ts, outcome, int(attempts), notes),
    )
    conn.commit()
    rospy.loginfo("Saved record -> patient=%s outcome=%s attempts=%d", patient, outcome, attempts)


# -----------------------------
# Bandit helpers (per-patient)
# -----------------------------

_BANDIT_ARMS = ("joke", "concern")


def _ensure_arm_rows(conn: sqlite3.Connection, patient: str) -> None:
    # Insert missing rows with (n=0,s=0) so queries are predictable.
    for arm in _BANDIT_ARMS:
        conn.execute(
            """
            INSERT OR IGNORE INTO bandit_arm_stats (patient, arm, n, s)
            VALUES (?, ?, 0, 0)
            """,
            (patient, arm),
        )
    conn.commit()


def get_bandit_stats(conn: sqlite3.Connection, patient: str) -> Dict[str, Tuple[int, int]]:
    """
    Returns dict: {arm: (n, s)} for arm in {"joke","concern"}.
    Ensures rows exist.
    """
    _ensure_arm_rows(conn, patient)

    cur = conn.execute(
        "SELECT arm, n, s FROM bandit_arm_stats WHERE patient = ?",
        (patient,),
    )
    out: Dict[str, Tuple[int, int]] = {}
    for arm, n, s in cur.fetchall():
        out[str(arm)] = (int(n), int(s))

    # Defensive: always include both arms
    for arm in _BANDIT_ARMS:
        out.setdefault(arm, (0, 0))

    return out


def update_bandit_stats(conn: sqlite3.Connection, patient: str, arm: str, reward: int) -> None:
    """
    reward is binary: 1 if pill taken after attempt 2, else 0.
    Updates (n,s) for the patient+arm.
    """
    if arm not in _BANDIT_ARMS:
        rospy.logwarn("update_bandit_stats: unknown arm='%s' (ignored)", arm)
        return

    _ensure_arm_rows(conn, patient)

    r = 1 if int(reward) != 0 else 0
    conn.execute(
        """
        UPDATE bandit_arm_stats
        SET n = n + 1,
            s = s + ?
        WHERE patient = ? AND arm = ?
        """,
        (r, patient, arm),
    )
    conn.commit()
    rospy.loginfo("Bandit update -> patient=%s arm=%s reward=%d", patient, arm, r)

#-----------------------------
# DB migration helpers
#-----------------------------

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