import os
import sqlite3
from datetime import datetime, timezone
import rospy

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
            notes TEXT
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
