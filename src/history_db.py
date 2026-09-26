import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import pandas as pd
import logging

from src.config import DB_PATH

logger = logging.getLogger(__name__)


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """
    Initializes SQLite table for prediction history if not exists.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                predicted_class INTEGER NOT NULL,
                predicted_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    logger.info(f"Database initialized at {db_path}")


def add_prediction_to_history(
    filename: str,
    predicted_class: int,
    predicted_name: str,
    confidence: float,
    db_path: Path = DB_PATH
) -> int:
    """
    Inserts a prediction record into the SQLite database.
    """
    init_db(db_path)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prediction_history (filename, predicted_class, predicted_name, confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (filename, predicted_class, predicted_name, round(confidence, 2), now_str))
        conn.commit()
        record_id = cursor.lastrowid
    return record_id


def get_prediction_history(limit: int = 100, db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Retrieves prediction history as a pandas DataFrame.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        df = pd.read_sql_query("""
            SELECT id, filename, predicted_class, predicted_name, confidence, created_at
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
        """, conn, params=(limit,))
    return df


def clear_prediction_history(db_path: Path = DB_PATH) -> None:
    """
    Clears all records in prediction_history.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM prediction_history")
        conn.commit()
    logger.info("Cleared prediction history database.")


if __name__ == "__main__":
    init_db()
    print("Database test init succeeded.")
