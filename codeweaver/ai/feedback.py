import sqlite3
from pathlib import Path
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

@dataclass
class UserInteraction:
    """Represents a single user interaction with the optimization results."""
    timestamp: float
    purpose: str
    selected_files: List[str]
    user_feedback: Dict[str, Any]  # e.g., {"file_path": "path/to/file.py", "action": "removed"}

class FeedbackLoop:
    """
    Manages the learning and feedback system for the optimization engine.
    """
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize the SQLite database for storing feedback."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    purpose TEXT NOT NULL,
                    selected_files TEXT NOT NULL,
                    user_feedback TEXT NOT NULL
                )
            """)

    def log_interaction(self, purpose: str, selected_files: List[Path], user_feedback: Dict[str, Any]):
        """Logs a user interaction to the database."""
        interaction = UserInteraction(
            timestamp=time.time(),
            purpose=purpose,
            selected_files=[str(p) for p in selected_files],
            user_feedback=user_feedback
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO interactions (timestamp, purpose, selected_files, user_feedback) VALUES (?, ?, ?, ?)",
                (interaction.timestamp, interaction.purpose, json.dumps(interaction.selected_files), json.dumps(interaction.user_feedback))
            )

    def get_historical_patterns(self, purpose: str) -> Optional[Dict[str, Any]]:
        """
        Analyzes historical interactions to find patterns for a given purpose.
        """
        # This is a simplified implementation. A more advanced version would use ML models.
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT user_feedback FROM interactions WHERE purpose = ?", (purpose,))
            rows = cursor.fetchall()

        if not rows:
            return None

        # Aggregate feedback
        file_actions = {}
        for row in rows:
            feedback = json.loads(row[0])
            file_path = feedback.get("file_path")
            action = feedback.get("action")
            if file_path and action:
                if file_path not in file_actions:
                    file_actions[file_path] = {"added": 0, "removed": 0}
                if action == "added":
                    file_actions[file_path]["added"] += 1
                elif action == "removed":
                    file_actions[file_path]["removed"] += 1

        return {"file_actions": file_actions}
