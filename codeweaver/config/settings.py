import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from .profiles import DEFAULT_IGNORE_PROFILES


class SettingsManager:
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path.home() / ".codeweaver_config.json"
        self._settings = {}
        self.load_settings()

    def load_settings(self) -> None:
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self._settings = json.load(f)
            else:
                self._settings = {}
        except Exception as e:
            print(f"Could not load settings: {e}")
            self._settings = {}

    def save_settings(self) -> None:
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._settings, f, indent=2)
        except Exception as e:
            print(f"Could not save settings: {e}")

    @property
    def project_dir(self) -> str:
        return self._settings.get("project_dir", "")

    @project_dir.setter
    def project_dir(self, value: str) -> None:
        self._settings["project_dir"] = value

    @property
    def ignore_profiles(self) -> Dict[str, List[str]]:
        return self._settings.get("ignore_profiles", DEFAULT_IGNORE_PROFILES)

    @ignore_profiles.setter
    def ignore_profiles(self, value: Dict[str, List[str]]) -> None:
        self._settings["ignore_profiles"] = value

    @property
    def last_ignore_list(self) -> List[str]:
        return self._settings.get("last_ignore_list", [])

    @last_ignore_list.setter
    def last_ignore_list(self, value: List[str]) -> None:
        self._settings["last_ignore_list"] = value

    def add_profile(self, name: str, patterns: List[str]) -> None:
        profiles = self.ignore_profiles.copy()
        profiles[name] = patterns
        self.ignore_profiles = profiles

    def get_profile(self, name: str) -> Optional[List[str]]:
        return self.ignore_profiles.get(name)

    def get_default_patterns(self) -> List[str]:
        profiles = self.ignore_profiles
        p1 = profiles.get("General Purpose", [])
        p2 = profiles.get("Media & Docs", [])
        p3 = profiles.get("Python (General)", [])
        return list(dict.fromkeys(p1 + p2 + p3))

    @property
    def recent_projects(self) -> List[Dict[str, str]]:
        """Returns list of recent projects with path, name, and last_used timestamp"""
        return self._settings.get("recent_projects", [])

    def add_recent_project(self, project_path: str, project_name: str, max_recent: int = 10) -> None:
        """Add or update a project in recent projects list"""
        recent = self.recent_projects.copy()
        timestamp = datetime.now().isoformat()
        
        # Remove existing entry if present
        recent = [p for p in recent if p.get("path") != project_path]
        
        # Add to front
        recent.insert(0, {
            "path": project_path,
            "name": project_name,
            "last_used": timestamp
        })
        
        # Limit size
        recent = recent[:max_recent]
        
        self._settings["recent_projects"] = recent

    def remove_recent_project(self, project_path: str) -> None:
        """Remove a project from recent projects list"""
        recent = [p for p in self.recent_projects if p.get("path") != project_path]
        self._settings["recent_projects"] = recent