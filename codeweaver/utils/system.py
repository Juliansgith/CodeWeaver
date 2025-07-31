import os
import sys
import subprocess
from pathlib import Path


class SystemUtils:
    @staticmethod
    def open_file(file_path: str) -> None:
        if sys.platform == "win32":
            os.startfile(file_path)
        else:
            subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", file_path])

    @staticmethod
    def show_in_explorer(file_path: str) -> None:
        path = os.path.normpath(file_path)
        if sys.platform == "win32":
            subprocess.run(['explorer', '/select,', path])
        elif sys.platform == "darwin":
            subprocess.run(['open', '-R', path])
        else:
            subprocess.run(['xdg-open', os.path.dirname(path)])

    @staticmethod
    def setup_dpi_awareness() -> None:
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except ImportError:
            pass

    @staticmethod
    def get_projects_in_directory(directory: str) -> list:
        if not os.path.isdir(directory):
            return []
        
        try:
            return [d for d in os.listdir(directory) 
                   if os.path.isdir(os.path.join(directory, d))]
        except Exception:
            return []