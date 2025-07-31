import tkinter as tk
from tkinter import messagebox, simpledialog
from typing import List
from pathlib import Path


class PreviewDialog:
    @staticmethod
    def show(parent, file_paths: List[Path], root_dir: str) -> None:
        preview_win = tk.Toplevel(parent)
        preview_win.title("Preview of Files to be Included")
        preview_win.geometry("600x500")
        
        listbox = tk.Listbox(preview_win, font=("Consolas", 9))
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for path in file_paths:
            try:
                relative_path = str(path.relative_to(root_dir))
                listbox.insert(tk.END, relative_path)
            except ValueError:
                listbox.insert(tk.END, str(path))


class ProfileDialog:
    @staticmethod
    def save_profile() -> str:
        return simpledialog.askstring(
            "Save Profile", 
            "Enter a name for the new profile:"
        )


class MessageDialog:
    @staticmethod
    def show_warning(title: str, message: str) -> None:
        messagebox.showwarning(title, message)
    
    @staticmethod
    def show_error(title: str, message: str) -> None:
        messagebox.showerror(title, message)
    
    @staticmethod
    def show_info(title: str, message: str) -> None:
        messagebox.showinfo(title, message)