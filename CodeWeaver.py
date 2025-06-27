import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import os
import json
from pathlib import Path
import threading
import queue
import subprocess
import sys

try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except ImportError:
    pass

CONFIG_FILE = Path.home() / ".codeweaver_config.json"

DEFAULT_IGNORE_PROFILES = {
    "Python (General)": [
        "__pycache__/", "venv/", ".venv/", "*/venv/*", "*.pyc", "*.egg-info/",
        "build/", "dist/", ".env", "data/", "notebooks/"
    ],
    "Node.js (React/Next)": [
        "node_modules/", ".next/", "build/", "dist/", "coverage/",
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", ".env*"
    ],
    "General Purpose": [
        ".git/", ".vscode/", ".idea/", "*.log", "*.tmp", "*.swp", ".DS_Store"
    ],
    "Media & Docs": [
        "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.svg", "*.ico",
        "*.mp3", "*.mp4", "*.avi", "*.mov", "*.webm",
        "*.pdf", "*.zip", "*.gz", "*.tar", "*.rar",
        "*.doc", "*.docx", "*.xls", "*.xlsx", "*.ppt", "*.pptx"
    ]
}

def _threaded_worker_function(mode, input_dir, ignore_patterns, size_limit_mb, q):
    try:
        q.put(('log', f"--- Starting {mode} for: {input_dir} ---"))
        q.put(('progress', ('indeterminate',)))
        
        project_files = []
        size_limit_bytes = size_limit_mb * 1024 * 1024
        input_path_obj = Path(input_dir)

        for root, dirs, files in os.walk(input_dir, topdown=True):
            root_path = Path(root)
            dirs[:] = [d for d in dirs if not any(Path(root, d).match(p) for p in ignore_patterns)]
            
            for file in files:
                file_path = root_path / file
                if not file_path.is_file(): continue

                if any(file_path.match(p) for p in ignore_patterns): continue
                
                try:
                    if file_path.stat().st_size > size_limit_bytes:
                        q.put(('log', f"  - Ignoring (too large): {file_path.relative_to(input_path_obj)}"))
                        continue
                except FileNotFoundError: continue 
                project_files.append(file_path)
        
        if not project_files:
            q.put(('done', {'success': False, 'message': 'No files to include after filtering.'}))
            return

        if mode == 'preview':
            q.put(('preview_result', {'files': sorted(project_files), 'root': input_dir}))
            q.put(('done', {'success': True}))
            return

        output_file = input_path_obj / "codebase.md"
        q.put(('log', f"Output will be: {output_file}"))
        q.put(('progress', ('determinate', len(project_files))))
        
        total_content = ""
        tree_string = generate_project_tree_static(input_dir, project_files)
        initial_content = f"# Project Structure\n\n```\n{tree_string}\n```\n\n---\n\n# File Contents\n\n"
        total_content += initial_content

        with open(output_file, 'w', encoding='utf-8', errors='ignore') as md_file:
            md_file.write(initial_content)
            sorted_files = sorted(project_files)
            for i, file_path in enumerate(sorted_files):
                relative_path_str = str(file_path.relative_to(input_path_obj)).replace("\\", "/")
                q.put(('log', f"  + Including: {relative_path_str}"))
                header = f"---\n\n### `{relative_path_str}`\n\n"
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
                    lang = file_path.suffix[1:].lower() if file_path.suffix else "text"
                    file_block = f"```{lang}\n{content.strip()}\n```\n\n"
                    md_file.write(header + file_block)
                    total_content += header + file_block
                except Exception as e:
                    error_block = f"```\n[Could not read file content: {e}]\n```\n\n"
                    md_file.write(header + error_block)
                    total_content += header + error_block
                q.put(('progress', ('update', i + 1)))

        file_size_kb = output_file.stat().st_size / 1024
        token_estimate = len(total_content) // 4
        
        results = {'success': True, 'output_path': str(output_file), 'stats': (file_size_kb, token_estimate)}
        q.put(('done', results))
            
    except Exception as e:
        import traceback
        q.put(('log', f"!!! AN ERROR OCCURRED: {e}"))
        q.put(('log', traceback.format_exc()))
        q.put(('done', {'success': False, 'message': str(e)}))

def generate_project_tree_static(root_dir, file_paths):
    tree = {}
    root_path = Path(root_dir)
    for path in file_paths:
        try: relative_path = path.relative_to(root_path)
        except ValueError: continue
        parts = relative_path.parts
        current_level = tree
        for part in parts:
            if part not in current_level: current_level[part] = {}
            current_level = current_level[part]

    def _build_tree_lines(d, prefix=''):
        lines = []
        items = sorted(d.keys())
        for i, name in enumerate(items):
            connector = '└── ' if i == len(items) - 1 else '├── '
            lines.append(prefix + connector + name)
            if d[name]:
                extension = '│   ' if i < len(items) - 1 else '    '
                lines.extend(_build_tree_lines(d[name], prefix + extension))
        return lines

    tree_lines = [os.path.basename(root_dir)]
    tree_lines.extend(_build_tree_lines(tree))
    return "\n".join(tree_lines)


class AIDigestApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.style = ttk.Style(self)
        self.style.theme_use('vista')
        self.style.configure('.', font=('Segoe UI', 10))
        self.style.configure('TButton', padding=6)
        
        self.title("CodeWeaver")
        self.geometry("1000x750")

        self.project_dir = ""
        self.ignore_profiles = {}
        self.is_processing = False
        self.queue = queue.Queue()
        self.current_output_path = None

        self.create_menu()
        self.create_widgets()
        self.load_settings()
        self.update_project_list()
        self.process_queue()
        
    def create_menu(self):
        self.menu_bar = tk.Menu(self)
        self.config(menu=self.menu_bar)
        profiles_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Profiles", menu=profiles_menu)
        self.build_profiles_menu(profiles_menu)
        profiles_menu.add_separator()
        profiles_menu.add_command(label="Save Current as Profile...", command=self.save_as_profile)

    def build_profiles_menu(self, menu):
        menu.delete(0, "end")
        for name in sorted(self.ignore_profiles.keys()):
            menu.add_command(label=name, command=lambda n=name: self.apply_profile(n))

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        left_pane = ttk.Frame(main_frame, width=300)
        left_pane.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        right_pane = ttk.Frame(main_frame)
        right_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Button(left_pane, text="Set Project Directory", command=self.select_project_directory).pack(fill=tk.X)
        self.project_dir_label = ttk.Label(left_pane, text="Not Set", wraplength=280, font=("Segoe UI", 8))
        self.project_dir_label.pack(fill=tk.X, pady=(5,0))
        ttk.Label(left_pane, text="Projects", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, pady=(10, 0))
        self.project_listbox = tk.Listbox(left_pane, selectmode=tk.SINGLE, exportselection=False, font=("Consolas", 10))
        self.project_listbox.pack(fill=tk.BOTH, expand=True, pady=(5,0))

        top_controls_frame = ttk.Frame(right_pane)
        top_controls_frame.pack(fill=tk.X, pady=(0, 10))
        self.digest_btn = ttk.Button(top_controls_frame, text="Digest Selected Project", command=lambda: self.start_processing(mode='digest'))
        self.digest_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5))
        self.preview_btn = ttk.Button(top_controls_frame, text="Preview Included Files...", command=lambda: self.start_processing(mode='preview'))
        self.preview_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        settings_frame = ttk.LabelFrame(right_pane, text="Settings", padding="10")
        settings_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        size_limit_frame = ttk.Frame(settings_frame)
        size_limit_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(size_limit_frame, text="Skip files larger than (MB):").pack(side=tk.LEFT, padx=(0, 5))
        self.size_limit_var = tk.StringVar(value="1.0")
        self.size_limit_entry = ttk.Entry(size_limit_frame, textvariable=self.size_limit_var, width=10)
        self.size_limit_entry.pack(side=tk.LEFT)

        ttk.Label(settings_frame, text="Ignore Patterns (one per line, supports wildcards like */dist/ and *.log)").pack(anchor=tk.W)
        self.ignore_text = tk.Text(settings_frame, height=10, width=60, font=("Consolas", 10))
        self.ignore_text.pack(fill=tk.BOTH, expand=True, pady=(5, 5))
        
        log_frame = ttk.LabelFrame(right_pane, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10,0))
        self.log_text = tk.Text(log_frame, state='disabled', height=8, font=("Consolas", 9), relief=tk.FLAT)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.progress_bar = ttk.Progressbar(right_pane, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(5, 5))
        
        self.post_actions_frame = ttk.Frame(right_pane)
        self.stats_label = ttk.Label(self.post_actions_frame, text="", font=("Segoe UI", 9, "italic"))
        self.stats_label.pack(side=tk.LEFT, padx=(0, 20))
        self.open_file_btn = ttk.Button(self.post_actions_frame, text="Open File", command=self.open_output_file)
        self.open_file_btn.pack(side=tk.RIGHT, padx=(5,0))
        self.show_in_explorer_btn = ttk.Button(self.post_actions_frame, text="Show in Explorer", command=self.show_in_explorer)
        self.show_in_explorer_btn.pack(side=tk.RIGHT)

    def start_processing(self, mode):
        if self.is_processing: return
        selection_indices = self.project_listbox.curselection()
        if not selection_indices:
            messagebox.showwarning("No Selection", "Please select a project from the list on the left.")
            return
        
        self.is_processing = True
        self.update_ui_state()
        project_name = self.project_listbox.get(selection_indices[0])
        project_path = os.path.join(self.project_dir, project_name)
        
        try: size_limit = float(self.size_limit_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "File size limit must be a number.")
            self.is_processing = False; self.update_ui_state()
            return
            
        raw_patterns = self.ignore_text.get("1.0", tk.END).strip().split("\n")
        ignore_patterns = [p for p in raw_patterns if p]
        
        thread = threading.Thread(target=_threaded_worker_function, args=(mode, project_path, ignore_patterns, size_limit, self.queue), daemon=True)
        thread.start()

    def process_queue(self):
        try:
            while True:
                msg_type, data = self.queue.get_nowait()
                if msg_type == 'log': self.log(data)
                elif msg_type == 'progress': self.update_progress(data)
                elif msg_type == 'preview_result': self.show_preview_window(data['files'], data['root'])
                elif msg_type == 'done':
                    self.is_processing = False
                    self.update_ui_state()
                    if data['success']:
                        self.log("--- PROCESS COMPLETE ---")
                        if 'output_path' in data:
                            self.current_output_path = data['output_path']
                            stats = data['stats']
                            self.stats_label.config(text=f"Size: {stats[0]:.1f} KB  |  Est. Tokens: ~{stats[1]:,}")
                            self.post_actions_frame.pack(fill=tk.X, pady=(5, 0))
                    else:
                        self.log(f"--- PROCESS FAILED: {data.get('message', '')} ---")
                        messagebox.showerror("Processing Failed", data.get('message', 'An unknown error occurred.'))
        except queue.Empty: pass
        finally: self.after(100, self.process_queue)
    
    def update_ui_state(self):
        state = 'disabled' if self.is_processing else 'normal'
        for btn in [self.digest_btn, self.preview_btn]: btn.config(state=state)
        self.project_listbox.config(state=state)
        self.size_limit_entry.config(state=state)
        self.ignore_text.config(state=state)
        self.menu_bar.entryconfig("Profiles", state=state)
        if self.is_processing:
            self.post_actions_frame.pack_forget()
            self.progress_bar.config(value=0)
            self.log_text.config(state='normal'); self.log_text.delete('1.0', tk.END); self.log_text.config(state='disabled')

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, str(message) + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def update_progress(self, data):
        mode, *args = data
        if mode == 'indeterminate': self.progress_bar.config(mode='indeterminate'); self.progress_bar.start()
        elif mode == 'determinate': self.progress_bar.stop(); self.progress_bar.config(mode='determinate', maximum=args[0])
        elif mode == 'update': self.progress_bar['value'] = args[0]

    def show_preview_window(self, file_paths, root_dir):
        preview_win = tk.Toplevel(self); preview_win.title("Preview of Files to be Included"); preview_win.geometry("600x500")
        listbox = tk.Listbox(preview_win, font=("Consolas", 9)); listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        for path in file_paths:
            try: listbox.insert(tk.END, str(path.relative_to(root_dir)))
            except ValueError: listbox.insert(tk.END, str(path))

    def show_in_explorer(self):
        if not self.current_output_path: return
        path = self.current_output_path
        if sys.platform == "win32": subprocess.run(['explorer', '/select,', os.path.normpath(path)])
        elif sys.platform == "darwin": subprocess.run(['open', '-R', os.path.normpath(path)])
        else: subprocess.run(['xdg-open', os.path.dirname(os.path.normpath(path))])

    def open_output_file(self):
        if not self.current_output_path: return
        path = self.current_output_path
        if sys.platform == "win32": os.startfile(path)
        else: subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", path])

    def apply_profile(self, name):
        patterns = self.ignore_profiles.get(name, [])
        self.ignore_text.delete("1.0", tk.END)
        self.ignore_text.insert("1.0", "\n".join(patterns))
        self.log(f"Applied profile: {name}")

    def save_as_profile(self):
        name = simpledialog.askstring("Save Profile", "Enter a name for the new profile:")
        if name and name not in self.ignore_profiles:
            patterns = [p for p in self.ignore_text.get("1.0", tk.END).strip().split("\n") if p]
            self.ignore_profiles[name] = patterns
            self.build_profiles_menu(self.menu_bar.winfo_children()[0]) 
            self.log(f"Saved new profile: {name}")

    def load_settings(self):
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, 'r') as f: settings = json.load(f)
                self.project_dir = settings.get("project_dir", "")
                self.ignore_profiles = settings.get("ignore_profiles", DEFAULT_IGNORE_PROFILES)
                self.ignore_text.insert("1.0", "\n".join(settings.get("last_ignore_list", [])))
            else:
                self.ignore_profiles = DEFAULT_IGNORE_PROFILES
                p1 = self.ignore_profiles.get("General Purpose", [])
                p2 = self.ignore_profiles.get("Media & Docs", [])
                p3 = self.ignore_profiles.get("Python (General)", [])
                combined_patterns = list(dict.fromkeys(p1 + p2 + p3))
                self.ignore_text.insert("1.0", "\n".join(combined_patterns))
                self.log("Loaded comprehensive default ignore patterns.")
        except Exception as e:
            self.log(f"Could not load settings: {e}")
            self.ignore_profiles = DEFAULT_IGNORE_PROFILES

    def save_settings(self):
        settings = {
            "project_dir": self.project_dir,
            "ignore_profiles": self.ignore_profiles,
            "last_ignore_list": [p for p in self.ignore_text.get("1.0", tk.END).strip().split("\n") if p]
        }
        with open(CONFIG_FILE, 'w') as f: json.dump(settings, f, indent=2)

    def on_closing(self):
        self.save_settings(); self.destroy()

    def select_project_directory(self):
        path = filedialog.askdirectory(title="Select Main Projects Directory")
        if path:
            self.project_dir = path; self.project_dir_label.config(text=f"Projects in: {self.project_dir}")
            self.update_project_list(); self.save_settings()

    def update_project_list(self):
        self.project_listbox.delete(0, tk.END)
        if self.project_dir and os.path.isdir(self.project_dir):
            try:
                subfolders = [d for d in os.listdir(self.project_dir) if os.path.isdir(os.path.join(self.project_dir, d))]
                for folder in sorted(subfolders): self.project_listbox.insert(tk.END, folder)
            except Exception as e: self.log(f"Error reading project dir: {e}")

if __name__ == "__main__":
    app = AIDigestApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()