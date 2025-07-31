import tkinter as tk
from tkinter import ttk
from typing import Optional, List, Dict, Callable
from datetime import datetime, timedelta


class LogWidget:
    def __init__(self, parent, height: int = 8):
        self.text_widget = tk.Text(
            parent, 
            state='disabled', 
            height=height, 
            font=("Consolas", 11), 
            relief=tk.FLAT
        )
    
    def log(self, message: str) -> None:
        self.text_widget.config(state='normal')
        self.text_widget.insert(tk.END, str(message) + "\n")
        self.text_widget.see(tk.END)
        self.text_widget.config(state='disabled')
    
    def clear(self) -> None:
        self.text_widget.config(state='normal')
        self.text_widget.delete('1.0', tk.END)
        self.text_widget.config(state='disabled')
    
    def pack(self, **kwargs):
        self.text_widget.pack(**kwargs)


class ProgressWidget:
    def __init__(self, parent):
        self.progress_bar = ttk.Progressbar(parent, mode='determinate')
    
    def update(self, data) -> None:
        mode, *args = data
        if mode == 'indeterminate':
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start()
        elif mode == 'determinate':
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate', maximum=args[0])
        elif mode == 'update':
            self.progress_bar['value'] = args[0]
    
    def reset(self) -> None:
        self.progress_bar.config(value=0)
    
    def pack(self, **kwargs):
        self.progress_bar.pack(**kwargs)


class ProjectListWidget:
    def __init__(self, parent, width: int = 300):
        self.frame = ttk.Frame(parent, width=width)
        self.setup_widgets()
    
    def setup_widgets(self):
        self.dir_button = ttk.Button(
            self.frame, 
            text="Set Project Directory"
        )
        self.dir_button.pack(fill=tk.X)
        
        self.dir_label = ttk.Label(
            self.frame, 
            text="Not Set", 
            wraplength=320, 
            font=("Segoe UI", 10)
        )
        self.dir_label.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(
            self.frame, 
            text="Projects", 
            font=("Segoe UI", 12, "bold")
        ).pack(anchor=tk.W, pady=(10, 0))
        
        self.listbox = tk.Listbox(
            self.frame, 
            selectmode=tk.SINGLE, 
            exportselection=False, 
            font=("Segoe UI", 11)
        )
        self.listbox.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
    
    def update_directory_label(self, path: str) -> None:
        self.dir_label.config(text=f"Projects in: {path}")
    
    def update_project_list(self, projects: list) -> None:
        self.listbox.delete(0, tk.END)
        for project in sorted(projects):
            self.listbox.insert(tk.END, project)
    
    def get_selected_project(self) -> Optional[str]:
        selection = self.listbox.curselection()
        if selection:
            return self.listbox.get(selection[0])
        return None
    
    def set_state(self, state: str) -> None:
        self.listbox.config(state=state)
    
    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class RecentProjectsWidget:
    def __init__(self, parent, on_project_selected: Callable[[str, str], None] = None):
        self.frame = ttk.LabelFrame(parent, text="Recent Projects", padding="5")
        self.on_project_selected = on_project_selected
        self.setup_widgets()
    
    def setup_widgets(self):
        # Create treeview for recent projects
        self.tree = ttk.Treeview(
            self.frame, 
            columns=("path", "last_used"), 
            show="tree headings",
            height=6
        )
        
        # Configure columns
        self.tree.heading("#0", text="Project")
        self.tree.heading("path", text="Path")
        self.tree.heading("last_used", text="Last Used")
        
        self.tree.column("#0", width=180, minwidth=120)
        self.tree.column("path", width=300, minwidth=180)
        self.tree.column("last_used", width=120, minwidth=100)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Layout
        self.tree.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)
        
        # Bind double-click event
        self.tree.bind("<Double-1>", self._on_double_click)
        
        # Context menu
        self.context_menu = tk.Menu(self.frame, tearoff=0)
        self.context_menu.add_command(label="Open Project", command=self._open_selected)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Remove from Recent", command=self._remove_selected)
        
        self.tree.bind("<Button-3>", self._show_context_menu)  # Right-click
    
    def update_recent_projects(self, recent_projects: List[Dict[str, str]]):
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add recent projects
        for project in recent_projects:
            name = project.get("name", "Unknown")
            path = project.get("path", "")
            last_used = project.get("last_used", "")
            
            # Format last used time
            try:
                dt = datetime.fromisoformat(last_used)
                now = datetime.now()
                
                if (now - dt).days == 0:
                    time_str = dt.strftime("%H:%M")
                elif (now - dt).days == 1:
                    time_str = "Yesterday"
                elif (now - dt).days < 7:
                    time_str = f"{(now - dt).days}d ago"
                else:
                    time_str = dt.strftime("%m/%d")
            except (ValueError, TypeError):
                time_str = "Unknown"
            
            self.tree.insert("", "end", text=name, values=(path, time_str))
    
    def _on_double_click(self, event):
        self._open_selected()
    
    def _open_selected(self):
        selection = self.tree.selection()
        if selection and self.on_project_selected:
            item = self.tree.item(selection[0])
            project_name = item["text"]
            project_path = item["values"][0]
            self.on_project_selected(project_path, project_name)
    
    def _remove_selected(self):
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            project_path = item["values"][0]
            # Emit event for removal (will be handled by main window)
            if hasattr(self, 'on_project_removed') and self.on_project_removed:
                self.on_project_removed(project_path)
    
    def _show_context_menu(self, event):
        # Select the item under cursor
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            self.context_menu.post(event.x_root, event.y_root)
    
    def pack(self, **kwargs):
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        self.frame.grid(**kwargs)