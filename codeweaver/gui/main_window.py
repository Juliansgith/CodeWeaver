import tkinter as tk
from tkinter import ttk, filedialog
import os
import queue
from typing import Optional

from ..config import SettingsManager
from ..core.models import ProcessingOptions
from ..core import TokenEstimator, LLMProvider
from ..utils import SystemUtils, BackgroundProcessor
from .components import LogWidget, ProgressWidget, ProjectListWidget, RecentProjectsWidget
from .dialogs import PreviewDialog, ProfileDialog, MessageDialog, TemplateDialog
from .token_analysis_dialog import TokenAnalysisDialog


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        
        SystemUtils.setup_dpi_awareness()
        self.setup_window()
        self.setup_variables()
        self.create_widgets()
        self.create_menu()
        self.load_settings()
        self.update_project_list()
        self.process_queue()
    
    def setup_window(self):
        self.style = ttk.Style(self)
        self.style.theme_use('vista')
        
        # Improved font sizes for 4K monitors
        self.style.configure('.', font=('Segoe UI', 12))
        self.style.configure('TButton', padding=8, font=('Segoe UI', 11))
        self.style.configure('TLabel', font=('Segoe UI', 11))
        self.style.configure('TLabelframe.Label', font=('Segoe UI', 12, 'bold'))
        self.style.configure('Heading', font=('Segoe UI', 13, 'bold'))
        
        self.title("CodeWeaver")
        self.geometry("1200x900")  # Larger default window for 4K
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_variables(self):
        self.settings = SettingsManager()
        self.queue = queue.Queue()
        self.background_processor = BackgroundProcessor(self.queue)
        self.current_output_path: Optional[str] = None
        self.current_token_analysis: Optional[dict] = None
        self.size_limit_var = tk.StringVar(value="1.0")
    
    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left pane - Project selection and recent projects
        left_pane = ttk.Frame(main_frame, width=400)
        left_pane.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        
        # Project directory selection
        self.project_list = ProjectListWidget(left_pane)
        self.project_list.pack(fill=tk.BOTH, expand=True)
        self.project_list.dir_button.config(command=self.select_project_directory)
        
        # Recent projects
        self.recent_projects = RecentProjectsWidget(
            left_pane, 
            on_project_selected=self.open_recent_project
        )
        self.recent_projects.on_project_removed = self.remove_recent_project
        self.recent_projects.pack(fill=tk.X, pady=(10, 0))
        
        # Right pane - Controls and settings
        right_pane = ttk.Frame(main_frame)
        right_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.create_control_buttons(right_pane)
        self.create_settings_frame(right_pane)
        self.create_log_frame(right_pane)
        self.create_progress_widgets(right_pane)
        self.create_post_actions_frame(right_pane)
    
    def create_control_buttons(self, parent):
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.digest_btn = ttk.Button(
            controls_frame, 
            text="Digest Selected Project", 
            command=lambda: self.start_processing('digest')
        )
        self.digest_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        
        self.preview_btn = ttk.Button(
            controls_frame, 
            text="Preview Included Files...", 
            command=lambda: self.start_processing('preview')
        )
        self.preview_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
    
    def create_settings_frame(self, parent):
        settings_frame = ttk.LabelFrame(parent, text="Settings", padding="10")
        settings_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Size limit
        size_frame = ttk.Frame(settings_frame)
        size_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(size_frame, text="Skip files larger than (MB):").pack(side=tk.LEFT, padx=(0, 5))
        self.size_limit_entry = ttk.Entry(size_frame, textvariable=self.size_limit_var, width=10)
        self.size_limit_entry.pack(side=tk.LEFT)

        # Content filtering options
        filter_frame = ttk.Frame(settings_frame)
        filter_frame.pack(fill=tk.X, pady=(10, 0))
        self.strip_comments_var = tk.BooleanVar()
        self.optimize_whitespace_var = tk.BooleanVar()
        self.intelligent_sampling_var = tk.BooleanVar()
        ttk.Checkbutton(filter_frame, text="Strip Comments", variable=self.strip_comments_var).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(filter_frame, text="Optimize Whitespace", variable=self.optimize_whitespace_var).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(filter_frame, text="Intelligent Sampling", variable=self.intelligent_sampling_var).pack(side=tk.LEFT)
        
        # Ignore patterns
        ttk.Label(settings_frame, text="Ignore Patterns (one per line, supports wildcards like */dist/ and *.log)", 
                 font=("Segoe UI", 11)).pack(anchor=tk.W)
        self.ignore_text = tk.Text(settings_frame, height=10, width=60, font=("Consolas", 11))
        self.ignore_text.pack(fill=tk.BOTH, expand=True, pady=(5, 5))
    
    def create_log_frame(self, parent):
        log_frame = ttk.LabelFrame(parent, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.log_widget = LogWidget(log_frame)
        self.log_widget.pack(fill=tk.BOTH, expand=True)
    
    def create_progress_widgets(self, parent):
        self.progress_widget = ProgressWidget(parent)
        self.progress_widget.pack(fill=tk.X, pady=(5, 5))
    
    def create_post_actions_frame(self, parent):
        self.post_actions_frame = ttk.Frame(parent)
        
        # Stats frame with detailed token info
        stats_frame = ttk.Frame(self.post_actions_frame)
        stats_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
        
        self.stats_label = ttk.Label(
            stats_frame,
            text="", 
            font=("Segoe UI", 11, "italic")
        )
        self.stats_label.pack(anchor=tk.W)
        
        self.token_details_label = ttk.Label(
            stats_frame,
            text="",
            font=("Segoe UI", 10),
            foreground="gray"
        )
        self.token_details_label.pack(anchor=tk.W)
        
        self.open_file_btn = ttk.Button(
            self.post_actions_frame, 
            text="Open File", 
            command=self.open_output_file
        )
        self.open_file_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        self.show_explorer_btn = ttk.Button(
            self.post_actions_frame, 
            text="Show in Explorer", 
            command=self.show_in_explorer
        )
        self.show_explorer_btn.pack(side=tk.RIGHT)
        
        self.token_analysis_btn = ttk.Button(
            self.post_actions_frame, 
            text="Token Analysis", 
            command=self.show_token_analysis
        )
        self.token_analysis_btn.pack(side=tk.RIGHT, padx=(0, 5))
    
    def create_menu(self):
        self.menu_bar = tk.Menu(self)
        self.config(menu=self.menu_bar)
        
        profiles_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Profiles", menu=profiles_menu)
        self.build_profiles_menu(profiles_menu)
        profiles_menu.add_separator()
        profiles_menu.add_command(label="Save Current as Profile...", command=self.save_as_profile)

        templates_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Templates", menu=templates_menu)
        templates_menu.add_command(label="Apply Template...", command=self.apply_template)
        templates_menu.add_command(label="Save Current as Template...", command=self.save_as_template)
    
    def build_profiles_menu(self, menu):
        menu.delete(0, "end")
        for name in sorted(self.settings.ignore_profiles.keys()):
            menu.add_command(label=name, command=lambda n=name: self.apply_profile(n))
    
    def start_processing(self, mode: str):
        if self.background_processor.is_processing:
            return
        
        selected_project = self.project_list.get_selected_project()
        if not selected_project:
            MessageDialog.show_warning("No Selection", "Please select a project from the list on the left.")
            return
        
        try:
            size_limit = float(self.size_limit_var.get())
        except ValueError:
            MessageDialog.show_error("Invalid Input", "File size limit must be a number.")
            return
        
        project_path = os.path.join(self.settings.project_dir, selected_project)
        raw_patterns = self.ignore_text.get("1.0", tk.END).strip().split("\n")
        ignore_patterns = [p for p in raw_patterns if p]
        
        options = ProcessingOptions(
            input_dir=project_path,
            ignore_patterns=ignore_patterns,
            size_limit_mb=size_limit,
            mode=mode,
            strip_comments=self.strip_comments_var.get(),
            optimize_whitespace=self.optimize_whitespace_var.get(),
            intelligent_sampling=self.intelligent_sampling_var.get()
        )
        
        # Add to recent projects if processing (not just previewing)
        if mode == 'digest':
            self.settings.add_recent_project(project_path, selected_project)
            self.update_recent_projects()
        
        self.update_ui_state(processing=True)
        self.background_processor.start_processing(options)
    
    def process_queue(self):
        try:
            while True:
                msg_type, data = self.queue.get_nowait()
                if msg_type == 'log':
                    self.log_widget.log(data)
                elif msg_type == 'progress':
                    self.progress_widget.update(data)
                elif msg_type == 'preview_result':
                    PreviewDialog.show(self, data['files'], data['root'])
                elif msg_type == 'done':
                    self.handle_processing_complete(data)
        except queue.Empty:
            pass
        finally:
            self.after(100, self.process_queue)
    
    def handle_processing_complete(self, result):
        self.background_processor.is_processing = False
        self.update_ui_state(processing=False)
        
        if result.success:
            self.log_widget.log("--- PROCESS COMPLETE ---")
            if result.output_path:
                self.current_output_path = result.output_path
                stats = result.stats
                
                # Store token analysis data
                self.current_token_analysis = stats.token_analysis
                
                # Update main stats
                self.stats_label.config(
                    text=f"Size: {stats.file_size_kb:.1f} KB  |  Files: {stats.file_count}  |  Tokens: ~{stats.estimated_tokens:,}"
                )
                
                # Show detailed token estimates
                if stats.token_estimates:
                    token_details = self._format_token_details(stats.token_estimates)
                    self.token_details_label.config(text=token_details)
                
                self.post_actions_frame.pack(fill=tk.X, pady=(5, 0))
        else:
            self.log_widget.log(f"--- PROCESS FAILED: {result.message or ''} ---")
            MessageDialog.show_error(
                "Processing Failed", 
                result.message or 'An unknown error occurred.'
            )
    
    def update_ui_state(self, processing: bool):
        state = 'disabled' if processing else 'normal'
        
        for btn in [self.digest_btn, self.preview_btn]:
            btn.config(state=state)
        
        self.project_list.set_state(state)
        self.size_limit_entry.config(state=state)
        self.ignore_text.config(state=state)
        self.menu_bar.entryconfig("Profiles", state=state)
        
        if processing:
            self.post_actions_frame.pack_forget()
            self.progress_widget.reset()
            self.log_widget.clear()
            self.current_token_analysis = None  # Clear previous analysis
    
    def select_project_directory(self):
        path = filedialog.askdirectory(title="Select Main Projects Directory")
        if path:
            self.settings.project_dir = path
            self.project_list.update_directory_label(path)
            self.update_project_list()
            self.settings.save_settings()
    
    def update_project_list(self):
        if self.settings.project_dir:
            projects = SystemUtils.get_projects_in_directory(self.settings.project_dir)
            self.project_list.update_project_list(projects)
    
    def apply_profile(self, name: str):
        patterns = self.settings.get_profile(name)
        if patterns:
            self.ignore_text.delete("1.0", tk.END)
            self.ignore_text.insert("1.0", "\n".join(patterns))
            self.log_widget.log(f"Applied profile: {name}")
    
    def save_as_profile(self):
        name = ProfileDialog.save_profile()
        if name and name not in self.settings.ignore_profiles:
            patterns = [p for p in self.ignore_text.get("1.0", tk.END).strip().split("\n") if p]
            self.settings.add_profile(name, patterns)
            self.build_profiles_menu(self.menu_bar.winfo_children()[0])
            self.log_widget.log(f"Saved new profile: {name}")

    def apply_template(self):
        available_templates = self.template_manager.get_available_templates()
        if not available_templates:
            MessageDialog.show_info("No Templates", "There are no saved templates.")
            return

        dialog = TemplateDialog(self, "Apply Template", available_templates)
        if dialog.result:
            template = self.template_manager.get_template(dialog.result)
            if template and 'ignore_patterns' in template:
                self.ignore_text.delete("1.0", tk.END)
                self.ignore_text.insert("1.0", "\n".join(template['ignore_patterns']))
                self.log_widget.log(f"Applied template: {dialog.result}")

    def save_as_template(self):
        name = simpledialog.askstring("Save Template", "Enter a name for the new template:")
        if name:
            patterns = [p for p in self.ignore_text.get("1.0", tk.END).strip().split("\n") if p]
            self.template_manager.create_template_from_project(name, Path(self.settings.project_dir), patterns)
            self.log_widget.log(f"Saved new template: {name}")
    
    def open_output_file(self):
        if self.current_output_path:
            SystemUtils.open_file(self.current_output_path)
    
    def show_in_explorer(self):
        if self.current_output_path:
            SystemUtils.show_in_explorer(self.current_output_path)
    
    def show_token_analysis(self):
        """Show the token analysis dialog."""
        if self.current_token_analysis:
            dialog = TokenAnalysisDialog(self, self.current_token_analysis)
            dialog.show()
        else:
            MessageDialog.show_warning(
                "No Analysis Available",
                "Token analysis is only available after processing a project.\nPlease digest a project first."
            )
    
    def open_recent_project(self, project_path: str, project_name: str):
        """Handle opening a recent project"""
        if os.path.exists(project_path):
            # Update settings to use this project
            parent_dir = os.path.dirname(project_path)
            if parent_dir != self.settings.project_dir:
                self.settings.project_dir = parent_dir
                self.project_list.update_directory_label(parent_dir)
                self.update_project_list()
            
            # Select the project in the list
            for i in range(self.project_list.listbox.size()):
                if self.project_list.listbox.get(i) == project_name:
                    self.project_list.listbox.selection_set(i)
                    break
            
            # Update recent projects (moves to top)
            self.settings.add_recent_project(project_path, project_name)
            self.update_recent_projects()
            self.settings.save_settings()
            
            self.log_widget.log(f"Opened recent project: {project_name}")
        else:
            MessageDialog.show_warning(
                "Project Not Found", 
                f"The project path no longer exists:\n{project_path}"
            )
            self.remove_recent_project(project_path)
    
    def remove_recent_project(self, project_path: str):
        """Remove a project from recent projects"""
        self.settings.remove_recent_project(project_path)
        self.update_recent_projects()
        self.settings.save_settings()
    
    def update_recent_projects(self):
        """Update the recent projects display"""
        self.recent_projects.update_recent_projects(self.settings.recent_projects)
    
    def _format_token_details(self, token_estimates: dict) -> str:
        """Format detailed token estimates for display"""
        details = []
        
        # Show estimates for preferred providers: Claude and Gemini 2.5
        if "claude" in token_estimates:
            claude_tokens = token_estimates["claude"].get("claude-3.5-sonnet", 0)
            usage_pct, status = TokenEstimator.get_context_usage(
                claude_tokens, LLMProvider.CLAUDE, "claude-3.5-sonnet"
            )
            details.append(f"Claude: {claude_tokens:,} tokens ({usage_pct:.1f}%) {status}")
        
        if "gemini" in token_estimates:
            gemini_tokens = token_estimates["gemini"].get("gemini-2.5-flash", 0)
            usage_pct, status = TokenEstimator.get_context_usage(
                gemini_tokens, LLMProvider.GEMINI, "gemini-2.5-flash"
            )
            details.append(f"Gemini 2.5: {gemini_tokens:,} tokens ({usage_pct:.1f}%) {status}")
        
        return " | ".join(details)
    
    def load_settings(self):
        if self.settings.project_dir:
            self.project_list.update_directory_label(self.settings.project_dir)
        
        # Load recent projects
        self.update_recent_projects()
        
        # Load ignore patterns
        if self.settings.last_ignore_list:
            patterns = self.settings.last_ignore_list
        else:
            patterns = self.settings.get_default_patterns()
            self.log_widget.log("Loaded comprehensive default ignore patterns.")
        
        self.ignore_text.insert("1.0", "\n".join(patterns))
    
    def on_closing(self):
        # Save current ignore patterns
        raw_patterns = self.ignore_text.get("1.0", tk.END).strip().split("\n")
        self.settings.last_ignore_list = [p for p in raw_patterns if p]
        self.settings.save_settings()
        self.destroy()