import tkinter as tk
from tkinter import ttk, filedialog
import os
from pathlib import Path
from typing import Dict, List, Optional, Callable, Set
import threading
import queue
import time

from ..core.dependency_analyzer import DependencyGraphAnalyzer
from ..core.importance_scorer import FileImportanceScorer, FileImportanceInfo
from ..core.token_budget import SmartTokenBudget, BudgetStrategy
from ..core.tokenizer import TokenEstimator, LLMProvider
from ..core.processor import CodebaseProcessor
from ..core.models import ProcessingOptions


class FileTreeItem:
    """Represents an item in the file tree."""
    def __init__(self, path: Path, is_directory: bool = False):
        self.path = path
        self.is_directory = is_directory
        self.children: List[FileTreeItem] = []
        self.parent: Optional[FileTreeItem] = None
        self.included = True
        self.tokens = 0
        self.importance_score = 0.0
        self.file_type = "unknown"
        self.treeview_id: Optional[str] = None


class InteractiveFileExplorer(ttk.Frame):
    """
    Interactive file explorer with live preview and real-time token counting.
    Allows users to select/deselect files and see immediate feedback.
    """
    
    def __init__(self, parent, project_path: Optional[Path] = None):
        super().__init__(parent)
        self.project_path = project_path
        self.file_tree: Dict[str, FileTreeItem] = {}
        self.root_items: List[FileTreeItem] = []
        self.included_files: Set[str] = set()
        self.ignored_patterns: List[str] = []
        
        # Callbacks
        self.on_selection_changed: Optional[Callable[[List[Path], int], None]] = None
        self.on_pattern_changed: Optional[Callable[[List[str]], None]] = None
        
        # Threading for background operations
        self.analysis_queue = queue.Queue()
        self.analysis_thread: Optional[threading.Thread] = None
        self.analysis_running = False
        
        # Analysis components
        self.dependency_analyzer: Optional[DependencyGraphAnalyzer] = None
        self.importance_scorer: Optional[FileImportanceScorer] = None
        self.token_budget: Optional[SmartTokenBudget] = None
        
        # File analysis cache
        self.file_analysis_cache: Dict[str, FileImportanceInfo] = {}
        self.last_analysis_time = 0
        
        self.create_widgets()
        self.setup_analysis_components()
        
        if self.project_path:
            self.load_project(self.project_path)
    
    def create_widgets(self):
        """Create the UI widgets."""
        # Main container with paned window
        self.paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - File tree
        self.create_file_tree_panel()
        
        # Right panel - Preview and stats
        self.create_preview_panel()
        
        # Bottom panel - Controls
        self.create_controls_panel()
    
    def create_file_tree_panel(self):
        """Create the file tree panel."""
        left_frame = ttk.Frame(self.paned)
        self.paned.add(left_frame, weight=2)
        
        # Header
        header_frame = ttk.Frame(left_frame)
        header_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(header_frame, text="Project Files", font=('TkDefaultFont', 12, 'bold')).pack(side=tk.LEFT)
        
        # Refresh button
        ttk.Button(header_frame, text="↻", width=3, command=self.refresh_tree).pack(side=tk.RIGHT)
        
        # Search box
        search_frame = ttk.Frame(left_frame)
        search_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        ttk.Label(search_frame, text="Filter:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.on_search_changed)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # File tree
        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        
        # Create treeview with columns
        columns = ('tokens', 'importance', 'type')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='tree headings')
        
        # Configure columns
        self.tree.heading('#0', text='File/Directory')
        self.tree.heading('tokens', text='Tokens')
        self.tree.heading('importance', text='Importance')
        self.tree.heading('type', text='Type')
        
        self.tree.column('#0', width=300, minwidth=200)
        self.tree.column('tokens', width=80, minwidth=60, anchor=tk.E)
        self.tree.column('importance', width=80, minwidth=60, anchor=tk.E)
        self.tree.column('type', width=100, minwidth=80)
        
        # Scrollbars
        tree_scroll_v = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        tree_scroll_h = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=tree_scroll_v.set, xscrollcommand=tree_scroll_h.set)
        
        # Pack tree and scrollbars
        self.tree.grid(row=0, column=0, sticky='nsew')
        tree_scroll_v.grid(row=0, column=1, sticky='ns')
        tree_scroll_h.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Bind events
        self.tree.bind('<Button-1>', self.on_tree_click)
        self.tree.bind('<Double-1>', self.on_tree_double_click)
        self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)
    
    def create_preview_panel(self):
        """Create the preview panel."""
        right_frame = ttk.Frame(self.paned)
        self.paned.add(right_frame, weight=1)
        
        # Notebook for different views
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Statistics tab
        self.create_stats_tab()
        
        # File preview tab
        self.create_preview_tab()
        
        # Pattern wizard tab
        self.create_pattern_tab()
    
    def create_stats_tab(self):
        """Create the statistics tab."""
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="Statistics")
        
        # Token budget section
        budget_frame = ttk.LabelFrame(stats_frame, text="Token Budget")
        budget_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Budget input
        budget_input_frame = ttk.Frame(budget_frame)
        budget_input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(budget_input_frame, text="Budget:").pack(side=tk.LEFT)
        self.budget_var = tk.StringVar(value="100000")
        self.budget_var.trace('w', self.on_budget_changed)
        budget_entry = ttk.Entry(budget_input_frame, textvariable=self.budget_var, width=10)
        budget_entry.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(budget_input_frame, text="Strategy:").pack(side=tk.LEFT)
        self.strategy_var = tk.StringVar(value="balanced")
        strategy_combo = ttk.Combobox(budget_input_frame, textvariable=self.strategy_var, 
                                    values=['balanced', 'importance', 'efficiency', 'coverage', 'smart'],
                                    state='readonly', width=12)
        strategy_combo.pack(side=tk.LEFT, padx=(5, 0))
        strategy_combo.bind('<<ComboboxSelected>>', self.on_strategy_changed)
        
        # Current stats
        self.stats_text = tk.Text(budget_frame, height=15, width=40, font=('Consolas', 9))
        stats_scroll = ttk.Scrollbar(budget_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        stats_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Update button
        ttk.Button(budget_frame, text="Update Analysis", command=self.trigger_analysis).pack(pady=5)
    
    def create_preview_tab(self):
        """Create the file preview tab."""
        preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(preview_frame, text="File Preview")
        
        # File info
        info_frame = ttk.Frame(preview_frame)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.file_info_text = tk.Text(info_frame, height=4, font=('TkDefaultFont', 9))
        self.file_info_text.pack(fill=tk.X)
        
        # File content preview
        content_frame = ttk.LabelFrame(preview_frame, text="Content Preview")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.preview_text = tk.Text(content_frame, font=('Consolas', 9), wrap=tk.NONE)
        preview_scroll_v = ttk.Scrollbar(content_frame, orient=tk.VERTICAL, command=self.preview_text.yview)
        preview_scroll_h = ttk.Scrollbar(content_frame, orient=tk.HORIZONTAL, command=self.preview_text.xview)
        self.preview_text.configure(yscrollcommand=preview_scroll_v.set, xscrollcommand=preview_scroll_h.set)
        
        self.preview_text.grid(row=0, column=0, sticky='nsew')
        preview_scroll_v.grid(row=0, column=1, sticky='ns')
        preview_scroll_h.grid(row=1, column=0, sticky='ew')
        
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
    
    def create_pattern_tab(self):
        """Create the pattern wizard tab."""
        pattern_frame = ttk.Frame(self.notebook)
        self.notebook.add(pattern_frame, text="Ignore Patterns")
        
        # Pattern list
        pattern_list_frame = ttk.LabelFrame(pattern_frame, text="Current Patterns")
        pattern_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.pattern_listbox = tk.Listbox(pattern_list_frame)
        pattern_scroll = ttk.Scrollbar(pattern_list_frame, orient=tk.VERTICAL, command=self.pattern_listbox.yview)
        self.pattern_listbox.configure(yscrollcommand=pattern_scroll.set)
        
        self.pattern_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        pattern_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Pattern controls
        controls_frame = ttk.Frame(pattern_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add pattern
        add_frame = ttk.Frame(controls_frame)
        add_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(add_frame, text="Add pattern:").pack(side=tk.LEFT)
        self.pattern_entry = ttk.Entry(add_frame)
        self.pattern_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.pattern_entry.bind('<Return>', self.add_pattern)
        ttk.Button(add_frame, text="Add", command=self.add_pattern).pack(side=tk.RIGHT)
        
        # Pattern buttons
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Remove Selected", command=self.remove_pattern).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Clear All", command=self.clear_patterns).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Load Defaults", command=self.load_default_patterns).pack(side=tk.LEFT)
        
        # Suggestions
        suggestions_frame = ttk.LabelFrame(pattern_frame, text="Suggestions")
        suggestions_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.suggestions_text = tk.Text(suggestions_frame, height=6, font=('TkDefaultFont', 9))
        suggestions_scroll = ttk.Scrollbar(suggestions_frame, orient=tk.VERTICAL, command=self.suggestions_text.yview)
        self.suggestions_text.configure(yscrollcommand=suggestions_scroll.set)
        
        self.suggestions_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        suggestions_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
    
    def create_controls_panel(self):
        """Create the bottom controls panel."""
        controls_frame = ttk.Frame(self)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Left side - Selection info
        info_frame = ttk.Frame(controls_frame)
        info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.selection_label = ttk.Label(info_frame, text="Ready", font=('TkDefaultFont', 9))
        self.selection_label.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(info_frame, variable=self.progress_var, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Right side - Action buttons
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(side=tk.RIGHT)
        
        ttk.Button(button_frame, text="Select All", command=self.select_all).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Select None", command=self.select_none).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Smart Select", command=self.smart_select).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Generate Digest", command=self.generate_digest).pack(side=tk.LEFT)
    
    def setup_analysis_components(self):
        """Setup the analysis components."""
        if self.project_path:
            self.dependency_analyzer = DependencyGraphAnalyzer(self.project_path)
            self.importance_scorer = FileImportanceScorer(self.dependency_analyzer)
            self.token_budget = SmartTokenBudget(self.importance_scorer)
    
    def load_project(self, project_path: Path):
        """Load a project into the file explorer."""
        self.project_path = project_path
        self.setup_analysis_components()
        self.load_default_patterns()
        self.refresh_tree()
    
    def refresh_tree(self):
        """Refresh the file tree."""
        if not self.project_path or not self.project_path.exists():
            return
        
        self.selection_label.config(text="Loading project files...")
        self.progress_var.set(0)
        
        # Clear existing tree
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        self.file_tree.clear()
        self.root_items.clear()
        
        # Start loading in background thread
        if not self.analysis_running:
            self.analysis_running = True
            self.analysis_thread = threading.Thread(target=self._load_project_files, daemon=True)
            self.analysis_thread.start()
            
            # Schedule UI updates
            self.after(100, self._check_loading_progress)
    
    def _load_project_files(self):
        """Load project files in background thread."""
        try:
            # Get files using processor
            processor = CodebaseProcessor()
            options = ProcessingOptions(
                input_dir=str(self.project_path),
                ignore_patterns=self.ignored_patterns,
                size_limit_mb=10.0,
                mode='preview'
            )
            
            result = processor.process(options)
            
            if result.success and result.files:
                # Build file tree structure
                for file_path in result.files:
                    self._add_file_to_tree(file_path)
                
            self.analysis_queue.put(('files_loaded', len(result.files) if result.files else 0))
            
        except Exception as e:
            self.analysis_queue.put(('error', str(e)))
        finally:
            self.analysis_running = False
    
    def _add_file_to_tree(self, file_path: Path):
        """Add a file to the internal tree structure."""
        relative_path = file_path.relative_to(self.project_path)
        path_parts = relative_path.parts
        
        current_level = self.root_items
        current_tree_path = ""
        
        # Create directory structure
        for i, part in enumerate(path_parts[:-1]):
            current_tree_path = os.path.join(current_tree_path, part) if current_tree_path else part
            
            # Find existing directory or create new one
            existing_dir = None
            for item in current_level:
                if item.path.name == part and item.is_directory:
                    existing_dir = item
                    break
            
            if not existing_dir:
                dir_path = self.project_path / current_tree_path
                dir_item = FileTreeItem(dir_path, is_directory=True)
                current_level.append(dir_item)
                self.file_tree[current_tree_path] = dir_item
                current_level = dir_item.children
            else:
                current_level = existing_dir.children
        
        # Add the file
        file_item = FileTreeItem(file_path, is_directory=False)
        current_level.append(file_item)
        self.file_tree[str(relative_path)] = file_item
        self.included_files.add(str(relative_path))
    
    def _check_loading_progress(self):
        """Check loading progress and update UI."""
        try:
            while True:
                try:
                    msg_type, data = self.analysis_queue.get_nowait()
                    
                    if msg_type == 'files_loaded':
                        self._populate_tree_view()
                        self.selection_label.config(text=f"Loaded {data} files")
                        self.progress_var.set(100)
                        self.trigger_analysis()
                        return
                    elif msg_type == 'error':
                        self.selection_label.config(text=f"Error: {data}")
                        self.progress_var.set(0)
                        return
                        
                except queue.Empty:
                    break
            
            # Continue checking if still loading
            if self.analysis_running:
                self.after(100, self._check_loading_progress)
                
        except Exception as e:
            self.selection_label.config(text=f"Error: {e}")
    
    def _populate_tree_view(self):
        """Populate the treeview widget with file tree data."""
        def add_items(items: List[FileTreeItem], parent=""):
            for item in sorted(items, key=lambda x: (not x.is_directory, x.path.name.lower())):
                # Determine display values
                if item.is_directory:
                    icon = "📁"
                    tokens_text = ""
                    importance_text = ""
                    type_text = "Directory"
                else:
                    icon = "📄"
                    tokens_text = str(item.tokens) if item.tokens > 0 else ""
                    importance_text = f"{item.importance_score:.1f}" if item.importance_score > 0 else ""
                    type_text = item.file_type
                
                # Insert item
                item_id = self.tree.insert(parent, 'end', 
                                         text=f"{icon} {item.path.name}",
                                         values=(tokens_text, importance_text, type_text),
                                         tags=('included' if item.path.name in self.included_files else 'excluded',))
                
                item.treeview_id = item_id
                
                # Add children recursively
                if item.is_directory and item.children:
                    add_items(item.children, item_id)
        
        # Configure tags for visual feedback
        self.tree.tag_configure('included', background='#e8f5e8')
        self.tree.tag_configure('excluded', background='#f5e8e8', foreground='gray')
        
        add_items(self.root_items)
    
    def trigger_analysis(self):
        """Trigger file analysis in background."""
        if not self.analysis_running and self.file_tree:
            self.analysis_running = True
            self.selection_label.config(text="Analyzing files...")
            self.progress_var.set(0)
            
            analysis_thread = threading.Thread(target=self._analyze_files, daemon=True)
            analysis_thread.start()
            
            self.after(100, self._check_analysis_progress)
    
    def _analyze_files(self):
        """Analyze files for importance and tokens."""
        try:
            if not self.importance_scorer or not self.file_tree:
                return
            
            # Get all file paths
            file_paths = [item.path for item in self.file_tree.values() if not item.is_directory]
            
            if not file_paths:
                return
            
            # Get token information
            token_info = {}
            for i, file_path in enumerate(file_paths):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    estimates = TokenEstimator.estimate_tokens(content, LLMProvider.CLAUDE)
                    relative_path = str(file_path.relative_to(self.project_path)).replace("\\", "/")
                    token_count = estimates.get("claude-3.5-sonnet", 0)
                    token_info[relative_path] = token_count
                    
                    # Update progress
                    progress = (i + 1) / len(file_paths) * 50  # First 50% for token analysis
                    self.analysis_queue.put(('progress', progress))
                    
                except Exception:
                    continue
            
            # Analyze importance
            scored_files = self.importance_scorer.score_files(file_paths, self.project_path, token_info)
            
            # Update file items with analysis results
            for file_info in scored_files:
                if file_info.relative_path in self.file_tree:
                    item = self.file_tree[file_info.relative_path]
                    item.tokens = file_info.tokens
                    item.importance_score = file_info.importance_score
                    item.file_type = file_info.file_type.value
            
            # Cache results
            self.file_analysis_cache = {f.relative_path: f for f in scored_files}
            self.last_analysis_time = time.time()
            
            self.analysis_queue.put(('analysis_complete', len(scored_files)))
            
        except Exception as e:
            self.analysis_queue.put(('error', str(e)))
        finally:
            self.analysis_running = False
    
    def _check_analysis_progress(self):
        """Check analysis progress and update UI."""
        try:
            while True:
                try:
                    msg_type, data = self.analysis_queue.get_nowait()
                    
                    if msg_type == 'progress':
                        self.progress_var.set(data)
                    elif msg_type == 'analysis_complete':
                        self._update_tree_view_analysis()
                        self._update_statistics()
                        self.selection_label.config(text=f"Analysis complete - {data} files analyzed")
                        self.progress_var.set(100)
                        return
                    elif msg_type == 'error':
                        self.selection_label.config(text=f"Analysis error: {data}")
                        self.progress_var.set(0)
                        return
                        
                except queue.Empty:
                    break
            
            # Continue checking if still analyzing
            if self.analysis_running:
                self.after(100, self._check_analysis_progress)
                
        except Exception as e:
            self.selection_label.config(text=f"Error: {e}")
    
    def _update_tree_view_analysis(self):
        """Update treeview with analysis results."""
        for item in self.file_tree.values():
            if not item.is_directory and item.treeview_id:
                tokens_text = str(item.tokens) if item.tokens > 0 else ""
                importance_text = f"{item.importance_score:.1f}" if item.importance_score > 0 else ""
                
                self.tree.set(item.treeview_id, 'tokens', tokens_text)
                self.tree.set(item.treeview_id, 'importance', importance_text)
                self.tree.set(item.treeview_id, 'type', item.file_type)
    
    def _update_statistics(self):
        """Update the statistics display."""
        if not self.file_analysis_cache:
            return
        
        try:
            budget = int(self.budget_var.get())
        except ValueError:
            budget = 100000
        
        # Get selected files
        selected_files = [info for path, info in self.file_analysis_cache.items() 
                         if path in self.included_files]
        
        if not selected_files:
            return
        
        # Calculate budget allocation
        strategy_map = {
            'importance': BudgetStrategy.IMPORTANCE_FIRST,
            'efficiency': BudgetStrategy.EFFICIENCY_FIRST,
            'balanced': BudgetStrategy.BALANCED,
            'coverage': BudgetStrategy.COVERAGE_FIRST,
            'smart': BudgetStrategy.SMART_SAMPLING
        }
        
        strategy = strategy_map.get(self.strategy_var.get(), BudgetStrategy.BALANCED)
        allocation = self.token_budget.allocate_budget(selected_files, budget, strategy)
        
        # Update statistics text
        stats = f"""CURRENT SELECTION
Files Selected: {len(selected_files)}
Total Files: {len(self.file_analysis_cache)}

TOKEN ANALYSIS
Budget: {budget:,}
Used: {allocation.used_tokens:,}
Utilization: {allocation.budget_utilization:.1f}%
Remaining: {allocation.remaining_tokens:,}

EFFICIENCY METRICS
Efficiency Score: {allocation.efficiency_score:.2f}
Coverage Score: {allocation.coverage_score:.1f}%
Strategy: {allocation.strategy_used.value}

SELECTION BREAKDOWN
Files in Budget: {len(allocation.selected_files)}
Files Rejected: {len(allocation.rejected_files)}

TOP FILES BY IMPORTANCE:"""
        
        # Add top files
        for i, file_info in enumerate(allocation.selected_files[:10]):
            stats += f"\n{i+1:2d}. {file_info.relative_path[:40]:<40} {file_info.importance_score:6.1f} {file_info.tokens:>6,}"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats)
        
        # Update selection label
        total_tokens = sum(info.tokens for info in selected_files)
        self.selection_label.config(
            text=f"{len(selected_files)} files selected • {total_tokens:,} tokens • Budget: {allocation.budget_utilization:.1f}%"
        )
        
        # Update suggestions
        if self.importance_scorer:
            suggestions = self.importance_scorer.suggest_ignore_patterns(list(self.file_analysis_cache.values()))
            
            suggestions_text = "SUGGESTED IGNORE PATTERNS:\n\n"
            for pattern in suggestions[:10]:
                suggestions_text += f"• {pattern}\n"
            
            if not suggestions:
                suggestions_text += "No optimization suggestions available.\nAll files appear to be important."
            
            self.suggestions_text.delete(1.0, tk.END)
            self.suggestions_text.insert(1.0, suggestions_text)
    
    # Event handlers
    def on_tree_click(self, event):
        """Handle tree click events."""
        item_id = self.tree.identify('item', event.x, event.y)
        if not item_id:
            return
        
        # Find the corresponding file item
        file_item = None
        for item in self.file_tree.values():
            if item.treeview_id == item_id:
                file_item = item
                break
        
        if file_item and not file_item.is_directory:
            # Toggle inclusion
            relative_path = str(file_item.path.relative_to(self.project_path)).replace("\\", "/")
            
            if relative_path in self.included_files:
                self.included_files.remove(relative_path)
                self.tree.item(item_id, tags=('excluded',))
            else:
                self.included_files.add(relative_path)
                self.tree.item(item_id, tags=('included',))
            
            self._update_statistics()
    
    def on_tree_double_click(self, event):
        """Handle tree double-click events."""
        item_id = self.tree.identify('item', event.x, event.y)
        if not item_id:
            return
        
        # Find the corresponding file item
        file_item = None
        for item in self.file_tree.values():
            if item.treeview_id == item_id:
                file_item = item
                break
        
        if file_item and not file_item.is_directory:
            self._show_file_preview(file_item)
    
    def on_tree_select(self, event):
        """Handle tree selection events."""
        pass  # Could be used for additional selection handling
    
    def on_search_changed(self, *args):
        """Handle search filter changes."""
        search_term = self.search_var.get().lower()
        if not search_term:
            # Show all items
            for item_id in self.tree.get_children():
                self._show_tree_item(item_id, True)
        else:
            # Filter items
            for item_id in self.tree.get_children():
                self._filter_tree_item(item_id, search_term)
    
    def _show_tree_item(self, item_id, show):
        """Show or hide tree item."""
        if show:
            self.tree.move(item_id, '', 'end')
        else:
            self.tree.detach(item_id)
        
        # Handle children
        for child_id in self.tree.get_children(item_id):
            self._show_tree_item(child_id, show)
    
    def _filter_tree_item(self, item_id, search_term):
        """Filter tree item based on search term."""
        item_text = self.tree.item(item_id, 'text').lower()
        matches = search_term in item_text
        
        # Check children
        child_matches = False
        for child_id in self.tree.get_children(item_id):
            if self._filter_tree_item(child_id, search_term):
                child_matches = True
        
        show = matches or child_matches
        self._show_tree_item(item_id, show)
        return show
    
    def on_budget_changed(self, *args):
        """Handle budget change."""
        self._update_statistics()
    
    def on_strategy_changed(self, *args):
        """Handle strategy change."""
        self._update_statistics()
    
    def _show_file_preview(self, file_item: FileTreeItem):
        """Show file preview in the preview tab."""
        self.notebook.select(1)  # Switch to preview tab
        
        # Update file info
        info_text = f"""File: {file_item.path.name}
Path: {file_item.path.relative_to(self.project_path)}
Type: {file_item.file_type}
Tokens: {file_item.tokens:,}
Importance: {file_item.importance_score:.2f}
Size: {file_item.path.stat().st_size:,} bytes
"""
        
        self.file_info_text.delete(1.0, tk.END)
        self.file_info_text.insert(1.0, info_text)
        
        # Load file content
        try:
            with open(file_item.path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(1.0, content)
            
        except Exception as e:
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(1.0, f"Could not read file: {e}")
    
    # Pattern management
    def add_pattern(self, event=None):
        """Add ignore pattern."""
        pattern = self.pattern_entry.get().strip()
        if pattern and pattern not in self.ignored_patterns:
            self.ignored_patterns.append(pattern)
            self.pattern_listbox.insert(tk.END, pattern)
            self.pattern_entry.delete(0, tk.END)
            self.refresh_tree()
    
    def remove_pattern(self):
        """Remove selected pattern."""
        selection = self.pattern_listbox.curselection()
        if selection:
            index = selection[0]
            pattern = self.pattern_listbox.get(index)
            self.ignored_patterns.remove(pattern)
            self.pattern_listbox.delete(index)
            self.refresh_tree()
    
    def clear_patterns(self):
        """Clear all patterns."""
        self.ignored_patterns.clear()
        self.pattern_listbox.delete(0, tk.END)
        self.refresh_tree()
    
    def load_default_patterns(self):
        """Load default ignore patterns."""
        default_patterns = [
            "*.pyc", "__pycache__", ".git", "node_modules", 
            "*.min.js", "*.bundle.js", "dist", "build",
            "*.log", "*.tmp", ".env", "*.class"
        ]
        
        self.clear_patterns()
        for pattern in default_patterns:
            self.ignored_patterns.append(pattern)
            self.pattern_listbox.insert(tk.END, pattern)
        
        self.refresh_tree()
    
    # Selection controls
    def select_all(self):
        """Select all files."""
        for item in self.file_tree.values():
            if not item.is_directory:
                relative_path = str(item.path.relative_to(self.project_path)).replace("\\", "/")
                self.included_files.add(relative_path)
                if item.treeview_id:
                    self.tree.item(item.treeview_id, tags=('included',))
        
        self._update_statistics()
    
    def select_none(self):
        """Deselect all files."""
        self.included_files.clear()
        
        for item in self.file_tree.values():
            if not item.is_directory and item.treeview_id:
                self.tree.item(item.treeview_id, tags=('excluded',))
        
        self._update_statistics()
    
    def smart_select(self):
        """Smart file selection based on budget and importance."""
        if not self.token_budget or not self.file_analysis_cache:
            return
        
        try:
            budget = int(self.budget_var.get())
        except ValueError:
            budget = 100000
        
        # Get all files
        all_files = list(self.file_analysis_cache.values())
        
        # Allocate budget
        strategy_map = {
            'importance': BudgetStrategy.IMPORTANCE_FIRST,
            'efficiency': BudgetStrategy.EFFICIENCY_FIRST,
            'balanced': BudgetStrategy.BALANCED,
            'coverage': BudgetStrategy.COVERAGE_FIRST,
            'smart': BudgetStrategy.SMART_SAMPLING
        }
        
        strategy = strategy_map.get(self.strategy_var.get(), BudgetStrategy.BALANCED)
        allocation = self.token_budget.allocate_budget(all_files, budget, strategy)
        
        # Update selection
        self.included_files.clear()
        for file_info in allocation.selected_files:
            self.included_files.add(file_info.relative_path)
        
        # Update tree view
        for item in self.file_tree.values():
            if not item.is_directory and item.treeview_id:
                relative_path = str(item.path.relative_to(self.project_path)).replace("\\", "/")
                if relative_path in self.included_files:
                    self.tree.item(item.treeview_id, tags=('included',))
                else:
                    self.tree.item(item.treeview_id, tags=('excluded',))
        
        self._update_statistics()
    
    def generate_digest(self):
        """Generate digest with selected files."""
        if self.on_selection_changed:
            selected_paths = [self.file_tree[path].path for path in self.included_files 
                            if path in self.file_tree]
            total_tokens = sum(self.file_tree[path].tokens for path in self.included_files 
                             if path in self.file_tree)
            self.on_selection_changed(selected_paths, total_tokens)
    
    def get_selected_files(self) -> List[Path]:
        """Get list of currently selected file paths."""
        return [self.file_tree[path].path for path in self.included_files 
                if path in self.file_tree]
    
    def get_selected_token_count(self) -> int:
        """Get total token count of selected files."""
        return sum(self.file_tree[path].tokens for path in self.included_files 
                  if path in self.file_tree)