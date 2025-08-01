import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict, Set, Optional, Callable, Tuple
from pathlib import Path
import re
from dataclasses import dataclass
from enum import Enum

from ..core.importance_scorer import FileImportanceScorer, FileImportanceInfo


class PatternType(Enum):
    FILE_EXTENSION = "file_extension"
    DIRECTORY = "directory"
    FILENAME = "filename"
    PATH_PATTERN = "path_pattern"
    SIZE_BASED = "size_based"
    CONTENT_BASED = "content_based"


@dataclass
class PatternSuggestion:
    pattern: str
    pattern_type: PatternType
    description: str
    files_affected: int
    tokens_saved: int
    confidence: float
    reasoning: str


@dataclass
class ProjectAnalysis:
    total_files: int
    total_tokens: int
    file_types: Dict[str, int]  # extension -> count
    large_files: List[Tuple[str, int]]  # (path, tokens)
    directories: Dict[str, int]  # directory -> file_count
    generated_files: List[str]
    test_files: List[str]
    config_files: List[str]


class PatternWizard(tk.Toplevel):
    """
    Interactive pattern wizard that helps users build optimal ignore patterns
    through visual analysis and intelligent suggestions.
    """
    
    # Common file patterns by category
    PATTERN_CATEGORIES = {
        "Generated Files": [
            ("*.pyc", "Python bytecode"),
            ("*.class", "Java compiled classes"),
            ("*.o", "Object files"),
            ("*.so", "Shared objects"),
            ("*.dll", "Dynamic libraries"),
            ("*.exe", "Executables"),
            ("*_pb2.py", "Protocol buffer generated files"),
            ("*.generated.*", "Generated source files"),
            ("*.g.py", "ANTLR generated Python"),
            ("*.g.java", "ANTLR generated Java"),
        ],
        "Build Artifacts": [
            ("build/", "Build output directory"),
            ("dist/", "Distribution directory"),
            ("target/", "Maven/Gradle target directory"),
            ("out/", "Output directory"),
            ("bin/", "Binary directory"),
            ("obj/", "Object files directory"),
            (".gradle/", "Gradle cache"),
            ("node_modules/", "npm dependencies"),
            ("__pycache__/", "Python cache"),
        ],
        "Development Tools": [
            (".git/", "Git repository data"),
            (".svn/", "Subversion data"),
            (".hg/", "Mercurial data"),
            (".vscode/", "VS Code settings"),
            (".idea/", "IntelliJ IDEA settings"),
            ("*.log", "Log files"),
            ("*.tmp", "Temporary files"),
            ("*.swp", "Vim swap files"),
            ("*.bak", "Backup files"),
            (".DS_Store", "macOS metadata"),
        ],
        "Test Files": [
            ("test/", "Test directory"),
            ("tests/", "Tests directory"),
            ("*test.py", "Python test files"),
            ("*Test.java", "Java test files"),
            ("*.test.js", "JavaScript test files"),
            ("*.spec.js", "JavaScript spec files"),
            ("spec/", "Spec directory"),
        ],
        "Documentation": [
            ("docs/", "Documentation directory"),
            ("*.md", "Markdown files"),
            ("*.rst", "reStructuredText files"),
            ("*.txt", "Text files"),
            ("README*", "README files"),
            ("CHANGELOG*", "Changelog files"),
            ("LICENSE*", "License files"),
        ],
        "Media & Assets": [
            ("assets/", "Assets directory"),
            ("static/", "Static files"),
            ("public/", "Public files"),
            ("*.png", "PNG images"),
            ("*.jpg", "JPEG images"),
            ("*.gif", "GIF images"),
            ("*.svg", "SVG images"),
            ("*.ico", "Icon files"),
            ("*.woff*", "Web fonts"),
            ("*.css", "Stylesheets"),
            ("*.scss", "Sass stylesheets"),
        ],
        "Configuration": [
            ("*.json", "JSON configuration"),
            ("*.yaml", "YAML configuration"),
            ("*.yml", "YAML configuration"),
            ("*.toml", "TOML configuration"),
            ("*.ini", "INI configuration"),
            ("*.cfg", "Configuration files"),
            (".env*", "Environment files"),
            ("config/", "Configuration directory"),
        ]
    }
    
    def __init__(self, parent, project_path: Path, current_patterns: List[str] = None):
        super().__init__(parent)
        
        self.project_path = project_path
        self.current_patterns = current_patterns or []
        self.suggested_patterns: List[PatternSuggestion] = []
        self.project_analysis: Optional[ProjectAnalysis] = None
        
        # Callbacks
        self.on_patterns_updated: Optional[Callable[[List[str]], None]] = None
        
        # UI state
        self.pattern_vars: Dict[str, tk.BooleanVar] = {}
        self.custom_patterns: List[str] = []
        
        self.setup_window()
        self.create_widgets()
        self.analyze_project()
    
    def setup_window(self):
        """Setup the wizard window."""
        self.title("Pattern Wizard - Smart Ignore Pattern Builder")
        self.geometry("900x700")
        self.resizable(True, True)
        
        # Make modal
        self.transient(self.master)
        self.grab_set()
        
        # Center on parent
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (self.winfo_width() // 2)
        y = (self.winfo_screenheight() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")
    
    def create_widgets(self):
        """Create the wizard widgets."""
        # Main container
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        self.create_header(main_frame)
        
        # Notebook for different sections
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Analysis tab
        self.create_analysis_tab()
        
        # Suggestions tab
        self.create_suggestions_tab()
        
        # Categories tab
        self.create_categories_tab()
        
        # Custom patterns tab
        self.create_custom_tab()
        
        # Preview tab
        self.create_preview_tab()
        
        # Bottom buttons
        self.create_button_panel(main_frame)
    
    def create_header(self, parent):
        """Create the header section."""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title and description
        title_label = ttk.Label(header_frame, text="Smart Ignore Pattern Builder", 
                               font=('TkDefaultFont', 14, 'bold'))
        title_label.pack(anchor=tk.W)
        
        desc_label = ttk.Label(header_frame, 
                              text=f"Project: {self.project_path.name} • Build optimal ignore patterns to reduce token usage",
                              font=('TkDefaultFont', 9), foreground='gray')
        desc_label.pack(anchor=tk.W, pady=(2, 0))
        
        # Progress indicator
        self.progress_frame = ttk.Frame(header_frame)
        self.progress_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.analysis_progress = ttk.Progressbar(self.progress_frame, mode='indeterminate')
        self.analysis_progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.status_label = ttk.Label(self.progress_frame, text="Analyzing project...")
        self.status_label.pack(side=tk.RIGHT, padx=(10, 0))
    
    def create_analysis_tab(self):
        """Create the project analysis tab."""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="📊 Analysis")
        
        # Scrollable frame
        canvas = tk.Canvas(analysis_frame)
        scrollbar = ttk.Scrollbar(analysis_frame, orient="vertical", command=canvas.yview)
        self.analysis_content = ttk.Frame(canvas)
        
        self.analysis_content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.analysis_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Initial placeholder
        ttk.Label(self.analysis_content, text="Analyzing project... Please wait.", 
                 font=('TkDefaultFont', 12)).pack(pady=50)
    
    def create_suggestions_tab(self):
        """Create the AI suggestions tab."""
        suggestions_frame = ttk.Frame(self.notebook)
        self.notebook.add(suggestions_frame, text="🤖 Smart Suggestions")
        
        # Header
        header = ttk.Frame(suggestions_frame)
        header.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(header, text="AI-Generated Pattern Suggestions", 
                 font=('TkDefaultFont', 12, 'bold')).pack(side=tk.LEFT)
        
        ttk.Button(header, text="Refresh Suggestions", 
                  command=self.refresh_suggestions).pack(side=tk.RIGHT)
        
        # Suggestions list
        suggestions_container = ttk.Frame(suggestions_frame)
        suggestions_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Create treeview for suggestions
        columns = ('pattern', 'files', 'tokens', 'confidence')
        self.suggestions_tree = ttk.Treeview(suggestions_container, columns=columns, show='headings', height=15)
        
        # Configure columns
        self.suggestions_tree.heading('pattern', text='Pattern')
        self.suggestions_tree.heading('files', text='Files Affected')
        self.suggestions_tree.heading('tokens', text='Tokens Saved')
        self.suggestions_tree.heading('confidence', text='Confidence')
        
        self.suggestions_tree.column('pattern', width=300)
        self.suggestions_tree.column('files', width=100, anchor=tk.CENTER)
        self.suggestions_tree.column('tokens', width=100, anchor=tk.CENTER)
        self.suggestions_tree.column('confidence', width=100, anchor=tk.CENTER)
        
        # Scrollbar for suggestions
        suggestions_scroll = ttk.Scrollbar(suggestions_container, orient=tk.VERTICAL, command=self.suggestions_tree.yview)
        self.suggestions_tree.configure(yscrollcommand=suggestions_scroll.set)
        
        self.suggestions_tree.grid(row=0, column=0, sticky='nsew')
        suggestions_scroll.grid(row=0, column=1, sticky='ns')
        
        suggestions_container.grid_rowconfigure(0, weight=1)
        suggestions_container.grid_columnconfigure(0, weight=1)
        
        # Suggestion details
        details_frame = ttk.LabelFrame(suggestions_frame, text="Suggestion Details", padding=10)
        details_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.suggestion_details = tk.Text(details_frame, height=6, wrap=tk.WORD, font=('TkDefaultFont', 9))
        details_scroll = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self.suggestion_details.yview)
        self.suggestion_details.configure(yscrollcommand=details_scroll.set)
        
        self.suggestion_details.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Apply suggestions buttons
        buttons_frame = ttk.Frame(suggestions_frame)
        buttons_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(buttons_frame, text="Apply Selected", command=self.apply_selected_suggestions).pack(side=tk.LEFT)
        ttk.Button(buttons_frame, text="Apply All High Confidence", command=self.apply_high_confidence).pack(side=tk.LEFT, padx=(10, 0))
        
        # Bind selection event
        self.suggestions_tree.bind('<<TreeviewSelect>>', self.on_suggestion_select)
    
    def create_categories_tab(self):
        """Create the pattern categories tab."""
        categories_frame = ttk.Frame(self.notebook)
        self.notebook.add(categories_frame, text="📁 Categories")
        
        # Scrollable frame
        canvas = tk.Canvas(categories_frame)
        scrollbar = ttk.Scrollbar(categories_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create category sections
        for category, patterns in self.PATTERN_CATEGORIES.items():
            self.create_category_section(scrollable_frame, category, patterns)
    
    def create_category_section(self, parent, category: str, patterns: List[Tuple[str, str]]):
        """Create a section for a pattern category."""
        # Category frame
        category_frame = ttk.LabelFrame(parent, text=category, padding=10)
        category_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Select all for category
        header_frame = ttk.Frame(category_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        category_var = tk.BooleanVar()
        category_check = ttk.Checkbutton(header_frame, text=f"Select All {category}", 
                                        variable=category_var,
                                        command=lambda: self.toggle_category(category, category_var.get()))
        category_check.pack(side=tk.LEFT)
        
        # Pattern list
        patterns_frame = ttk.Frame(category_frame)
        patterns_frame.pack(fill=tk.X)
        
        for pattern, description in patterns:
            pattern_frame = ttk.Frame(patterns_frame)
            pattern_frame.pack(fill=tk.X, pady=2)
            
            var = tk.BooleanVar()
            var.set(pattern in self.current_patterns)
            self.pattern_vars[pattern] = var
            
            check = ttk.Checkbutton(pattern_frame, text=pattern, variable=var, width=20)
            check.pack(side=tk.LEFT)
            
            desc_label = ttk.Label(pattern_frame, text=description, foreground='gray', font=('TkDefaultFont', 8))
            desc_label.pack(side=tk.LEFT, padx=(10, 0))
    
    def create_custom_tab(self):
        """Create the custom patterns tab."""
        custom_frame = ttk.Frame(self.notebook)
        self.notebook.add(custom_frame, text="✏️ Custom")
        
        # Instructions
        instructions = ttk.Label(custom_frame, 
                                text="Add custom ignore patterns. Use wildcards (*) and directory separators (/).",
                                font=('TkDefaultFont', 10))
        instructions.pack(pady=10)
        
        # Pattern entry
        entry_frame = ttk.Frame(custom_frame)
        entry_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(entry_frame, text="Pattern:").pack(side=tk.LEFT)
        self.custom_pattern_entry = ttk.Entry(entry_frame)
        self.custom_pattern_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        self.custom_pattern_entry.bind('<Return>', self.add_custom_pattern)
        
        ttk.Button(entry_frame, text="Add", command=self.add_custom_pattern).pack(side=tk.RIGHT)
        
        # Pattern list
        list_frame = ttk.Frame(custom_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(list_frame, text="Custom Patterns:", font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W)
        
        # Listbox with scrollbar
        listbox_frame = ttk.Frame(list_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.custom_listbox = tk.Listbox(listbox_frame)
        custom_scroll = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.custom_listbox.yview)
        self.custom_listbox.configure(yscrollcommand=custom_scroll.set)
        
        self.custom_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        custom_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Custom pattern buttons
        custom_buttons = ttk.Frame(list_frame)
        custom_buttons.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(custom_buttons, text="Remove Selected", command=self.remove_custom_pattern).pack(side=tk.LEFT)
        ttk.Button(custom_buttons, text="Clear All", command=self.clear_custom_patterns).pack(side=tk.LEFT, padx=(10, 0))
        
        # Pattern testing
        test_frame = ttk.LabelFrame(custom_frame, text="Pattern Tester", padding=10)
        test_frame.pack(fill=tk.X, padx=10, pady=10)
        
        test_entry_frame = ttk.Frame(test_frame)
        test_entry_frame.pack(fill=tk.X)
        
        ttk.Label(test_entry_frame, text="Test Path:").pack(side=tk.LEFT)
        self.test_path_entry = ttk.Entry(test_entry_frame)
        self.test_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        
        ttk.Button(test_entry_frame, text="Test", command=self.test_pattern).pack(side=tk.RIGHT)
        
        self.test_result_label = ttk.Label(test_frame, text="", font=('TkDefaultFont', 9))
        self.test_result_label.pack(pady=(10, 0))
    
    def create_preview_tab(self):
        """Create the preview tab."""
        preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(preview_frame, text="👁️ Preview")
        
        # Header
        header_frame = ttk.Frame(preview_frame)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(header_frame, text="Pattern Preview", font=('TkDefaultFont', 12, 'bold')).pack(side=tk.LEFT)
        ttk.Button(header_frame, text="Refresh Preview", command=self.refresh_preview).pack(side=tk.RIGHT)
        
        # Summary stats
        stats_frame = ttk.LabelFrame(preview_frame, text="Impact Summary", padding=10)
        stats_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=6, font=('Consolas', 9))
        self.stats_text.pack(fill=tk.X)
        
        # Files affected
        files_frame = ttk.LabelFrame(preview_frame, text="Files That Will Be Ignored", padding=10)
        files_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Create treeview for affected files
        files_columns = ('file', 'pattern', 'tokens')
        self.preview_tree = ttk.Treeview(files_frame, columns=files_columns, show='headings', height=12)
        
        self.preview_tree.heading('file', text='File Path')
        self.preview_tree.heading('pattern', text='Matched Pattern')
        self.preview_tree.heading('tokens', text='Tokens')
        
        self.preview_tree.column('file', width=400)
        self.preview_tree.column('pattern', width=150)
        self.preview_tree.column('tokens', width=100, anchor=tk.CENTER)
        
        preview_scroll = ttk.Scrollbar(files_frame, orient=tk.VERTICAL, command=self.preview_tree.yview)
        self.preview_tree.configure(yscrollcommand=preview_scroll.set)
        
        self.preview_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        preview_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_button_panel(self, parent):
        """Create the bottom button panel."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Left side - Reset
        ttk.Button(button_frame, text="Reset All", command=self.reset_patterns).pack(side=tk.LEFT)
        
        # Right side - Main actions
        right_buttons = ttk.Frame(button_frame)
        right_buttons.pack(side=tk.RIGHT)
        
        ttk.Button(right_buttons, text="Cancel", command=self.destroy).pack(side=tk.LEFT)
        ttk.Button(right_buttons, text="Apply Patterns", command=self.apply_patterns).pack(side=tk.LEFT, padx=(10, 0))
    
    def analyze_project(self):
        """Analyze the project to understand its structure."""
        # Start progress animation
        self.analysis_progress.start()
        
        # Run analysis in background thread
        import threading
        threading.Thread(target=self._analyze_project_worker, daemon=True).start()
    
    def _analyze_project_worker(self):
        """Worker thread for project analysis."""
        try:
            # Analyze project structure
            analysis = self._perform_project_analysis()
            
            # Generate suggestions
            suggestions = self._generate_suggestions(analysis)
            
            # Update UI in main thread
            self.after(0, lambda: self._update_analysis_ui(analysis, suggestions))
            
        except Exception as e:
            self.after(0, lambda: self._handle_analysis_error(str(e)))
    
    def _perform_project_analysis(self) -> ProjectAnalysis:
        """Perform comprehensive project analysis."""
        file_types = {}
        large_files = []
        directories = {}
        generated_files = []
        test_files = []
        config_files = []
        total_files = 0
        total_tokens = 0
        
        # Walk through project directory
        for root, dirs, files in self.project_path.walk():
            rel_root = root.relative_to(self.project_path)
            dir_key = str(rel_root) if rel_root != Path('.') else 'root'
            directories[dir_key] = len(files)
            
            for file in files:
                file_path = root / file
                rel_path = str(file_path.relative_to(self.project_path))
                
                total_files += 1
                
                # File extension analysis
                ext = file_path.suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1
                
                # Token estimation (simplified)
                try:
                    if file_path.stat().st_size < 100000:  # Only analyze files < 100KB
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Simple token estimation
                        tokens = len(content.split()) * 1.3  # Rough approximation
                        total_tokens += tokens
                        
                        if tokens > 1000:  # Large files
                            large_files.append((rel_path, int(tokens)))
                    
                except Exception:
                    continue
                
                # Categorize files
                if self._is_generated_file(file_path):
                    generated_files.append(rel_path)
                elif self._is_test_file(file_path):
                    test_files.append(rel_path)
                elif self._is_config_file(file_path):
                    config_files.append(rel_path)
        
        # Sort large files by token count
        large_files.sort(key=lambda x: x[1], reverse=True)
        
        return ProjectAnalysis(
            total_files=total_files,
            total_tokens=int(total_tokens),
            file_types=file_types,
            large_files=large_files[:20],  # Top 20 largest
            directories=directories,
            generated_files=generated_files,
            test_files=test_files,
            config_files=config_files
        )
    
    def _is_generated_file(self, file_path: Path) -> bool:
        """Check if file appears to be generated."""
        name = file_path.name.lower()
        return any(pattern in name for pattern in [
            '_pb2.py', '.generated.', '.g.py', '.g.java', 
            'generated', 'auto', '.min.js', '.bundle.'
        ])
    
    def _is_test_file(self, file_path: Path) -> bool:
        """Check if file appears to be a test file."""
        path_str = str(file_path).lower()
        return any(pattern in path_str for pattern in [
            'test', 'spec', '__test__', '_test.', '.test.'
        ])
    
    def _is_config_file(self, file_path: Path) -> bool:
        """Check if file appears to be a configuration file."""
        name = file_path.name.lower()
        ext = file_path.suffix.lower()
        return ext in ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg'] or \
               name.startswith('.env') or 'config' in name
    
    def _generate_suggestions(self, analysis: ProjectAnalysis) -> List[PatternSuggestion]:
        """Generate intelligent pattern suggestions."""
        suggestions = []
        
        # Suggest patterns for large file counts
        for ext, count in analysis.file_types.items():
            if count > 10 and ext in ['.pyc', '.class', '.o', '.log', '.tmp']:
                tokens_saved = count * 50  # Estimate
                suggestions.append(PatternSuggestion(
                    pattern=f"*{ext}",
                    pattern_type=PatternType.FILE_EXTENSION,
                    description=f"Ignore {ext[1:].upper()} files ({count} files)",
                    files_affected=count,
                    tokens_saved=tokens_saved,
                    confidence=0.9,
                    reasoning=f"Found {count} {ext} files. These are typically generated/compiled files."
                ))
        
        # Suggest patterns for test directories
        test_dirs = [d for d in analysis.directories.keys() if 'test' in d.lower()]
        if test_dirs:
            total_test_files = sum(analysis.directories[d] for d in test_dirs)
            suggestions.append(PatternSuggestion(
                pattern="test*/",
                pattern_type=PatternType.DIRECTORY,
                description=f"Ignore test directories ({total_test_files} files)",
                files_affected=total_test_files,
                tokens_saved=total_test_files * 200,  # Estimate
                confidence=0.8,
                reasoning=f"Found {len(test_dirs)} test directories with {total_test_files} files."
            ))
        
        # Suggest patterns for generated files
        if analysis.generated_files:
            suggestions.append(PatternSuggestion(
                pattern="*.generated.*",
                pattern_type=PatternType.FILENAME,
                description=f"Ignore generated files ({len(analysis.generated_files)} files)",
                files_affected=len(analysis.generated_files),
                tokens_saved=len(analysis.generated_files) * 100,
                confidence=0.95,
                reasoning="Generated files are automatically created and don't need to be reviewed."
            ))
        
        # Suggest patterns for large files
        if analysis.large_files:
            largest_files = [f for f, tokens in analysis.large_files[:5] if tokens > 5000]
            if largest_files:
                suggestions.append(PatternSuggestion(
                    pattern="# Large files - consider individual patterns",
                    pattern_type=PatternType.SIZE_BASED,
                    description=f"Consider ignoring very large files ({len(largest_files)} files > 5000 tokens)",
                    files_affected=len(largest_files),
                    tokens_saved=sum(tokens for _, tokens in analysis.large_files[:5] if tokens > 5000),
                    confidence=0.6,
                    reasoning="Very large files may contain auto-generated content or data."
                ))
        
        # Sort by confidence and token savings
        suggestions.sort(key=lambda s: (s.confidence, s.tokens_saved), reverse=True)
        
        return suggestions
    
    def _update_analysis_ui(self, analysis: ProjectAnalysis, suggestions: List[PatternSuggestion]):
        """Update the UI with analysis results."""
        # Stop progress animation
        self.analysis_progress.stop()
        self.status_label.config(text="Analysis complete")
        
        # Store results
        self.project_analysis = analysis
        self.suggested_patterns = suggestions
        
        # Update analysis tab
        self._populate_analysis_tab(analysis)
        
        # Update suggestions tab
        self._populate_suggestions_tab(suggestions)
        
        # Enable tabs
        for i in range(self.notebook.index('end')):
            self.notebook.tab(i, state='normal')
    
    def _populate_analysis_tab(self, analysis: ProjectAnalysis):
        """Populate the analysis tab with results."""
        # Clear existing content
        for widget in self.analysis_content.winfo_children():
            widget.destroy()
        
        # Project overview
        overview_frame = ttk.LabelFrame(self.analysis_content, text="Project Overview", padding=10)
        overview_frame.pack(fill=tk.X, padx=10, pady=10)
        
        overview_text = f"""Total Files: {analysis.total_files:,}
Total Estimated Tokens: {analysis.total_tokens:,}
Average Tokens per File: {analysis.total_tokens // max(analysis.total_files, 1):,}
Unique File Types: {len(analysis.file_types)}"""
        
        ttk.Label(overview_frame, text=overview_text, font=('Consolas', 10)).pack(anchor=tk.W)
        
        # File types breakdown
        types_frame = ttk.LabelFrame(self.analysis_content, text="File Types", padding=10)
        types_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Sort file types by count
        sorted_types = sorted(analysis.file_types.items(), key=lambda x: x[1], reverse=True)
        
        types_text = "Extension | Count | Percentage\n" + "-" * 35 + "\n"
        for ext, count in sorted_types[:15]:  # Top 15 types
            percentage = (count / analysis.total_files) * 100
            ext_display = ext if ext else "(no ext)"
            types_text += f"{ext_display:<10} | {count:>5} | {percentage:>6.1f}%\n"
        
        types_label = ttk.Label(types_frame, text=types_text, font=('Consolas', 9))
        types_label.pack(anchor=tk.W)
        
        # Large files
        if analysis.large_files:
            large_frame = ttk.LabelFrame(self.analysis_content, text="Largest Files (by tokens)", padding=10)
            large_frame.pack(fill=tk.X, padx=10, pady=10)
            
            large_text = "File Path | Tokens\n" + "-" * 50 + "\n"
            for file_path, tokens in analysis.large_files[:10]:
                large_text += f"{file_path[:40]:<40} | {tokens:>8,}\n"
            
            ttk.Label(large_frame, text=large_text, font=('Consolas', 9)).pack(anchor=tk.W)
        
        # Special file categories
        categories_frame = ttk.LabelFrame(self.analysis_content, text="File Categories", padding=10)
        categories_frame.pack(fill=tk.X, padx=10, pady=10)
        
        categories_text = f"""Generated Files: {len(analysis.generated_files)}
Test Files: {len(analysis.test_files)}
Configuration Files: {len(analysis.config_files)}"""
        
        ttk.Label(categories_frame, text=categories_text, font=('Consolas', 10)).pack(anchor=tk.W)
    
    def _populate_suggestions_tab(self, suggestions: List[PatternSuggestion]):
        """Populate the suggestions tab."""
        # Clear existing items
        for item in self.suggestions_tree.get_children():
            self.suggestions_tree.delete(item)
        
        # Add suggestions
        for suggestion in suggestions:
            confidence_text = f"{suggestion.confidence * 100:.0f}%"
            self.suggestions_tree.insert('', 'end', values=(
                suggestion.pattern,
                suggestion.files_affected,
                f"{suggestion.tokens_saved:,}",
                confidence_text
            ))
    
    def _handle_analysis_error(self, error_msg: str):
        """Handle analysis errors."""
        self.analysis_progress.stop()
        self.status_label.config(text=f"Analysis failed: {error_msg}")
        messagebox.showerror("Analysis Error", f"Failed to analyze project:\n{error_msg}")
    
    # Event handlers
    def on_suggestion_select(self, event):
        """Handle suggestion selection."""
        selection = self.suggestions_tree.selection()
        if not selection:
            return
        
        item = selection[0]
        values = self.suggestions_tree.item(item, 'values')
        
        if not values or not self.suggested_patterns:
            return
        
        # Find the suggestion
        pattern = values[0]
        suggestion = next((s for s in self.suggested_patterns if s.pattern == pattern), None)
        
        if suggestion:
            details_text = f"""Pattern: {suggestion.pattern}
Type: {suggestion.pattern_type.value.replace('_', ' ').title()}
Description: {suggestion.description}

Files Affected: {suggestion.files_affected}
Tokens Saved: {suggestion.tokens_saved:,}
Confidence: {suggestion.confidence * 100:.0f}%

Reasoning: {suggestion.reasoning}"""
            
            self.suggestion_details.delete(1.0, tk.END)
            self.suggestion_details.insert(1.0, details_text)
    
    def toggle_category(self, category: str, selected: bool):
        """Toggle all patterns in a category."""
        if category in self.PATTERN_CATEGORIES:
            for pattern, _ in self.PATTERN_CATEGORIES[category]:
                if pattern in self.pattern_vars:
                    self.pattern_vars[pattern].set(selected)
    
    def add_custom_pattern(self, event=None):
        """Add a custom pattern."""
        pattern = self.custom_pattern_entry.get().strip()
        if pattern and pattern not in self.custom_patterns:
            self.custom_patterns.append(pattern)
            self.custom_listbox.insert(tk.END, pattern)
            self.custom_pattern_entry.delete(0, tk.END)
    
    def remove_custom_pattern(self):
        """Remove selected custom pattern."""
        selection = self.custom_listbox.curselection()
        if selection:
            index = selection[0]
            pattern = self.custom_listbox.get(index)
            self.custom_patterns.remove(pattern)
            self.custom_listbox.delete(index)
    
    def clear_custom_patterns(self):
        """Clear all custom patterns."""
        self.custom_patterns.clear()
        self.custom_listbox.delete(0, tk.END)
    
    def test_pattern(self):
        """Test a pattern against a path."""
        test_path = self.test_path_entry.get().strip()
        if not test_path:
            return
        
        # Get all current patterns
        patterns = self.get_selected_patterns()
        
        # Test against patterns
        matches = []
        for pattern in patterns:
            if self._pattern_matches(test_path, pattern):
                matches.append(pattern)
        
        if matches:
            self.test_result_label.config(
                text=f"✓ Matches: {', '.join(matches)}", 
                foreground='green'
            )
        else:
            self.test_result_label.config(
                text="✗ No matches", 
                foreground='red'
            )
    
    def _pattern_matches(self, path: str, pattern: str) -> bool:
        """Check if a path matches a pattern."""
        import fnmatch
        
        # Convert pattern to regex-like matching
        if pattern.endswith('/'):
            # Directory pattern
            return any(part == pattern[:-1] for part in path.split('/'))
        else:
            # File pattern
            return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(Path(path).name, pattern)
    
    def refresh_suggestions(self):
        """Refresh AI suggestions."""
        if self.project_analysis:
            suggestions = self._generate_suggestions(self.project_analysis)
            self.suggested_patterns = suggestions
            self._populate_suggestions_tab(suggestions)
    
    def apply_selected_suggestions(self):
        """Apply selected suggestions."""
        selected_items = self.suggestions_tree.selection()
        for item in selected_items:
            values = self.suggestions_tree.item(item, 'values')
            pattern = values[0]
            
            # Add to custom patterns if not already there
            if pattern not in self.custom_patterns:
                self.custom_patterns.append(pattern)
                self.custom_listbox.insert(tk.END, pattern)
    
    def apply_high_confidence(self):
        """Apply all high confidence suggestions."""
        high_confidence = [s for s in self.suggested_patterns if s.confidence >= 0.8]
        
        for suggestion in high_confidence:
            if suggestion.pattern not in self.custom_patterns:
                self.custom_patterns.append(suggestion.pattern)
                self.custom_listbox.insert(tk.END, suggestion.pattern)
        
        messagebox.showinfo("Applied", f"Applied {len(high_confidence)} high confidence suggestions.")
    
    def refresh_preview(self):
        """Refresh the preview of pattern effects."""
        patterns = self.get_selected_patterns()
        
        if not patterns or not self.project_analysis:
            return
        
        # Simulate pattern matching
        affected_files = []
        total_tokens_saved = 0
        
        # This would need to be implemented with actual file matching
        # For now, show placeholder data
        
        stats_text = f"""Current Patterns: {len(patterns)}
Files That Will Be Ignored: {len(affected_files)}
Estimated Tokens Saved: {total_tokens_saved:,}
Reduction Percentage: {(total_tokens_saved / max(self.project_analysis.total_tokens, 1)) * 100:.1f}%"""
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
    
    def get_selected_patterns(self) -> List[str]:
        """Get all selected patterns."""
        patterns = []
        
        # Category patterns
        for pattern, var in self.pattern_vars.items():
            if var.get():
                patterns.append(pattern)
        
        # Custom patterns
        patterns.extend(self.custom_patterns)
        
        return patterns
    
    def reset_patterns(self):
        """Reset all patterns to original state."""
        # Reset category checkboxes
        for var in self.pattern_vars.values():
            var.set(False)
        
        # Set current patterns
        for pattern in self.current_patterns:
            if pattern in self.pattern_vars:
                self.pattern_vars[pattern].set(True)
        
        # Clear custom patterns
        self.clear_custom_patterns()
    
    def apply_patterns(self):
        """Apply the selected patterns."""
        patterns = self.get_selected_patterns()
        
        if self.on_patterns_updated:
            self.on_patterns_updated(patterns)
        
        self.destroy()


def show_pattern_wizard(parent, project_path: Path, current_patterns: List[str] = None) -> List[str]:
    """Show the pattern wizard and return selected patterns."""
    wizard = PatternWizard(parent, project_path, current_patterns)
    
    # Make it modal and wait for result
    result_patterns = []
    
    def on_patterns_updated(patterns):
        nonlocal result_patterns
        result_patterns = patterns
    
    wizard.on_patterns_updated = on_patterns_updated
    wizard.wait_window()
    
    return result_patterns