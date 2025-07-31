import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, List

from ..core import FileTokenInfo, DirectoryTokenInfo


class TokenAnalysisDialog:
    def __init__(self, parent, token_analysis: Dict[str, Any]):
        self.parent = parent
        self.analysis = token_analysis
        self.window = None
        
    def show(self):
        """Display the token analysis dialog."""
        self.window = tk.Toplevel(self.parent)
        self.window.title("Token Analysis - File & Directory Breakdown")
        self.window.geometry("1000x700")
        self.window.transient(self.parent)
        self.window.grab_set()
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top Files tab
        self.create_files_tab(notebook)
        
        # Top Directories tab
        self.create_directories_tab(notebook)
        
        # Suggestions tab
        self.create_suggestions_tab(notebook)
        
        # Close button
        close_frame = ttk.Frame(self.window)
        close_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(
            close_frame, 
            text="Close", 
            command=self.window.destroy
        ).pack(side=tk.RIGHT)
        
        # Center the window
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (self.window.winfo_width() // 2)
        y = (self.window.winfo_screenheight() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f"+{x}+{y}")
    
    def create_files_tab(self, notebook):
        """Create the top files analysis tab."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="📄 Top Files by Tokens")
        
        # Description
        desc_label = ttk.Label(
            frame, 
            text="Files consuming the most tokens (top 20):",
            font=("Segoe UI", 12, "bold")
        )
        desc_label.pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        # Create treeview
        columns = ("tokens", "percentage", "size", "path")
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=20)
        
        # Configure columns
        tree.heading("tokens", text="Tokens")
        tree.heading("percentage", text="% of Total")
        tree.heading("size", text="File Size")
        tree.heading("path", text="File Path")
        
        tree.column("tokens", width=100, anchor="e")
        tree.column("percentage", width=100, anchor="e")
        tree.column("size", width=100, anchor="e")
        tree.column("path", width=600, anchor="w")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate with data
        top_files = self.analysis.get('top_files', [])
        for file_info in top_files:
            size_str = self._format_file_size(file_info.size_bytes)
            tree.insert("", "end", values=(
                f"{file_info.tokens:,}",
                f"{file_info.percentage:.1f}%",
                size_str,
                file_info.relative_path
            ))
    
    def create_directories_tab(self, notebook):
        """Create the top directories analysis tab."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="📁 Top Directories by Tokens")
        
        # Description
        desc_label = ttk.Label(
            frame, 
            text="Directories consuming the most tokens (including subdirectories):",
            font=("Segoe UI", 12, "bold")
        )
        desc_label.pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        # Create treeview
        columns = ("tokens", "percentage", "files", "path")
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=20)
        
        # Configure columns
        tree.heading("tokens", text="Tokens")
        tree.heading("percentage", text="% of Total")
        tree.heading("files", text="File Count")
        tree.heading("path", text="Directory Path")
        
        tree.column("tokens", width=120, anchor="e")
        tree.column("percentage", width=100, anchor="e")
        tree.column("files", width=100, anchor="e")
        tree.column("path", width=600, anchor="w")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate with data
        top_dirs = self.analysis.get('top_directories', [])
        for dir_info in top_dirs:
            tree.insert("", "end", values=(
                f"{dir_info.tokens:,}",
                f"{dir_info.percentage:.1f}%",
                f"{dir_info.file_count}",
                dir_info.path or "(root)"
            ))
    
    def create_suggestions_tab(self, notebook):
        """Create the optimization suggestions tab."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="💡 Optimization Suggestions")
        
        # Description
        desc_label = ttk.Label(
            frame, 
            text="Directories that might be worth ignoring to reduce token usage:",
            font=("Segoe UI", 12, "bold")
        )
        desc_label.pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        # Create text widget for suggestions
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        text_widget = tk.Text(
            text_frame, 
            wrap=tk.WORD, 
            font=("Consolas", 11),
            relief=tk.SUNKEN,
            borderwidth=1
        )
        
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add suggestions content
        suggestions = self.analysis.get('suggestions', [])
        
        if suggestions:
            content = "🔍 DIRECTORIES TO CONSIDER IGNORING:\n\n"
            
            for suggestion in suggestions:
                content += f"• {suggestion}\n"
            
            content += "\n" + "="*60 + "\n\n"
            content += "💡 HOW TO USE THESE SUGGESTIONS:\n\n"
            content += "1. Add directory patterns to your ignore list\n"
            content += "   Example: tests/ or */tests/* or app/tests/\n\n"
            content += "2. Use wildcards for flexibility:\n"
            content += "   • */logs/* - ignores 'logs' in any directory\n"
            content += "   • **/__pycache__/ - ignores Python cache recursively\n"
            content += "   • *.test.js - ignores all test files\n\n"
            content += "3. Consider the trade-off:\n"
            content += "   • Ignoring tests saves tokens but removes test context\n"
            content += "   • Ignoring logs saves tokens but removes runtime info\n"
            content += "   • Ignoring docs saves tokens but removes documentation\n\n"
            content += "4. Test your changes with 'Preview Included Files' first!\n"
            
        else:
            content = "✅ No optimization suggestions found.\n\n"
            content += "Your current ignore patterns are doing a good job of filtering out\n"
            content += "non-essential directories. All remaining directories appear to contain\n"
            content += "important code files.\n\n"
            content += "💡 TIP: If you still want to reduce tokens, look at the 'Top Files'\n"
            content += "tab to identify individual large files that might not be essential."
        
        text_widget.insert("1.0", content)
        text_widget.config(state="disabled")
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"