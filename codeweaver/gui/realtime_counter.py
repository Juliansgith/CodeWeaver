import tkinter as tk
from tkinter import ttk
import threading
import queue
import time
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from ..core.tokenizer import TokenEstimator, LLMProvider
from ..core.models import ProcessingOptions


class CounterState(Enum):
    IDLE = "idle"
    COUNTING = "counting"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class TokenCountUpdate:
    """Represents a token count update."""
    file_path: str
    tokens: int
    progress: float
    total_files: int
    processed_files: int


@dataclass
class LLMProviderInfo:
    """Information about an LLM provider and its models."""
    name: str
    display_name: str
    models: Dict[str, int]  # model_name -> context_limit
    color: str  # Color for display


class RealTimeTokenCounter(ttk.Frame):
    """
    Real-time token counter that provides live feedback as files are selected/deselected.
    Shows token counts for multiple LLM providers and models simultaneously.
    """
    
    # LLM Provider configurations
    LLM_PROVIDERS = {
        LLMProvider.CLAUDE: LLMProviderInfo(
            name="claude",
            display_name="Claude",
            models={
                "claude-3-haiku": 200000,
                "claude-3-sonnet": 200000,
                "claude-3-opus": 200000,
                "claude-3.5-sonnet": 200000,
            },
            color="#FF6B35"
        ),
        LLMProvider.GPT: LLMProviderInfo(
            name="gpt",
            display_name="OpenAI GPT",
            models={
                "gpt-4": 128000,
                "gpt-4-turbo": 128000,
                "gpt-4o": 128000,
                "gpt-3.5-turbo": 16385,
            },
            color="#00A67E"
        ),
        LLMProvider.GEMINI: LLMProviderInfo(
            name="gemini",
            display_name="Google Gemini",
            models={
                "gemini-pro": 32768,
                "gemini-1.5-pro": 2000000,
                "gemini-2.0-flash": 1000000,
            },
            color="#4285F4"
        ),
        LLMProvider.LLAMA: LLMProviderInfo(
            name="llama",
            display_name="Meta LLaMA",
            models={
                "llama-2-70b": 4096,
                "llama-3-70b": 8192,
                "codellama-34b": 16384,
            },
            color="#FF8C00"
        )
    }
    
    def __init__(self, parent):
        super().__init__(parent)
        
        # State management
        self.state = CounterState.IDLE
        self.current_files: List[Path] = []
        self.token_cache: Dict[str, Dict[str, int]] = {}  # file_path -> {provider_model: tokens}
        self.total_tokens: Dict[str, int] = {}  # provider_model -> total_tokens
        
        # Threading
        self.count_queue = queue.Queue()
        self.count_thread: Optional[threading.Thread] = None
        self.counting_active = False
        
        # Callbacks
        self.on_count_updated: Optional[Callable[[Dict[str, int]], None]] = None
        
        # UI Components storage
        self.provider_frames: Dict[str, ttk.Frame] = {}
        self.model_labels: Dict[str, Dict[str, ttk.Label]] = {}
        self.progress_bars: Dict[str, ttk.Progressbar] = {}
        self.status_labels: Dict[str, ttk.Label] = {}
        
        self.create_widgets()
        self.start_queue_processor()
    
    def create_widgets(self):
        """Create the token counter widgets."""
        # Main container
        main_frame = ttk.LabelFrame(self, text="Real-Time Token Counter", padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        self.create_control_panel(main_frame)
        
        # Provider panels
        self.create_provider_panels(main_frame)
        
        # Summary panel
        self.create_summary_panel(main_frame)
    
    def create_control_panel(self, parent):
        """Create the control panel."""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Status indicator
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.status_indicator = tk.Canvas(status_frame, width=12, height=12)
        self.status_indicator.pack(side=tk.LEFT, padx=(0, 5))
        self.status_indicator.create_oval(2, 2, 10, 10, fill="gray", outline="")
        
        self.main_status_label = ttk.Label(status_frame, text="Ready", font=('TkDefaultFont', 9))
        self.main_status_label.pack(side=tk.LEFT)
        
        # Controls
        controls_frame = ttk.Frame(control_frame)
        controls_frame.pack(side=tk.RIGHT)
        
        self.recount_button = ttk.Button(controls_frame, text="Recount", command=self.recount_tokens)
        self.recount_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.clear_button = ttk.Button(controls_frame, text="Clear Cache", command=self.clear_cache)
        self.clear_button.pack(side=tk.LEFT)
    
    def create_provider_panels(self, parent):
        """Create panels for each LLM provider."""
        providers_frame = ttk.Frame(parent)
        providers_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create a scrollable frame for providers
        canvas = tk.Canvas(providers_frame)
        scrollbar = ttk.Scrollbar(providers_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create provider panels
        for provider, info in self.LLM_PROVIDERS.items():
            self.create_provider_panel(scrollable_frame, provider, info)
    
    def create_provider_panel(self, parent, provider: LLMProvider, info: LLMProviderInfo):
        """Create a panel for a specific LLM provider."""
        # Provider frame
        provider_frame = ttk.LabelFrame(parent, text=info.display_name, padding=8)
        provider_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.provider_frames[provider.value] = provider_frame
        self.model_labels[provider.value] = {}
        
        # Progress bar for this provider
        progress_frame = ttk.Frame(provider_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 5))
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, mode='determinate')
        progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        status_label = ttk.Label(progress_frame, text="Ready", font=('TkDefaultFont', 8))
        status_label.pack(side=tk.RIGHT, padx=(5, 0))
        
        self.progress_bars[provider.value] = progress_bar
        self.status_labels[provider.value] = status_label
        
        # Model information grid
        models_frame = ttk.Frame(provider_frame)
        models_frame.pack(fill=tk.X)
        
        # Headers
        ttk.Label(models_frame, text="Model", font=('TkDefaultFont', 9, 'bold')).grid(
            row=0, column=0, sticky=tk.W, padx=(0, 20)
        )
        ttk.Label(models_frame, text="Tokens", font=('TkDefaultFont', 9, 'bold')).grid(
            row=0, column=1, sticky=tk.E, padx=(0, 20)
        )
        ttk.Label(models_frame, text="Context Limit", font=('TkDefaultFont', 9, 'bold')).grid(
            row=0, column=2, sticky=tk.E, padx=(0, 20)
        )
        ttk.Label(models_frame, text="Usage", font=('TkDefaultFont', 9, 'bold')).grid(
            row=0, column=3, sticky=tk.E
        )
        
        # Model rows
        for row, (model_name, context_limit) in enumerate(info.models.items(), 1):
            # Model name
            model_label = ttk.Label(models_frame, text=model_name)
            model_label.grid(row=row, column=0, sticky=tk.W, padx=(0, 20), pady=2)
            
            # Token count
            tokens_label = ttk.Label(models_frame, text="0", font=('Consolas', 9))
            tokens_label.grid(row=row, column=1, sticky=tk.E, padx=(0, 20), pady=2)
            
            # Context limit
            limit_label = ttk.Label(models_frame, text=f"{context_limit:,}", 
                                  font=('Consolas', 9), foreground="gray")
            limit_label.grid(row=row, column=2, sticky=tk.E, padx=(0, 20), pady=2)
            
            # Usage percentage
            usage_label = ttk.Label(models_frame, text="0.0%", font=('Consolas', 9))
            usage_label.grid(row=row, column=3, sticky=tk.E, pady=2)
            
            # Store references
            model_key = f"{provider.value}_{model_name}"
            self.model_labels[provider.value][model_name] = {
                'tokens': tokens_label,
                'usage': usage_label,
                'limit': context_limit
            }
    
    def create_summary_panel(self, parent):
        """Create the summary panel."""
        summary_frame = ttk.LabelFrame(parent, text="Summary", padding=8)
        summary_frame.pack(fill=tk.X)
        
        # Summary grid
        summary_grid = ttk.Frame(summary_frame)
        summary_grid.pack(fill=tk.X)
        
        # File count
        ttk.Label(summary_grid, text="Files:", font=('TkDefaultFont', 9, 'bold')).grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10)
        )
        self.files_count_label = ttk.Label(summary_grid, text="0")
        self.files_count_label.grid(row=0, column=1, sticky=tk.W)
        
        # Processing time
        ttk.Label(summary_grid, text="Processing Time:", font=('TkDefaultFont', 9, 'bold')).grid(
            row=0, column=2, sticky=tk.W, padx=(20, 10)
        )
        self.processing_time_label = ttk.Label(summary_grid, text="0.0s")
        self.processing_time_label.grid(row=0, column=3, sticky=tk.W)
        
        # Cache hit rate
        ttk.Label(summary_grid, text="Cache Hit Rate:", font=('TkDefaultFont', 9, 'bold')).grid(
            row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0)
        )
        self.cache_hit_rate_label = ttk.Label(summary_grid, text="0%")
        self.cache_hit_rate_label.grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        
        # Average tokens per file
        ttk.Label(summary_grid, text="Avg Tokens/File:", font=('TkDefaultFont', 9, 'bold')).grid(
            row=1, column=2, sticky=tk.W, padx=(20, 10), pady=(5, 0)
        )
        self.avg_tokens_label = ttk.Label(summary_grid, text="0")
        self.avg_tokens_label.grid(row=1, column=3, sticky=tk.W, pady=(5, 0))
    
    def start_queue_processor(self):
        """Start the queue processor for UI updates."""
        self.after(50, self.process_queue)
    
    def process_queue(self):
        """Process updates from the counting thread."""
        try:
            while True:
                try:
                    msg_type, data = self.count_queue.get_nowait()
                    
                    if msg_type == 'progress':
                        self._handle_progress_update(data)
                    elif msg_type == 'file_complete':
                        self._handle_file_complete(data)
                    elif msg_type == 'provider_complete':
                        self._handle_provider_complete(data)
                    elif msg_type == 'all_complete':
                        self._handle_all_complete(data)
                    elif msg_type == 'error':
                        self._handle_error(data)
                        
                except queue.Empty:
                    break
            
        except Exception as e:
            print(f"Queue processing error: {e}")
        
        # Schedule next check
        self.after(50, self.process_queue)
    
    def update_files(self, files: List[Path]):
        """Update the list of files to count tokens for."""
        if files != self.current_files:
            self.current_files = files[:]
            self.files_count_label.config(text=str(len(files)))
            
            if files:
                self.count_tokens_async()
            else:
                self.clear_display()
    
    def count_tokens_async(self):
        """Start counting tokens asynchronously."""
        if self.counting_active:
            return
        
        self.counting_active = True
        self.state = CounterState.COUNTING
        self._update_status_indicator()
        
        self.main_status_label.config(text="Counting tokens...")
        
        # Reset progress bars
        for progress_bar in self.progress_bars.values():
            progress_bar['value'] = 0
        
        # Start counting thread
        self.count_thread = threading.Thread(target=self._count_tokens_worker, daemon=True)
        self.count_thread.start()
    
    def _count_tokens_worker(self):
        """Worker thread for counting tokens."""
        start_time = time.time()
        cache_hits = 0
        total_processed = 0
        
        try:
            total_files = len(self.current_files)
            if total_files == 0:
                return
            
            # Process each provider
            for provider, info in self.LLM_PROVIDERS.items():
                provider_name = provider.value
                self.count_queue.put(('progress', {
                    'provider': provider_name,
                    'status': f"Processing {info.display_name}..."
                }))
                
                for file_index, file_path in enumerate(self.current_files):
                    file_str = str(file_path)
                    
                    # Check cache first
                    if file_str in self.token_cache and provider_name in self.token_cache[file_str]:
                        cache_hits += 1
                        estimates = self.token_cache[file_str][provider_name]
                    else:
                        # Read file and estimate tokens
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            
                            estimates = TokenEstimator.estimate_tokens(content, provider)
                            
                            # Cache results
                            if file_str not in self.token_cache:
                                self.token_cache[file_str] = {}
                            self.token_cache[file_str][provider_name] = estimates
                            
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")
                            estimates = {}
                    
                    total_processed += 1
                    
                    # Update progress
                    progress = ((file_index + 1) / total_files) * 100
                    self.count_queue.put(('progress', {
                        'provider': provider_name,
                        'progress': progress,
                        'file_path': file_str,
                        'estimates': estimates
                    }))
                
                self.count_queue.put(('provider_complete', {
                    'provider': provider_name
                }))
            
            # Calculate final statistics
            processing_time = time.time() - start_time
            cache_hit_rate = (cache_hits / total_processed) * 100 if total_processed > 0 else 0
            
            self.count_queue.put(('all_complete', {
                'processing_time': processing_time,
                'cache_hit_rate': cache_hit_rate,
                'total_processed': total_processed
            }))
            
        except Exception as e:
            self.count_queue.put(('error', str(e)))
        finally:
            self.counting_active = False
    
    def _handle_progress_update(self, data):
        """Handle progress updates."""
        provider = data['provider']
        
        if 'progress' in data:
            self.progress_bars[provider]['value'] = data['progress']
            
        if 'status' in data:
            self.status_labels[provider].config(text=data['status'])
            
        if 'estimates' in data:
            self._update_provider_display(provider, data['estimates'])
    
    def _handle_file_complete(self, data):
        """Handle file completion."""
        pass  # Could be used for per-file updates
    
    def _handle_provider_complete(self, data):
        """Handle provider completion."""
        provider = data['provider']
        self.status_labels[provider].config(text="Complete")
        self.progress_bars[provider]['value'] = 100
    
    def _handle_all_complete(self, data):
        """Handle completion of all providers."""
        self.state = CounterState.COMPLETED
        self._update_status_indicator()
        
        self.main_status_label.config(text="Complete")
        
        # Update summary
        self.processing_time_label.config(text=f"{data['processing_time']:.2f}s")
        self.cache_hit_rate_label.config(text=f"{data['cache_hit_rate']:.1f}%")
        
        # Calculate average tokens per file
        if self.current_files:
            total_tokens_all_providers = sum(self.total_tokens.values())
            avg_tokens = total_tokens_all_providers / (len(self.current_files) * len(self.LLM_PROVIDERS))
            self.avg_tokens_label.config(text=f"{avg_tokens:.0f}")
        
        # Notify callback
        if self.on_count_updated:
            self.on_count_updated(self.total_tokens.copy())
    
    def _handle_error(self, error_msg):
        """Handle errors."""
        self.state = CounterState.ERROR
        self._update_status_indicator()
        self.main_status_label.config(text=f"Error: {error_msg}")
        self.counting_active = False
    
    def _update_provider_display(self, provider_name: str, estimates: Dict[str, int]):
        """Update the display for a specific provider."""
        if provider_name not in self.model_labels:
            return
        
        provider_total = 0
        
        for model_name, model_info in self.model_labels[provider_name].items():
            tokens = estimates.get(model_name, 0)
            context_limit = model_info['limit']
            
            # Update token count
            model_info['tokens'].config(text=f"{tokens:,}")
            
            # Update usage percentage
            usage_percent = (tokens / context_limit) * 100 if context_limit > 0 else 0
            model_info['usage'].config(text=f"{usage_percent:.1f}%")
            
            # Color code based on usage
            if usage_percent > 90:
                color = "red"
            elif usage_percent > 75:
                color = "orange"
            elif usage_percent > 50:
                color = "blue"
            else:
                color = "black"
            
            model_info['usage'].config(foreground=color)
            
            provider_total += tokens
        
        # Update total for this provider
        model_key = f"{provider_name}_total"
        self.total_tokens[model_key] = provider_total
    
    def _update_status_indicator(self):
        """Update the status indicator color."""
        colors = {
            CounterState.IDLE: "gray",
            CounterState.COUNTING: "orange", 
            CounterState.COMPLETED: "green",
            CounterState.ERROR: "red"
        }
        
        color = colors.get(self.state, "gray")
        self.status_indicator.delete("all")
        self.status_indicator.create_oval(2, 2, 10, 10, fill=color, outline="")
    
    def recount_tokens(self):
        """Force recount of all tokens."""
        if self.current_files:
            # Clear cache for current files
            for file_path in self.current_files:
                file_str = str(file_path)
                if file_str in self.token_cache:
                    del self.token_cache[file_str]
            
            self.count_tokens_async()
    
    def clear_cache(self):
        """Clear the token cache."""
        self.token_cache.clear()
        self.total_tokens.clear()
        self.clear_display()
        
        if self.current_files:
            self.count_tokens_async()
    
    def clear_display(self):
        """Clear all displays."""
        # Clear model displays
        for provider_models in self.model_labels.values():
            for model_info in provider_models.values():
                model_info['tokens'].config(text="0")
                model_info['usage'].config(text="0.0%", foreground="black")
        
        # Clear progress bars
        for progress_bar in self.progress_bars.values():
            progress_bar['value'] = 0
        
        # Clear status labels
        for status_label in self.status_labels.values():
            status_label.config(text="Ready")
        
        # Clear summary
        self.processing_time_label.config(text="0.0s")
        self.cache_hit_rate_label.config(text="0%")
        self.avg_tokens_label.config(text="0")
        
        # Update main status
        self.state = CounterState.IDLE
        self._update_status_indicator()
        self.main_status_label.config(text="Ready")
    
    def get_current_tokens(self) -> Dict[str, int]:
        """Get current token counts for all providers/models."""
        current_tokens = {}
        
        for provider_name, models in self.model_labels.items():
            for model_name, model_info in models.items():
                # Extract token count from label
                token_text = model_info['tokens'].cget('text')
                try:
                    tokens = int(token_text.replace(',', ''))
                    current_tokens[f"{provider_name}_{model_name}"] = tokens
                except ValueError:
                    current_tokens[f"{provider_name}_{model_name}"] = 0
        
        return current_tokens
    
    def get_recommended_models(self, target_usage: float = 0.8) -> Dict[str, List[str]]:
        """Get recommended models based on current usage and target utilization."""
        recommendations = {}
        current_tokens = self.get_current_tokens()
        
        for provider, info in self.LLM_PROVIDERS.items():
            provider_name = provider.value
            suitable_models = []
            
            for model_name, context_limit in info.models.items():
                model_key = f"{provider_name}_{model_name}"
                current_count = current_tokens.get(model_key, 0)
                
                if current_count > 0:
                    usage = current_count / context_limit
                    if usage <= target_usage:
                        suitable_models.append({
                            'name': model_name,
                            'usage': usage * 100,
                            'tokens': current_count,
                            'limit': context_limit
                        })
            
            if suitable_models:
                # Sort by usage (ascending - prefer models with more headroom)
                suitable_models.sort(key=lambda x: x['usage'])
                recommendations[info.display_name] = suitable_models
        
        return recommendations
    
    def export_token_analysis(self) -> Dict[str, Any]:
        """Export comprehensive token analysis data."""
        analysis = {
            'timestamp': time.time(),
            'files_count': len(self.current_files),
            'cache_size': len(self.token_cache),
            'providers': {},
            'summary': {
                'total_tokens_by_provider': {},
                'average_tokens_per_file': {},
                'recommendations': self.get_recommended_models()
            }
        }
        
        current_tokens = self.get_current_tokens()
        
        # Analyze each provider
        for provider, info in self.LLM_PROVIDERS.items():
            provider_name = provider.value
            provider_data = {
                'display_name': info.display_name,
                'models': {},
                'total_tokens': 0
            }
            
            for model_name, context_limit in info.models.items():
                model_key = f"{provider_name}_{model_name}"
                tokens = current_tokens.get(model_key, 0)
                usage = (tokens / context_limit) * 100 if context_limit > 0 else 0
                
                provider_data['models'][model_name] = {
                    'tokens': tokens,
                    'context_limit': context_limit,
                    'usage_percent': usage,
                    'fits_in_context': tokens <= context_limit
                }
                
                provider_data['total_tokens'] += tokens
            
            analysis['providers'][provider_name] = provider_data
            analysis['summary']['total_tokens_by_provider'][info.display_name] = provider_data['total_tokens']
            
            if self.current_files:
                analysis['summary']['average_tokens_per_file'][info.display_name] = (
                    provider_data['total_tokens'] / len(self.current_files)
                )
        
        return analysis