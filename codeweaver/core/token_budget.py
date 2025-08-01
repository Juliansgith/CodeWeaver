from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .importance_scorer import FileImportanceInfo, FileImportanceScorer
from .tokenizer import LLMProvider, TokenEstimator


class BudgetStrategy(Enum):
    """Different strategies for token budget allocation."""
    IMPORTANCE_FIRST = "importance_first"      # Prioritize most important files
    EFFICIENCY_FIRST = "efficiency_first"     # Prioritize high importance-to-token ratio
    BALANCED = "balanced"                      # Balance importance and efficiency
    COVERAGE_FIRST = "coverage_first"          # Maximize file coverage within budget
    SMART_SAMPLING = "smart_sampling"          # Smart sampling across file types


@dataclass
class BudgetAllocation:
    """Represents how token budget is allocated."""
    selected_files: List[FileImportanceInfo]
    rejected_files: List[FileImportanceInfo]
    total_budget: int
    used_tokens: int
    remaining_tokens: int
    efficiency_score: float  # Overall importance per token
    coverage_score: float    # Percentage of important files included
    strategy_used: BudgetStrategy
    
    @property
    def budget_utilization(self) -> float:
        """Percentage of budget used."""
        return (self.used_tokens / self.total_budget) * 100 if self.total_budget > 0 else 0
    
    @property
    def file_count(self) -> int:
        """Number of files selected."""
        return len(self.selected_files)


@dataclass
class BudgetConstraints:
    """Constraints for budget allocation."""
    max_tokens: int
    min_file_count: Optional[int] = None
    max_file_count: Optional[int] = None
    required_file_types: Optional[Set[str]] = None  # Must include files of these types
    excluded_file_types: Optional[Set[str]] = None  # Must exclude files of these types
    min_importance_threshold: Optional[float] = None
    max_tokens_per_file: Optional[int] = None


class SmartTokenBudget:
    """
    Intelligent token budget allocation system that optimally selects files
    based on importance, efficiency, and coverage constraints.
    """
    
    def __init__(self, importance_scorer: FileImportanceScorer):
        self.importance_scorer = importance_scorer
    
    def allocate_budget(self, 
                       scored_files: List[FileImportanceInfo],
                       budget: int,
                       strategy: BudgetStrategy = BudgetStrategy.BALANCED,
                       constraints: Optional[BudgetConstraints] = None) -> BudgetAllocation:
        """
        Allocate token budget optimally based on the chosen strategy.
        
        Args:
            scored_files: List of files with importance scores
            budget: Maximum token budget
            strategy: Budget allocation strategy
            constraints: Additional constraints for allocation
            
        Returns:
            BudgetAllocation with selected files and statistics
        """
        constraints = constraints or BudgetConstraints(max_tokens=budget)
        
        # Apply pre-filtering based on constraints
        filtered_files = self._apply_constraints(scored_files, constraints)
        
        # Choose allocation strategy
        if strategy == BudgetStrategy.IMPORTANCE_FIRST:
            allocation = self._allocate_importance_first(filtered_files, budget, constraints)
        elif strategy == BudgetStrategy.EFFICIENCY_FIRST:
            allocation = self._allocate_efficiency_first(filtered_files, budget, constraints)
        elif strategy == BudgetStrategy.COVERAGE_FIRST:
            allocation = self._allocate_coverage_first(filtered_files, budget, constraints)
        elif strategy == BudgetStrategy.SMART_SAMPLING:
            allocation = self._allocate_smart_sampling(filtered_files, budget, constraints)
        else:  # BALANCED
            allocation = self._allocate_balanced(filtered_files, budget, constraints)
        
        allocation.strategy_used = strategy
        return allocation
    
    def _apply_constraints(self, files: List[FileImportanceInfo], 
                          constraints: BudgetConstraints) -> List[FileImportanceInfo]:
        """Apply initial filtering based on constraints."""
        filtered = files[:]
        
        # Filter by file types
        if constraints.excluded_file_types:
            filtered = [f for f in filtered if f.file_type.value not in constraints.excluded_file_types]
        
        # Filter by importance threshold
        if constraints.min_importance_threshold is not None:
            filtered = [f for f in filtered if f.importance_score >= constraints.min_importance_threshold]
        
        # Filter by token limit per file
        if constraints.max_tokens_per_file is not None:
            filtered = [f for f in filtered if f.tokens <= constraints.max_tokens_per_file]
        
        return filtered
    
    def _allocate_importance_first(self, files: List[FileImportanceInfo], 
                                  budget: int, constraints: BudgetConstraints) -> BudgetAllocation:
        """Allocate budget prioritizing files with highest importance scores."""
        selected = []
        rejected = []
        used_tokens = 0
        
        # Sort by importance score (descending)
        sorted_files = sorted(files, key=lambda f: f.importance_score, reverse=True)
        
        # Ensure required file types are included first
        if constraints.required_file_types:
            required_files = [f for f in sorted_files if f.file_type.value in constraints.required_file_types]
            for file_info in required_files:
                if used_tokens + file_info.tokens <= budget:
                    selected.append(file_info)
                    used_tokens += file_info.tokens
                    sorted_files.remove(file_info)
        
        # Add remaining files by importance
        for file_info in sorted_files:
            if used_tokens + file_info.tokens <= budget:
                # Check file count constraints
                if constraints.max_file_count and len(selected) >= constraints.max_file_count:
                    rejected.append(file_info)
                    continue
                
                selected.append(file_info)
                used_tokens += file_info.tokens
            else:
                rejected.append(file_info)
        
        # Check minimum file count constraint
        if constraints.min_file_count and len(selected) < constraints.min_file_count:
            # Try to include more files by removing some large ones
            selected, rejected, used_tokens = self._adjust_for_min_files(
                selected, rejected, budget, constraints.min_file_count
            )
        
        return self._create_allocation(selected, rejected, budget, used_tokens)
    
    def _allocate_efficiency_first(self, files: List[FileImportanceInfo], 
                                  budget: int, constraints: BudgetConstraints) -> BudgetAllocation:
        """Allocate budget prioritizing files with highest importance-to-token ratio."""
        # Filter out files with zero tokens to avoid division by zero
        files_with_tokens = [f for f in files if f.tokens > 0]
        
        # Sort by efficiency ratio (importance per token)
        sorted_files = sorted(files_with_tokens, key=lambda f: f.efficiency_ratio, reverse=True)
        
        return self._greedy_allocation(sorted_files, budget, constraints)
    
    def _allocate_coverage_first(self, files: List[FileImportanceInfo], 
                                budget: int, constraints: BudgetConstraints) -> BudgetAllocation:
        """Allocate budget to maximize number of files included."""
        # Sort by token count (ascending) to fit more files
        sorted_files = sorted(files, key=lambda f: f.tokens)
        
        return self._greedy_allocation(sorted_files, budget, constraints)
    
    def _allocate_balanced(self, files: List[FileImportanceInfo], 
                          budget: int, constraints: BudgetConstraints) -> BudgetAllocation:
        """Balanced allocation considering both importance and efficiency."""
        # Create a composite score balancing importance and efficiency
        for file_info in files:
            # Normalize scores to 0-1 range
            max_importance = max(f.importance_score for f in files) if files else 1
            max_efficiency = max(f.efficiency_ratio for f in files if f.tokens > 0) if files else 1
            
            normalized_importance = file_info.importance_score / max_importance if max_importance > 0 else 0
            normalized_efficiency = file_info.efficiency_ratio / max_efficiency if max_efficiency > 0 else 0
            
            # Composite score (60% importance, 40% efficiency)
            file_info.composite_score = (0.6 * normalized_importance) + (0.4 * normalized_efficiency)
        
        # Sort by composite score
        sorted_files = sorted(files, key=lambda f: getattr(f, 'composite_score', 0), reverse=True)
        
        return self._greedy_allocation(sorted_files, budget, constraints)
    
    def _allocate_smart_sampling(self, files: List[FileImportanceInfo], 
                                budget: int, constraints: BudgetConstraints) -> BudgetAllocation:
        """Smart sampling across different file types and importance levels."""
        from collections import defaultdict
        
        # Group files by type
        files_by_type = defaultdict(list)
        for file_info in files:
            files_by_type[file_info.file_type.value].append(file_info)
        
        # Sort each group by importance
        for file_type in files_by_type:
            files_by_type[file_type].sort(key=lambda f: f.importance_score, reverse=True)
        
        selected = []
        used_tokens = 0
        
        # Allocate budget proportionally to each file type based on importance
        type_budgets = self._calculate_type_budgets(files_by_type, budget)
        
        for file_type, type_budget in type_budgets.items():
            type_files = files_by_type[file_type]
            type_selected = []
            type_used = 0
            
            for file_info in type_files:
                if type_used + file_info.tokens <= type_budget:
                    type_selected.append(file_info)
                    type_used += file_info.tokens
            
            selected.extend(type_selected)
            used_tokens += type_used
        
        # Add remaining files if budget allows
        remaining_files = [f for f in files if f not in selected]
        remaining_files.sort(key=lambda f: f.importance_score, reverse=True)
        
        for file_info in remaining_files:
            if used_tokens + file_info.tokens <= budget:
                selected.append(file_info)
                used_tokens += file_info.tokens
        
        rejected = [f for f in files if f not in selected]
        
        return self._create_allocation(selected, rejected, budget, used_tokens)
    
    def _greedy_allocation(self, sorted_files: List[FileImportanceInfo], 
                          budget: int, constraints: BudgetConstraints) -> BudgetAllocation:
        """Generic greedy allocation algorithm."""
        selected = []
        rejected = []
        used_tokens = 0
        
        for file_info in sorted_files:
            if used_tokens + file_info.tokens <= budget:
                # Check file count constraints
                if constraints.max_file_count and len(selected) >= constraints.max_file_count:
                    rejected.append(file_info)
                    continue
                
                selected.append(file_info)
                used_tokens += file_info.tokens
            else:
                rejected.append(file_info)
        
        return self._create_allocation(selected, rejected, budget, used_tokens)
    
    def _calculate_type_budgets(self, files_by_type: Dict[str, List[FileImportanceInfo]], 
                               total_budget: int) -> Dict[str, int]:
        """Calculate budget allocation for each file type."""
        type_importance = {}
        total_importance = 0
        
        for file_type, type_files in files_by_type.items():
            # Calculate average importance for this type
            if type_files:
                avg_importance = sum(f.importance_score for f in type_files) / len(type_files)
                type_importance[file_type] = max(avg_importance, 1.0)  # Ensure minimum allocation
                total_importance += type_importance[file_type]
        
        # Allocate budget proportionally
        type_budgets = {}
        for file_type, importance in type_importance.items():
            proportion = importance / total_importance if total_importance > 0 else 1 / len(type_importance)
            type_budgets[file_type] = int(total_budget * proportion)
        
        return type_budgets
    
    def _adjust_for_min_files(self, selected: List[FileImportanceInfo], 
                            rejected: List[FileImportanceInfo], 
                            budget: int, min_files: int) -> Tuple[List[FileImportanceInfo], List[FileImportanceInfo], int]:
        """Adjust selection to meet minimum file count requirement."""
        if len(selected) >= min_files:
            return selected, rejected, sum(f.tokens for f in selected)
        
        # Sort rejected files by efficiency to add best candidates
        rejected.sort(key=lambda f: f.efficiency_ratio, reverse=True)
        
        # Remove largest files from selected to make room
        selected.sort(key=lambda f: f.tokens, reverse=True)
        
        new_selected = selected[:]
        new_rejected = rejected[:]
        
        while len(new_selected) < min_files and new_rejected:
            # Try to add smallest rejected file
            candidate = min(new_rejected, key=lambda f: f.tokens)
            candidate_tokens = candidate.tokens
            
            # Make room by removing files if necessary
            current_tokens = sum(f.tokens for f in new_selected)
            while current_tokens + candidate_tokens > budget and new_selected:
                removed = new_selected.pop(0)  # Remove largest file
                new_rejected.append(removed)
                current_tokens = sum(f.tokens for f in new_selected)
            
            if current_tokens + candidate_tokens <= budget:
                new_selected.append(candidate)
                new_rejected.remove(candidate)
            else:
                break
        
        used_tokens = sum(f.tokens for f in new_selected)
        return new_selected, new_rejected, used_tokens
    
    def _create_allocation(self, selected: List[FileImportanceInfo], 
                          rejected: List[FileImportanceInfo], 
                          budget: int, used_tokens: int) -> BudgetAllocation:
        """Create BudgetAllocation object with calculated metrics."""
        remaining_tokens = budget - used_tokens
        
        # Calculate efficiency score (average importance per token)
        efficiency_score = 0.0
        if selected and used_tokens > 0:
            total_importance = sum(f.importance_score for f in selected)
            efficiency_score = total_importance / used_tokens
        
        # Calculate coverage score (percentage of high-importance files included)
        coverage_score = 0.0
        all_files = selected + rejected
        if all_files:
            # Define high-importance as top 50% of files
            sorted_all = sorted(all_files, key=lambda f: f.importance_score, reverse=True)
            top_half_count = len(sorted_all) // 2
            top_half_files = set(f.relative_path for f in sorted_all[:top_half_count])
            selected_top_half = set(f.relative_path for f in selected) & top_half_files
            coverage_score = (len(selected_top_half) / top_half_count) * 100 if top_half_count > 0 else 0
        
        return BudgetAllocation(
            selected_files=selected,
            rejected_files=rejected,
            total_budget=budget,
            used_tokens=used_tokens,
            remaining_tokens=remaining_tokens,
            efficiency_score=efficiency_score,
            coverage_score=coverage_score,
            strategy_used=BudgetStrategy.BALANCED  # Will be overridden by caller
        )
    
    def compare_strategies(self, scored_files: List[FileImportanceInfo], 
                          budget: int, constraints: Optional[BudgetConstraints] = None) -> Dict[BudgetStrategy, BudgetAllocation]:
        """Compare different budget allocation strategies."""
        results = {}
        
        for strategy in BudgetStrategy:
            allocation = self.allocate_budget(scored_files, budget, strategy, constraints)
            results[strategy] = allocation
        
        return results
    
    def optimize_budget(self, scored_files: List[FileImportanceInfo], 
                       target_efficiency: float, max_budget: int) -> Optional[BudgetAllocation]:
        """Find the minimum budget needed to achieve target efficiency."""
        # Binary search for optimal budget
        min_budget = min(f.tokens for f in scored_files if f.tokens > 0) if scored_files else 0
        best_allocation = None
        
        while min_budget <= max_budget:
            mid_budget = (min_budget + max_budget) // 2
            allocation = self.allocate_budget(scored_files, mid_budget, BudgetStrategy.BALANCED)
            
            if allocation.efficiency_score >= target_efficiency:
                best_allocation = allocation
                max_budget = mid_budget - 1
            else:
                min_budget = mid_budget + 1
        
        return best_allocation
    
    def suggest_budget_for_llm(self, scored_files: List[FileImportanceInfo], 
                              llm_provider: LLMProvider, model: str, 
                              reserve_ratio: float = 0.2) -> Tuple[int, BudgetAllocation]:
        """
        Suggest optimal budget for a specific LLM model, considering context limits.
        
        Args:
            scored_files: Files to allocate budget for
            llm_provider: Target LLM provider
            model: Specific model name
            reserve_ratio: Percentage of context to reserve for response (0.0-1.0)
            
        Returns:
            Tuple of (suggested_budget, allocation)
        """
        # Get context limit for the model
        context_limits = TokenEstimator.CONTEXT_LIMITS.get(llm_provider, {})
        max_context = context_limits.get(model, 100000)  # Default fallback
        
        # Reserve space for response and overhead
        available_budget = int(max_context * (1 - reserve_ratio))
        
        # Allocate budget using balanced strategy
        allocation = self.allocate_budget(
            scored_files, 
            available_budget, 
            BudgetStrategy.BALANCED
        )
        
        return available_budget, allocation