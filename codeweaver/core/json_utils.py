"""
JSON serialization utilities for CodeWeaver.
Handles serialization of complex objects including enums, dataclasses, and Path objects.
"""

import json
import datetime
from pathlib import Path
from enum import Enum
from dataclasses import is_dataclass, asdict
from typing import Any, Dict, Union
import numpy as np


class CodeWeaverJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles CodeWeaver-specific data types."""
    
    def default(self, obj: Any) -> Any:
        """Convert non-serializable objects to serializable formats."""
        
        # Handle enums
        if isinstance(obj, Enum):
            return obj.value
        
        # Handle Path objects
        if isinstance(obj, Path):
            return str(obj)
        
        # Handle datetime objects
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        
        # Handle numpy arrays and types
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        
        # Handle dataclasses
        if is_dataclass(obj):
            return asdict(obj)
        
        # Handle sets
        if isinstance(obj, set):
            return list(obj)
        
        # Let the base class handle everything else
        return super().default(obj)


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely serialize an object to JSON string.
    Uses the custom CodeWeaverJSONEncoder to handle complex types.
    """
    return json.dumps(obj, cls=CodeWeaverJSONEncoder, **kwargs)


def safe_asdict_json(obj: Any, **kwargs) -> str:
    """
    Convert a dataclass to dict and then serialize to JSON safely.
    Handles nested enums and other complex types.
    """
    if is_dataclass(obj):
        data = convert_for_json(asdict(obj))
    else:
        data = convert_for_json(obj)
    
    return json.dumps(data, **kwargs)


def convert_for_json(obj: Any) -> Any:
    """
    Recursively convert an object to be JSON-serializable.
    This is useful when you need to prepare data before passing to json.dumps().
    """
    
    # Handle enums
    if isinstance(obj, Enum):
        return obj.value
    
    # Handle Path objects
    if isinstance(obj, Path):
        return str(obj)
    
    # Handle datetime objects
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    
    # Handle numpy arrays and types
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {key: convert_for_json(value) for key, value in obj.items()}
    
    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    
    # Handle sets
    if isinstance(obj, set):
        return [convert_for_json(item) for item in obj]
    
    # Handle dataclasses
    if is_dataclass(obj):
        return convert_for_json(asdict(obj))
    
    # Return primitive types as-is
    return obj


def create_json_response(data: Any, **kwargs) -> str:
    """
    Create a JSON response string from any data structure.
    Automatically handles all CodeWeaver data types.
    """
    return safe_json_dumps(data, **kwargs)


# Convenience functions for common use cases
def serialize_optimization_result(result) -> Dict[str, Any]:
    """Serialize an OptimizationResult for JSON response."""
    return convert_for_json({
        'selected_files': [str(f) for f in result.selected_files],
        'file_scores': [convert_for_json(asdict(score)) for score in result.file_scores],
        'budget_allocation': convert_for_json(asdict(result.budget_allocation)) if result.budget_allocation else None,
        'optimization_strategy': result.optimization_strategy,
        'confidence_score': result.confidence_score,
        'recommendations': result.recommendations,
        'execution_time': result.execution_time
    })


def serialize_template_config(template) -> Dict[str, Any]:
    """Serialize a TemplateConfig for JSON response."""
    template_dict = asdict(template)
    return convert_for_json(template_dict)


def serialize_project_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize project metadata ensuring all types are JSON-safe."""
    return convert_for_json(metadata)