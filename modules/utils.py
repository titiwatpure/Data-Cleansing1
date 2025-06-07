"""
Utility Functions
================

Common utility functions used across the data cleansing pipeline.
"""

import logging
import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union
from datetime import datetime


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration for the application.
    
    Args:
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    log_format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.FileHandler(
                log_dir / f"cleansing_{datetime.now().strftime('%Y%m%d')}.log"
            ),
            logging.StreamHandler()
        ]
    )
    
    # Return logger instance
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)


def validate_file_path(file_path: str, must_exist: bool = True) -> Path:
    """
    Validate and return Path object for file path.
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must exist (default: True)
        
    Returns:
        Validated Path object
        
    Raises:
        FileNotFoundError: If file doesn't exist and must_exist is True
        ValueError: If path is invalid
    """
    path = Path(file_path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not must_exist:
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
    
    return path


def get_file_extension(file_path: str) -> str:
    """
    Get file extension from file path.
    
    Args:
        file_path: Path to file
        
    Returns:
        File extension (lowercase, without dot)
    """
    return Path(file_path).suffix.lower().lstrip('.')


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def sanitize_column_name(column_name: str) -> str:
    """
    Sanitize column name for consistency.
    
    Args:
        column_name: Original column name
        
    Returns:
        Sanitized column name
    """
    # Convert to lowercase
    sanitized = column_name.lower()
    
    # Replace spaces and special characters with underscores
    import re
    sanitized = re.sub(r'[^a-z0-9_]', '_', sanitized)
    
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = f"col_{sanitized}"
    
    return sanitized


def create_backup(file_path: str) -> str:
    """
    Create backup of a file.
    
    Args:
        file_path: Path to file to backup
        
    Returns:
        Path to backup file
    """
    original_path = Path(file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = original_path.with_name(
        f"{original_path.stem}_backup_{timestamp}{original_path.suffix}"
    )
    
    if original_path.exists():
        import shutil
        shutil.copy2(original_path, backup_path)
    
    return str(backup_path)


def load_json_schema(schema_path: str) -> Dict[str, Any]:
    """
    Load JSON schema for data validation.
    
    Args:
        schema_path: Path to schema file
        
    Returns:
        Schema dictionary
    """
    schema_file = Path(schema_path)
    
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    with open(schema_file, 'r', encoding='utf-8') as file:
        schema = json.load(file)
    
    return schema


def generate_unique_filename(base_path: str, prefix: str = "", suffix: str = "") -> str:
    """
    Generate unique filename to avoid overwrites.
    
    Args:
        base_path: Base file path
        prefix: Optional prefix
        suffix: Optional suffix
        
    Returns:
        Unique file path
    """
    path = Path(base_path)
    counter = 1
    
    # Add prefix and suffix
    if prefix:
        name = f"{prefix}_{path.stem}"
    else:
        name = path.stem
    
    if suffix:
        name = f"{name}_{suffix}"
    
    new_path = path.with_name(f"{name}{path.suffix}")
    
    # Check if file exists and increment counter
    while new_path.exists():
        if prefix:
            name = f"{prefix}_{path.stem}_{counter}"
        else:
            name = f"{path.stem}_{counter}"
        
        if suffix:
            name = f"{name}_{suffix}"
        
        new_path = path.with_name(f"{name}{path.suffix}")
        counter += 1
    
    return str(new_path)


class ProgressTracker:
    """
    Simple progress tracker for long-running operations.
    """
    
    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items to process
            description: Description of the operation
        """
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1) -> None:
        """
        Update progress.
        
        Args:
            increment: Number of items completed
        """
        self.current += increment
        percentage = (self.current / self.total) * 100
        
        # Calculate elapsed time and ETA
        elapsed = datetime.now() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
        else:
            eta = None
        
        # Print progress
        print(f"\r{self.description}: {percentage:.1f}% "
              f"({self.current}/{self.total})", end="")
        
        if eta:
            print(f" - ETA: {eta}", end="")
    
    def finish(self) -> None:
        """
        Mark progress as complete.
        """
        self.current = self.total
        elapsed = datetime.now() - self.start_time
        print(f"\r{self.description}: 100.0% "
              f"({self.total}/{self.total}) - "
              f"Completed in {elapsed}")


def memory_usage_mb() -> float:
    """
    Get current memory usage in MB.
    
    Returns:
        Memory usage in megabytes
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024
