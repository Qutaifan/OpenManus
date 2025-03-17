"""
Error recovery and intelligent retry system for OpenManus.

This module provides mechanisms for detecting when agents are stuck in loops,
automatically recovering from errors, and implementing sophisticated retry
strategies with fallbacks.
"""

import asyncio
import inspect
import time
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union, cast

from loguru import logger
from pydantic import BaseModel, Field

from app.exceptions import ToolError


class ErrorCategory(Enum):
    """Categories of errors for recovery strategies."""
    
    NETWORK = auto()  # Network-related errors
    TIMEOUT = auto()  # Timeout errors
    VALIDATION = auto()  # Validation errors
    LOOP_DETECTED = auto()  # Loop detection errors
    RATE_LIMIT = auto()  # Rate limiting errors
    AUTHENTICATION = auto()  # Authentication errors
    UNKNOWN = auto()  # Unknown errors
    PERMISSION = auto()  # Permission errors
    RESOURCE = auto()  # Resource-related errors


class RecoveryStrategy(BaseModel):
    """Base model for all recovery strategies."""
    
    name: str = Field(..., description="Name of the recovery strategy")
    description: str = Field(..., description="Description of the recovery strategy")
    error_categories: List[ErrorCategory] = Field(
        ..., description="Categories of errors this strategy can handle"
    )
    max_retries: int = Field(3, description="Maximum number of retries")
    
    async def apply(
        self, 
        error: Exception, 
        context: Dict[str, Any],
        retry_count: int,
    ) -> Dict[str, Any]:
        """
        Apply the recovery strategy to the given error and context.
        
        Args:
            error: The error to recover from
            context: The context in which the error occurred
            retry_count: Number of times this strategy has been applied
            
        Returns:
            Modified context for the next attempt
            
        Raises:
            Exception: If recovery is not possible
        """
        raise NotImplementedError("Recovery strategies must implement apply()")


class LoopDetector(BaseModel):
    """
    Detects when an agent is stuck in a loop.
    
    This class uses pattern recognition to detect repetitive behavior
    in agent actions and provides mechanisms to break out of loops.
    """
    
    # Maximum number of items to keep in history
    max_history: int = Field(20, description="Maximum items to keep in history")
    
    # Minimum number of items to consider for loop detection
    min_items: int = Field(3, description="Minimum items required for detection")
    
    # History of actions or messages
    history: List[str] = Field(default_factory=list, description="History of actions")
    
    # Timestamps of actions
    timestamps: List[float] = Field(default_factory=list, description="Timestamps")
    
    def add_item(self, item: str) -> None:
        """
        Add an item to the history and check for loops.
        
        Args:
            item: The item to add (message content, action, etc.)
        """
        # Add to history
        self.history.append(item)
        self.timestamps.append(time.time())
        
        # Maintain max history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.timestamps.pop(0)
    
    def is_in_loop(
        self, 
        pattern_size_range: Tuple[int, int] = (2, 5),
        min_repetitions: int = 2
    ) -> Optional[List[str]]:
        """
        Check if the history shows a loop pattern.
        
        Args:
            pattern_size_range: Range of pattern sizes to check (min, max)
            min_repetitions: Minimum number of repetitions to consider a loop
            
        Returns:
            The detected pattern if a loop is found, None otherwise
        """
        # Need enough history for detection
        if len(self.history) < self.min_items:
            return None
            
        # Check for various pattern sizes
        min_size, max_size = pattern_size_range
        for pattern_size in range(min_size, min(max_size + 1, len(self.history) // 2 + 1)):
            # Check each possible pattern
            for start in range(len(self.history) - pattern_size * min_repetitions + 1):
                pattern = self.history[start:start + pattern_size]
                
                # Count repetitions
                repetitions = 1
                pos = start + pattern_size
                
                while pos + pattern_size <= len(self.history):
                    if self.history[pos:pos + pattern_size] == pattern:
                        repetitions += 1
                        pos += pattern_size
                    else:
                        break
                
                # If enough repetitions and the pattern ends at the end of history
                if repetitions >= min_repetitions and pos >= len(self.history) - pattern_size:
                    return pattern
        
        return None
    
    def detect_time_based_loops(
        self, 
        min_repetitions: int = 3, 
        max_time_diff: float = 0.1
    ) -> bool:
        """
        Detect loops based on timing patterns.
        
        Args:
            min_repetitions: Minimum number of repeating time intervals
            max_time_diff: Maximum difference between intervals to consider them the same
            
        Returns:
            True if a time-based loop is detected
        """
        if len(self.timestamps) < min_repetitions + 1:
            return False
            
        # Calculate time differences
        diffs = [self.timestamps[i+1] - self.timestamps[i] for i in range(len(self.timestamps)-1)]
        
        # Look for repeating time intervals
        for i in range(len(diffs) - min_repetitions + 1):
            base_diff = diffs[i]
            is_loop = True
            
            # Check if following diffs are similar
            for j in range(1, min_repetitions):
                if abs(diffs[i+j] - base_diff) > max_time_diff:
                    is_loop = False
                    break
            
            if is_loop:
                return True
                
        return False
    
    def clear(self) -> None:
        """Clear the detector history."""
        self.history.clear()
        self.timestamps.clear()


class RetryableError(Exception):
    """
    An exception that can be retried with recovery strategies.
    
    This wrapper provides additional context about an error that
    helps the recovery system decide how to handle it.
    """
    
    def __init__(
        self, 
        original_error: Exception,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.original_error = original_error
        self.category = category
        self.context = context or {}
        super().__init__(str(original_error))


class WaitStrategy(Enum):
    """Strategy for waiting between retries."""
    
    CONSTANT = "constant"  # Wait a constant amount of time
    LINEAR = "linear"  # Wait time increases linearly
    EXPONENTIAL = "exponential"  # Wait time increases exponentially
    RANDOM = "random"  # Wait a random amount of time within a range


class RecoveryManager:
    """
    Manages error recovery strategies and coordinates recovery attempts.
    
    This class maintains a registry of recovery strategies and applies them
    when errors occur, based on the error type and context.
    """
    
    def __init__(self):
        # Registry of recovery strategies by name
        self.strategies: Dict[str, RecoveryStrategy] = {}
        
        # Map of error categories to strategy names
        self.category_map: Dict[ErrorCategory, List[str]] = {
            category: [] for category in ErrorCategory
        }
        
        # Map of exception types to categories
        self.exception_categories: Dict[Type[Exception], ErrorCategory] = {}
        
        # Register common exception types
        self._register_common_exceptions()
    
    def _register_common_exceptions(self) -> None:
        """Register common exception types with their categories."""
        # Network errors
        self.register_exception_category(TimeoutError, ErrorCategory.TIMEOUT)
        self.register_exception_category(ConnectionError, ErrorCategory.NETWORK)
        self.register_exception_category(ConnectionRefusedError, ErrorCategory.NETWORK)
        self.register_exception_category(ConnectionResetError, ErrorCategory.NETWORK)
        
        # Value errors
        self.register_exception_category(ValueError, ErrorCategory.VALIDATION)
        self.register_exception_category(TypeError, ErrorCategory.VALIDATION)
        
        # Resource errors
        self.register_exception_category(MemoryError, ErrorCategory.RESOURCE)
        self.register_exception_category(PermissionError, ErrorCategory.PERMISSION)
    
    def register_strategy(self, strategy: RecoveryStrategy) -> None:
        """
        Register a recovery strategy.
        
        Args:
            strategy: The strategy to register
            
        Raises:
            ValueError: If a strategy with the same name already exists
        """
        if strategy.name in self.strategies:
            raise ValueError(f"Strategy '{strategy.name}' is already registered")
            
        self.strategies[strategy.name] = strategy
        
        # Register with categories
        for category in strategy.error_categories:
            self.category_map[category].append(strategy.name)
    
    def register_exception_category(
        self, 
        exception_type: Type[Exception],
        category: ErrorCategory
    ) -> None:
        """
        Register an exception type with a category.
        
        Args:
            exception_type: The exception type to register
            category: The category to associate with the exception
        """
        self.exception_categories[exception_type] = category
    
    def get_category_for_exception(self, exc: Exception) -> ErrorCategory:
        """
        Determine the category for an exception.
        
        Args:
            exc: The exception to categorize
            
        Returns:
            The error category
        """
        # Check for direct match
        for exc_type, category in self.exception_categories.items():
            if isinstance(exc, exc_type):
                return category
        
        # Default to UNKNOWN
        return ErrorCategory.UNKNOWN
    
    def get_strategies_for_category(self, category: ErrorCategory) -> List[RecoveryStrategy]:
        """
        Get all strategies for a specific error category.
        
        Args:
            category: The error category
            
        Returns:
            List of applicable recovery strategies
        """
        strategy_names = self.category_map.get(category, [])
        return [self.strategies[name] for name in strategy_names if name in self.strategies]
    
    def get_strategies_for_exception(self, exc: Exception) -> List[RecoveryStrategy]:
        """
        Get all applicable strategies for an exception.
        
        Args:
            exc: The exception to handle
            
        Returns:
            List of applicable recovery strategies
        """
        if isinstance(exc, RetryableError):
            category = exc.category
        else:
            category = self.get_category_for_exception(exc)
            
        return self.get_strategies_for_category(category)
    
    async def apply_recovery(
        self,
        exc: Exception,
        context: Dict[str, Any],
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Apply recovery strategies to an exception.
        
        Args:
            exc: The exception to recover from
            context: The context in which the error occurred
            max_attempts: Maximum number of recovery attempts
            
        Returns:
            Updated context after recovery
            
        Raises:
            Exception: If recovery is not possible
        """
        if isinstance(exc, RetryableError):
            original_exc = exc.original_error
            context.update(exc.context)
        else:
            original_exc = exc
        
        strategies = self.get_strategies_for_exception(exc)
        if not strategies:
            raise exc
        
        # Try each strategy in order
        for strategy in strategies:
            retry_count = 0
            recovery_context = context.copy()
            
            while retry_count < min(strategy.max_retries, max_attempts):
                try:
                    recovery_context = await strategy.apply(
                        original_exc, 
                        recovery_context,
                        retry_count
                    )
                    return recovery_context
                except Exception as e:
                    retry_count += 1
                    logger.warning(
                        f"Recovery attempt {retry_count}/{strategy.max_retries} with "
                        f"strategy '{strategy.name}' failed: {e}"
                    )
        
        # If all strategies failed, re-raise the original exception
        raise original_exc


class SimpleRetryStrategy(RecoveryStrategy):
    """
    A simple retry strategy with wait options.
    
    This strategy retries the operation with the same inputs after a delay.
    """
    
    name: str = "simple_retry"
    description: str = "Retry the operation after a delay."
    error_categories: List[ErrorCategory] = [
        ErrorCategory.NETWORK, 
        ErrorCategory.TIMEOUT,
        ErrorCategory.RATE_LIMIT
    ]
    max_retries: int = 3
    
    # Wait strategy settings
    wait_strategy: WaitStrategy = Field(
        WaitStrategy.EXPONENTIAL,
        description="Strategy for determining wait time between retries"
    )
    base_wait_time: float = Field(
        1.0, 
        description="Base time to wait between retries in seconds"
    )
    max_wait_time: float = Field(
        60.0,
        description="Maximum time to wait between retries in seconds"
    )
    
    def calculate_wait_time(self, retry_count: int) -> float:
        """
        Calculate the wait time based on the strategy and retry count.
        
        Args:
            retry_count: The current retry attempt (0-based)
            
        Returns:
            Wait time in seconds
        """
        if self.wait_strategy == WaitStrategy.CONSTANT:
            return min(self.base_wait_time, self.max_wait_time)
            
        elif self.wait_strategy == WaitStrategy.LINEAR:
            wait_time = self.base_wait_time * (retry_count + 1)
            return min(wait_time, self.max_wait_time)
            
        elif self.wait_strategy == WaitStrategy.EXPONENTIAL:
            wait_time = self.base_wait_time * (2 ** retry_count)
            return min(wait_time, self.max_wait_time)
            
        elif self.wait_strategy == WaitStrategy.RANDOM:
            import random
            min_wait = self.base_wait_time
            max_wait = min(self.base_wait_time * (retry_count + 2), self.max_wait_time)
            return random.uniform(min_wait, max_wait)
            
        return self.base_wait_time
    
    async def apply(
        self, 
        error: Exception, 
        context: Dict[str, Any],
        retry_count: int,
    ) -> Dict[str, Any]:
        """
        Apply the recovery strategy by waiting and then allowing a retry.
        
        Args:
            error: The error to recover from
            context: The context in which the error occurred
            retry_count: Number of times this strategy has been applied
            
        Returns:
            Modified context for the next attempt
        """
        # Calculate wait time
        wait_time = self.calculate_wait_time(retry_count)
        
        # Wait before retry
        logger.info(f"Waiting {wait_time:.2f}s before retry {retry_count + 1}/{self.max_retries}")
        await asyncio.sleep(wait_time)
        
        # Add retry metadata to context
        context["retry_count"] = retry_count + 1
        context["retry_strategy"] = self.name
        
        return context


class LoopBreakStrategy(RecoveryStrategy):
    """
    A strategy for breaking out of detected loops.
    
    This strategy analyzes the context to identify repetitive patterns
    and modifies the context to break the loop.
    """
    
    name: str = "loop_break"
    description: str = "Break out of detected loops by modifying the context."
    error_categories: List[ErrorCategory] = [ErrorCategory.LOOP_DETECTED]
    max_retries: int = 2
    
    async def apply(
        self, 
        error: Exception, 
        context: Dict[str, Any],
        retry_count: int,
    ) -> Dict[str, Any]:
        """
        Apply the loop breaking strategy.
        
        Args:
            error: The error to recover from
            context: The context in which the error occurred
            retry_count: Number of times this strategy has been applied
            
        Returns:
            Modified context for the next attempt
            
        Raises:
            Exception: If recovery is not possible
        """
        # Add a hint to the context to break the loop
        context["break_loop"] = True
        context["retry_count"] = retry_count + 1
        
        # Change strategy based on retry count
        if retry_count == 0:
            # First retry: Add a hint to try a different approach
            context["loop_break_hint"] = (
                "A repetitive pattern has been detected. "
                "Try a different approach to solve the problem."
            )
        elif retry_count == 1:
            # Second retry: Add more specific guidance
            context["loop_break_hint"] = (
                "Still detecting a loop. Consider simplifying the problem or "
                "breaking it down into smaller steps."
            )
        else:
            # Last resort: Skip the problematic step
            context["skip_step"] = True
            context["loop_break_hint"] = (
                "Unable to break the loop. Skipping this step and continuing."
            )
        
        logger.info(f"Applied loop breaking strategy (attempt {retry_count + 1})")
        return context


# Singleton instance
recovery_manager = RecoveryManager()

# Register default strategies
recovery_manager.register_strategy(SimpleRetryStrategy())
recovery_manager.register_strategy(LoopBreakStrategy())


def recoverable(
    max_retries: int = 3,
    error_types: Optional[List[Type[Exception]]] = None,
    wait_strategy: WaitStrategy = WaitStrategy.EXPONENTIAL,
    base_wait_time: float = 1.0,
    max_wait_time: float = 60.0,
):
    """
    Decorator to make a function or method recoverable.
    
    Args:
        max_retries: Maximum number of retry attempts
        error_types: List of exception types to handle (None for all)
        wait_strategy: Strategy for determining wait time between retries
        base_wait_time: Base time to wait between retries in seconds
        max_wait_time: Maximum time to wait between retries in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            context = {
                "args": args,
                "kwargs": kwargs,
                "function": func.__name__,
            }
            
            if not error_types:
                # Handle all exceptions
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    for _ in range(max_retries):
                        try:
                            # Create a simple retry strategy
                            strategy = SimpleRetryStrategy(
                                name="dynamic_retry",
                                wait_strategy=wait_strategy,
                                base_wait_time=base_wait_time,
                                max_wait_time=max_wait_time,
                                max_retries=1  # Only try once per loop iteration
                            )
                            
                            # Apply the strategy
                            await strategy.apply(e, context, _)
                            
                            # Retry the function
                            return await func(*args, **kwargs)
                        except Exception:
                            # Continue to next retry
                            pass
                    
                    # If all retries failed, re-raise the original exception
                    raise
            else:
                # Handle specific error types
                try:
                    return await func(*args, **kwargs)
                except tuple(error_types) as e:
                    for _ in range(max_retries):
                        try:
                            # Create a simple retry strategy
                            strategy = SimpleRetryStrategy(
                                name="dynamic_retry",
                                wait_strategy=wait_strategy,
                                base_wait_time=base_wait_time,
                                max_wait_time=max_wait_time,
                                max_retries=1  # Only try once per loop iteration
                            )
                            
                            # Apply the strategy
                            await strategy.apply(e, context, _)
                            
                            # Retry the function
                            return await func(*args, **kwargs)
                        except tuple(error_types):
                            # Continue to next retry
                            pass
                    
                    # If all retries failed, re-raise the original exception
                    raise
        
        return wrapper
    
    return decorator
