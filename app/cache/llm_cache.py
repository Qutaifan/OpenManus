import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from diskcache import Cache
from loguru import logger

from app.config import config
from app.schema import Message


class LLMCache:
    """
    Cache for LLM responses to improve performance and reduce API costs.
    
    This class uses diskcache to provide a persistent cache for LLM responses
    with configurable TTL, size limits, and deterministic-only options.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, cache_dir: Optional[str] = None, max_size: Optional[int] = None, 
                 ttl: Optional[int] = None, enabled: Optional[bool] = None,
                 deterministic_only: Optional[bool] = None):
        if self._initialized:
            return
            
        cache_config = config.llm_cache_config
        
        # Initialize settings from parameters or config
        self.enabled = enabled if enabled is not None else cache_config.enabled
        self.cache_dir = cache_dir or cache_config.directory
        self.max_size = max_size or cache_config.max_size
        self.ttl = ttl or cache_config.ttl
        self.deterministic_only = (deterministic_only if deterministic_only is not None 
                                   else cache_config.deterministic_only)
        
        # Create cache directory if it doesn't exist
        cache_path = Path(self.cache_dir)
        if not cache_path.exists():
            os.makedirs(cache_path, exist_ok=True)
        
        # Initialize diskcache with settings
        self.cache = Cache(
            directory=str(cache_path),
            size_limit=self.max_size,
            cull_limit=10,  # Cull 10% of cache when size limit is reached
        )
        
        self._initialized = True
        logger.info(f"LLM Cache initialized at {self.cache_dir} (enabled: {self.enabled})")
    
    def _generate_key(self, messages: List[Dict[str, Any]], model: str, temperature: float) -> str:
        """
        Generate a cache key based on the input parameters.
        
        The key includes the model name, temperature, and a hash of the messages.
        For consistent hashing, we normalize the messages to a canonical form.
        
        Args:
            messages: List of messages to hash
            model: LLM model identifier
            temperature: Sampling temperature
            
        Returns:
            str: Cache key
        """
        # Serialize messages to a canonical form
        serialized = json.dumps(messages, sort_keys=True)
        
        # Create hash of the serialized messages
        message_hash = hashlib.md5(serialized.encode()).hexdigest()
        
        # Combine model, temperature, and hash for the full key
        # Round temperature to handle floating point precision issues
        temp_str = str(round(temperature, 4))
        return f"{model}:{temp_str}:{message_hash}"
    
    def get(self, messages: List[Union[Dict, Message]], model: str, temperature: float) -> Optional[str]:
        """
        Get a cached response if available.
        
        Args:
            messages: List of messages
            model: LLM model identifier
            temperature: Sampling temperature
            
        Returns:
            Optional[str]: Cached response if found, None otherwise
        """
        if not self.enabled:
            return None
            
        # Skip cache for non-deterministic requests if configured
        if self.deterministic_only and temperature > 0:
            return None
        
        # Convert Message objects to dicts if needed
        message_dicts = []
        for message in messages:
            if isinstance(message, Message):
                message_dicts.append(message.to_dict())
            else:
                message_dicts.append(message)
        
        # Generate cache key
        key = self._generate_key(message_dicts, model, temperature)
        
        # Try to get cached response
        try:
            cached = self.cache.get(key, default=None, expire_time=True)
            if cached is None:
                return None
                
            value, timestamp = cached
            logger.info(f"Cache hit for {model} (key: {key[:10]}...)")
            return value
        except Exception as e:
            logger.warning(f"Error retrieving from cache: {e}")
            return None
    
    def set(self, messages: List[Union[Dict, Message]], model: str, temperature: float, 
            response: str) -> None:
        """
        Cache an LLM response.
        
        Args:
            messages: List of messages that prompted the response
            model: LLM model identifier
            temperature: Sampling temperature
            response: LLM response text to cache
        """
        if not self.enabled:
            return
            
        # Skip cache for non-deterministic requests if configured
        if self.deterministic_only and temperature > 0:
            return
        
        # Convert Message objects to dicts if needed
        message_dicts = []
        for message in messages:
            if isinstance(message, Message):
                message_dicts.append(message.to_dict())
            else:
                message_dicts.append(message)
        
        # Generate cache key
        key = self._generate_key(message_dicts, model, temperature)
        
        # Set cache with TTL
        try:
            self.cache.set(key, response, expire=self.ttl)
            logger.info(f"Cached response for {model} (key: {key[:10]}...)")
        except Exception as e:
            logger.warning(f"Error caching response: {e}")
    
    def clear(self) -> None:
        """Clear the entire cache."""
        try:
            self.cache.clear()
            logger.info("LLM cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache stats including size, item count, etc.
        """
        try:
            stats = {
                "size": self.cache.size,
                "item_count": len(self.cache),
                "directory": self.cache_dir,
                "enabled": self.enabled,
                "deterministic_only": self.deterministic_only,
                "ttl": self.ttl,
                "max_size": self.max_size,
            }
            return stats
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            return {
                "error": str(e),
                "enabled": self.enabled,
            }


# Singleton instance
llm_cache = LLMCache()
