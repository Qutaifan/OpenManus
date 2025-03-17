"""
Plugin registry system for OpenManus.

This module provides a central registry for OpenManus plugins and tools,
allowing for dynamic loading, registration, and discovery of extensions.
"""

import importlib
import inspect
import os
import pkgutil
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union, cast

from loguru import logger
from pydantic import BaseModel, Field

from app.config import config
from app.tool.base import BaseTool


T = TypeVar('T', bound=BaseTool)


class PluginMetadata(BaseModel):
    """Metadata for a plugin."""
    
    name: str = Field(..., description="Name of the plugin")
    version: str = Field(..., description="Version of the plugin")
    description: Optional[str] = Field(None, description="Description of the plugin")
    author: Optional[str] = Field(None, description="Author of the plugin")
    homepage: Optional[str] = Field(None, description="Homepage URL of the plugin")
    requires: List[str] = Field(default_factory=list, description="Required dependencies")
    tags: List[str] = Field(default_factory=list, description="Tags for categorizing plugins")
    enabled: bool = Field(True, description="Whether the plugin is enabled")


class PluginRegistry:
    """
    Central registry for OpenManus plugins.
    
    This class serves as a singleton registry for all plugins and provides
    methods for discovering, registering, and retrieving plugins.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Dictionary of registered tools by name
        self._tools: Dict[str, Type[BaseTool]] = {}
        
        # Dictionary of plugin metadata by plugin name
        self._plugins: Dict[str, PluginMetadata] = {}
        
        # Set of already loaded modules to prevent duplicates
        self._loaded_modules: Set[str] = set()
        
        # Set of search paths for plugins
        self._search_paths: List[Path] = []
        
        # Default search paths
        self._add_default_search_paths()
        
        self._initialized = True
        logger.info("Plugin registry initialized")
    
    def _add_default_search_paths(self):
        """Add default search paths for plugins."""
        from app.config import get_project_root
        
        # Built-in tools directory
        self._search_paths.append(Path(get_project_root()) / "app" / "tool")
        
        # User plugins directory
        self._search_paths.append(Path(get_project_root()) / "plugins")
        
        # Site-packages installed plugins
        for path in sys.path:
            if "site-packages" in path:
                plugin_path = Path(path) / "openmanus_plugins"
                if plugin_path.exists() and plugin_path.is_dir():
                    self._search_paths.append(plugin_path)
    
    def add_search_path(self, path: Union[str, Path]) -> bool:
        """
        Add a new search path for plugins.
        
        Args:
            path: Path to add to the search paths
            
        Returns:
            bool: True if path was added, False if it doesn't exist
        """
        path_obj = Path(path)
        if path_obj.exists() and path_obj.is_dir():
            self._search_paths.append(path_obj)
            return True
        return False
    
    def register_tool(self, tool_class: Type[T]) -> Type[T]:
        """
        Register a tool class with the registry.
        
        This method can be used as a decorator:
        
        @registry.register_tool
        class MyTool(BaseTool):
            ...
        
        Args:
            tool_class: The tool class to register
            
        Returns:
            The registered tool class
        
        Raises:
            ValueError: If the tool is already registered
        """
        if not inspect.isclass(tool_class) or not issubclass(tool_class, BaseTool):
            raise ValueError(f"Can only register subclasses of BaseTool, got {tool_class}")
        
        # Create instance to get tool name
        instance = tool_class()
        name = instance.name
        
        if name in self._tools:
            # Skip if the same class is already registered
            if self._tools[name] == tool_class:
                return tool_class
            raise ValueError(f"Tool '{name}' is already registered")
        
        self._tools[name] = tool_class
        logger.info(f"Registered tool: {name}")
        return tool_class
    
    def register_plugin(self, metadata: PluginMetadata) -> None:
        """
        Register plugin metadata.
        
        Args:
            metadata: Plugin metadata
            
        Raises:
            ValueError: If the plugin is already registered
        """
        if metadata.name in self._plugins:
            raise ValueError(f"Plugin '{metadata.name}' is already registered")
        
        self._plugins[metadata.name] = metadata
        logger.info(f"Registered plugin: {metadata.name} v{metadata.version}")
    
    def get_tool(self, name: str) -> Optional[Type[BaseTool]]:
        """
        Get a tool class by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            The tool class or None if not found
        """
        return self._tools.get(name)
    
    def get_tool_instance(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool instance by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            A new instance of the tool or None if not found
        """
        tool_class = self.get_tool(name)
        if tool_class:
            return tool_class()
        return None
    
    def get_all_tools(self) -> Dict[str, Type[BaseTool]]:
        """
        Get all registered tools.
        
        Returns:
            Dictionary of tool name to tool class
        """
        return self._tools.copy()
    
    def get_plugin(self, name: str) -> Optional[PluginMetadata]:
        """
        Get plugin metadata by name.
        
        Args:
            name: Name of the plugin
            
        Returns:
            Plugin metadata or None if not found
        """
        return self._plugins.get(name)
    
    def get_all_plugins(self) -> Dict[str, PluginMetadata]:
        """
        Get all registered plugins.
        
        Returns:
            Dictionary of plugin name to plugin metadata
        """
        return self._plugins.copy()
    
    def discover_plugins(self) -> int:
        """
        Discover and load plugins from all search paths.
        
        Returns:
            Number of plugins loaded
        """
        count = 0
        for path in self._search_paths:
            count += self._discover_in_path(path)
        return count
    
    def _discover_in_path(self, path: Path) -> int:
        """
        Discover plugins in a specific path.
        
        Args:
            path: Path to search for plugins
            
        Returns:
            Number of plugins loaded
        """
        if not path.exists() or not path.is_dir():
            return 0
        
        count = 0
        # Walk through the directory
        for entry in path.iterdir():
            # Skip hidden files and directories
            if entry.name.startswith('.'):
                continue
                
            # Load Python modules and packages
            if entry.is_file() and entry.suffix == '.py':
                if self._load_module_from_file(entry):
                    count += 1
            elif entry.is_dir() and (entry / '__init__.py').exists():
                if self._load_package(entry):
                    count += 1
                    
        return count
    
    def _load_module_from_file(self, file_path: Path) -> bool:
        """
        Load a module from a Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            True if module was loaded, False otherwise
        """
        # Skip if already loaded
        if str(file_path) in self._loaded_modules:
            return False
        
        try:
            module_name = file_path.stem
            spec = spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                return False
                
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            
            self._loaded_modules.add(str(file_path))
            
            # Check for plugin setup function
            if hasattr(module, 'setup_plugin'):
                setup_func = getattr(module, 'setup_plugin')
                if callable(setup_func):
                    setup_func(self)
            
            return True
        except Exception as e:
            logger.error(f"Failed to load module from {file_path}: {e}")
            return False
    
    def _load_package(self, package_path: Path) -> bool:
        """
        Load a package containing multiple modules.
        
        Args:
            package_path: Path to the package directory
            
        Returns:
            True if package was loaded, False otherwise
        """
        # Skip if already loaded
        if str(package_path) in self._loaded_modules:
            return False
        
        try:
            # Add package to path temporarily
            sys.path.insert(0, str(package_path.parent))
            
            package_name = package_path.name
            module = importlib.import_module(package_name)
            
            self._loaded_modules.add(str(package_path))
            
            # Check for plugin setup function
            if hasattr(module, 'setup_plugin'):
                setup_func = getattr(module, 'setup_plugin')
                if callable(setup_func):
                    setup_func(self)
            
            return True
        except Exception as e:
            logger.error(f"Failed to load package from {package_path}: {e}")
            return False
        finally:
            # Remove package path from sys.path
            if str(package_path.parent) in sys.path:
                sys.path.remove(str(package_path.parent))


# Global singleton instance
registry = PluginRegistry()
