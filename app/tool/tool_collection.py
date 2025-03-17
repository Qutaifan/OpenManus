"""Collection classes for managing multiple tools with plugin support."""
import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from loguru import logger

from app.exceptions import ToolError
from app.plugin.registry import registry
from app.tool.base import BaseTool, ToolFailure, ToolResult


class ToolCollection:
    """
    A collection of tools with plugin system integration and parallel execution support.
    
    This class manages a set of tools and provides methods for executing them both
    sequentially and in parallel. It integrates with the plugin registry to discover
    and load tools dynamically.
    """

    def __init__(self, *tools: BaseTool, discover_plugins: bool = True):
        """
        Initialize the tool collection with optional tools and plugin discovery.
        
        Args:
            *tools: Initial tools to include in the collection
            discover_plugins: Whether to discover and load plugins automatically
        """
        self.tools = list(tools)
        self.tool_map = {tool.name: tool for tool in tools}
        
        # Discover plugins if enabled
        if discover_plugins:
            self.discover_tools()

    def __iter__(self):
        return iter(self.tools)
    
    def __len__(self):
        return len(self.tools)

    def to_params(self) -> List[Dict[str, Any]]:
        """Convert all tools to OpenAI function parameters format."""
        return [tool.to_param() for tool in self.tools]
    
    def discover_tools(self) -> int:
        """
        Discover and load tools from the plugin registry.
        
        Returns:
            Number of new tools loaded
        """
        # Ensure plugins are discovered
        registry.discover_plugins()
        
        # Get all tool classes from registry
        tool_classes = registry.get_all_tools()
        
        # Track newly added tools
        count = 0
        
        # Create instances of tools not already in the collection
        for name, tool_class in tool_classes.items():
            if name not in self.tool_map:
                try:
                    tool = tool_class()
                    self.add_tool(tool)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to instantiate tool {name}: {e}")
        
        logger.info(f"Discovered {count} new tools via plugin system")
        return count

    async def execute(
        self, *, name: str, tool_input: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """
        Execute a single tool by name.
        
        Args:
            name: Name of the tool to execute
            tool_input: Input parameters for the tool
            
        Returns:
            ToolResult from the execution
        """
        if tool_input is None:
            tool_input = {}
            
        # Check in-memory tools
        tool = self.tool_map.get(name)
        
        # If not found, try to load from plugin registry
        if not tool:
            tool_class = registry.get_tool(name)
            if tool_class:
                try:
                    tool = tool_class()
                    self.add_tool(tool)
                except Exception as e:
                    return ToolFailure(error=f"Failed to instantiate tool {name}: {str(e)}")
        
        if not tool:
            return ToolFailure(error=f"Tool '{name}' not found")
            
        try:
            result = await tool(**tool_input)
            return result
        except ToolError as e:
            return ToolFailure(error=e.message)
        except Exception as e:
            return ToolFailure(error=f"Error executing tool '{name}': {str(e)}")

    async def execute_all(self) -> List[ToolResult]:
        """
        Execute all tools in the collection sequentially.
        
        Returns:
            List of results from all tools
        """
        results = []
        for tool in self.tools:
            try:
                result = await tool()
                results.append(result)
            except ToolError as e:
                results.append(ToolFailure(error=e.message))
            except Exception as e:
                results.append(ToolFailure(error=f"Error executing tool '{tool.name}': {str(e)}"))
        return results
    
    async def execute_parallel(
        self, 
        tools: List[Tuple[str, Dict[str, Any]]], 
        max_concurrent: int = 5
    ) -> List[Tuple[str, ToolResult]]:
        """
        Execute multiple tools in parallel with concurrency limit.
        
        Args:
            tools: List of (tool_name, tool_input) tuples
            max_concurrent: Maximum number of concurrent executions
            
        Returns:
            List of (tool_name, tool_result) tuples, in the same order as input
        """
        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(name: str, inputs: Dict[str, Any]) -> Tuple[str, ToolResult]:
            async with semaphore:
                result = await self.execute(name=name, tool_input=inputs)
                return (name, result)
        
        # Create tasks for each tool execution
        tasks = [
            execute_with_semaphore(name, inputs) 
            for name, inputs in tools
        ]
        
        # Execute all tasks concurrently and gather results
        results = await asyncio.gather(*tasks)
        return results

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name from the collection or registry.
        
        Args:
            name: Name of the tool
            
        Returns:
            Tool instance or None if not found
        """
        # Check in-memory tools first
        tool = self.tool_map.get(name)
        
        # If not found, try to load from plugin registry
        if not tool:
            tool_class = registry.get_tool(name)
            if tool_class:
                try:
                    tool = tool_class()
                    self.add_tool(tool)
                except Exception as e:
                    logger.error(f"Failed to instantiate tool {name}: {e}")
                    return None
                    
        return tool

    def add_tool(self, tool: BaseTool) -> "ToolCollection":
        """
        Add a tool to the collection.
        
        Args:
            tool: Tool to add
            
        Returns:
            Self for chaining
        """
        # Check if tool with same name exists and replace it
        if tool.name in self.tool_map:
            # If same class, skip
            if isinstance(tool, type(self.tool_map[tool.name])):
                return self
                
            # Remove old tool from list
            self.tools = [t for t in self.tools if t.name != tool.name]
            
        # Add new tool
        self.tools.append(tool)
        self.tool_map[tool.name] = tool
        return self

    def add_tools(self, *tools: BaseTool) -> "ToolCollection":
        """
        Add multiple tools to the collection.
        
        Args:
            *tools: Tools to add
            
        Returns:
            Self for chaining
        """
        for tool in tools:
            self.add_tool(tool)
        return self
    
    def get_tool_names(self) -> List[str]:
        """
        Get the names of all tools in the collection.
        
        Returns:
            List of tool names
        """
        return list(self.tool_map.keys())
    
    def filter_by_tags(self, tags: List[str]) -> "ToolCollection":
        """
        Create a new collection with tools matching the given tags.
        
        Args:
            tags: List of tags to filter by
            
        Returns:
            New tool collection with matching tools
        """
        matching_tools = []
        for tool in self.tools:
            # Check if tool has tags attribute
            if hasattr(tool, "tags") and any(tag in tool.tags for tag in tags):
                matching_tools.append(tool)
                
        return ToolCollection(*matching_tools, discover_plugins=False)
