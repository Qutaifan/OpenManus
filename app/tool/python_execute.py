import multiprocessing
import sys
import importlib
import platform
from io import StringIO
from typing import Dict, List, Optional
import threading
import time

# The resource module is only available on Unix-like systems, not on Windows
is_windows = platform.system() == "Windows"
if not is_windows:
    import resource

# Import what we can from RestrictedPython
from RestrictedPython import compile_restricted, safe_builtins, limited_builtins, utility_builtins

from pydantic import Field
from loguru import logger

from app.tool.base import BaseTool
from app.config import config


class PythonExecute(BaseTool):
    """A tool for executing Python code with enhanced security restrictions using RestrictedPython."""

    name: str = "python_execute"
    description: str = "Executes Python code in a secure sandbox with limited access to system resources. Only print outputs are visible. Use print statements to see results."
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum execution time in seconds (default: 5)",
                "default": 5,
            },
            "memory_limit": {
                "type": "integer",
                "description": "Maximum memory usage in MB (default: 100)",
                "default": 100,
            },
        },
        "required": ["code"],
    }

    # List of allowed modules that can be imported in the sandbox
    allowed_modules: List[str] = Field(
        default=["math", "datetime", "random", "json", "re", "collections", 
                "itertools", "functools", "statistics", "copy", "uuid"]
    )
    
    # Maximum size of output to capture (prevents memory exhaustion)
    max_output_size: int = Field(default=1024 * 1024)  # 1MB
    
    def __init__(self, **data):
        super().__init__(**data)
        # Load allowed modules from config if available
        if hasattr(config, "python_execute_config") and hasattr(config.python_execute_config, "allowed_modules"):
            self.allowed_modules = config.python_execute_config.allowed_modules
    
    def _create_restricted_globals(self) -> dict:
        """
        Create a restricted globals dictionary with safe builtins and allowed modules.
        """
        # Define basic protection functions that work across RestrictedPython versions
        def _safe_getattr(obj, name, default=None):
            return getattr(obj, name, default)
            
        def _safe_setattr(obj, name, value):
            setattr(obj, name, value)
            return value
            
        def _safe_getitem(obj, key):
            return obj[key]
            
        def _safe_setitem(obj, key, value):
            obj[key] = value
            return value
            
        # Start with a safe set of builtins
        restricted_globals = {
            "__builtins__": {
                **safe_builtins,
                **limited_builtins,
                **utility_builtins,
                "_getattr_": _safe_getattr,
                "_setattr_": _safe_setattr,
                "_getitem_": _safe_getitem,
                "_setitem_": _safe_setitem,
                "_print_": self._create_print_handler(),
                "_getiter_": iter,
                "_iter_unpack_sequence_": lambda seq, spec, target: [
                    target[i](v) for i, v in enumerate(seq)
                ],
                "_unpack_sequence_": lambda seq, spec: list(seq),
            }
        }
        
        # Add allowed modules to the restricted globals
        for module_name in self.allowed_modules:
            try:
                restricted_globals[module_name] = importlib.import_module(module_name)
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(f"Failed to import module {module_name}: {e}")
        
        return restricted_globals
    
    def _create_print_handler(self):
        """
        Create a print handler function for the restricted environment.
        This function is used to capture print output and prevent buffer overflow.
        """
        def _safe_print(*args, **kwargs):
            output = " ".join(str(arg) for arg in args)
            if len(self._output_buffer.getvalue()) + len(output) < self.max_output_size:
                print(output, **kwargs, file=self._output_buffer)
            else:
                self._output_buffer.truncate(self.max_output_size - 100)
                self._output_buffer.write("\n... output truncated due to size limits ...")
                self._overflow = True
                
        return _safe_print
    
    def _set_memory_limit(self, memory_mb: int):
        """
        Set the maximum memory limit for the process.
        """
        # Skip if we're on Windows as it doesn't have the resource module
        if is_windows:
            logger.warning("Memory limits not supported on Windows")
            return
        
        try:
            # Convert MB to bytes (for most systems)
            memory_bytes = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except (AttributeError, ValueError, NameError) as e:
            # RLIMIT_AS might not be available on some systems
            logger.warning(f"Failed to set memory limit: {e}")
    
    def _run_code(self, code: str, result_dict: dict, memory_limit: Optional[int] = None) -> None:
        """
        Compile and run the restricted code with safety measures.
        """
        self._output_buffer = StringIO()
        self._overflow = False
        original_stdout = sys.stdout
        
        try:
            # Set memory limit if supported and not on Windows
            if memory_limit and not is_windows:
                self._set_memory_limit(memory_limit)
            
            # Create restricted globals for safe execution
            restricted_globals = self._create_restricted_globals()
            
            # Compile the code with RestrictedPython
            try:
                compiled_code = compile_restricted(code, filename="<string>", mode="exec")
            except SyntaxError as e:
                result_dict["observation"] = f"Syntax error: {str(e)}"
                result_dict["success"] = False
                return
            
            # Redirect stdout to capture output
            sys.stdout = self._output_buffer
            
            # Execute the compiled code
            exec(compiled_code, restricted_globals)
            
            # Get the captured output
            output = self._output_buffer.getvalue()
            if self._overflow:
                output += "\n[Note: Output was truncated due to size limits]"
                
            result_dict["observation"] = output
            result_dict["success"] = True
            
        except Exception as e:
            result_dict["observation"] = f"Error during execution: {type(e).__name__}: {str(e)}"
            result_dict["success"] = False
        finally:
            sys.stdout = original_stdout

    async def execute(
        self,
        code: str,
        timeout: int = 5,
        memory_limit: int = 100,
    ) -> Dict:
        """
        Executes the provided Python code with timeout and memory limits.

        Args:
            code (str): The Python code to execute.
            timeout (int): Execution timeout in seconds (default: 5).
            memory_limit (int): Maximum memory usage in MB (default: 100).

        Returns:
            Dict: Contains 'observation' with execution output or error message and 'success' status.
        """
        with multiprocessing.Manager() as manager:
            result = manager.dict({"observation": "", "success": False})
            
            # Create a process to run the code in isolation
            proc = multiprocessing.Process(
                target=self._run_code, 
                args=(code, result, memory_limit)
            )
            
            # Start the process and wait for it to complete or timeout
            proc.start()
            proc.join(timeout)

            # Handle timeout
            if proc.is_alive():
                proc.terminate()
                proc.join(1)
                return {
                    "observation": f"Execution timeout after {timeout} seconds. This may indicate an infinite loop or excessive computation.",
                    "success": False,
                }
                
            # Return the result
            return dict(result)
