"""
Parallel planning flow implementation for OpenManus.

This module provides enhanced planning flow with parallel execution capabilities,
allowing multiple independent steps to be executed concurrently for improved performance.
"""

import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from loguru import logger
from pydantic import Field

from app.agent.base import BaseAgent
from app.flow.base import BaseFlow, PlanStepStatus
from app.flow.planning import PlanningFlow
from app.llm import LLM
from app.recovery.recovery_manager import LoopDetector, recoverable
from app.schema import AgentState, Message, ToolChoice
from app.tool import PlanningTool


class StepDependency:
    """
    Represents a dependency between plan steps.
    
    This class tracks which steps depend on other steps and helps
    determine which steps can be executed in parallel.
    """
    
    def __init__(self):
        # Map of step index to list of step indexes it depends on
        self.dependencies: Dict[int, Set[int]] = {}
        
        # Map of step index to list of step indexes that depend on it
        self.dependents: Dict[int, Set[int]] = {}
    
    def add_dependency(self, step_index: int, depends_on: int) -> None:
        """
        Add a dependency between steps.
        
        Args:
            step_index: The step that depends on another
            depends_on: The step it depends on
        """
        # Initialize sets if not already present
        if step_index not in self.dependencies:
            self.dependencies[step_index] = set()
        if depends_on not in self.dependents:
            self.dependents[depends_on] = set()
            
        # Add dependency
        self.dependencies[step_index].add(depends_on)
        self.dependents[depends_on].add(step_index)
    
    def get_dependencies(self, step_index: int) -> Set[int]:
        """
        Get all dependencies of a step.
        
        Args:
            step_index: The step to get dependencies for
            
        Returns:
            Set of step indexes that this step depends on
        """
        return self.dependencies.get(step_index, set())
    
    def get_dependents(self, step_index: int) -> Set[int]:
        """
        Get all dependents of a step.
        
        Args:
            step_index: The step to get dependents for
            
        Returns:
            Set of step indexes that depend on this step
        """
        return self.dependents.get(step_index, set())
    
    def has_dependencies(self, step_index: int) -> bool:
        """
        Check if a step has any dependencies.
        
        Args:
            step_index: The step to check
            
        Returns:
            True if the step has dependencies, False otherwise
        """
        deps = self.dependencies.get(step_index, set())
        return len(deps) > 0
    
    def has_incomplete_dependencies(
        self, 
        step_index: int, 
        completed_steps: Set[int]
    ) -> bool:
        """
        Check if a step has incomplete dependencies.
        
        Args:
            step_index: The step to check
            completed_steps: Set of completed step indexes
            
        Returns:
            True if the step has incomplete dependencies, False otherwise
        """
        deps = self.dependencies.get(step_index, set())
        return any(dep not in completed_steps for dep in deps)
    
    def get_executable_steps(self, all_steps: List[int], completed_steps: Set[int]) -> List[int]:
        """
        Get all steps that can be executed now.
        
        Args:
            all_steps: List of all step indexes
            completed_steps: Set of completed step indexes
            
        Returns:
            List of executable step indexes
        """
        executable = []
        for step in all_steps:
            if step in completed_steps:
                continue
                
            if not self.has_incomplete_dependencies(step, completed_steps):
                executable.append(step)
                
        return executable


class ParallelPlanningFlow(PlanningFlow):
    """
    Enhanced planning flow with parallel execution capabilities.
    
    This class extends the basic PlanningFlow with the ability to identify
    and execute independent steps in parallel, improving performance for
    complex tasks.
    """
    
    # Maximum number of concurrent steps to execute
    max_concurrent_steps: int = Field(3, description="Maximum concurrent steps")
    
    # Step dependencies
    dependencies: Optional[StepDependency] = Field(
        None, description="Dependencies between steps"
    )
    
    # The set of completed step indexes
    completed_steps: Set[int] = Field(
        default_factory=set, description="Set of completed step indexes"
    )
    
    # Running steps
    running_steps: Dict[int, asyncio.Task] = Field(
        default_factory=dict, description="Currently running step tasks"
    )
    
    # Loop detector
    loop_detector: LoopDetector = Field(
        default_factory=LoopDetector, description="Loop detector for steps"
    )
    
    def __init__(
        self, 
        agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], 
        **data
    ):
        super().__init__(agents, **data)
        
        # Initialize dependency tracking
        self.dependencies = StepDependency()
        self.completed_steps = set()
    
    @recoverable(max_retries=3)
    async def execute(self, input_text: str) -> str:
        """
        Execute the planning flow with parallel step execution for speed.
        
        Args:
            input_text: The user input to execute
            
        Returns:
            Execution result text
        """
        try:
            if not self.primary_agent:
                raise ValueError("No primary agent available")

            # Create initial plan
            if input_text:
                await self._create_initial_plan(input_text)

                # Verify plan was created successfully
                if self.active_plan_id not in self.planning_tool.plans:
                    logger.error(
                        f"Plan creation failed. Plan ID {self.active_plan_id} not found in planning tool."
                    )
                    return f"Failed to create plan for: {input_text}"
                    
                # Analyze dependencies (after plan creation)
                await self._analyze_dependencies()

            result = []
            
            # Track completed steps
            self.completed_steps = set()
            
            # Main execution loop
            while True:
                # Get executable steps
                executable_steps = await self._get_executable_steps()
                
                # Exit if no more steps
                if not executable_steps:
                    # Check if there are pending steps but they're blocked
                    all_steps = await self._get_all_step_indexes()
                    if len(self.completed_steps) < len(all_steps):
                        logger.warning("Some steps are blocked and cannot be executed")
                        # Try to force-complete blocked steps
                        await self._handle_blocked_steps(all_steps)
                        # Try again with executable steps
                        executable_steps = await self._get_executable_steps()
                        if not executable_steps:
                            break
                    else:
                        break
                
                # Execute steps in parallel
                step_results = await self._execute_steps_parallel(executable_steps)
                
                # Add results to overall result
                result.extend(step_results)
                
                # Check for loops
                self._check_for_execution_loops()

            # Finalize plan
            final_summary = await self._finalize_plan()
            result.append(final_summary)

            return "\n\n".join(result)
        except Exception as e:
            logger.error(f"Error in ParallelPlanningFlow: {str(e)}")
            return f"Execution failed: {str(e)}"
    
    async def _analyze_dependencies(self) -> None:
        """
        Analyze the plan to identify dependencies between steps.
        
        This method uses the LLM to extract explicit and implicit dependencies
        from the step descriptions.
        """
        # Get plan data
        plan_data = self.planning_tool.plans[self.active_plan_id]
        steps = plan_data.get("steps", [])
        
        if not steps:
            return
            
        # Create a clean StepDependency
        self.dependencies = StepDependency()
        
        # For small plans (under 5 steps), don't bother with complex dependency analysis
        if len(steps) < 5:
            # Simple sequential dependencies
            for i in range(1, len(steps)):
                self.dependencies.add_dependency(i, i - 1)
            return
        
        # Analyze with LLM for larger plans
        prompt = self._generate_dependency_analysis_prompt(steps)
        
        system_message = Message.system_message(
            "You are an expert in plan analysis. Your task is to identify dependencies "
            "between steps in a plan. A step depends on another if it requires the output "
            "or results of that step. Represent dependencies as pairs of step indexes, "
            "where the first number depends on the second."
        )
        
        try:
            response = await self.llm.ask(
                messages=[Message.user_message(prompt)],
                system_msgs=[system_message],
                temperature=0.0,  # Need deterministic output
                stream=False,
            )
            
            # Parse dependencies from response
            dependencies = self._parse_dependencies_from_response(response, len(steps))
            
            # Add dependencies to tracking
            for dep_step, req_step in dependencies:
                self.dependencies.add_dependency(dep_step, req_step)
                
            logger.info(f"Analyzed dependencies: found {len(dependencies)} dependencies")
        except Exception as e:
            logger.error(f"Error analyzing dependencies: {e}")
            # Fall back to simple sequential dependencies
            for i in range(1, len(steps)):
                self.dependencies.add_dependency(i, i - 1)
    
    def _generate_dependency_analysis_prompt(self, steps: List[str]) -> str:
        """
        Generate a prompt for dependency analysis.
        
        Args:
            steps: List of step texts
            
        Returns:
            Prompt text
        """
        prompt = "I need to identify dependencies between the following plan steps:\n\n"
        
        for i, step in enumerate(steps):
            prompt += f"Step {i}: {step}\n"
        
        prompt += "\n"
        prompt += "For each step, list the step indexes it depends on. A step depends on another if it requires the output or results of that step.\n"
        prompt += "Format your response as step_index -> [dependency_indexes], one per line. For example:\n"
        prompt += "1 -> [0]\n"
        prompt += "3 -> [1, 2]\n"
        prompt += "If a step has no dependencies, use an empty list, like: 0 -> []\n"
        
        return prompt
    
    def _parse_dependencies_from_response(
        self, 
        response: str, 
        num_steps: int
    ) -> List[Tuple[int, int]]:
        """
        Parse dependencies from an LLM response.
        
        Args:
            response: LLM response text
            num_steps: Total number of steps
            
        Returns:
            List of (dependent_step, required_step) pairs
        """
        dependencies = []
        
        # Define a regex pattern for parsing dependency lines
        pattern = r'(\d+)\s*->\s*\[([\d\s,]*)\]'
        matches = re.finditer(pattern, response)
        
        for match in matches:
            step_str, deps_str = match.groups()
            step = int(step_str)
            
            # Parse dependency list
            if deps_str.strip():
                deps = [int(d.strip()) for d in deps_str.split(',') if d.strip()]
                
                # Add each dependency
                for dep in deps:
                    # Validate step indexes
                    if 0 <= step < num_steps and 0 <= dep < num_steps:
                        dependencies.append((step, dep))
            
        return dependencies
    
    async def _get_all_step_indexes(self) -> List[int]:
        """
        Get indexes of all steps in the plan.
        
        Returns:
            List of step indexes
        """
        if (
            not self.active_plan_id
            or self.active_plan_id not in self.planning_tool.plans
        ):
            return []
            
        plan_data = self.planning_tool.plans[self.active_plan_id]
        steps = plan_data.get("steps", [])
        
        return list(range(len(steps)))
    
    async def _get_executable_steps(self) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Get all steps that can be executed now, with step info.
        
        Returns:
            List of (step_index, step_info) tuples
        """
        if (
            not self.active_plan_id
            or self.active_plan_id not in self.planning_tool.plans
        ):
            logger.error(f"Plan with ID {self.active_plan_id} not found")
            return []

        try:
            # Get plan data
            plan_data = self.planning_tool.plans[self.active_plan_id]
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])
            
            # Ensure step_statuses is complete
            while len(step_statuses) < len(steps):
                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
                
            # Get all incomplete steps
            incomplete_steps = []
            
            for i, step in enumerate(steps):
                status = step_statuses[i]
                
                # Skip completed or running steps
                if (
                    status not in PlanStepStatus.get_active_statuses() 
                    or i in self.completed_steps
                    or i in self.running_steps
                ):
                    continue
                    
                # Skip steps with incomplete dependencies
                if self.dependencies and self.dependencies.has_incomplete_dependencies(
                    i, self.completed_steps
                ):
                    continue
                    
                # Extract step info
                step_info = {"text": step}
                
                # Extract step type if available
                type_match = re.search(r"\[([A-Z_]+)\]", step)
                if type_match:
                    step_info["type"] = type_match.group(1).lower()
                    
                incomplete_steps.append((i, step_info))
            
            # Limit to max_concurrent_steps
            remaining_slots = self.max_concurrent_steps - len(self.running_steps)
            return incomplete_steps[:remaining_slots]
                
        except Exception as e:
            logger.warning(f"Error finding executable steps: {e}")
            return []
    
    async def _execute_steps_parallel(
        self, 
        executable_steps: List[Tuple[int, Dict[str, Any]]]
    ) -> List[str]:
        """
        Execute multiple steps in parallel.
        
        Args:
            executable_steps: List of (step_index, step_info) tuples
            
        Returns:
            List of step results
        """
        if not executable_steps:
            return []
            
        # Create tasks for each step
        tasks = []
        for step_index, step_info in executable_steps:
            # Mark step as in progress
            await self._mark_step_in_progress(step_index)
            
            # Create task
            step_type = step_info.get("type")
            executor = self.get_executor(step_type)
            
            task = asyncio.create_task(
                self._execute_step_with_tracking(step_index, executor, step_info)
            )
            
            # Track running step
            self.running_steps[step_index] = task
            tasks.append(task)
            
            logger.info(f"Started execution of step {step_index} in parallel")
        
        # Wait for all tasks to complete
        results = []
        for task in asyncio.as_completed(tasks):
            try:
                step_index, result = await task
                results.append(result)
                
                # Mark step as completed
                await self._mark_step_completed(step_index)
                
                # Track completion
                self.completed_steps.add(step_index)
                
                # Remove from running steps
                if step_index in self.running_steps:
                    del self.running_steps[step_index]
                    
            except Exception as e:
                logger.error(f"Error in parallel step execution: {e}")
                results.append(f"Error in step execution: {str(e)}")
        
        return results
    
    async def _execute_step_with_tracking(
        self, 
        step_index: int,
        executor: BaseAgent,
        step_info: Dict[str, Any]
    ) -> Tuple[int, str]:
        """
        Execute a step and track its completion.
        
        Args:
            step_index: Index of the step
            executor: Agent to execute the step
            step_info: Information about the step
            
        Returns:
            Tuple of (step_index, result)
        """
        try:
            # Get plan status
            plan_status = await self._get_plan_text()
            step_text = step_info.get("text", f"Step {step_index}")
            
            # Create a prompt for the agent
            step_prompt = f"""
            CURRENT PLAN STATUS:
            {plan_status}

            YOUR CURRENT TASK:
            You are now working on step {step_index}: "{step_text}"

            Please execute this step using the appropriate tools. When you're done, provide a summary of what you accomplished.
            """
            
            # Add any loop breaking hints
            if hasattr(self, "_loop_break_hint") and self._loop_break_hint:
                step_prompt += f"\n\nIMPORTANT: {self._loop_break_hint}"
                
            # Track step for loop detection
            self.loop_detector.add_item(f"Step {step_index}: {step_text}")
            
            # Execute the step
            result = await executor.run(step_prompt)
            
            # Format result
            formatted_result = f"Step {step_index} Result:\n{result}"
            
            return step_index, formatted_result
            
        except Exception as e:
            logger.error(f"Error executing step {step_index}: {e}")
            await self._mark_step_failed(step_index)
            return step_index, f"Error executing step {step_index}: {str(e)}"
    
    async def _mark_step_in_progress(self, step_index: int) -> None:
        """
        Mark a step as in progress.
        
        Args:
            step_index: Index of the step
        """
        try:
            await self.planning_tool.execute(
                command="mark_step",
                plan_id=self.active_plan_id,
                step_index=step_index,
                step_status=PlanStepStatus.IN_PROGRESS.value,
            )
        except Exception as e:
            logger.warning(f"Error marking step {step_index} as in progress: {e}")
            # Update directly in storage
            if self.active_plan_id in self.planning_tool.plans:
                plan_data = self.planning_tool.plans[self.active_plan_id]
                step_statuses = plan_data.get("step_statuses", [])
                
                # Ensure the step_statuses list is long enough
                while len(step_statuses) <= step_index:
                    step_statuses.append(PlanStepStatus.NOT_STARTED.value)
                
                # Update status
                step_statuses[step_index] = PlanStepStatus.IN_PROGRESS.value
                plan_data["step_statuses"] = step_statuses
    
    async def _mark_step_failed(self, step_index: int) -> None:
        """
        Mark a step as blocked/failed.
        
        Args:
            step_index: Index of the step
        """
        try:
            await self.planning_tool.execute(
                command="mark_step",
                plan_id=self.active_plan_id,
                step_index=step_index,
                step_status=PlanStepStatus.BLOCKED.value,
            )
        except Exception as e:
            logger.warning(f"Error marking step {step_index} as blocked: {e}")
            # Update directly in storage
            if self.active_plan_id in self.planning_tool.plans:
                plan_data = self.planning_tool.plans[self.active_plan_id]
                step_statuses = plan_data.get("step_statuses", [])
                
                # Ensure the step_statuses list is long enough
                while len(step_statuses) <= step_index:
                    step_statuses.append(PlanStepStatus.NOT_STARTED.value)
                
                # Update status
                step_statuses[step_index] = PlanStepStatus.BLOCKED.value
                plan_data["step_statuses"] = step_statuses
    
    async def _handle_blocked_steps(self, all_steps: List[int]) -> None:
        """
        Try to handle blocked steps by force-completing them.
        
        Args:
            all_steps: List of all step indexes
        """
        # Find steps that are not completed
        blocked_steps = [s for s in all_steps if s not in self.completed_steps]
        
        for step in blocked_steps:
            logger.warning(f"Force-completing blocked step {step}")
            
            # Mark as completed
            await self._mark_step_completed(step)
            self.completed_steps.add(step)
    
    def _check_for_execution_loops(self) -> None:
        """
        Check if the execution is stuck in a loop and take corrective action.
        """
        # Check for loops
        loop_pattern = self.loop_detector.is_in_loop()
        
        if loop_pattern or self.loop_detector.detect_time_based_loops():
            logger.warning("Detected execution loop in planning flow")
            
            # Create a loop breaking hint
            self._loop_break_hint = (
                "A repetitive pattern has been detected in the execution. "
                "Try a different approach or skip this step if it's not critical."
            )
            
            # If we've detected loops multiple times, be more aggressive
            if hasattr(self, "_loop_count"):
                self._loop_count += 1
                
                if self._loop_count > 2:
                    logger.warning("Multiple loops detected, clearing running steps")
                    
                    # Cancel all running tasks
                    for step_index, task in list(self.running_steps.items()):
                        if not task.done():
                            task.cancel()
                        
                        # Remove from running steps
                        if step_index in self.running_steps:
                            del self.running_steps[step_index]
            else:
                self._loop_count = 1
                
            # Clear the detector
            self.loop_detector.clear()
        else:
            # Clear any previous loop hints
            if hasattr(self, "_loop_break_hint"):
                delattr(self, "_loop_break_hint")
