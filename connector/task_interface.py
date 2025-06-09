# Core task abstraction and TaskRunner will be defined here.

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

from web_automation.core.factory import BrowserAgentFactory 
from web_automation.models.instructions import InstructionSet, BrowserInstruction, InstructionSetResult
from web_automation.config.config_models import Mem0AdapterConfig, VisualSystemConfig, BrowserConfig
from web_automation.config.settings import general_config 

from .models import TaskConfig, TaskResult, TaskStatus

logger = logging.getLogger(__name__)

PROFILES_DIR = Path(general_config.LOGS_DIR).parent / 'profiles' 

class TaskRunner:
    def __init__(self, profiles_dir: Optional[str] = None):
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_results: Dict[str, TaskResult] = {}
        self.profiles_dir = Path(profiles_dir) if profiles_dir else PROFILES_DIR
        if not self.profiles_dir.exists():
            logger.warning(f"Profiles directory {self.profiles_dir} does not exist. Profile loading will fail.")
            # Consider creating it: self.profiles_dir.mkdir(parents=True, exist_ok=True)

    async def _execute_task(self, task_config: TaskConfig) -> TaskResult:
        task_id = str(uuid.uuid4())
        task_result = TaskResult(
            task_id=task_id,
            task_name=task_config.name,
            status=TaskStatus.RUNNING,
            start_time=datetime.utcnow()
        )
        self.task_results[task_id] = task_result
        agent = None
        try:
            logger.info(f"Starting task: {task_config.name} (ID: {task_id})")

            agent_kwargs = {
                'identity_id': f"agent_for_task_{task_id}",
            }
            if task_config.browser_config:
                agent_kwargs.update(task_config.browser_config.model_dump(exclude_none=True))
            
            memory_cfg_dict = task_config.memory_config.model_dump(exclude_none=True) if task_config.memory_config else None
            visual_cfg_dict = task_config.visual_config.model_dump(exclude_none=True) if task_config.visual_config else None

            agent = BrowserAgentFactory.create_agent(
                memory_config=memory_cfg_dict,
                visual_config_input=visual_cfg_dict,
                **agent_kwargs
            )
            await agent.launch_browser()

            instruction_set_to_run: Optional[InstructionSet] = None
            if task_config.instruction_set:
                instruction_set_to_run = task_config.instruction_set
            elif task_config.instructions:
                instruction_set_to_run = InstructionSet(instructions=task_config.instructions)
            elif task_config.profile_name:
                raise ValueError("TaskConfig for _execute_task must have instructions or instruction_set.")
            else:
                 raise ValueError("No instructions or profile provided for the task.")

            logger.info(f"Executing instruction set for task {task_id} with {len(instruction_set_to_run.instructions)} instructions.")
            instruction_result: InstructionSetResult = await agent.execute_instruction_set(instruction_set_to_run)
            
            task_result.instruction_set_result = instruction_result
            task_result.status = TaskStatus.COMPLETED if instruction_result.overall_success else TaskStatus.FAILED
            if not instruction_result.overall_success:
                # Aggregate error messages from individual instruction results or overall errors
                error_messages_from_results = [res.error for res in instruction_result.results if res.error]
                if error_messages_from_results:
                    task_result.error_message = "; ".join(error_messages_from_results)
                elif instruction_result.errors:
                    task_result.error_message = "; ".join(instruction_result.errors)
                else:
                    task_result.error_message = "Unknown error in instruction set execution."
            
            task_result.screenshots = instruction_result.screenshots
            task_result.extracted_data = instruction_result.extracted_data
            
            # Prefer a dedicated method for memory stats if available
            if hasattr(agent, 'get_memory_stats') and callable(getattr(agent, 'get_memory_stats')):
                if asyncio.iscoroutinefunction(agent.get_memory_stats):
                    task_result.memory_stats = await agent.get_memory_stats()
                else:
                    task_result.memory_stats = agent.get_memory_stats()
            elif agent.memory_manager and hasattr(agent, 'interactions_stored_session'): 
                 task_result.memory_stats = {"interactions_stored_this_session": agent.interactions_stored_session}

        except Exception as e:
            logger.error(f"Error executing task {task_config.name} (ID: {task_id}): {e}", exc_info=True)
            task_result.status = TaskStatus.FAILED
            task_result.error_message = str(e)
        finally:
            if agent:
                await agent.close()
            task_result.end_time = datetime.utcnow()
            task_result.update_duration()
            logger.info(f"Task {task_config.name} (ID: {task_id}) finished with status: {task_result.status}")
        
        return task_result

    async def submit_task(self, task_config: TaskConfig) -> str:
        task_id = str(uuid.uuid4()) 
        placeholder_result = TaskResult(
            task_id=task_id,
            task_name=task_config.name,
            status=TaskStatus.PENDING
        )
        self.task_results[task_id] = placeholder_result

        loop = asyncio.get_event_loop()
        async_task = loop.create_task(self._execute_task(task_config)) 
        self.active_tasks[task_id] = async_task
        logger.info(f"Task {task_config.name} submitted with ID: {task_id}")
        return task_id

    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        result = self.task_results.get(task_id)
        return result.status if result else None

    async def get_active_tasks_status(self) -> Dict[str, Dict[str, Any]]:
        """Returns a dictionary of active tasks (not COMPLETED, FAILED, or CANCELLED) with their details."""
        active_tasks = {}
        async with self.lock:
            for task_id, task_result_or_future in self.tasks.items():
                current_status_obj = self.task_statuses.get(task_id)
                current_status = current_status_obj.status if current_status_obj else TaskStatus.UNKNOWN

                if current_status not in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    # Try to get more details from task_results if available (e.g., profile name)
                    # This part might need refinement based on what's stored early in task_results
                    task_details_from_result = self.task_results.get(task_id)
                    profile_name = "N/A"
                    description = "N/A"
                    submitted_at = "N/A"

                    if task_details_from_result:
                        profile_name = task_details_from_result.profile_name
                        description = task_details_from_result.description
                        submitted_at = task_details_from_result.submitted_at.isoformat() if task_details_from_result.submitted_at else "N/A"
                    elif isinstance(task_result_or_future, TaskResult): # Should not happen for active tasks normally
                        profile_name = task_result_or_future.profile_name
                        description = task_result_or_future.description
                        submitted_at = task_result_or_future.submitted_at.isoformat() if task_result_or_future.submitted_at else "N/A"
                    
                    active_tasks[task_id] = {
                        "profile_name": profile_name,
                        "description": description,
                        "status": current_status.value if hasattr(current_status, 'value') else str(current_status),
                        "submitted_at": submitted_at
                    }
        return active_tasks

    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        if task_id in self.active_tasks and not self.active_tasks[task_id].done():
            await self.active_tasks[task_id] 
        return self.task_results.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            if not task.done():
                task.cancel()
                try:
                    await task 
                except asyncio.CancelledError:
                    logger.info(f"Task {task_id} cancelled successfully.")
                if task_id in self.task_results:
                    self.task_results[task_id].status = TaskStatus.CANCELLED
                    self.task_results[task_id].end_time = datetime.utcnow()
                    self.task_results[task_id].update_duration()
                return True
        logger.warning(f"Task {task_id} not found or already completed.")
        return False

    def list_available_profiles(self) -> List[str]:
        if not self.profiles_dir.exists() or not self.profiles_dir.is_dir():
            logger.warning(f"Profiles directory '{self.profiles_dir}' not found or not a directory.")
            return []
        return [f.stem for f in self.profiles_dir.glob('*.json')] 

    async def run_task_from_profile(self, profile_name: str, 
                                    memory_config_override: Optional[Mem0AdapterConfig] = None,
                                    visual_config_override: Optional[VisualSystemConfig] = None,
                                    browser_config_override: Optional[BrowserConfig] = None) -> str:
        profile_path = self.profiles_dir / f"{profile_name}.json"
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile '{profile_name}' not found at {profile_path}")
        
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)
        
        task_name = profile_data.get('name', f"task_from_profile_{profile_name}")
        
        instructions_data = profile_data.get('instructions')
        instruction_set_data = profile_data.get('instruction_set')

        if not instructions_data and not instruction_set_data:
            raise ValueError(f"Profile '{profile_name}' must contain 'instructions' or 'instruction_set'.")

        task_config_data = {
            'name': task_name,
        }

        if instructions_data:
            task_config_data['instructions'] = [BrowserInstruction(**instr) for instr in instructions_data]
        elif instruction_set_data:
            task_config_data['instruction_set'] = InstructionSet(**instruction_set_data)

        if memory_config_override:
            task_config_data['memory_config'] = memory_config_override
        elif 'memory_config' in profile_data:
            task_config_data['memory_config'] = Mem0AdapterConfig(**profile_data['memory_config'])
        
        if visual_config_override:
            task_config_data['visual_config'] = visual_config_override
        elif 'visual_config' in profile_data:
            task_config_data['visual_config'] = VisualSystemConfig(**profile_data['visual_config'])

        if browser_config_override:
            task_config_data['browser_config'] = browser_config_override
        elif 'browser_config' in profile_data:
            task_config_data['browser_config'] = BrowserConfig(**profile_data['browser_config'])

        task_config = TaskConfig(**task_config_data)
        return await self.submit_task(task_config)

    async def run_instructions(self, 
                             instructions: List[BrowserInstruction], 
                             task_name: Optional[str] = None,
                             memory_config: Optional[Mem0AdapterConfig] = None,
                             visual_config: Optional[VisualSystemConfig] = None,
                             browser_config: Optional[BrowserConfig] = None) -> str:
        if not task_name:
            task_name = f"adhoc_task_{uuid.uuid4().hex[:8]}"
        
        task_config = TaskConfig(
            name=task_name,
            instructions=instructions,
            memory_config=memory_config,
            visual_config=visual_config,
            browser_config=browser_config
        )
        return await self.submit_task(task_config)

    async def shutdown(self, wait: bool = True):
        logger.info(f"Shutting down TaskRunner. Cancelling {len(self.active_tasks)} active tasks.")
        active_task_ids = list(self.active_tasks.keys())
        for task_id in active_task_ids:
            await self.cancel_task(task_id) 
        
        if wait:
            logger.info("Waiting for all tasks to finalize cancellation...")
            await asyncio.gather(*[task for task in self.active_tasks.values() if not task.done()], return_exceptions=True)
        
        self.active_tasks.clear()
        logger.info("TaskRunner shutdown complete.")

async def main():
    logging.basicConfig(level=logging.INFO)
    task_runner = TaskRunner()

    print("Available profiles:", task_runner.list_available_profiles())

    sample_instructions = [
        BrowserInstruction(type="navigate", params={"url": "https://www.example.com"}),
        BrowserInstruction(type="screenshot", params={"filename": "example.png"})
    ]
    
    if not PROFILES_DIR.exists():
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    sample_profile_content = {
        "name": "Sample Profile Task",
        "instructions": [
            {"type": "navigate", "params": {"url": "https://www.google.com"}},
            {"type": "type", "params": {"selector": "textarea[name=q]", "text": "Async Python"}},
            {"type": "click", "params": {"selector": "input[name=btnK]"}}, 
            {"type": "wait", "params": {"duration_ms": 2000}},
            {"type": "screenshot", "params": {"filename": "google_search.png"}}
        ]
    }
    with open(PROFILES_DIR / "sample_profile.json", "w") as f:
        json.dump(sample_profile_content, f, indent=2)
    print("Created sample_profile.json for testing.")
    print("Available profiles after creation:", task_runner.list_available_profiles())

    try:
        profile_task_id = await task_runner.run_task_from_profile("sample_profile")
        print(f"Submitted task from profile 'sample_profile' with ID: {profile_task_id}")
        profile_task_result = await task_runner.get_task_result(profile_task_id)
        if profile_task_result:
            print(f"Profile Task '{profile_task_result.task_name}' completed with status: {profile_task_result.status}")
            if profile_task_result.error_message:
                print(f"Error: {profile_task_result.error_message}")
            if profile_task_result.instruction_set_result:
                print(f"Instruction results: {profile_task_result.instruction_set_result.model_dump_json(indent=2)}")
        else:
            print(f"Could not retrieve result for profile task {profile_task_id}")

        adhoc_task_id = await task_runner.run_instructions(sample_instructions, task_name="My Adhoc Example Task")
        print(f"Submitted adhoc task 'My Adhoc Example Task' with ID: {adhoc_task_id}")
        adhoc_task_result = await task_runner.get_task_result(adhoc_task_id)
        if adhoc_task_result:
            print(f"Adhoc Task '{adhoc_task_result.task_name}' completed with status: {adhoc_task_result.status}")
            if adhoc_task_result.error_message:
                print(f"Error: {adhoc_task_result.error_message}")
        else:
            print(f"Could not retrieve result for adhoc task {adhoc_task_id}")

    except FileNotFoundError as e:
        print(f"Error running profile task: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        await task_runner.shutdown()
        if (PROFILES_DIR / "sample_profile.json").exists():
            os.remove(PROFILES_DIR / "sample_profile.json")
            print("Cleaned up sample_profile.json")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
