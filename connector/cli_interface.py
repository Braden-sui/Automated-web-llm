import asyncio
import json
import logging
import atexit
from pathlib import Path
from typing import Optional, List, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.json import JSON as RichJSON # To pretty print JSON

from .task_interface import TaskRunner
from .models import TaskConfig, TaskStatus, TaskResult # Assuming models are in connector.models
from ..web_automation.models.instructions import InstructionSet, BrowserInstruction, ActionType # For ad-hoc instructions

# Configure basic logging for the CLI
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = typer.Typer(help="CLI for managing and running browser automation tasks.")
console = Console()

# Global TaskRunner instance
# Profiles will be expected in ./connector_cli_logs/profiles/
CLI_LOGS_DIR = Path("./connector_cli_logs")
PROFILES_DIR = CLI_LOGS_DIR / "profiles"

# Ensure profiles directory exists for clarity, though TaskRunner also handles this
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

task_runner = TaskRunner(logs_dir=str(CLI_LOGS_DIR), profiles_dir=str(PROFILES_DIR))

@atexit.register
def shutdown_task_runner():
    console.print("[yellow]Shutting down TaskRunner...[/yellow]")
    asyncio.run(task_runner.shutdown()) # Ensure shutdown is called
    console.print("[green]TaskRunner shutdown complete.[/green]")

@app.command(name="list-profiles", help="Lists all available task profiles.")
def list_profiles_command():
    """Lists all available task profiles found in the profiles directory."""
    try:
        profiles = task_runner.list_profiles()
        if not profiles:
            console.print("[yellow]No profiles found.[/yellow] Searched in: ", str(task_runner.profiles_dir))
            return

        table = Table(title="Available Task Profiles")
        table.add_column("Profile Name", style="cyan", no_wrap=True)
        table.add_column("File Path", style="magenta")

        for name, path in profiles.items():
            table.add_row(name, str(path))
        
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error listing profiles: {e}[/red]")
        logger.error("Error in list-profiles command", exc_info=True)



@app.command(name="run-profile", help="Runs a task defined by a profile name.")
def run_profile_command(
    profile_name: str = typer.Argument(..., help="The name of the profile to run (e.g., 'example_profile')."),
    memory_config_json: Optional[str] = typer.Option(None, "--memory-config", help='JSON string to override memory config. E.g., \'{"enabled": true, "scope": "TASK"}\''),
    visual_config_json: Optional[str] = typer.Option(None, "--visual-config", help='JSON string to override visual config. E.g., \'{"enabled": true, "auto_capture": true}\''),
    browser_config_json: Optional[str] = typer.Option(None, "--browser-config", help='JSON string to override browser config. E.g., \'{"headless": false}\'')
):
    """Runs a task based on the specified profile name. 
    Allows overriding memory, visual, and browser configurations via JSON strings.
    """
    console.print(f"[bold blue]Attempting to run profile:[/bold blue] {profile_name}")
    
    memory_override: Optional[Dict[str, Any]] = None
    visual_override: Optional[Dict[str, Any]] = None
    browser_override: Optional[Dict[str, Any]] = None

    try:
        if memory_config_json:
            memory_override = json.loads(memory_config_json)
            console.print(f"[dim]Using memory config override:[/dim] {memory_override}")
        if visual_config_json:
            visual_override = json.loads(visual_config_json)
            console.print(f"[dim]Using visual config override:[/dim] {visual_override}")
        if browser_config_json:
            browser_override = json.loads(browser_config_json)
            console.print(f"[dim]Using browser config override:[/dim] {browser_override}")
    except json.JSONDecodeError as e:
        console.print(f"[red]Error parsing JSON override: {e}[/red]")
        raise typer.Exit(code=1)

    try:
        task_id = asyncio.run(task_runner.run_task_from_profile(
            profile_name=profile_name,
            memory_config_override=memory_override,
            visual_config_override=visual_override,
            browser_config_override=browser_override
        ))
        console.print(f"[green]Profile task submitted successfully![/green]")
        console.print(f"  Task ID: [bold cyan]{task_id}[/bold cyan]")
        
        console.print("Monitoring task progress...")
        final_status = asyncio.run(task_runner.wait_for_task_completion(task_id))
        
        if final_status:
            status_color = 'green' if final_status == TaskStatus.COMPLETED else 'red'
            console.print(f"Task [bold cyan]{task_id}[/bold cyan] finished with status: [bold {status_color}]{final_status.value}[/bold {status_color}]")
            
            task_result = asyncio.run(task_runner.get_task_result(task_id))
            if task_result:
                console.print(f"  Screenshots taken: {len(task_result.screenshots)}")
                console.print(f"  Data extracted: {'Yes' if task_result.extracted_data else 'No'}")
                if task_result.error_message:
                    console.print(f"  Error: [red]{task_result.error_message}[/red]")
            elif final_status != TaskStatus.COMPLETED:
                 console.print(f"  Additional details might be available via 'result {task_id}' command.")
        else:
            console.print(f"[yellow]Could not determine final status for task {task_id} after waiting, or task still in progress. Use 'status {task_id}' to check.[/yellow]")

    except FileNotFoundError:
        console.print(f"[red]Error: Profile '{profile_name}.json' not found in {str(task_runner.profiles_dir)}.[/red]")
        logger.error(f"Profile not found: {profile_name} in {task_runner.profiles_dir}")
    except ValueError as ve:
        console.print(f"[red]Error running profile (likely invalid JSON in profile or override): {ve}[/red]")
        logger.error(f"ValueError in run_profile_command for {profile_name}", exc_info=True)
    except Exception as e:
        console.print(f"[red]An unexpected error occurred while running profile '{profile_name}': {e}[/red]")
        logger.error(f"Error in run_profile_command for {profile_name}", exc_info=True)


@app.command(help="Gets the current status of a specific task.")
def status(
    task_id: str = typer.Argument(..., help="The ID of the task to query.")
):
    """Retrieves and displays the current status of the specified task."""
    console.print(f"Querying status for Task ID: [bold cyan]{task_id}[/bold cyan]")
    try:
        task_status = asyncio.run(task_runner.get_task_status(task_id))
        if task_status:
            console.print(f"  Status: [bold]{task_status.value}[/bold]")
        else:
            console.print(f"[yellow]Task ID '{task_id}' not found or status not available.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error retrieving status for task '{task_id}': {e}[/red]")
        logger.error(f"Error in status command for {task_id}", exc_info=True)


@app.command(help="Gets the full result of a completed or failed task.")
def result(
    task_id: str = typer.Argument(..., help="The ID of the task to get results for.")
):
    """Retrieves and displays the full result object for the specified task."""
    console.print(f"Fetching result for Task ID: [bold cyan]{task_id}[/bold cyan]")
    try:
        task_result = asyncio.run(task_runner.get_task_result(task_id))
        if task_result:
            # Pretty print the Pydantic model as JSON
            result_json_str = task_result.model_dump_json(indent=2)
            console.print(Syntax(result_json_str, "json", theme="native", line_numbers=True))
        else:
            console.print(f"[yellow]Result for task ID '{task_id}' not found. Task may still be running, not exist, or already cleaned up.[/yellow]")
            current_status = asyncio.run(task_runner.get_task_status(task_id))
            if current_status:
                console.print(f"  Current status: {current_status.value}. Results are typically available once a task is COMPLETED or FAILED.")
            else:
                console.print(f"  Task ID '{task_id}' not found in active/completed tasks.")

    except Exception as e:
        console.print(f"[red]Error retrieving result for task '{task_id}': {e}[/red]")
        logger.error(f"Error in result command for {task_id}", exc_info=True)


@app.command(name="run-instructions", help="Runs a task from a JSON string of ad-hoc instructions.")
def run_instructions_command(
    instructions_json: str = typer.Argument(..., help='JSON string representing a list of browser instructions. E.g., \'[{"type": "NAVIGATE", "url": "https://example.com"}]\''),
    profile_name: Optional[str] = typer.Option("ad_hoc_task", help="Optional name for this ad-hoc task profile."),
    description: Optional[str] = typer.Option("Ad-hoc instructions submitted via CLI.", help="Optional description for this task."),
    memory_config_json: Optional[str] = typer.Option(None, "--memory-config", help='JSON string for memory config. E.g., \'{"enabled": true}\''),
    visual_config_json: Optional[str] = typer.Option(None, "--visual-config", help='JSON string for visual config. E.g., \'{"enabled": false}\''),
    browser_config_json: Optional[str] = typer.Option(None, "--browser-config", help='JSON string for browser config. E.g., \'{"headless": true}\'')
):
    """Runs a task based on a JSON string of ad-hoc browser instructions."""
    console.print(f"[bold blue]Attempting to run ad-hoc instructions...[/bold blue]")

    try:
        instructions_list_data = json.loads(instructions_json)
        # Basic validation: ensure it's a list
        if not isinstance(instructions_list_data, list):
            console.print("[red]Error: Instructions JSON must be a list of instruction objects.[/red]")
            raise typer.Exit(code=1)
        
        # Further Pydantic validation will happen inside TaskRunner/TaskConfig
        # For CLI, we just ensure it's a list of dicts basically
        parsed_instructions = [BrowserInstruction(**instr) for instr in instructions_list_data]

    except json.JSONDecodeError as e:
        console.print(f"[red]Error parsing instructions JSON: {e}[/red]")
        raise typer.Exit(code=1)
    except Exception as e: # Catch potential Pydantic validation errors early
        console.print(f"[red]Error validating instructions structure: {e}[/red]")
        raise typer.Exit(code=1)

    memory_config: Optional[Dict[str, Any]] = json.loads(memory_config_json) if memory_config_json else None
    visual_config: Optional[Dict[str, Any]] = json.loads(visual_config_json) if visual_config_json else None
    browser_config: Optional[Dict[str, Any]] = json.loads(browser_config_json) if browser_config_json else None

    task_config_dict = {
        "profile_name": profile_name,
        "description": description,
        "instructions": instructions_list_data, # Pass the raw dicts for Pydantic parsing in TaskConfig
    }
    if memory_config: task_config_dict["memory_config"] = memory_config
    if visual_config: task_config_dict["visual_config"] = visual_config
    if browser_config: task_config_dict["browser_config"] = browser_config
    
    try:
        # TaskConfig will validate the structure including instructions
        task_config = TaskConfig(**task_config_dict)
        
        task_id = asyncio.run(task_runner.submit_task(task_config))
        console.print(f"[green]Ad-hoc task submitted successfully![/green]")
        console.print(f"  Task ID: [bold cyan]{task_id}[/bold cyan]")

        console.print("Monitoring task progress...")
        final_status = asyncio.run(task_runner.wait_for_task_completion(task_id))

        if final_status:
            status_color = 'green' if final_status == TaskStatus.COMPLETED else 'red'
            console.print(f"Task [bold cyan]{task_id}[/bold cyan] finished with status: [bold {status_color}]{final_status.value}[/bold {status_color}]")
            task_result = asyncio.run(task_runner.get_task_result(task_id))
            if task_result:
                console.print(f"  Screenshots taken: {len(task_result.screenshots)}")
                console.print(f"  Data extracted: {'Yes' if task_result.extracted_data else 'No'}")
                if task_result.error_message:
                    console.print(f"  Error: [red]{task_result.error_message}[/red]")
        else:
            console.print(f"[yellow]Could not determine final status for task {task_id}. Use 'status {task_id}' to check.[/yellow]")

    except ValueError as ve: # Catches Pydantic validation errors from TaskConfig
        console.print(f"[red]Error creating task configuration: {ve}[/red]")
        logger.error(f"ValueError in run_instructions_command", exc_info=True)
    except Exception as e:
        console.print(f"[red]An unexpected error occurred: {e}[/red]")
        logger.error(f"Error in run_instructions_command", exc_info=True)


@app.command(help="Requests cancellation of a running task.")
def cancel(
    task_id: str = typer.Argument(..., help="The ID of the task to cancel.")
):
    """Attempts to cancel the specified task if it is currently running or pending."""
    console.print(f"Requesting cancellation for Task ID: [bold cyan]{task_id}[/bold cyan]")
    try:
        success = asyncio.run(task_runner.cancel_task(task_id))
        if success:
            console.print(f"[green]Cancellation request for task '{task_id}' sent. Check status to confirm.[/green]")
        else:
            console.print(f"[yellow]Could not cancel task '{task_id}'. It may have already completed, failed, or does not exist.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error cancelling task '{task_id}': {e}[/red]")
        logger.error(f"Error in cancel command for {task_id}", exc_info=True)


@app.command(name="active-tasks", help="Lists all tasks that are currently active (submitted, pending, or running).")
def active_tasks_command():
    """Displays a list of all tasks currently being managed by the TaskRunner that are not in a final state."""
    console.print("Fetching active tasks...")
    try:
        active_tasks_dict = asyncio.run(task_runner.get_active_tasks_status()) # Assuming this method exists
        
        if not active_tasks_dict:
            console.print("[yellow]No active tasks found.[/yellow]")
            return

        table = Table(title="Active Tasks")
        table.add_column("Task ID", style="cyan", no_wrap=True)
        table.add_column("Profile Name/Description", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Submitted At", style="blue")

        for task_id, task_info in active_tasks_dict.items():
            # Assuming task_info is a dict or object with 'profile_name', 'status', 'submitted_at'
            # This part depends on the actual structure returned by get_active_tasks_status
            # For now, let's assume task_info is the TaskResult or similar structure
            profile_name = task_info.get('profile_name', task_info.get('description', 'N/A'))
            status_val = task_info.get('status', 'UNKNOWN').value if hasattr(task_info.get('status'), 'value') else str(task_info.get('status', 'UNKNOWN'))
            submitted_at = task_info.get('submitted_at', 'N/A')
            if submitted_at != 'N/A' and not isinstance(submitted_at, str):
                submitted_at = submitted_at.isoformat() if hasattr(submitted_at, 'isoformat') else str(submitted_at)

            table.add_row(task_id, profile_name, status_val, submitted_at)
        
        console.print(table)
    except AttributeError as ae:
        # This might happen if get_active_tasks_status is not yet implemented
        console.print(f"[red]Error: The 'get_active_tasks_status' method might not be implemented in TaskRunner yet: {ae}[/red]")
        logger.error("Likely missing get_active_tasks_status in TaskRunner", exc_info=True)
    except Exception as e:
        console.print(f"[red]Error fetching active tasks: {e}[/red]")
        logger.error("Error in active-tasks command", exc_info=True)


if __name__ == "__main__":
    # Example: Create a dummy profile for testing if it doesn't exist
    dummy_profile_path = PROFILES_DIR / "example_profile.json"
    if not dummy_profile_path.exists():
        dummy_profile_content = {
            "profile_name": "example_profile",
            "description": "A simple example profile to navigate and take a screenshot.",
            "memory_config": {"enabled": False},
            "visual_config": {"enabled": False},
            "browser_config": {"headless": True},
            "instructions": [
                {"type": "NAVIGATE", "url": "https://example.com"},
                {"type": "SCREENSHOT", "filename": "example_page.png"}
            ]
        }
        with open(dummy_profile_path, 'w') as f:
            json.dump(dummy_profile_content, f, indent=2)
        console.print(f"[dim]Created dummy profile: {dummy_profile_path}[/dim]")
    
    app()

