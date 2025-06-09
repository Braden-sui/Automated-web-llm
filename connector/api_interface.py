import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Body, Query, status as http_status
from pydantic import BaseModel, Field # For request/response models

from .task_interface import TaskRunner
from .models import TaskConfig, TaskStatus, TaskResult, HealthStatus # Connector models
from ..web_automation.models.instructions import BrowserInstruction # For ad-hoc instructions structure

# Configure basic logging for the API
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Automated Web LLM Connector API",
    description="API for managing and running browser automation tasks.",
    version="0.1.0"
)

# Global TaskRunner instance for the API
API_LOGS_DIR = Path("./api_connector_logs")
API_PROFILES_DIR = API_LOGS_DIR / "profiles"

# Ensure profiles directory exists
API_PROFILES_DIR.mkdir(parents=True, exist_ok=True)

# Initialize TaskRunner
# In a production scenario, TaskRunner might be managed differently (e.g., as a singleton service)
api_task_runner = TaskRunner(logs_dir=str(API_LOGS_DIR), profiles_dir=str(API_PROFILES_DIR))

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application startup...")
    # You can add any async startup tasks for TaskRunner if needed, though it's mostly self-contained.
    # For now, just ensure directories are there.
    if not API_PROFILES_DIR.exists():
        API_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created profiles directory at {API_PROFILES_DIR}")
    logger.info(f"TaskRunner initialized. Profiles expected in: {api_task_runner.profiles_dir}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application shutting down...")
    await api_task_runner.shutdown()
    logger.info("TaskRunner shutdown complete.")

# --- Pydantic Models for API Requests/Responses (can be expanded) ---
class TaskSubmissionResponse(BaseModel):
    task_id: str
    message: str
    status_endpoint: str
    result_endpoint: str

class ProfileResponseModel(BaseModel):
    name: str
    path: str

class AdhocInstructionPayload(BaseModel):
    instructions: List[Dict[str, Any]] # Allow raw dicts, TaskConfig will validate
    profile_name: Optional[str] = "ad_hoc_api_task"
    description: Optional[str] = "Ad-hoc instructions submitted via API."
    memory_config: Optional[Dict[str, Any]] = None
    visual_config: Optional[Dict[str, Any]] = None
    browser_config: Optional[Dict[str, Any]] = None

# --- API Endpoints ---

@app.get("/health", response_model=HealthStatus, tags=["General"])
async def get_health():
    """Returns the health status of the API and TaskRunner."""
    # Basic health check, can be expanded to check TaskRunner's internal state
    return HealthStatus(status="OK", message="API is running.", task_runner_status="OPERATIONAL")

@app.get("/profiles", response_model=List[ProfileResponseModel], tags=["Profiles"])
async def list_profiles():
    """Lists all available task profiles."""
    try:
        profiles_dict = api_task_runner.list_profiles() # This is sync, but TaskRunner methods are generally async where IO is involved
        return [ProfileResponseModel(name=name, path=str(path)) for name, path in profiles_dict.items()]
    except Exception as e:
        logger.error(f"Error listing profiles: {e}", exc_info=True)
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/tasks/run-profile/{profile_name}", response_model=TaskSubmissionResponse, tags=["Tasks"])
async def run_task_by_profile(
    profile_name: str,
    memory_config_override: Optional[Dict[str, Any]] = Body(None, description="Optional memory config override."),
    visual_config_override: Optional[Dict[str, Any]] = Body(None, description="Optional visual config override."),
    browser_config_override: Optional[Dict[str, Any]] = Body(None, description="Optional browser config override.")
):
    """Submits a task based on a predefined profile name. Allows for configuration overrides in the request body."""
    try:
        task_id = await api_task_runner.run_task_from_profile(
            profile_name=profile_name,
            memory_config_override=memory_config_override,
            visual_config_override=visual_config_override,
            browser_config_override=browser_config_override
        )
        return TaskSubmissionResponse(
            task_id=task_id,
            message=f"Task from profile '{profile_name}' submitted successfully.",
            status_endpoint=f"/tasks/{task_id}/status",
            result_endpoint=f"/tasks/{task_id}/result"
        )
    except FileNotFoundError:
        logger.warning(f"Profile '{profile_name}.json' not found during API request.")
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=f"Profile '{profile_name}.json' not found.")
    except ValueError as ve: # Catches Pydantic validation errors from TaskConfig if profile is malformed
        logger.error(f"ValueError running profile '{profile_name}': {ve}", exc_info=True)
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=f"Error processing profile '{profile_name}': {ve}")
    except Exception as e:
        logger.error(f"Unexpected error running profile '{profile_name}': {e}", exc_info=True)
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")


@app.post("/tasks/run-instructions", response_model=TaskSubmissionResponse, tags=["Tasks"])
async def run_task_adhoc(payload: AdhocInstructionPayload):
    """Submits a task based on an ad-hoc list of instructions and configurations provided in the request body."""
    try:
        task_config_data = {
            "profile_name": payload.profile_name or "ad_hoc_api_task",
            "description": payload.description or "Ad-hoc instructions submitted via API.",
            "instructions": payload.instructions,
            "memory_config": payload.memory_config,
            "visual_config": payload.visual_config,
            "browser_config": payload.browser_config
        }
        # Filter out None values for cleaner TaskConfig creation if some configs are optional and not provided
        task_config_data_cleaned = {k: v for k, v in task_config_data.items() if v is not None}
        
        task_config = TaskConfig(**task_config_data_cleaned)
        
        task_id = await api_task_runner.submit_task(task_config)
        return TaskSubmissionResponse(
            task_id=task_id,
            message="Ad-hoc task submitted successfully.",
            status_endpoint=f"/tasks/{task_id}/status",
            result_endpoint=f"/tasks/{task_id}/result"
        )
    except ValueError as ve: # Catches Pydantic validation errors from TaskConfig
        logger.error(f"ValueError creating ad-hoc task: {ve}", exc_info=True)
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=f"Invalid task configuration: {ve}")
    except Exception as e:
        logger.error(f"Unexpected error running ad-hoc task: {e}", exc_info=True)
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")


@app.get("/tasks/{task_id}/status", response_model=Dict[str, str], tags=["Tasks"])
async def get_task_status_api(task_id: str):
    """Retrieves the current status of a specific task."""
    status = await api_task_runner.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=f"Task ID '{task_id}' not found.")
    return {"task_id": task_id, "status": status.value}

@app.get("/tasks/{task_id}/result", response_model=TaskResult, tags=["Tasks"])
async def get_task_result_api(task_id: str):
    """Retrieves the full result of a completed or failed task."""
    # Ensure task is finished before attempting to get result, or allow TaskRunner to handle this
    # For simplicity, we'll let TaskRunner's get_task_result handle waiting if necessary or returning None
    task_result = await api_task_runner.get_task_result(task_id)
    if not task_result:
        current_status = await api_task_runner.get_task_status(task_id)
        detail_msg = f"Result for task ID '{task_id}' not found."
        if current_status:
            detail_msg += f" Current status: {current_status.value}. Results are typically available once a task is COMPLETED or FAILED."
        else:
            detail_msg += " Task ID may not exist or has been cleaned up."
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=detail_msg)
    return task_result

@app.post("/tasks/{task_id}/cancel", response_model=Dict[str, str], tags=["Tasks"])
async def cancel_task_api(task_id: str):
    """Requests cancellation of a running or pending task."""
    success = await api_task_runner.cancel_task(task_id)
    if not success:
        # Check current status to provide more context
        current_status = await api_task_runner.get_task_status(task_id)
        detail_msg = f"Could not cancel task '{task_id}'."
        if current_status:
            detail_msg += f" Current status: {current_status.value}. Task may have already completed, failed, or does not exist."
        else:
            detail_msg += " Task ID may not exist."
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=detail_msg)
    return {"task_id": task_id, "message": "Cancellation request sent. Check status to confirm."}

@app.get("/tasks/active", response_model=Dict[str, Dict[str, Any]], tags=["Tasks"])
async def list_active_tasks_api():
    """Lists all tasks that are currently active (submitted, pending, or running)."""
    try:
        active_tasks = await api_task_runner.get_active_tasks_status()
        if not active_tasks:
            return {}
        return active_tasks
    except Exception as e:
        logger.error(f"Error fetching active tasks for API: {e}", exc_info=True)
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error fetching active tasks: {e}")


# Placeholder for other endpoints to be added

if __name__ == "__main__":
    import uvicorn
    # Create a dummy profile for testing if it doesn't exist
    dummy_profile_path = API_PROFILES_DIR / "api_example_profile.json"
    if not dummy_profile_path.exists():
        dummy_profile_content = {
            "profile_name": "api_example_profile",
            "description": "A simple example profile for API testing.",
            "memory_config": {"enabled": False},
            "visual_config": {"enabled": False},
            "browser_config": {"headless": True},
            "instructions": [
                {"type": "NAVIGATE", "url": "https://httpbin.org/get"},
                {"type": "SCREENSHOT", "filename": "api_example_page.png"}
            ]
        }
        import json
        with open(dummy_profile_path, 'w') as f:
            json.dump(dummy_profile_content, f, indent=2)
        logger.info(f"Created dummy profile for API: {dummy_profile_path}")

    uvicorn.run(app, host="0.0.0.0", port=8000)

