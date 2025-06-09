from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

from web_automation.config.config_models import Mem0AdapterConfig, VisualSystemConfig, BrowserConfig
from web_automation.models.instructions import BrowserInstruction, InstructionSet, InstructionSetResult

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskConfig(BaseModel):
    name: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    instructions: List[BrowserInstruction] = Field(..., description="List of browser instructions for the task.")
    instruction_set: Optional[InstructionSet] = Field(None, description="Alternatively, provide a full InstructionSet.")
    profile_name: Optional[str] = Field(None, description="Name of the profile to use from web_automation/profiles.")
    memory_config: Optional[Mem0AdapterConfig] = Field(None, description="Memory configuration for the task.")
    visual_config: Optional[VisualSystemConfig] = Field(None, description="Visual system configuration for the task.")
    browser_config: Optional[BrowserConfig] = Field(None, description="Browser-specific configurations for this task.")
    # Add other relevant configurations like proxy, anti-detection if they can be task-specific

    @field_validator('instructions', 'instruction_set')
    def check_instructions_or_set(cls, v, values, **kwargs):
        if 'instructions' in values.data and values.data['instructions'] and 'instruction_set' in values.data and values.data['instruction_set']:
            raise ValueError("Provide either 'instructions' or 'instruction_set', not both.")
        if not values.data.get('instructions') and not values.data.get('instruction_set') and not values.data.get('profile_name'):
            raise ValueError("Either 'instructions', 'instruction_set', or 'profile_name' must be provided.")
        return v

class TaskResult(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_name: str
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    memory_stats: Optional[Dict[str, Any]] = Field(None, description="Memory statistics from the agent after task completion.")
    screenshots: List[str] = Field(default_factory=list, description="List of paths to screenshots taken during the task.")
    extracted_data: Dict[str, Any] = Field(default_factory=dict, description="Data extracted during the task.")
    instruction_set_result: Optional[InstructionSetResult] = Field(None, description="Detailed results from the instruction set execution.")
    error_message: Optional[str] = None

    def update_duration(self):
        if self.start_time and self.end_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()

# Example for health monitor
class HealthStatus(BaseModel):
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    dependencies: Dict[str, str] = Field(default_factory=dict) # e.g., {"ollama": "healthy", "qdrant": "healthy"}

# Models for API request/response if they differ significantly from TaskConfig/Result
class CreateTaskRequest(BaseModel):
    task_config: TaskConfig

class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    details: Optional[str] = None

class ExecuteInstructionsRequest(BaseModel):
    instructions: List[BrowserInstruction]
    memory_config: Optional[Mem0AdapterConfig] = None
    visual_config: Optional[VisualSystemConfig] = None
    browser_config: Optional[BrowserConfig] = None

# Models for OS Integration (as per plan)
class MemoryScope(str, Enum):
    APP_LOCAL = "app_local"
    OS_SHARED = "os_shared"
    SESSION_ONLY = "session"

class ResourceLimits(BaseModel):
    max_cpu_percent: Optional[float] = None
    max_memory_mb: Optional[int] = None

class OSAppConfig(BaseModel):
    app_id: str = "browser_automation"
    resource_limits: Optional[ResourceLimits] = None
    memory_sharing_policy: MemoryScope = MemoryScope.APP_LOCAL
    os_memory_endpoint: Optional[str] = None
    shared_services: Dict[str, str] = Field(default_factory=dict)
    communication_channels: List[str] = Field(default_factory=list)
    browser_config: Optional[BrowserConfig] = None # App-specific, can be overridden by task
    memory_config: Optional[Mem0AdapterConfig] = None # Always local to app, can be overridden by task
    visual_config: Optional[VisualSystemConfig] = None # App-specific, can be overridden by task

class ServiceConfig(BaseModel): # For service_interface.py
    os_app_config: OSAppConfig = Field(default_factory=OSAppConfig)
    # Add other service-level configurations if needed
