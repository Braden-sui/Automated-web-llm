from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, validator

class ActionType(str, Enum):
    CLICK = "click"
    TYPE = "type"
    WAIT = "wait"
    SCROLL = "scroll"
    HOVER = "hover"
    SCREENSHOT = "screenshot"
    EXTRACT = "extract"
    NAVIGATE = "navigate"
    EVALUATE = "evaluate"
    UPLOAD = "upload"
    DOWNLOAD = "download"
    # Additional actions for advanced automation
    SELECT = "select"  # For dropdown selections
    CHECKBOX = "checkbox"  # For checkbox/radio button interactions
    DRAG_DROP = "drag_drop"  # For drag and drop operations
    KEYBOARD = "keyboard"  # For complex keyboard interactions

class WaitCondition(str, Enum):
    NAVIGATION = "navigation"
    ELEMENT_VISIBLE = "element_visible"
    ELEMENT_HIDDEN = "element_hidden"
    TIMEOUT = "timeout"
    NETWORK_IDLE = "network_idle"
    LOAD = "load"
    DOM_CONTENT_LOADED = "domcontentloaded"
    # Additional wait conditions
    ELEMENT_CLICKABLE = "element_clickable"
    ELEMENT_ATTACHED = "element_attached"
    ELEMENT_DETACHED = "element_detached"
    URL_CONTAINS = "url_contains"
    TITLE_CONTAINS = "title_contains"

class Instruction(BaseModel):
    """Base instruction model for all browser actions."""
    type: ActionType
    selector: Optional[str] = None
    wait_for: Optional[Union[WaitCondition, int]] = None
    timeout: Optional[int] = 5000  # Default 5 seconds
    retry_attempts: int = 3
    retry_delay: int = 1000  # ms
    # Human-like behavior settings
    human_delay: bool = True  # Add random delays to mimic human behavior
    description: Optional[str] = None  # For debugging/logging

class ClickInstruction(Instruction):
    """Instruction for clicking an element."""
    type: ActionType = ActionType.CLICK
    button: str = "left"  # 'left', 'right', 'middle'
    click_count: int = 1
    delay: int = 100  # ms between mousedown and mouseup
    force: bool = False  # Force click even if element is not visible
    position: Optional[Dict[str, float]] = None  # {x: 0.5, y: 0.5} for relative positioning

class TypeInstruction(Instruction):
    """Instruction for typing text into an element."""  # Fixed missing docstring quote
    type: ActionType = ActionType.TYPE
    text: str
    clear: bool = True
    delay: int = 50  # ms between keystrokes
    press_enter: bool = False  # Press enter after typing

class WaitInstruction(Instruction):
    """Instruction for waiting for a condition."""
    type: ActionType = ActionType.WAIT
    condition: WaitCondition
    timeout: int = 30000  # 30 seconds default for wait operations
    expected_value: Optional[str] = None  # For conditions like url_contains, title_contains

class ScrollInstruction(Instruction):
    """Instruction for scrolling the page or an element."""
    type: ActionType = ActionType.SCROLL
    x: Optional[int] = None
    y: Optional[int] = None
    behavior: str = "smooth"  # 'auto' or 'smooth'
    scroll_into_view: bool = False
    # Additional scroll options
    wheel_delta: Optional[int] = None  # For precise wheel scrolling
    scroll_by: Optional[Dict[str, int]] = None  # {x: 100, y: 200}

class HoverInstruction(Instruction):
    """Instruction for hovering over an element."""
    type: ActionType = ActionType.HOVER
    duration: int = 1000  # How long to hover in ms
    force: bool = False

class ScreenshotInstruction(Instruction):
    """Instruction for taking screenshots."""
    type: ActionType = ActionType.SCREENSHOT
    filename: Optional[str] = None
    full_page: bool = False
    save_to_disk: bool = True
    return_as_base64: bool = True
    # Advanced screenshot options
    element_only: bool = False  # Screenshot only the selected element
    hide_elements: Optional[List[str]] = None  # Selectors to hide before screenshot
    quality: int = 80  # JPEG quality (0-100)
    clip: Optional[Dict[str, int]] = None  # {x, y, width, height}

class ExtractInstruction(Instruction):
    """Instruction for extracting data from elements."""
    type: ActionType = ActionType.EXTRACT
    attribute: Optional[str] = None  # If None, extracts text content
    multiple: bool = False
    as_json: bool = True
    # Enhanced extraction options
    extract_type: str = "text"  # 'text', 'html', 'attribute', 'property'
    filter_regex: Optional[str] = None  # Regex to filter extracted data
    transform: Optional[str] = None  # 'lowercase', 'uppercase', 'trim', 'strip_html'

class NavigateInstruction(Instruction):
    """Instruction for navigating to a URL."""
    type: ActionType = ActionType.NAVIGATE
    url: str
    wait_until: WaitCondition = WaitCondition.LOAD
    timeout: int = 30000
    # Navigation options
    referer: Optional[str] = None
    replace_current: bool = False  # Replace current page in history
    user_agent: Optional[str] = None  # Override user agent for this navigation

class EvaluateInstruction(Instruction):
    """Instruction for executing JavaScript."""
    type: ActionType = ActionType.EVALUATE
    script: str
    return_by_value: bool = True
    # Enhanced JS execution options
    await_promise: bool = False  # Wait for promise to resolve
    world: str = "main"  # 'main' or 'isolated' execution context

class UploadInstruction(Instruction):
    """Instruction for file uploads."""
    type: ActionType = ActionType.UPLOAD
    files: List[str]  # List of file paths to upload
    no_wait_after: bool = False
    # File upload options
    stream: bool = False  # Stream large files
    multiple: bool = False  # Whether input accepts multiple files

class DownloadInstruction(Instruction):
    """Instruction for file downloads."""
    type: ActionType = ActionType.DOWNLOAD
    save_as: str  # Path to save the downloaded file
    accept_downloads: bool = True
    # Download options
    suggest_filename: Optional[str] = None
    save_path: Optional[str] = None

class SelectInstruction(Instruction):
    """Instruction for selecting options in dropdowns."""
    type: ActionType = ActionType.SELECT
    value: Optional[str] = None  # Value to select
    index: Optional[int] = None  # Index to select
    label: Optional[str] = None  # Label text to select
    force: bool = False

class CheckboxInstruction(Instruction):
    """Instruction for checkbox/radio button interactions."""
    type: ActionType = ActionType.CHECKBOX
    checked: bool = True  # True to check, False to uncheck
    force: bool = False

class DragDropInstruction(Instruction):
    """Instruction for drag and drop operations."""
    type: ActionType = ActionType.DRAG_DROP
    target_selector: str  # Where to drop the element
    source_position: Optional[Dict[str, float]] = None
    target_position: Optional[Dict[str, float]] = None
    trial: bool = False  # Perform trial run without actual drop

class KeyboardInstruction(Instruction):
    """Instruction for complex keyboard interactions."""
    type: ActionType = ActionType.KEYBOARD
    key: str  # Key to press (e.g., 'Enter', 'Tab', 'Escape')
    modifiers: Optional[List[str]] = None  # ['Control', 'Shift', etc.]
    delay: int = 0

# Union type for all possible instructions
BrowserInstruction = Union[
    ClickInstruction,
    TypeInstruction,
    WaitInstruction,
    ScrollInstruction,
    HoverInstruction,
    ScreenshotInstruction,
    ExtractInstruction,
    NavigateInstruction,
    EvaluateInstruction,
    UploadInstruction,
    DownloadInstruction,
    SelectInstruction,
    CheckboxInstruction,
    DragDropInstruction,
    KeyboardInstruction
]

class InstructionSet(BaseModel):
    """A set of instructions to be executed in sequence."""
    url: Optional[str] = None
    instructions: List[BrowserInstruction]
    options: Dict[str, Any] = Field(default_factory=dict)
    # Enhanced execution options
    continue_on_error: bool = False  # Continue executing even if an instruction fails
    max_execution_time: Optional[int] = None  # Maximum time for entire instruction set
    parallel_execution: bool = False  # Execute compatible instructions in parallel
    browser_context: Dict[str, Any] = Field(default_factory=dict)  # Browser settings

    @validator('instructions', pre=True)
    def parse_instructions(cls, v):
        if not isinstance(v, list):
            raise ValueError("Instructions must be a list")
        
        parsed = []
        for item in v:
            if isinstance(item, dict):
                instruction_type = item.get('type')
                if not instruction_type:
                    raise ValueError("Instruction missing 'type' field")
                
                # Map the instruction type to the appropriate model
                instruction_class = {
                    ActionType.CLICK: ClickInstruction,
                    ActionType.TYPE: TypeInstruction,
                    ActionType.WAIT: WaitInstruction,
                    ActionType.SCROLL: ScrollInstruction,
                    ActionType.HOVER: HoverInstruction,
                    ActionType.SCREENSHOT: ScreenshotInstruction,
                    ActionType.EXTRACT: ExtractInstruction,
                    ActionType.NAVIGATE: NavigateInstruction,
                    ActionType.EVALUATE: EvaluateInstruction,
                    ActionType.UPLOAD: UploadInstruction,
                    ActionType.DOWNLOAD: DownloadInstruction,
                    ActionType.SELECT: SelectInstruction,
                    ActionType.CHECKBOX: CheckboxInstruction,
                    ActionType.DRAG_DROP: DragDropInstruction,
                    ActionType.KEYBOARD: KeyboardInstruction,
                }.get(ActionType(instruction_type))
                
                if not instruction_class:
                    raise ValueError(f"Unknown instruction type: {instruction_type}")
                
                parsed.append(instruction_class(**item))
            elif isinstance(item, BaseModel):
                parsed.append(item)
            else:
                raise ValueError(f"Invalid instruction type: {type(item)}")
        
        return parsed

# Additional models for results and errors
class ExecutionResult(BaseModel):
    """Result of executing a single instruction."""
    instruction_index: int
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float
    screenshot_path: Optional[str] = None
    retry_count: int = 0

class InstructionSetResult(BaseModel):
    """Result of executing an entire instruction set."""
    success: bool
    total_execution_time: float
    instructions_executed: int
    instructions_failed: int
    results: List[ExecutionResult]
    final_url: Optional[str] = None
    screenshots: List[str] = Field(default_factory=list)
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    captchas_solved: int = 0
    browser_context: Dict[str, Any] = Field(default_factory=dict)