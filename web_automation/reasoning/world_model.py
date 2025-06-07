import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from reasoners import WorldModel

logger = logging.getLogger(__name__)

@dataclass
class WebAutomationState:
    """State representation for web automation reasoning"""
    url: str
    page_content_hash: str  # Hash of key page elements
    instruction_index: int
    instruction_data: Dict[str, Any]
    user_id: str
    execution_history: list
    memory_context: Dict[str, Any]
    error_count: int = 0
    confidence_score: float = 1.0

@dataclass 
class WebAutomationAction:
    """Action representation for web automation reasoning"""
    action_type: str
    selector: Optional[str] = None
    parameters: Dict[str, Any] = None
    reasoning: str = ""
    confidence: float = 1.0

class WebAutomationWorldModel(WorldModel[WebAutomationState, WebAutomationAction]):
    """World model for web automation reasoning using existing browser agent"""
    
    def __init__(self, browser_agent, memory_manager=None):
        super().__init__()
        self.browser_agent = browser_agent
        self.memory_manager = memory_manager
        
    async def init_state(self, instruction_set) -> WebAutomationState:
        """Initialize state from instruction set"""
        current_url = self.browser_agent._page.url if self.browser_agent._page else "about:blank"
        page_hash = await self._get_page_content_hash()
        
        return WebAutomationState(
            url=current_url,
            page_content_hash=page_hash,
            instruction_index=0,
            instruction_data=instruction_set.instructions[0].dict() if instruction_set.instructions else {},
            user_id=self.browser_agent.identity_id,
            execution_history=[],
            memory_context=self._get_memory_context(instruction_set.instructions[0] if instruction_set.instructions else None)
        )
    
    async def step(self, state: WebAutomationState, action: WebAutomationAction) -> Tuple[WebAutomationState, Dict[str, Any]]:
        """Execute action and return new state"""
        try:
            # Execute the action using existing browser agent methods
            success = await self._execute_action(action)
            
            # Create new state
            new_state = self._create_new_state(state, action, success)
            
            # Additional info for reward calculation
            info = {
                "success": success,
                "action_executed": action,
                "page_changed": await self._detect_page_change(state),
                "errors_encountered": new_state.error_count > state.error_count
            }
            
            return new_state, info
            
        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            error_state = self._create_error_state(state, action, str(e))
            return error_state, {"success": False, "error": str(e)}
    
    def is_terminal(self, state: WebAutomationState) -> bool:
        """Check if state is terminal (all instructions completed or max errors)"""
        return (state.instruction_index >= len(self.current_instruction_set.instructions) or 
                state.error_count >= 3)
    
    async def _execute_action(self, action: WebAutomationAction) -> bool:
        """Execute action using existing browser agent infrastructure"""
        try:
            # Map action to existing instruction format
            instruction_dict = {
                "type": action.action_type,
                "selector": action.selector,
                **(action.parameters or {})
            }
            
            # Use existing instruction execution with memory
            await self.browser_agent._execute_instruction_with_memory(
                instruction_dict, 
                action.parameters.get("user_id", self.browser_agent.identity_id)
            )
            return True
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return False
    
    async def _get_page_content_hash(self) -> str:
        """Generate hash of key page elements for state comparison"""
        if not self.browser_agent._page:
            return "no_page"
        
        try:
            # Get key page identifiers
            title = await self.browser_agent._page.title()
            url = self.browser_agent._page.url
            # Simple hash of title + url for state differentiation
            import hashlib
            return hashlib.md5(f"{title}:{url}".encode()).hexdigest()[:12]
        except:
            return "unknown_page"
    
    def _get_memory_context(self, instruction) -> Dict[str, Any]:
        """Get relevant memory context for reasoning"""
        if not self.memory_manager or not instruction:
            return {}
        
        try:
            patterns = self.memory_manager.search_automation_patterns(
                pattern_query=f"{instruction.type} {getattr(instruction, 'selector', '')}",
                user_id=self.browser_agent.identity_id,
                limit=3
            )
            return {"relevant_patterns": patterns}
        except Exception as e:
            logger.warning(f"Error getting memory context: {e}")
            return {}
    
    def _create_new_state(self, old_state: WebAutomationState, action: WebAutomationAction, success: bool) -> WebAutomationState:
        """Create new state after action execution"""
        new_history = old_state.execution_history + [{
            "action": action,
            "success": success,
            "timestamp": asyncio.get_event_loop().time()
        }]
        
        return WebAutomationState(
            url=old_state.url,  # Will be updated if navigation occurred
            page_content_hash=old_state.page_content_hash,  # Will be updated if page changed
            instruction_index=old_state.instruction_index + (1 if success else 0),
            instruction_data=old_state.instruction_data,
            user_id=old_state.user_id,
            execution_history=new_history,
            memory_context=old_state.memory_context,
            error_count=old_state.error_count + (0 if success else 1),
            confidence_score=old_state.confidence_score * (0.9 if not success else 1.0)
        )
    
    def _create_error_state(self, old_state: WebAutomationState, action: WebAutomationAction, error: str) -> WebAutomationState:
        """Create error state"""
        return self._create_new_state(old_state, action, False)
    
    async def _detect_page_change(self, old_state: WebAutomationState) -> bool:
        """Detect if page has changed significantly"""
        current_hash = await self._get_page_content_hash()
        return current_hash != old_state.page_content_hash
