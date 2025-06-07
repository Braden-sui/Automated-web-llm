import logging
import math
from typing import List, Tuple, Dict, Any
from reasoners import SearchConfig
from .world_model import WebAutomationState, WebAutomationAction

logger = logging.getLogger(__name__)

class WebAutomationSearchConfig(SearchConfig[WebAutomationState, WebAutomationAction]):
    """Search configuration for web automation reasoning"""
    
    def __init__(self, browser_agent, memory_manager=None, config=None):
        super().__init__()
        self.browser_agent = browser_agent
        self.memory_manager = memory_manager
        self.config = config or {}
        
    def get_actions(self, state: WebAutomationState) -> List[WebAutomationAction]:
        """Generate possible actions from current state"""
        actions = []
        
        # Get current instruction
        if state.instruction_index < len(self.current_instruction_set.instructions):
            instruction = self.current_instruction_set.instructions[state.instruction_index]
            
            # Primary action: execute instruction as specified
            primary_action = WebAutomationAction(
                action_type=instruction.type.value,
                selector=getattr(instruction, 'selector', None),
                parameters=instruction.dict(),
                reasoning="Direct execution of current instruction",
                confidence=1.0
            )
            actions.append(primary_action)
            
            # Alternative actions based on memory patterns
            if self.memory_manager:
                alternative_actions = self._get_memory_based_alternatives(state, instruction)
                actions.extend(alternative_actions)
            
            # Error recovery actions if previous attempts failed
            if state.error_count > 0:
                recovery_actions = self._get_recovery_actions(state, instruction)
                actions.extend(recovery_actions)
        
        return actions
    
    def reward(self, state: WebAutomationState, action: WebAutomationAction, **kwargs) -> Tuple[float, Dict[str, Any]]:
        """Calculate reward for state-action pair"""
        base_reward = 0.0
        reward_details = {}
        
        # Success/failure reward
        if kwargs.get("success", False):
            base_reward += 10.0
            reward_details["success_bonus"] = 10.0
        else:
            base_reward -= 5.0
            reward_details["failure_penalty"] = -5.0
        
        # Memory-based reward
        if self.memory_manager:
            memory_reward = self._calculate_memory_reward(state, action)
            base_reward += memory_reward
            reward_details["memory_reward"] = memory_reward
        
        # Efficiency reward (prefer fewer actions)
        efficiency_reward = max(0, 5 - len(state.execution_history)) * 0.5
        base_reward += efficiency_reward
        reward_details["efficiency_reward"] = efficiency_reward
        
        # Confidence-based reward
        confidence_reward = action.confidence * 2.0
        base_reward += confidence_reward
        reward_details["confidence_reward"] = confidence_reward
        
        return base_reward, reward_details
    
    def fast_reward(self, state: WebAutomationState, action: WebAutomationAction) -> Tuple[float, Dict[str, Any]]:
        """Quick reward estimation for MCTS rollouts"""
        # Simple heuristic-based reward for fast evaluation
        base_reward = 0.0
        
        # Action type preferences
        action_type_rewards = {
            "click": 3.0,
            "type": 2.0,
            "navigate": 1.0,
            "wait": 0.5,
            "screenshot": 0.2
        }
        base_reward += action_type_rewards.get(action.action_type, 0.0)
        
        # Selector quality (prefer specific selectors)
        if action.selector:
            if "#" in action.selector:  # ID selector
                base_reward += 2.0
            elif "." in action.selector:  # Class selector  
                base_reward += 1.0
            elif "[" in action.selector:  # Attribute selector
                base_reward += 1.5
        
        # Memory pattern bonus
        if self.memory_manager:
            patterns = self.memory_manager.search_automation_patterns(
                pattern_query=f"{action.action_type} {action.selector}",
                user_id=state.user_id,
                limit=1
            )
            if patterns and patterns[0].get("metadata", {}).get("success", False):
                base_reward += 3.0
        
        return base_reward, {"fast_reward_components": "heuristic_based"}
    
    def _get_memory_based_alternatives(self, state: WebAutomationState, instruction) -> List[WebAutomationAction]:
        """Generate alternative actions based on memory patterns"""
        alternatives = []
        
        try:
            patterns = self.memory_manager.search_automation_patterns(
                pattern_query=f"{instruction.type}",
                user_id=state.user_id,
                limit=3
            )
            
            for pattern in patterns:
                metadata = pattern.get("metadata", {})
                if metadata.get("success") and metadata.get("selector_used"):
                    alt_action = WebAutomationAction(
                        action_type=instruction.type.value,
                        selector=metadata["selector_used"],
                        parameters=instruction.dict(),
                        reasoning=f"Memory-based alternative: {pattern.get('memory', '')[:50]}...",
                        confidence=0.8
                    )
                    alternatives.append(alt_action)
        except Exception as e:
            logger.warning(f"Error generating memory alternatives: {e}")
        
        return alternatives
    
    def _get_recovery_actions(self, state: WebAutomationState, instruction) -> List[WebAutomationAction]:
        """Generate recovery actions for error scenarios"""
        recovery_actions = []
        
        # Add wait action for timing issues
        wait_action = WebAutomationAction(
            action_type="wait",
            parameters={"condition": "timeout", "wait_for": 2000},
            reasoning="Wait for page to stabilize before retry",
            confidence=0.6
        )
        recovery_actions.append(wait_action)
        
        # Add screenshot for debugging
        screenshot_action = WebAutomationAction(
            action_type="screenshot",
            parameters={"filename": f"debug_{state.instruction_index}.png"},
            reasoning="Capture current state for debugging",
            confidence=0.5
        )
        recovery_actions.append(screenshot_action)
        
        return recovery_actions
    
    def _calculate_memory_reward(self, state: WebAutomationState, action: WebAutomationAction) -> float:
        """Calculate reward based on memory patterns"""
        try:
            patterns = self.memory_manager.search_automation_patterns(
                pattern_query=f"{action.action_type} {action.selector}",
                user_id=state.user_id,
                limit=5
            )
            
            if not patterns:
                return 0.0
            
            # Calculate success rate from patterns
            successes = sum(1 for p in patterns if p.get("metadata", {}).get("success", False))
            success_rate = successes / len(patterns)
            
            # Return reward based on success rate
            return success_rate * 5.0 - 2.5  # Range: -2.5 to 2.5
            
        except Exception as e:
            logger.warning(f"Error calculating memory reward: {e}")
            return 0.0
