import asyncio
import logging
from typing import Optional, Dict, Any
from reasoners import Reasoner
from reasoners.algorithm import MCTS, BeamSearch, BFS, DFS

from .world_model import WebAutomationWorldModel, WebAutomationState
from .search_config import WebAutomationSearchConfig
from ..config.settings import reasoning_config

logger = logging.getLogger(__name__)

class WebAutomationReasoner:
    """Main reasoning engine for web automation"""
    
    def __init__(self, browser_agent, memory_manager=None):
        self.browser_agent = browser_agent
        self.memory_manager = memory_manager
        self.config = reasoning_config
        
        # Initialize components
        self.world_model = WebAutomationWorldModel(browser_agent, memory_manager)
        self.search_config = WebAutomationSearchConfig(browser_agent, memory_manager, self.config)
        
        # Initialize algorithm
        self.algorithm = self._create_algorithm()
        
        # Create reasoner
        self.reasoner = Reasoner(
            world_model=self.world_model,
            search_config=self.search_config,
            search_algo=self.algorithm
        )
        
        self.reasoning_stats = {
            "total_reasonings": 0,
            "successful_reasonings": 0,
            "average_depth": 0.0,
            "average_time": 0.0
        }
    
    def _create_algorithm(self):
        """Create reasoning algorithm based on configuration"""
        algorithm_name = self.config.algorithm.upper()
        
        if algorithm_name == "MCTS":
            return MCTS(
                n_iters=self.config.max_iterations,
                w_exp=self.config.exploration_weight,
                output_trace_in_each_iter=self.config.log_reasoning_traces
            )
        elif algorithm_name == "BEAMSEARCH":
            return BeamSearch(
                beam_size=self.config.max_iterations,
                max_depth=self.config.max_depth
            )
        elif algorithm_name == "BFS":
            return BFS(max_depth=self.config.max_depth)
        elif algorithm_name == "DFS":
            return DFS(max_depth=self.config.max_depth)
        else:
            logger.warning(f"Unknown algorithm {algorithm_name}, defaulting to MCTS")
            return MCTS(n_iters=self.config.max_iterations)
    
    async def reason_about_instruction_set(self, instruction_set) -> Dict[str, Any]:
        """Apply reasoning to an entire instruction set"""
        if not self.config.enabled:
            return {"reasoning_applied": False, "message": "Reasoning disabled in config"}
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Set instruction set in world model and search config
            self.world_model.current_instruction_set = instruction_set
            self.search_config.current_instruction_set = instruction_set
            
            # Initialize state
            initial_state = await self.world_model.init_state(instruction_set)
            
            # Apply reasoning
            result = await asyncio.wait_for(
                self._run_reasoning(initial_state),
                timeout=self.config.reasoning_timeout
            )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Update stats
            self._update_stats(result, execution_time)
            
            return {
                "reasoning_applied": True,
                "result": result,
                "execution_time": execution_time,
                "stats": self.reasoning_stats
            }
            
        except asyncio.TimeoutError:
            logger.warning(f"Reasoning timeout after {self.config.reasoning_timeout}s")
            return {"reasoning_applied": False, "error": "timeout"}
        except Exception as e:
            logger.error(f"Reasoning error: {e}", exc_info=True)
            return {"reasoning_applied": False, "error": str(e)}
    
    async def _run_reasoning(self, initial_state: WebAutomationState):
        """Run the reasoning algorithm"""
        # Note: reasoners library may not be fully async, so we run in executor
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # The following line was causing an issue: `result = await asyncio.get_event_loop().run_in_executor(None, future.result)`
            # It should be `result = await asyncio.get_event_loop().run_in_executor(executor, future.result)`
            # However, the library's `reasoner()` call itself might not be awaitable directly if it's purely synchronous.
            # A common pattern for running sync code in an async context is:
            # `result = await asyncio.get_event_loop().run_in_executor(executor, self.reasoner, initial_state)`
            # Assuming `self.reasoner(initial_state)` is the synchronous call.
            result = await asyncio.get_event_loop().run_in_executor(executor, self.reasoner, initial_state)
        
        return result
    
    def _update_stats(self, result, execution_time):
        """Update reasoning statistics"""
        self.reasoning_stats["total_reasonings"] += 1
        
        # Update success rate
        if result and getattr(result, 'terminal_state', None):
            if result.terminal_state.error_count == 0:
                self.reasoning_stats["successful_reasonings"] += 1
        
        # Update average time
        total = self.reasoning_stats["total_reasonings"]
        current_avg = self.reasoning_stats["average_time"]
        self.reasoning_stats["average_time"] = ((current_avg * (total - 1)) + execution_time) / total
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get reasoning performance statistics"""
        stats = self.reasoning_stats.copy()
        if stats["total_reasonings"] > 0:
            stats["success_rate"] = stats["successful_reasonings"] / stats["total_reasonings"]
        else:
            stats["success_rate"] = 0.0
        return stats
    
    def reset_stats(self):
        """Reset reasoning statistics"""
        self.reasoning_stats = {
            "total_reasonings": 0,
            "successful_reasonings": 0,
            "average_depth": 0.0,
            "average_time": 0.0
        }
