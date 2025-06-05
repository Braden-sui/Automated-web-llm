from mem0 import Memory
import logging
import os
from web_automation.config.config_models import Mem0AdapterConfig

logger = logging.getLogger(__name__)

class Mem0BrowserAdapter:
    def __init__(self, mem0_config: Mem0AdapterConfig = None):
        """
        Initializes the Mem0BrowserAdapter.

        Args:
            mem0_config: Configuration object for Mem0 AI.
        """
        self.memory = None
        mem0_initialization_config = {}

        if mem0_config:
            if mem0_config.api_key:
                os.environ.setdefault("MEM0_API_KEY", mem0_config.api_key)
            if mem0_config.agent_id:
                mem0_initialization_config["agent_id"] = mem0_config.agent_id

        try:
            if mem0_initialization_config:
                self.memory = Memory(**mem0_initialization_config)
            else:
                self.memory = Memory()
            logger.info("Mem0 Memory initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Mem0 Memory: {e}")
            self.memory = None # Fallback or raise error

    # USER MEMORY: Preferences, behavior patterns, learned strategies
    def store_user_preference(self, user_id: str, preference: str, metadata: dict = None):
        if not self.memory:
            logger.warning("Memory not initialized. Cannot store user preference.")
            return
        
        full_metadata = {"type": "preference"}
        if metadata:
            full_metadata.update(metadata)
            
        try:
            self.memory.add(
                data=f"User preference: {preference}", 
                user_id=user_id, 
                metadata=full_metadata
            )
            logger.debug(f"Stored user preference for {user_id}: {preference}")
        except Exception as e:
            logger.error(f"Error storing user preference for {user_id}: {e}")

    # SESSION MEMORY: Current conversation, active tasks, context
    def store_session_context(self, session_id: str, context: str, metadata: dict = None):
        if not self.memory:
            logger.warning("Memory not initialized. Cannot store session context.")
            return
            
        full_metadata = {"type": "session"}
        if metadata:
            full_metadata.update(metadata)

        try:
            self.memory.add(
                data=f"Session context: {context}", 
                user_id=session_id, # Using session_id as user_id for mem0
                metadata=full_metadata
            )
            logger.debug(f"Stored session context for {session_id}: {context}")
        except Exception as e:
            logger.error(f"Error storing session context for {session_id}: {e}")

    # AUTOMATION MEMORY: Successful patterns, failed attempts, optimizations
    def store_automation_pattern(self, pattern: str, success: bool, user_id: str, metadata: dict = None):
        if not self.memory:
            logger.warning("Memory not initialized. Cannot store automation pattern.")
            return
            
        result_text = "successful" if success else "failed"
        full_metadata = {"type": "automation", "success": success}
        if metadata:
            full_metadata.update(metadata)
            
        try:
            self.memory.add(
                f"Automation pattern {result_text}: {pattern}", 
                user_id=user_id, 
                metadata=full_metadata
            )
            logger.debug(f"Stored automation pattern for {user_id} (Success: {success}): {pattern}")
        except Exception as e:
            logger.error(f"Error storing automation pattern for {user_id}: {e}")

    # Generic search method (can be specialized later as needed)
    def search_memory(self, query: str, user_id: str, limit: int = 5, metadata_filter: dict = None):
        if not self.memory:
            logger.warning("Memory not initialized. Cannot search memory.")
            return [] # Return empty list or handle error appropriately
        
        try:
            raw_output = self.memory.search(query=query, user_id=user_id, limit=limit)
            logger.debug(f"Raw output from mem0.search for {user_id} with query '{query}': {raw_output}")
            results_list = []

            if isinstance(raw_output, dict) and 'results' in raw_output and isinstance(raw_output['results'], list):
                results_list = raw_output['results']
            elif isinstance(raw_output, list):
                results_list = raw_output # Older behavior or direct list
            else:
                logger.warning(f"Unexpected output format from mem0.search for {user_id} with query '{query}': {type(raw_output)} - {raw_output}")
                return [] # Return empty list on unexpected format

            if metadata_filter:
                filtered_results = []
                for res_idx, res_item in enumerate(results_list):
                    try:
                        if isinstance(res_item, dict) and res_item.get('metadata') and all(item in res_item['metadata'].items() for item in metadata_filter.items()):
                            filtered_results.append(res_item)
                        elif not isinstance(res_item, dict):
                            logger.warning(f"Search result item at index {res_idx} is not a dict: {res_item}")
                    except Exception as item_exc:
                        logger.error(f"Error processing search result item at index {res_idx}: {res_item}. Exception: {item_exc}")
                logger.debug(f"Searched memory for {user_id} with query '{query}', filter {metadata_filter}. Found {len(filtered_results)} after filtering from {len(results_list)} raw results.")
                return filtered_results
            else:
                # Ensure all items in results_list are dicts if no filter is applied, or handle non-dicts gracefully
                # For now, assume if no filter, client expects list of dicts or whatever mem0 returns in 'results'
                # Adding a simple check to ensure it's a list of dicts primarily
                processed_results_list = []
                for res_idx, res_item in enumerate(results_list):
                    if isinstance(res_item, dict):
                        processed_results_list.append(res_item)
                    else:
                        logger.warning(f"Search result item at index {res_idx} (no filter) is not a dict: {res_item}. Skipping.")
                logger.debug(f"Searched memory for {user_id} with query '{query}'. Found {len(processed_results_list)} items.")
                return processed_results_list
        except Exception as e:
            logger.error(f"Error searching memory for {user_id} with query '{query}': {e}")
            return []

    def search_automation_patterns(self, pattern_query: str, user_id: str, limit: int = 5):
        return self.search_memory(
            query=pattern_query, 
            user_id=user_id, 
            limit=limit,
            metadata_filter={"type": "automation"}
        )

    def get_session_context(self, session_id: str, limit: int = 10) -> list[str]:
        if not self.memory:
            logger.warning("Memory not initialized. Cannot get session context.")
            return []
        
        try:
            raw_output = self.memory.get_all(user_id=session_id, limit=limit)
            logger.debug(f"Raw output from mem0.get_all for {session_id}: {raw_output}")
            session_memories = []
            memories_list = []

            if isinstance(raw_output, dict) and 'results' in raw_output and isinstance(raw_output['results'], list):
                memories_list = raw_output['results']
            elif isinstance(raw_output, list):
                # Older behavior, or direct list of memories
                memories_list = raw_output
            else:
                logger.warning(f"Unexpected output format from mem0.get_all for {session_id}: {type(raw_output)} - {raw_output}")
                # Fall through to return empty session_memories

            for m_idx, m in enumerate(memories_list):
                try:
                    if isinstance(m, dict) and m.get('metadata', {}).get('type') == 'session':
                        session_memories.append(m['memory'])
                    elif not isinstance(m, dict):
                        logger.warning(f"Memory item at index {m_idx} is not a dict: {m}")
                except Exception as item_exc:
                    logger.error(f"Error processing memory item at index {m_idx}: {m}. Exception: {item_exc}")
            logger.debug(f"Retrieved {len(session_memories)} session context items for {session_id}.")
            return session_memories
        except Exception as e:
            logger.error(f"Error retrieving session context for {session_id}: {e}")
            return []

    def learn_from_execution(self, instruction: dict, result: dict, user_id: str):
        self.store_automation_pattern(
            pattern=f"{instruction.get('type', 'unknown_type')}: {instruction.get('selector', '')}",
            success=result.get('success', False),
            user_id=user_id
        )
        logger.debug(f"Learned from execution for user {user_id}: instruction type {instruction.get('type')}")

    def update_session_context(self, instructions: list[dict], user_id: str):
        context_summary = f"Processed {len(instructions)} instructions. Last type: {instructions[-1].get('type', 'N/A') if instructions else 'N/A'}"
        self.store_session_context(session_id=user_id, context=context_summary)
        logger.debug(f"Updated session context for user {user_id} after processing instructions.")
