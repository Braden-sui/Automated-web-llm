"""
Original quality memory manager with minimal fix for Windows file locking
This preserves the original clean code structure
"""
from mem0 import Memory
import logging
import os
from web_automation.config.config_models import Mem0AdapterConfig
import time
import shutil
from pathlib import Path
from typing import Optional # Import Optional

logger = logging.getLogger(__name__)

class Mem0BrowserAdapter:
    def __init__(self, mem0_config: Mem0AdapterConfig = None):
        """
        Initializes the Mem0BrowserAdapter.

        Args:
            mem0_config: Configuration object for Mem0 AI.
        """
        self.memory = None
        final_mem0_init_config = {}
        if mem0_config: # If a specific config is provided
            self.mem0_config = mem0_config
            logger.info(f"Mem0BrowserAdapter initialized with provided Mem0AdapterConfig: {mem0_config.model_dump_json(indent=2)}")

            # Build up the Mem0 configuration dictionary with defaults and overrides
            final_mem0_init_config = {}

            # Vector Store Configuration (Qdrant)
            # According to Mem0 schema, we need to use specific fields
            qdrant_config = {
                "collection_name": self.mem0_config.qdrant_collection_name or "mem0_default_collection",
                "on_disk": self.mem0_config.qdrant_on_disk,  # Key parameter for in-memory vs disk storage
            }
            
            # Add embedding dimensions - critical for the embedder to work correctly
            if self.mem0_config.qdrant_embedding_model_dims:
                qdrant_config["embedding_model_dims"] = self.mem0_config.qdrant_embedding_model_dims
                logger.info(f"Mem0BrowserAdapter: Setting Qdrant embedding_model_dims to {self.mem0_config.qdrant_embedding_model_dims}")
            else:
                # Fallback dimension if not specified - critical for sentence-transformers
                logger.warning("qdrant_embedding_model_dims not specified in Mem0AdapterConfig. This is critical for non-OpenAI embedders.")
                # Defaulting to 384 for all-MiniLM-L6-v2
                qdrant_config["embedding_model_dims"] = 384
                logger.info("Mem0BrowserAdapter: Using default embedding_model_dims=384 for all-MiniLM-L6-v2") 

            final_mem0_init_config["vector_store"] = {
                "provider": "qdrant",
                "config": qdrant_config
            }

            # Embedder Configuration (HuggingFace/sentence-transformers)
            # Assuming we always want to use the local sentence-transformer for tests if memory is enabled
            final_mem0_init_config["embedder"] = {
                "provider": "huggingface",
                "config": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2"  # Full model name as per guide
                }
            }
            logger.info(f"Mem0BrowserAdapter: Configuring embedder: {final_mem0_init_config['embedder']}")

            # LLM Configuration
            llm_config_data = {
                "model": self.mem0_config.llm_model,
                "temperature": self.mem0_config.llm_temperature
            }

            if self.mem0_config.llm_provider.lower() == "ollama":
                if self.mem0_config.llm_base_url:
                    llm_config_data["base_url"] = self.mem0_config.llm_base_url
                # For Ollama, API key is not typically used in the config block this way.
                logger.info(f"Configuring Ollama LLM provider with model: {self.mem0_config.llm_model}")
            elif self.mem0_config.llm_provider.lower() == "openai":
                # OpenAI API key is expected to be set as an environment variable (OPENAI_API_KEY)
                # which Mem0's OpenAI client should pick up automatically.
                # The self.mem0_config.api_key is available if direct injection was ever needed, but standard is env var.
                if not os.getenv("OPENAI_API_KEY") and self.mem0_config.api_key:
                    # This is a fallback, ideally env var is set before this point.
                    logger.warning("OpenAI API key not found in environment, consider setting OPENAI_API_KEY.")
                logger.info(f"Configuring OpenAI LLM provider with model: {self.mem0_config.llm_model}")
            else:
                logger.warning(f"Unsupported LLM provider: {self.mem0_config.llm_provider}. Configuration may be incomplete.")
                # If other providers need specific config keys (like api_key directly in config), add here.
                if self.mem0_config.api_key:
                    llm_config_data["api_key"] = self.mem0_config.api_key # Example for a generic provider

            final_mem0_init_config["llm"] = {
                "provider": self.mem0_config.llm_provider,
                "config": llm_config_data
            }
            logger.info(f"Mem0BrowserAdapter: Final LLM config: {final_mem0_init_config['llm']}")

            # Version Configuration
            final_mem0_init_config["version"] = self.mem0_config.mem0_version
            logger.info(f"Mem0BrowserAdapter: Setting Mem0 version: {final_mem0_init_config['version']}")

        else: # No mem0_config provided, attempt default initialization
            logger.warning("Mem0BrowserAdapter: No Mem0AdapterConfig provided. Attempting default Mem0 initialization.")
            # Default init might fail if OPENAI_API_KEY is not set, or use a default local setup if Mem0 supports it.

        # Force garbage collection to help release any file handles
        import gc
        gc.collect()
        
        # Try to initialize with retries and cleanup
        for attempt in range(3):
            try:
                if final_mem0_init_config:
                    logger.info(f"Attempting to initialize Mem0 with config: {final_mem0_init_config}")
                    self.memory = Memory.from_config(final_mem0_init_config)
                    logger.info(f"Mem0 Memory initialized with custom config on attempt {attempt + 1}")
                else:
                    self.memory = Memory()
                    logger.info(f"Mem0 Memory initialized with default config (no custom config provided) on attempt {attempt + 1}.")
                
                if self.memory and mem0_config:
                    if mem0_config.qdrant_on_disk and mem0_config.qdrant_path:
                        logger.info(f"Mem0 Qdrant backend is using disk storage at: {mem0_config.qdrant_path}")
                    elif not mem0_config.qdrant_on_disk and mem0_config.qdrant_path is None:
                        logger.info("Mem0 Qdrant backend is configured for in-memory storage.")
                break  # Success, exit loop
            except Exception as e:
                logger.warning(f"Mem0 initialization attempt {attempt + 1} failed: {e}")
                if "WinError 32" in str(e) or "database is locked" in str(e).lower():
                    configured_qdrant_path_for_cleanup = None
                    if mem0_config and mem0_config.qdrant_path and mem0_config.qdrant_on_disk:
                        configured_qdrant_path_for_cleanup = Path(mem0_config.qdrant_path)
                    self._handle_database_lock(configured_qdrant_path=configured_qdrant_path_for_cleanup)
                    time.sleep(1)  # Wait before retry
                
                if attempt == 2:  # Last attempt
                    logger.error(f"Failed to initialize Mem0 Memory after 3 attempts: {e}")
                    self.memory = None # Ensure memory is None if all attempts fail
                    break # Exit loop

    def _handle_database_lock(self, configured_qdrant_path: Optional[Path] = None):
        """
        Handle database locking issues.
        If configured_qdrant_path is provided (and was used for on-disk Qdrant),
        this function will attempt to remove that directory.
        Otherwise, it falls back to cleaning the default ~/.mem0 location.
        """
        try:
            if configured_qdrant_path:
                if configured_qdrant_path.exists():
                    logger.warning(f"Attempting to clean up Qdrant storage at configured path: {configured_qdrant_path}...")
                    try:
                        shutil.rmtree(configured_qdrant_path)
                        logger.info(f"Successfully removed entire directory: {configured_qdrant_path}")
                    except Exception as dir_error:
                        logger.error(f"Critical: Could not remove Qdrant directory {configured_qdrant_path}: {dir_error}. The lock might persist.")
                else:
                    logger.info(f"Configured Qdrant path {configured_qdrant_path} does not exist, no cleanup needed for it by rmtree.")
                return # Attempted cleanup of custom path, so exit.

            default_mem0_dir = Path.home() / ".mem0"
            if default_mem0_dir.exists():
                logger.warning(f"Attempting to clean up locked Mem0 database at default location: {default_mem0_dir}...")
                internal_qdrant_storage_dir = default_mem0_dir / "migrations_qdrant" / "collection" / "mem0migrations"
                
                if internal_qdrant_storage_dir.exists():
                    locked_files = [
                        internal_qdrant_storage_dir / "storage.sqlite",
                        internal_qdrant_storage_dir / "storage.sqlite-wal",
                        internal_qdrant_storage_dir / "storage.sqlite-shm"
                    ]
                    files_removed_count = 0
                    for file_path in locked_files:
                        if file_path.exists():
                            try:
                                file_path.unlink()
                                logger.info(f"Removed locked file: {file_path}")
                                files_removed_count +=1
                            except Exception as cleanup_error:
                                logger.warning(f"Could not remove {file_path}: {cleanup_error}")
                    
                    main_sqlite_file = internal_qdrant_storage_dir / "storage.sqlite"
                    if main_sqlite_file.exists() or (files_removed_count > 0 and files_removed_count < 3):
                        logger.warning(f"Lock issue might persist in {internal_qdrant_storage_dir}. Attempting to remove entire {default_mem0_dir} directory.")
                        try:
                            shutil.rmtree(default_mem0_dir)
                            logger.info(f"Successfully removed entire {default_mem0_dir} directory.")
                        except Exception as dir_error:
                            logger.error(f"Critical: Could not remove {default_mem0_dir} directory: {dir_error}.")
                else:
                    logger.warning(f"Internal Qdrant storage dir {internal_qdrant_storage_dir} not found. Attempting to remove entire {default_mem0_dir} directory.")
                    try:
                        shutil.rmtree(default_mem0_dir)
                        logger.info(f"Successfully removed entire {default_mem0_dir} directory.")
                    except Exception as dir_error:
                        logger.error(f"Critical: Could not remove {default_mem0_dir} directory: {dir_error}.")
            else:
                logger.info(f"Default Mem0 directory {default_mem0_dir} does not exist, no cleanup needed for it.")
        except Exception as e:
            logger.error(f"Error during database cleanup process: {e}")

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
                f"User preference: {preference}", 
                user_id=user_id, 
                metadata=full_metadata
            )
            logger.debug(f"Stored user preference for {user_id}: {preference}")
        except Exception as e:
            logger.error(f"Error storing user preference for {user_id}: {e}")

    # SESSION MEMORY: Current conversation, active tasks, context
    async def store_session_context(self, user_id: str, context_data: dict, metadata: dict = None):
        if not self.memory:
            logger.warning("Memory not initialized. Cannot store session context.")
            return
            
        full_metadata = {"type": "session"}
        if metadata:
            full_metadata.update(metadata)

        try:
            context_str = str(context_data)
            self.memory.add(
                f"Session context: {context_str}", 
                user_id=user_id,
                metadata=full_metadata
            )
            logger.debug(f"Stored session context for {user_id}: {context_str[:100]}...")
        except Exception as e:
            logger.error(f"Error storing session context for {user_id}: {e}")

    # AUTOMATION MEMORY: Successful patterns, failed attempts, optimizations
    def store_automation_pattern(self, description: str, selector: str, success: bool, user_id: str, metadata: dict = None):
        if not self.memory:
            logger.warning("Memory not initialized. Cannot store automation pattern.")
            return
            
        # result_text = "successful" if success else "failed" # No longer needed for main text
        full_metadata = {"type": "automation", "success": success, "selector": selector}
        if metadata:
            full_metadata.update(metadata) # Allow overriding/adding to selector if needed
            
        try:
            # Store the human-readable description as the main searchable text.
            # The actual selector is in metadata.
            add_result = self.memory.add(
                description, 
                user_id=user_id, 
                metadata=full_metadata
            )
            print(f"MEM0_ADD_RESULT for automation_pattern: {add_result}") # DEBUG PRINT
            logger.debug(f"Stored automation pattern for {user_id} (Success: {success}): Description='{description}', Selector='{selector}'")
        except Exception as e:
            logger.error(f"Error storing automation pattern for {user_id}: {e}")

    # Generic search method (can be specialized later as needed)
    def search_memory(self, query: str, user_id: str, limit: int = 5, metadata_filter: dict = None):
        if not self.memory:
            logger.warning("Memory not initialized. Cannot search memory.")
            return []
        
        try:
            raw_output = self.memory.search(query=query, user_id=user_id, limit=limit)
            logger.debug(f"Raw output from mem0.search for {user_id} with query '{query}': {raw_output}")
            results_list = []

            if isinstance(raw_output, dict) and 'results' in raw_output and isinstance(raw_output['results'], list):
                results_list = raw_output['results']
            elif isinstance(raw_output, list):
                results_list = raw_output
            else:
                logger.warning(f"Unexpected output format from mem0.search for {user_id} with query '{query}': {type(raw_output)} - {raw_output}")
                return []

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
        results = self.search_memory(
            query=pattern_query, 
            user_id=user_id, 
            limit=limit,
            metadata_filter={"type": "automation"}
        )
        print(f"MEM0_SEARCH_RESULTS for automation_patterns (query='{pattern_query}', filter={{'type': 'automation'}}): {results}") # DEBUG PRINT
        return results

    async def search_session_context(self, user_id: str, query: str, limit: int = 5):
        results = self.search_memory(
            query=query, 
            user_id=user_id, 
            limit=limit,
            metadata_filter={"type": "session"}
        )
        formatted_results = []
        for result in results:
            formatted_results.append({
                "data": result.get("memory", ""),
                "metadata": result.get("metadata", {})
            })
        return formatted_results

    def get_session_context(self, session_id: str, limit: int = 10) -> list:
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
                memories_list = raw_output
            else:
                logger.warning(f"Unexpected output format from mem0.get_all for {session_id}: {type(raw_output)} - {raw_output}")
                return []

            for m_idx, m in enumerate(memories_list):
                try:
                    if isinstance(m, dict) and m.get('metadata', {}).get('type') == 'session':
                        session_memories.append(m.get('memory', ''))
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

    def update_session_context(self, instructions: list, user_id: str):
        context_summary = f"Processed {len(instructions)} instructions. Last type: {instructions[-1].get('type', 'N/A') if instructions else 'N/A'}"
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self.store_session_context(user_id, {"summary": context_summary}))
        except:
            self.store_user_preference(user_id, f"Session: {context_summary}")
        logger.debug(f"Updated session context for user {user_id} after processing instructions.")