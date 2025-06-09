"""
Original quality memory manager with minimal fix for Windows file locking
This preserves the original clean code structure
"""
import os
os.environ["OPENAI_API_KEY"] = ""  # Prevent OpenAI/Ollama conflicts before mem0 import
from mem0 import Memory
import logging
from datetime import datetime
import pytz
from web_automation.config.config_models import Mem0AdapterConfig
import time
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise e

import functools

import functools
from collections import OrderedDict

class Mem0BrowserAdapter:
    def __init__(self, mem0_config: Mem0AdapterConfig = None):
        # ... existing code ...
        self._search_cache = OrderedDict()
        self._search_cache_size = 64
        self._cache_stats = {'hits': 0, 'misses': 0}

        """
        Initializes the Mem0BrowserAdapter.

        Args:
            mem0_config: Configuration object for Mem0 AI.
        """
        print(f"--- Mem0Adapter: __init__ ENTERING. Received mem0_config type: {type(mem0_config)}")
        self.memory = None
        final_mem0_init_config = {}
        if mem0_config: # If a specific config is provided
            self.mem0_config = mem0_config
            logger.info(f"Mem0BrowserAdapter initialized with provided Mem0AdapterConfig: {mem0_config.model_dump_json(indent=2)}")
            print(f"--- Mem0Adapter: Processing provided mem0_config.")

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
            print(f"--- Mem0Adapter: Built vector_store config: {final_mem0_init_config['vector_store']}")

            # Embedder Configuration (HuggingFace/sentence-transformers)
            # Assuming we always want to use the local sentence-transformer for tests if memory is enabled
            final_mem0_init_config["embedder"] = {
                "provider": "huggingface",
                "config": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2"  # Full model name as per guide
                }
            }
            logger.info(f"Mem0BrowserAdapter: Configuring embedder: {final_mem0_init_config['embedder']}")
            print(f"--- Mem0Adapter: Built embedder config: {final_mem0_init_config['embedder']}")

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
            print(f"--- Mem0Adapter: Built LLM config: {final_mem0_init_config['llm']}")

            # Agent ID Configuration
            if self.mem0_config.agent_id:
                final_mem0_init_config["agent_id"] = self.mem0_config.agent_id
                logger.info(f"Mem0BrowserAdapter: Setting agent_id for Mem0: {self.mem0_config.agent_id}")

            # Version Configuration
            final_mem0_init_config["version"] = self.mem0_config.mem0_version
            logger.info(f"Mem0BrowserAdapter: Setting Mem0 version: {final_mem0_init_config['version']}")

        else: # No mem0_config provided, attempt default initialization
            logger.warning("Mem0BrowserAdapter: No Mem0AdapterConfig provided. Attempting default Mem0 initialization.")
            print("--- Mem0Adapter: No mem0_config provided, will attempt default Mem0 init.")
            # Default init might fail if OPENAI_API_KEY is not set, or use a default local setup if Mem0 supports it.

        # Force garbage collection to help release any file handles
        import gc
        gc.collect()
        print("--- Mem0BrowserAdapter: Garbage collection completed.")
        
        # Try to initialize with retries and cleanup
        print(f"--- Mem0Adapter: Entering Mem0 instantiation loop. final_mem0_init_config IS_SET: {bool(final_mem0_init_config)}")
        for attempt in range(3):
            try:
                if final_mem0_init_config:
                    logger.info(f"Attempting to initialize Mem0 with config: {final_mem0_init_config}")
                    print(f"--- Mem0Adapter: Attempt {attempt + 1}: Calling Memory.from_config with: {final_mem0_init_config}")
                    self.memory = Memory.from_config(final_mem0_init_config)
                    logger.info(f"Mem0 Memory initialized with custom config on attempt {attempt + 1}")
                    print(f"--- Mem0Adapter: Attempt {attempt + 1}: Memory.from_config SUCCESSFUL.")
                else:
                    print(f"--- Mem0Adapter: Attempt {attempt + 1}: Calling Memory() (default config).")
                    self.memory = Memory()
                    logger.info(f"Mem0 Memory initialized with default config (no custom config provided) on attempt {attempt + 1}.")
                    print(f"--- Mem0Adapter: Attempt {attempt + 1}: Memory() SUCCESSFUL.")
                
                if self.memory and mem0_config:
                    if mem0_config.qdrant_on_disk and mem0_config.qdrant_path:
                        logger.info(f"Mem0 Qdrant backend is using disk storage at: {mem0_config.qdrant_path}")
                    elif not mem0_config.qdrant_on_disk and mem0_config.qdrant_path is None:
                        logger.info("Mem0 Qdrant backend is configured for in-memory storage.")
                break  # Success, exit loop
            except Exception as e:
                logger.warning(f"Mem0 initialization attempt {attempt + 1} failed: {e}")
                print(f"--- Mem0Adapter: Attempt {attempt + 1}: EXCEPTION during Mem0 init: {e}")
                import traceback
                traceback.print_exc() # Print full traceback to console
                if "WinError 32" in str(e) or "database is locked" in str(e).lower():
                    configured_qdrant_path_for_cleanup = None
                    if mem0_config and mem0_config.qdrant_path and mem0_config.qdrant_on_disk:
                        configured_qdrant_path_for_cleanup = Path(mem0_config.qdrant_path)
                    self._handle_database_lock(configured_qdrant_path=configured_qdrant_path_for_cleanup)
                    time.sleep(1)  # Wait before retry
                
                if attempt == 2:  # Last attempt
                    logger.error(f"Failed to initialize Mem0 Memory after 3 attempts: {e}")
                    self.memory = None # Ensure memory is None if all attempts fail
                    print(f"--- Mem0Adapter: Failed to init Mem0 after 3 attempts. self.memory is None.")
                    break # Exit loop
        print(f"--- Mem0BrowserAdapter: __init__ COMPLETED successfully.")

    def _handle_database_lock(self, configured_qdrant_path: Optional[Path] = None):
        """
        Handle database locking issues. Only allow destructive cleanup if MEM0_ALLOW_DESTRUCTIVE_TEST_CLEANUP is set.
        If configured_qdrant_path is provided (and was used for on-disk Qdrant),
        this function will attempt to remove that directory.
        Otherwise, it falls back to cleaning the default ~/.mem0 location.
        """
        allow_cleanup = os.getenv("MEM0_ALLOW_DESTRUCTIVE_TEST_CLEANUP", "0").lower() in ("1", "true")
        if not allow_cleanup:
            logger.warning("Destructive Qdrant cleanup is DISABLED. Set MEM0_ALLOW_DESTRUCTIVE_TEST_CLEANUP=1 to enable cleanup in test/dev only. Skipping cleanup for safety.")
            return
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
            # Add a short timeout if supported (Mem0 may not support directly, but wrap in a timeout if needed)
            self.memory.add(
                f"User preference: {preference}", 
                user_id=user_id, 
                metadata=full_metadata
            )
            logger.debug(f"Stored user preference for {user_id}: {preference}")
        except Exception as e:
            logger.error(f"Error storing user preference for {user_id}: {e}", exc_info=True)

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
            logger.error(f"Error storing session context for {user_id}: {e}", exc_info=True)

    # AUTOMATION MEMORY: Successful patterns, failed attempts, optimizations
    def store_automation_pattern(
        self,
        user_id: str,
        description: str = None,
        selector: str = None,
        success: bool = True,
        fallback_selector: Optional[str] = None,
        metadata: Optional[dict] = None,
        pattern: Optional[str] = None,
    ):
        if not self.memory:
            logger.warning("Memory not initialized. Cannot store automation pattern.")
            return

        # Support both legacy (description/selector) and new (pattern) usage
        if pattern is not None and description is None and selector is None:
            pattern_text = pattern
            metadata_to_store = {
                "type": "automation_pattern",
                "success": success,
                "timestamp": datetime.now(pytz.utc).isoformat(),
            }
            logger.debug(f"[DEBUG] Storing automation pattern (pattern mode): pattern_text='{pattern_text}', metadata={metadata_to_store}")
        else:
            if description is None or selector is None or user_id is None:
                logger.error("store_automation_pattern requires description and selector when no pattern is provided.")
                return

    def search_memory(self, query: str, user_id: str, limit: int = 5, metadata_filter: dict = None):
        if not self.memory:
            logger.warning("Memory not initialized. Cannot search memory.")
            return []
        try:
            # mem0 SDK expects 'filters' for metadata filtering
            raw_output = self.memory.search(
                query=query,
                user_id=user_id,
                limit=limit,
                filters=metadata_filter
            )
            logger.debug(f"Mem0 search raw output: {raw_output}")
            return raw_output
        except Exception as e:
            logger.error(f"Error searching memory for {user_id}: {e}")
            return []

    def get_visual_patterns_for_user(self, user_id: str, limit: int = 10):
        """
        Retrieve all visual patterns for a user (empty query for all patterns).
        """
        logger.info(f"Fetching all visual patterns for user {user_id} with limit {limit}")
        return self.search_memory("", user_id, limit=limit, metadata_filter={"type": "visual_pattern"})

    def search_automation_patterns(self, pattern_query: str, user_id: str, limit: int = 10):
        """
        Search for automation patterns for a user with optional query string.
        """
        logger.info(f"Searching automation patterns for user {user_id} with query '{pattern_query}' and limit {limit}")
        return self.search_memory(pattern_query, user_id, limit=limit, metadata_filter={"type": "automation_pattern"})

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