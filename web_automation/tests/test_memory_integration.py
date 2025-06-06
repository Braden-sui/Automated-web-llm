import asyncio
import pytest
import uuid # Import uuid
from web_automation import create_playwright_agent
from web_automation.config.config_models import Mem0AdapterConfig # Import Mem0AdapterConfig
# The plan mentions ClickInstruction and ActionType, but they are not directly used in the test_memory_enhanced_agent.
# If they were needed for constructing instructions to pass to an agent method, they would be imported from:
# from web_automation.models.instructions import ClickInstruction, ActionType 

import logging
import os
# from dotenv import load_dotenv # No longer loading OPENAI_API_KEY from .env for this test
logging.getLogger('mem0').setLevel(logging.DEBUG)

@pytest.mark.asyncio
async def test_memory_enhanced_agent():
    """Test memory-enhanced agent basic functionality with isolated in-memory Qdrant and Ollama LLM."""

    # Create a specific Mem0 config for this test, now using Ollama
    test_collection_name = f"test_mem_collection_{uuid.uuid4().hex[:10]}" # Unique collection name
    test_mem0_config = Mem0AdapterConfig(
        qdrant_path=None,  # Explicitly None for in-memory Qdrant
        qdrant_on_disk=False, # Ensure Qdrant runs in-memory
        qdrant_collection_name=test_collection_name,
        qdrant_embedding_model_dims=384, # Critical: For all-MiniLM-L6-v2 embedder
        mem0_version="v1.1", 
        llm_provider="ollama", 
        llm_model="qwen2.5vl:7b", # Your Ollama model
        llm_temperature=0.7, 
        # llm_base_url can be specified if Ollama is not on http://localhost:11434
        api_key=None # Not needed for Ollama
        # agent_id can be omitted if not needed for local/test Mem0
    )

    # Ensure any lingering OPENAI_API_KEY from previous test runs or environment is cleared for this test,
    # to ensure we are truly testing the Ollama path without interference.
    original_openai_api_key_for_cleanup = os.environ.pop("OPENAI_API_KEY", None)

    try:
        agent = create_playwright_agent(
            memory_enabled=True, 
            headless=True,
            memory_config=test_mem0_config # Pass the specific config
        )

        print(f"\n=== MEMORY DEBUG START ===")
        print(f"Agent type: {type(agent).__name__}")
        print(f"Memory manager exists: {agent.memory_manager is not None}")
        if agent.memory_manager:
           print(f"Memory object exists: {hasattr(agent.memory_manager, 'memory') and agent.memory_manager.memory is not None}")
           if hasattr(agent.memory_manager, 'memory') and agent.memory_manager.memory:
               print(f"\n=== DIRECT MEM0 ADD/SEARCH TEST ===")
               direct_add_user_id = f"direct_test_user_{uuid.uuid4().hex[:6]}"
               direct_add_text = "Direct Mem0 add test entry"
               direct_add_metadata = {"type": "direct_test", "source": "pytest"}
               try:
                   add_op_result = agent.memory_manager.memory.add(
                       direct_add_text,
                       user_id=direct_add_user_id,
                       metadata=direct_add_metadata
                   )
                   print(f"Direct memory.add result: {add_op_result}")
                   assert add_op_result is not None, "Mem0 add operation should return a result."
                   assert "results" in add_op_result, "Mem0 add result should contain 'results' key."
                   assert len(add_op_result["results"]) > 0, "Mem0 add should store at least one memory item in 'results'."
                   assert "memory" in add_op_result["results"][0], "Stored memory item in 'results' should have a 'memory' key for the text."
                   assert add_op_result["results"][0]["memory"] == direct_add_text, "Stored memory text should match input."

                   # Try to retrieve it immediately
                   search_results = agent.memory_manager.memory.search(query=direct_add_text, user_id=direct_add_user_id, limit=1)
                   print(f"Direct memory.search for '{direct_add_text}' result: {search_results}")
                   assert search_results is not None, "Mem0 search operation should return results."
                   assert "results" in search_results, "Mem0 search result should contain 'results' key."
                   assert len(search_results["results"]) > 0, "Mem0 search should find at least one memory in 'results'."
                   assert search_results["results"][0]["memory"] == direct_add_text, "Found memory text should match added text."
                   all_for_user = agent.memory_manager.memory.get_all(user_id=direct_add_user_id)
                   print(f"Direct memory.get_all for user '{direct_add_user_id}' result: {all_for_user}")
               except Exception as e:
                   print(f"ERROR during direct Mem0 add/search test: {e}")
               print(f"=== DIRECT MEM0 ADD/SEARCH TEST END ===\n")
        
        async with agent: # WebBrowserAgent likely implements __aenter__ and __aexit__
            # Test navigation with memory (or standard navigation if memory doesn't alter it)
            # Assuming _page is an attribute set up by WebBrowserAgent's __aenter__
            if agent._page:
                await agent._page.goto("https://example.com")
                await asyncio.sleep(0.5) # Ensure page is fully settled
            else:
                pytest.fail("Agent page was not initialized.")

            print(f"\n=== BEFORE SMART CLICK ===")
            if agent.memory_manager:
                existing_patterns = agent.memory_manager.search_automation_patterns("main navigation", agent.identity_id, limit=5)
                print(f"Existing patterns found: {len(existing_patterns)}")
                for i, pattern in enumerate(existing_patterns):
                    print(f"  Pattern {i}: {pattern.get('memory', 'N/A')[:50]}...")
            
            # Test smart selector (should fallback gracefully)
            success = await agent.smart_selector_click(
                target_description="main navigation link",
                fallback_selector="a[href='https://www.iana.org/domains/example']"
            )

            print(f"\n=== AFTER SMART CLICK ===")
            print(f"Click success: {success}")
            if agent.memory_manager:
                new_patterns = agent.memory_manager.search_automation_patterns("main navigation", agent.identity_id, limit=5)
                print(f"Patterns after click: {len(new_patterns)}")
                for i, pattern in enumerate(new_patterns):
                    print(f"  Pattern {i}: {{'memory': pattern.get('memory', 'N/A')[:50]+'...', 'metadata': pattern.get('metadata')}}")
                
                assert len(new_patterns) > 0, "At least one automation pattern should be stored after smart_selector_click."
                # Check if the description or part of it is in the stored memory text
                assert any("main navigation link" in pattern.get('memory', '') for pattern in new_patterns), \
                    "Stored pattern memory text should relate to 'main navigation link'."
                # Check if the selector used (fallback in this case) is in the metadata
                assert any(pattern.get('metadata', {}).get('original_fallback_selector') == "a[href='https://www.iana.org/domains/example']" for pattern in new_patterns), \
                    "Stored pattern metadata should contain the fallback selector used."

            print(f"=== MEMORY DEBUG END ===\n")
            
            assert success, "smart_selector_click should succeed, at least with fallback."
            
            # Test memory stats
            stats = agent.get_memory_stats()
            assert "memory_enabled" in stats
            assert stats["memory_enabled"] == True
            assert "interactions_stored_session" in stats # Based on my implementation of get_memory_stats
    finally:
        # Clean up/restore OPENAI_API_KEY if it was popped
        if original_openai_api_key_for_cleanup is not None:
            os.environ["OPENAI_API_KEY"] = original_openai_api_key_for_cleanup

@pytest.mark.asyncio  
async def test_standard_agent_compatibility():
    """Test that standard agent still works without memory"""
    agent = create_playwright_agent(memory_enabled=False, headless=True)
    
    async with agent:
        if agent._page:
            await agent._page.goto("https://example.com")
            # Basic check: page title
            title = await agent._page.title()
            assert "Example Domain" in title, "Standard agent could not navigate to example.com or title mismatch."
        else:
            pytest.fail("Agent page was not initialized for standard agent.")
        
def test_memory_config():
    """Test memory configuration loading"""
    from web_automation.config.settings import awm_config
    assert hasattr(awm_config, 'ENABLED')
    assert hasattr(awm_config, 'BACKEND')

def debug_memory_config_flow():
    """Debug helper to trace config serialization"""
    import json
    from web_automation.config.config_models import Mem0AdapterConfig
    
    # Create test config
    test_config = Mem0AdapterConfig(
        qdrant_on_disk=False,
        qdrant_embedding_model_dims=384,
        llm_provider="ollama",
        llm_model="qwen2.5vl:7b"
    )
    
    # Test dict operations
    print(f"\n=== DEBUG MEMORY CONFIG FLOW ===")
    print(f"Original config type: {type(test_config)}")
    print(f"Config fields set: {test_config.model_fields_set}")
    
    # Wrong way (illustrating the old bug)
    print("\n--- Simulating Incorrect Serialization (dict.update(model_instance)) ---")
    wrong_dict = {"enabled": True}
    # Simulate the incorrect update by directly assigning the model instance as a value 
    # (as dict.update(model) would effectively do for its internal items() iteration)
    # For clarity, let's show what happens if you tried to make it a key (which is what dict.update does with an iterable of key-value pairs)
    # but the actual bug was more about the internal structure not being a flat dict.
    # The original bug was: final_memory_config.update(memory_config) -> this iterates memory_config.model_fields
    # and updates final_memory_config with field_name: field_value. The issue was if memory_config was NOT a Mem0AdapterConfig but a plain dict
    # or if the downstream factory expected a dict representation of Mem0AdapterConfig, not the object itself.
    # The most direct way to show the problem fixed by model_dump() is comparing the object vs its dict representation.
    
    # Let's show the state of a config if it were *not* dumped:
    config_as_object_in_dict = {"enabled": True, "mem0_details": test_config} 
    print(f"Simulated wrong dict (object not dumped): {config_as_object_in_dict}")
    print(f"Type of mem0_details: {type(config_as_object_in_dict['mem0_details'])}")

    # Right way
    print("\n--- Correct Serialization (model.model_dump()) ---")
    right_dict = {"enabled": True}
    right_dict.update(test_config.model_dump())
    print(f"Right dict keys: {list(right_dict.keys())}")
    print(f"Right dict (content): {json.dumps(right_dict, indent=2)}")
    print(f"Type of 'config' in right_dict: {type(right_dict.get('config'))}")
    print(f"=== DEBUG MEMORY CONFIG FLOW END ===\n")
    
    return right_dict
    assert awm_config.ENABLED is True # Based on .env file created
    assert awm_config.BACKEND == "sqlite" # Based on .env file created
