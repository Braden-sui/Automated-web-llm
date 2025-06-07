import asyncio
import pytest
import uuid
import logging
import os
from web_automation.config.config_models import Mem0AdapterConfig
from web_automation.core.dependencies import BrowserAgentDependencies, BrowserAgentFactory
from web_automation.memory.memory_manager import Mem0BrowserAdapter

# Configure test logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('mem0').setLevel(logging.DEBUG)

@pytest.fixture
def test_mem0_config():
    """Create a test Mem0 configuration with in-memory Qdrant."""
    return Mem0AdapterConfig(
        qdrant_path=None,  # In-memory Qdrant
        qdrant_on_disk=False,
        qdrant_collection_name=f"test_mem_collection_{uuid.uuid4().hex[:10]}",
        qdrant_embedding_model_dims=384,  # For all-MiniLM-L6-v2 embedder
        mem0_version="v1.1",
        llm_provider="ollama",
        llm_model=os.getenv("MEMORY_LLM_MODEL", "qwen2.5vl:7b"),
        llm_temperature=0.7,
        api_key=None  # Not needed for Ollama
    )

@pytest.fixture
def test_memory_manager(test_mem0_config):
    from web_automation.memory.memory_manager import Mem0BrowserAdapter
    config = Mem0AdapterConfig(
        qdrant_path=None,
        qdrant_on_disk=False,
        qdrant_collection_name=f"test_mem_collection_{uuid.uuid4().hex[:10]}",
        qdrant_embedding_model_dims=384,
        mem0_version="v1.1",
        llm_provider="ollama",
        llm_model=os.getenv("MEMORY_LLM_MODEL", "qwen2.5vl:7b"),
        llm_temperature=0.7,
        api_key=None
    )
    return Mem0BrowserAdapter(config)

def test_mem0_config_instance(test_mem0_config):
    config = test_mem0_config
    assert config.llm_provider == "ollama"
    assert config.llm_model == os.getenv("MEMORY_LLM_MODEL", "qwen2.5vl:7b")
    assert config.llm_temperature == 0.7

@pytest.mark.asyncio
async def test_memory_enhanced_agent(test_mem0_config):
    """Test memory-enhanced agent basic functionality with isolated in-memory Qdrant and Ollama LLM."""
    # Create agent with memory configuration
    agent = BrowserAgentFactory.create_agent(
        memory_config={
            'enabled': True,
            **test_mem0_config.model_dump()
        },
        browser_type='chromium',
        headless=True,
        identity_id=f'test_agent_{uuid.uuid4().hex[:8]}'
    )

    # Test memory operations
    if agent.memory_manager:
        # Define a clear, factual statement for testing semantic search
        original_fact = "The official support channel for critical system outages is the emergency hotline: 555-123-4567."
        test_metadata = {"type": "emergency_contact", "source": "test_suite_infer_false_storage"}
        
        # Add memory with infer=False to GUARANTEE storage and get an ID
        print(f"Attempting to add memory with infer=False: '{original_fact}'")
        add_response = agent.memory_manager.memory.add(
            original_fact,
            user_id=agent.identity_id,
            metadata=test_metadata,
            infer=False  # Guarantee storage, bypassing LLM fact extraction for add
        )
        
        print(f"Add response (infer=False): {add_response}")
        assert add_response is not None, "Add operation with infer=False should return a response."

        # Extract the ID of the added memory. 
        # With infer=False, mem0 should directly return the ID of the stored item or a list containing it.
        added_memory_id = None
        if isinstance(add_response, list) and len(add_response) > 0 and add_response[0].get('id'):
            # Common response format for infer=False is a list of stored items
            added_memory_id = add_response[0]['id']
        elif isinstance(add_response, dict) and add_response.get('id'): 
            # Some versions might return a dict with an id directly
            added_memory_id = add_response['id']
        elif isinstance(add_response, dict) and add_response.get('results') and len(add_response['results']) > 0 and add_response['results'][0].get('id'):
            # Or nested within 'results'
            added_memory_id = add_response['results'][0]['id']

        assert added_memory_id is not None, f"Failed to retrieve an ID for the memory added with infer=False. Response: {add_response}"
        print(f"Stored memory with ID (infer=False): {added_memory_id}")

        # Allow some time for indexing, though infer=False might be quicker
        await asyncio.sleep(1) 
        
        # Search for a semantically related concept. The search itself will use mem0's LLM capabilities.
        search_query = "How do I report a critical system failure?"
        print(f"Searching for: '{search_query}' (tests semantic retrieval)")
        search_results = agent.memory_manager.memory.search(
            query=search_query,
            user_id=agent.identity_id,
            limit=3  # We expect our fact in the top 3
        )
        
        print(f"Search results: {search_results}")
        assert search_results is not None, "Search operation should return results."
        
        # Verify that the original memory (by ID) is in the top 3 results
        found_in_top_results = False
        retrieved_texts = []
        # mem0.search typically returns a list of dicts, or a dict with a 'results' key containing the list
        results_list = []
        if isinstance(search_results, list):
            results_list = search_results
        elif isinstance(search_results, dict) and 'results' in search_results:
            results_list = search_results['results']

        for result in results_list:
            retrieved_texts.append(result.get('text', ''))
            if result.get('id') == added_memory_id:
                found_in_top_results = True
                break
        
        assert found_in_top_results, \
            f"Original memory (ID: {added_memory_id}, Text: '{original_fact}') not found in top 3 search results for query '{search_query}'.\nRetrieved texts: {retrieved_texts}"

    
    async with agent:
        # Test page navigation
        if not hasattr(agent, '_page'):
            pytest.fail("Agent page was not initialized.")
        
        # Navigate to test page
        await agent._page.goto("https://example.com")
        await asyncio.sleep(0.5)  # Allow page to settle
        
        # Test smart selector functionality with more factual storage
        success = await agent.smart_selector_click(
            target_description="IANA documentation link on example.com that explains domain name registration",
            fallback_selector="a[href='https://www.iana.org/domains/example']"
        )
        
        # Verify navigation was successful
        assert "Example Domain" in await agent._page.title()

        print(f"\n=== AFTER SMART CLICK ===")
        print(f"Click success: {success}")
        
        # Verify memory operations
        if agent.memory_manager:
            # Search for any patterns for this user (empty query to get all)
            all_patterns = agent.memory_manager.search_automation_patterns(
                "",  # Empty query to get all patterns
                agent.identity_id,
                limit=10
            )
            
            print(f"All patterns for user: {len(all_patterns)}")
            for i, pattern in enumerate(all_patterns):
                print(f"  Pattern {i}: {{'memory': {pattern.get('memory', 'N/A')[:100]}..., 'metadata': {pattern.get('metadata', {})}}}")
            
            # Verify at least one pattern was stored
            assert len(all_patterns) > 0, "At least one automation pattern should be stored after smart_selector_click."
            
            # Get the most recent pattern (should be first in results)
            recent_pattern = all_patterns[0]
            recent_memory = recent_pattern.get('memory', '')
            recent_metadata = recent_pattern.get('metadata', {})
            
            # Verify the pattern contains our target description
            assert "IANA documentation" in recent_memory, \
                f"Most recent pattern should contain 'IANA documentation'. Got: {recent_memory}"
                
            # Verify the fallback selector is in the metadata
            assert recent_metadata.get('original_fallback_selector') == "a[href='https://www.iana.org/domains/example']", \
                f"Metadata should contain fallback selector. Got: {recent_metadata}"

        print("=== MEMORY DEBUG END ===\n")
        assert success, "smart_selector_click should succeed, at least with fallback."
        
        # Test memory stats if available
        if hasattr(agent, 'get_memory_stats'):
            stats = agent.get_memory_stats()
            assert "memory_enabled" in stats
            assert stats["memory_enabled"] is True
            assert "interactions_stored_session" in stats

@pytest.mark.asyncio  
async def test_standard_agent_compatibility():
    """Test that standard agent still works without memory"""
    # Create agent without memory
    agent = BrowserAgentFactory.create_agent(
        memory_config={'enabled': False},
        browser_type='chromium',
        headless=True,
        identity_id=f'test_agent_{uuid.uuid4().hex[:8]}'
    )
    
    async with agent:
        if not hasattr(agent, '_page'):
            pytest.fail("Agent page was not initialized for standard agent.")
            
        await agent._page.goto("https://example.com")
        # Basic check: page title
        title = await agent._page.title()
        assert "Example Domain" in title, "Standard agent could not navigate to example.com or title mismatch."


def test_memory_config():
    """Test memory configuration loading"""
    # Test creating a memory configuration
    config = Mem0AdapterConfig(
        qdrant_path=None,
        qdrant_on_disk=False,
        qdrant_collection_name=f"test_collection_{uuid.uuid4().hex[:8]}",
        qdrant_embedding_model_dims=384,
        mem0_version="v1.1",
        llm_provider="ollama",
        llm_model="qwen2.5vl:7b",
        llm_temperature=0.7,
        api_key=None
    )
    
    # Verify configuration
    assert config.llm_provider == "ollama"
    assert config.llm_model == "qwen2.5vl:7b"
    assert config.qdrant_embedding_model_dims == 384
    assert not config.qdrant_on_disk
