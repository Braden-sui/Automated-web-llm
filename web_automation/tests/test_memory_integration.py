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

@pytest.mark.asyncio
async def test_memory_enhanced_agent(test_mem0_config):
    """Test memory-enhanced agent basic functionality with isolated in-memory Qdrant and Ollama LLM."""
    # Create dependencies with memory manager
    dependencies = BrowserAgentDependencies(
        memory_manager=Mem0BrowserAdapter(mem0_config=test_mem0_config),
        config={
            'browser_type': 'chromium',
            'headless': True,
            'identity_id': f'test_agent_{uuid.uuid4().hex[:8]}'
        }
    )
    
    # Create agent with dependencies
    agent = create_playwright_agent(dependencies=dependencies)

    # Test memory operations
    if agent.memory_manager:
        # Test adding and retrieving a memory
        test_text = "Test memory entry"
        test_metadata = {"test": "value"}
        
        # Add memory
        add_result = agent.memory_manager.memory.add(
            test_text,
            user_id=agent.identity_id,
            metadata=test_metadata
        )
        
        # Verify add operation
        assert add_result is not None
        assert "results" in add_result
        assert len(add_result["results"]) > 0
        
        # Search for the memory
        search_results = agent.memory_manager.memory.search(
            query=test_text,
            user_id=agent.identity_id,
            limit=1
        )
        
        # Verify search results
        assert search_results is not None
        assert "results" in search_results
        assert len(search_results["results"]) > 0
        assert search_results["results"][0]["memory"] == test_text
    
    async with agent:
        # Test page navigation
        if not hasattr(agent, '_page'):
            pytest.fail("Agent page was not initialized.")
        
        # Navigate to test page
        await agent._page.goto("https://example.com")
        await asyncio.sleep(0.5)  # Allow page to settle
        
        # Test smart selector functionality
        success = await agent.smart_selector_click(
            target_description="main navigation link",
            fallback_selector="a[href='https://www.iana.org/domains/example']"
        )
        
        # Verify navigation was successful
        assert "Example Domain" in await agent._page.title()

        print(f"\n=== AFTER SMART CLICK ===")
        print(f"Click success: {success}")
        
        # Verify memory operations
        if agent.memory_manager:
            # Search for patterns related to our click
            new_patterns = agent.memory_manager.search_automation_patterns(
                "main navigation", 
                agent.identity_id, 
                limit=5
            )
            
            print(f"Patterns after click: {len(new_patterns)}")
            for i, pattern in enumerate(new_patterns):
                print(f"  Pattern {i}: {{'memory': pattern.get('memory', 'N/A')[:50]+'...', 'metadata': pattern.get('metadata')}}")
            
            # Verify pattern was stored
            assert len(new_patterns) > 0, "At least one automation pattern should be stored after smart_selector_click."
            
            # Check if the description or part of it is in the stored memory text
            assert any("main navigation link" in pattern.get('memory', '') for pattern in new_patterns), \
                "Stored pattern memory text should relate to 'main navigation link'."
                
            # Check if the selector used is in the metadata
            assert any(
                pattern.get('metadata', {}).get('original_fallback_selector') == 
                "a[href='https://www.iana.org/domains/example']" 
                for pattern in new_patterns
            ), "Stored pattern metadata should contain the fallback selector used."

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
    # Create dependencies without memory manager
    dependencies = BrowserAgentDependencies(
        memory_manager=None,
        config={
            'browser_type': 'chromium',
            'headless': True,
            'identity_id': f'test_agent_{uuid.uuid4().hex[:8]}'
        }
    )
    
    # Create agent with dependencies
    agent = create_playwright_agent(dependencies=dependencies)
    
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
        llm_model=os.getenv("MEMORY_LLM_MODEL", "qwen2.5vl:7b"),
        llm_temperature=0.7,
        api_key=None
    )
    
    # Verify configuration
    assert config.llm_provider == "ollama"
    assert config.llm_model == os.getenv("MEMORY_LLM_MODEL", "qwen2.5vl:7b")
    assert config.qdrant_embedding_model_dims == 384
    assert not config.qdrant_on_disk
