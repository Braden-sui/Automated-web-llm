import asyncio
import pytest
import uuid
import logging
import os
from web_automation.config.config_models import Mem0AdapterConfig
from web_automation.core.dependencies import BrowserAgentFactory
from web_automation.memory.memory_manager import Mem0BrowserAdapter
from unittest.mock import AsyncMock

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
    
    # Create agent using the factory
    agent = BrowserAgentFactory.create_agent(
        memory_config={
            'enabled': True,
            **test_mem0_config.model_dump()
        },
        browser_type='chromium',
        headless=True,
        identity_id=f'test_agent_{uuid.uuid4().hex[:8]}'
    )

    # Mock the browser for testing
    agent._page = AsyncMock()
    agent._page.goto = AsyncMock()
    agent._page.title = AsyncMock(return_value="Example Domain")
    agent._page.url = "https://example.com"
    
    # Test that agent was created properly
    assert agent is not None
    assert hasattr(agent, 'memory_manager')
    
    # If this test needs to actually test memory functionality,
    # we'd need to set up proper mocking or use a real memory system
    # For now, just verify the agent can be created
    
    print("Memory-enhanced agent created successfully")

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
    
    # Mock the browser
    agent._page = AsyncMock()
    agent._page.goto = AsyncMock()
    agent._page.title = AsyncMock(return_value="Example Domain")
    
    # Test basic functionality
    assert agent is not None
    print("Standard agent created successfully")

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
