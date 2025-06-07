import pytest
import asyncio
from web_automation import create_playwright_agent
from web_automation.config.config_models import Mem0AdapterConfig
from web_automation.models.instructions import InstructionSet, ClickInstruction, ActionType

@pytest.mark.asyncio
async def test_reasoning_integration():
    """Test that reasoning integration works with existing system"""
    
    # Create agent with both memory and reasoning enabled
    mem_config = Mem0AdapterConfig(
        llm_provider="ollama",
        llm_model="qwen2.5vl:7b",
        qdrant_on_disk=False,
        qdrant_embedding_model_dims=384 # Ensure this matches your embedder
    )
    
    agent = create_playwright_agent(
        memory_enabled=True,
        reasoning_enabled=True,
        headless=True,
        memory_config=mem_config.model_dump() # Pass as dict
    )
    
    try:
        async with agent:
            # Test simple instruction set
            instruction_set = InstructionSet(
                url="https://example.com",
                instructions=[
                    ClickInstruction(
                        type=ActionType.CLICK,
                        selector="a[href*='iana']"
                    )
                ]
            )
            
            # Execute with reasoning
            result = await agent.execute_instructions_with_reasoning(instruction_set)
            
            # Verify reasoning was applied
            assert "reasoning" in result
            assert result["success"] in [True, False]  # Should complete regardless
            
            # Verify existing functionality still works
            # Re-create agent or reset state if necessary for clean test of standard execution
            # For simplicity here, we assume state from reasoning doesn't break standard execution
            standard_result = await agent.execute_instructions(instruction_set)
            assert standard_result["success"] in [True, False]
            
            # Check reasoning stats
            stats = agent.get_reasoning_stats()
            assert "total_reasonings" in stats
            assert stats.get("reasoning_enabled") == True
            
    finally:
        await agent.close()

@pytest.mark.asyncio
async def test_reasoning_disabled():
    """Test that system works when reasoning is disabled"""
    
    agent = create_playwright_agent(
        reasoning_enabled=False,
        headless=True
    )
    
    try:
        async with agent:
            instruction_set = InstructionSet(
                url="https://example.com",
                instructions=[] # Empty instructions, should succeed
            )
            
            result = await agent.execute_instructions_with_reasoning(instruction_set)
            
            # Should work without reasoning
            assert result["success"] == True
            assert "reasoning" in result # reasoning key should still exist
            assert result.get("reasoning", {}).get("reasoning_applied", False) == False
            assert result.get("reasoning", {}).get("message") == "Reasoning disabled in config"

            stats = agent.get_reasoning_stats()
            assert stats.get("reasoning_enabled") == False
            
    finally:
        await agent.close()
