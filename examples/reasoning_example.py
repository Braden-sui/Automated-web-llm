import asyncio
import logging
from web_automation import create_playwright_agent
from web_automation.config.config_models import Mem0AdapterConfig
from web_automation.models.instructions import InstructionSet, ClickInstruction, TypeInstruction, ActionType

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Example of using CoT reasoning with web automation"""
    
    # Configure memory for pattern learning
    # Ensure your Ollama instance is running and has the 'qwen2.5vl:7b' model pulled.
    # `ollama pull qwen2.5vl:7b`
    mem_config = Mem0AdapterConfig(
        llm_provider="ollama",
        llm_model="qwen2.5vl:7b", # Make sure this model is available in your Ollama instance
        qdrant_on_disk=False, # Use in-memory Qdrant for this example
        qdrant_embedding_model_dims=384 # Matching all-MiniLM-L6-v2
    )
    
    # Create agent with reasoning enabled
    agent = create_playwright_agent(
        memory_enabled=True,
        reasoning_enabled=True,
        headless=False,  # Set to False to see reasoning in action (if applicable)
        memory_config=mem_config.model_dump() # Pass config as dict
    )
    
    try:
        async with agent:
            # Complex instruction set that benefits from reasoning
            instruction_set = InstructionSet(
                url="https://example.com",
                instructions=[
                    ClickInstruction(
                        type=ActionType.CLICK,
                        selector="a[href*='iana']" # This link exists on example.com
                    ),
                    # Example of a potentially problematic instruction that reasoning might help with
                    # On IANA page, there isn't a direct search input like this.
                    # Reasoning might try alternatives or log the issue.
                    TypeInstruction(
                        type=ActionType.TYPE,
                        selector="input[type='search']", # This selector might not exist on iana.org
                        text="web automation"
                    )
                ]
            )
            
            logger.info("Executing with CoT reasoning...")
            result = await agent.execute_instructions_with_reasoning(instruction_set)
            
            logger.info(f"Overall Success: {result['success']}")
            logger.info(f"Actions Completed: {result['actions_completed']}")
            
            if "reasoning" in result and result["reasoning"]:
                logger.info(f"Reasoning Applied: {result['reasoning'].get('reasoning_applied')}")
                if result['reasoning'].get('reasoning_applied'):
                    logger.info(f"Reasoning Execution Time: {result['reasoning'].get('execution_time', 0):.2f}s")
                    # logger.info(f"Reasoning Result: {result['reasoning'].get('result')}") # This can be very verbose
                else:
                    logger.warning(f"Reasoning Problem: {result['reasoning'].get('error') or result['reasoning'].get('message')}")
            
            if "reasoning_stats" in result:
                logger.info(f"Reasoning Stats: {result['reasoning_stats']}")

    except Exception as e:
        logger.error(f"An error occurred in the example: {e}", exc_info=True)
    finally:
        if 'agent' in locals() and agent:
            await agent.close()
            logger.info("Agent closed.")

if __name__ == "__main__":
    asyncio.run(main())
