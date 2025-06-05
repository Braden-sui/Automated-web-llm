"""
Tests for profile reuse functionality in the browser agent.
"""
import os
import json
import pytest
import asyncio
import shutil
from pathlib import Path
from web_automation import create_playwright_agent
from web_automation.config.config_models import Mem0AdapterConfig

# Test data
TEST_PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"
TEST_RUN_ID = f"test_run_{os.getpid()}"

def get_profile_path(profile_name):
    """Helper to get the full path to a profile file."""
    return TEST_PROFILES_DIR / f"{profile_name}.json"

def backup_existing_profiles():
    """Backup existing profiles to a temporary directory."""
    backup_dir = TEST_PROFILES_DIR.parent / f"profiles_backup_{TEST_RUN_ID}"
    if not backup_dir.exists() and TEST_PROFILES_DIR.exists():
        shutil.copytree(TEST_PROFILES_DIR, backup_dir)
    return backup_dir

def restore_profiles(backup_dir):
    """Restore profiles from backup and remove the backup directory."""
    # Remove any test-created profiles
    for profile_file in TEST_PROFILES_DIR.glob("test_*.json"):
        profile_file.unlink()
    
    # Restore from backup if it exists
    if backup_dir.exists():
        for item in backup_dir.glob("*"):
            if item.is_file():
                shutil.copy2(item, TEST_PROFILES_DIR / item.name)
        shutil.rmtree(backup_dir)

@pytest.fixture(scope="function")
async def test_setup():
    """Setup and teardown for profile tests."""
    # Backup existing profiles
    backup_dir = backup_existing_profiles()
    
    # Create a test memory config with in-memory Qdrant
    test_mem0_config = Mem0AdapterConfig(
        qdrant_path=None,
        qdrant_on_disk=False,
        qdrant_collection_name=f"test_profile_reuse_{TEST_RUN_ID}",
        qdrant_embedding_model_dims=384,
        mem0_version="v1.1",
        llm_provider="ollama",
        llm_model="qwen2.5vl:7b",
        llm_temperature=0.1,
        api_key=None
    )
    
    yield test_mem0_config
    
    # Cleanup after test
    await asyncio.sleep(0.1)  # Give time for any pending operations
    restore_profiles(backup_dir)

@pytest.mark.asyncio
async def test_profile_reuse(test_setup):
    """Test that agents with the same identity pattern reuse the same profile."""
    test_mem0_config = test_setup
    
    # Test 1: Multiple agents with test identity should use the same profile
    test_identities = ["test_1", "test_2", "test_3"]
    profile_files = set()
    
    for identity in test_identities:
        agent = create_playwright_agent(
            identity_id=identity,
            memory_enabled=True,
            headless=True,
            memory_config=test_mem0_config
        )
        
        # The profile should be created when the agent is initialized
        profile_path = get_profile_path("test_profile")
        assert profile_path.exists(), f"Profile file {profile_path} was not created"
        profile_files.add(str(profile_path))  # Convert to string for set comparison
        
        # Clean up agent
        await agent.close()
    
    # All test agents should use the same profile file
    assert len(profile_files) == 1, f"All test agents should use the same profile file, found: {profile_files}"
    
    # Test 2: Different identity patterns should use different profiles
    identity_patterns = [
        ("default", "default_profile"),
        ("production_1", "production_profile"),
        ("custom_identity", "custom_identity")  # Custom identity uses its own name
    ]
    
    for identity, expected_profile in identity_patterns:
        agent = create_playwright_agent(
            identity_id=identity,
            memory_enabled=True,
            headless=True,
            memory_config=test_mem0_config
        )
        
        # The profile should be created when the agent is initialized
        profile_path = get_profile_path(expected_profile)
        assert profile_path.exists(), f"Profile file {profile_path} was not created"
        
        # Clean up agent
        await agent.close()
    
    # Verify the expected profile files exist
    expected_profiles = {"test_profile", "default_profile", "production_profile", "custom_identity"}
    existing_profiles = {f.stem for f in TEST_PROFILES_DIR.glob("*.json")}
    
    missing = expected_profiles - existing_profiles
    assert not missing, f"Expected profile files not found: {missing}"

if __name__ == "__main__":
    asyncio.run(test_profile_reuse(None))
