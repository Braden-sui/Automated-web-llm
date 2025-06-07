import pytest
from unittest.mock import patch, MagicMock
import hashlib
import json

from web_automation.utils.fingerprint import (
    create_consistent_fingerprint,
    _get_platform_from_ua,
    _get_timezone_for_platform,
    _PLATFORM_TIMEZONES,
    get_random_user_agent, # For testing with realistic UAs
    get_random_platform, # For testing with realistic platforms
    get_random_timezone # For testing with realistic timezones
)

# Test cases for _get_platform_from_ua
UA_PLATFORM_CASES = [
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", "Win32"),
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36", "MacIntel"),
    ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36", "Linux x86_64"),
    ("Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15", "Linux x86_64"), # Current fallback for unknown
    ("Mozilla/5.0 (Linux; Android 10; SM-G975F) AppleWebKit/537.36", "Linux x86_64"), # Current fallback for unknown
    ("Unknown User Agent String", "Linux x86_64"), # Test fallback
    ("", "Linux x86_64") # Test empty string
]

@pytest.mark.parametrize("user_agent, expected_platform", UA_PLATFORM_CASES)
def test_get_platform_from_ua(user_agent, expected_platform):
    assert _get_platform_from_ua(user_agent) == expected_platform

# Test cases for _get_timezone_for_platform
PLATFORM_TIMEZONE_CASES = [
    ("Win32", _PLATFORM_TIMEZONES["Win32"][0]),
    ("MacIntel", _PLATFORM_TIMEZONES["MacIntel"][0]),
    ("Linux x86_64", _PLATFORM_TIMEZONES["Linux x86_64"][0]),
    ("UnknownPlatform", "America/New_York"), # Example from COMMON_TIMEZONES as fallback
    ("", "America/New_York") # Example from COMMON_TIMEZONES as fallback
]

# Note: The _get_timezone_for_platform function returns a random choice from the fallback list (COMMON_TIMEZONES)
# if the platform is not found. So we can't assert a specific UTC value unless COMMON_TIMEZONES only has UTC.
# Instead, we check if the returned timezone is *in* COMMON_TIMEZONES for unknown platforms.

@pytest.mark.parametrize("platform, expected_timezone_part", PLATFORM_TIMEZONE_CASES)
def test_get_timezone_for_platform(platform, expected_timezone_part):
    if platform in _PLATFORM_TIMEZONES:
        assert _get_timezone_for_platform(platform) in _PLATFORM_TIMEZONES[platform]
    else: # For UnknownPlatform or empty string, it should pick from COMMON_TIMEZONES
        from web_automation.utils.fingerprint import COMMON_TIMEZONES
        assert _get_timezone_for_platform(platform) in COMMON_TIMEZONES

# Test create_consistent_fingerprint
def test_create_consistent_fingerprint_basic():
    # Since create_consistent_fingerprint now uses random internal choices, 
    # we can only check structure and types, not exact values easily without mocking.
    fingerprint_data = create_consistent_fingerprint()

    assert isinstance(fingerprint_data, dict)
    assert "user_agent" in fingerprint_data
    assert isinstance(fingerprint_data["user_agent"], str)
    assert "platform" in fingerprint_data
    assert fingerprint_data["platform"] in ["Win32", "MacIntel", "Linux x86_64"]
    assert "languages" in fingerprint_data
    assert isinstance(fingerprint_data["languages"], list)
    assert "timezone" in fingerprint_data
    assert isinstance(fingerprint_data["timezone"], str)
    assert "viewport" in fingerprint_data
    assert isinstance(fingerprint_data["viewport"], dict)
    assert "width" in fingerprint_data["viewport"]
    assert "height" in fingerprint_data["viewport"]
    assert "webgl" in fingerprint_data
    assert isinstance(fingerprint_data["webgl"], dict)
    assert "vendor" in fingerprint_data["webgl"]
    assert "renderer" in fingerprint_data["webgl"]

# The current create_consistent_fingerprint is fully random, so consistency tests are not applicable.
# def test_create_consistent_fingerprint_consistency():
#     pass

# The current create_consistent_fingerprint does not support overrides.
# def test_create_consistent_fingerprint_with_overrides():
#    pass

# This test is covered by how UA_PLATFORM_CASES and PLATFORM_TIMEZONE_CASES are handled now.
# The create_consistent_fingerprint() itself doesn't take a UA string directly anymore.
# def test_create_consistent_fingerprint_unknown_ua_defaults():
#    pass

# Test that the PLATFORM_TIMEZONES structure is as expected
# (e.g., each platform has a non-empty list of timezones)
def test_platform_timezones_structure():
    assert isinstance(_PLATFORM_TIMEZONES, dict)
    for platform, timezones in _PLATFORM_TIMEZONES.items():
        assert isinstance(platform, str)
        assert isinstance(timezones, list)
        assert len(timezones) > 0
        for tz in timezones:
            assert isinstance(tz, str)
            assert len(tz) > 0 # Ensure timezone strings are not empty

# Test with randomly generated valid inputs to see if it handles them
@pytest.mark.parametrize("run", range(5)) # Run a few times with random inputs
def test_create_consistent_fingerprint_with_random_valid_inputs(run):
    # create_consistent_fingerprint() now internally calls get_random_user_agent()
    # so we can't pass one in to check against.
    # We can just call it and check the structure, similar to test_create_consistent_fingerprint_basic.
    fingerprint_data = create_consistent_fingerprint()

    assert isinstance(fingerprint_data, dict)
    assert "user_agent" in fingerprint_data
    assert "platform" in fingerprint_data
    assert "languages" in fingerprint_data
    assert "timezone" in fingerprint_data
    assert "viewport" in fingerprint_data
    assert "webgl" in fingerprint_data
    
    # Check that the platform and timezone are consistent with the generated user_agent
    ua = fingerprint_data["user_agent"]
    expected_platform = _get_platform_from_ua(ua)
    assert fingerprint_data["platform"] == expected_platform
    assert fingerprint_data["timezone"] == _get_timezone_for_platform(expected_platform)
