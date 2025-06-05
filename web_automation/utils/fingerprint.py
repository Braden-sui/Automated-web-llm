import random
import json

# A list of common, realistic user agents
# Sourced from various places, try to keep it updated with modern browser versions
COMMON_USER_AGENTS = [
    # Chrome (Windows)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    # Chrome (macOS)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Firefox (Windows)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    # Firefox (macOS)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:125.0) Gecko/20100101 Firefox/125.0",
    # Safari (macOS)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    # Edge (Windows)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
]

COMMON_ACCEPT_LANGUAGES = [
    "en-US,en;q=0.9",
    "en-GB,en;q=0.8",
    "en-CA,en;q=0.7",
    "fr-FR,fr;q=0.9,en;q=0.8",
    "de-DE,de;q=0.9,en;q=0.8",
    "es-ES,es;q=0.9,en;q=0.8",
    "it-IT,it;q=0.9,en;q=0.8",
]

COMMON_PLATFORMS = [
    "Win32",
    "MacIntel", 
    "Linux x86_64",
]

COMMON_VIEWPORTS = [
    {"width": 1920, "height": 1080},
    {"width": 1366, "height": 768}, 
    {"width": 1440, "height": 900},
    {"width": 1600, "height": 900},
    {"width": 1280, "height": 800},
    {"width": 1280, "height": 1024},
    {"width": 1680, "height": 1050},
    {"width": 2560, "height": 1440},  # Added more modern resolutions
    {"width": 1536, "height": 864},
]

COMMON_TIMEZONES = [
    "America/New_York",
    "America/Los_Angeles", 
    "America/Chicago",
    "America/Denver",
    "Europe/London",
    "Europe/Paris",
    "Europe/Berlin",
    "Europe/Madrid",
    "Asia/Tokyo",
    "Asia/Shanghai",
    "Australia/Sydney",
    "America/Toronto",
]

# WebGL renderer/vendor combinations that match common hardware
WEBGL_VENDORS = [
    {"vendor": "Google Inc. (Intel)", "renderer": "ANGLE (Intel, Intel(R) UHD Graphics Direct3D11 vs_5_0 ps_5_0, D3D11)"},
    {"vendor": "Google Inc. (NVIDIA)", "renderer": "ANGLE (NVIDIA, NVIDIA GeForce GTX 1060 Direct3D11 vs_5_0 ps_5_0, D3D11)"},
    {"vendor": "Google Inc. (AMD)", "renderer": "ANGLE (AMD, AMD Radeon RX 580 Series Direct3D11 vs_5_0 ps_5_0, D3D11)"},
    {"vendor": "Google Inc. (Intel)", "renderer": "ANGLE (Intel, Intel(R) Iris(R) Xe Graphics Direct3D11 vs_5_0 ps_5_0, D3D11)"},
]

def get_random_user_agent() -> str:
    """Returns a random user agent string from a predefined list."""
    return random.choice(COMMON_USER_AGENTS)

def get_random_accept_language() -> str:
    """Returns a random accept language string."""
    return random.choice(COMMON_ACCEPT_LANGUAGES)

def get_random_platform() -> str:
    """Returns a random platform string."""
    return random.choice(COMMON_PLATFORMS)

def get_random_viewport() -> dict:
    """Returns a random viewport dictionary with slight variations."""
    base_vp = random.choice(COMMON_VIEWPORTS).copy()
    # Add slight variations to make each session unique
    base_vp["width"] += random.randint(-50, 50)
    base_vp["height"] += random.randint(-30, 30)
    # Ensure minimum reasonable size
    base_vp["width"] = max(800, base_vp["width"])
    base_vp["height"] = max(600, base_vp["height"])
    return base_vp

def get_random_timezone() -> str:
    """Returns a random timezone ID."""
    return random.choice(COMMON_TIMEZONES)

def get_random_webgl_info() -> dict:
    """Returns random but realistic WebGL vendor/renderer info."""
    return random.choice(WEBGL_VENDORS)

def generate_canvas_noise() -> float:
    """Generate subtle canvas noise value."""
    return random.uniform(0.0001, 0.001)

def create_consistent_fingerprint() -> dict:
    """Create a consistent set of fingerprint values that work together."""
    user_agent = get_random_user_agent()
    viewport = get_random_viewport()
    
    # Extract OS info from user agent for consistent platform
    if "Windows" in user_agent:
        platform = "Win32"
        timezone = random.choice(["America/New_York", "America/Los_Angeles", "America/Chicago"])
    elif "Macintosh" in user_agent:
        platform = "MacIntel"
        timezone = random.choice(["America/New_York", "America/Los_Angeles", "America/Denver"])
    else:
        platform = "Linux x86_64"
        timezone = random.choice(["Europe/London", "Europe/Berlin", "America/New_York"])
    
    return {
        "user_agent": user_agent,
        "platform": platform,
        "languages": get_random_accept_language().split(','),
        "timezone": timezone,
        "viewport": viewport,
        "webgl": get_random_webgl_info()
    }

if __name__ == '__main__':
    print("=== Random Fingerprint Components ===")
    print("Random User Agent:", get_random_user_agent())
    print("Random Accept Language:", get_random_accept_language())
    print("Random Platform:", get_random_platform())
    print("Random Viewport:", get_random_viewport())
    print("Random Timezone:", get_random_timezone())
    print("Random WebGL:", get_random_webgl_info())
    
    print("\n=== Consistent Fingerprint Set ===")
    fingerprint = create_consistent_fingerprint()
    for key, value in fingerprint.items():
        print(f"{key}: {value}")