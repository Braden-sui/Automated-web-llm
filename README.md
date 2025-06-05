# LLM Web Automation Project

## Overview

Advanced web automation system using Playwright with anti-detection capabilities, CAPTCHA solving, and computer vision integration. Designed for robust web scraping and automation tasks.

## Features

### Core Capabilities

- **Browser Automation**:
  - Headless and non-headless modes
  - Multi-browser support (Chromium, Firefox, WebKit)
  - Human-like interaction patterns

- **Stealth Mode**:
  - Evade bot detection systems
  - Randomized user-agent rotation
  - Behavioral fingerprint masking

- **CAPTCHA Handling**:
  - Image-based CAPTCHA solving (placeholder implementation)
  - reCAPTCHA v2/v3 support (not fully implemented)
  - Audio CAPTCHA fallback

- **Data Processing**:
  - OCR text extraction
  - Image analysis with OpenCV
  - PDF content parsing

## Advanced Setup

### Environment Configuration

```bash
# Recommended Python version
python --version  # Requires 3.8+

# Install system dependencies (Windows)
choco install -y vcredist2015

# Create virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\activate

# Install dependencies
pip install -r web_automation/requirements.txt
```

### Configuration Files

1. Environment variables are loaded from a `.env` file in the project root. This allows for easy configuration of sensitive information and various settings. For example, to configure the CAPTCHA vision model:

```bash
CAPTCHA_VISION_MODEL_NAME=qwen2.5-vl:7b
```

1. `config/settings.py` - Main configuration in Python format:

```python
# Example settings from settings.py
BROWSER_HEADLESS = True
CAPTCHA_SOLVER = "two_captcha"
```

## Usage

```python
from core.browser_agent import WebBrowserAgent

engine = WebBrowserAgent()
engine.run_workflow("scraping_profile")
```

## Workflow Examples

### Basic Scraping

```python
from core.browser_agent import WebBrowserAgent

engine = WebBrowserAgent()
result = engine.scrape_url(
    url="https://example.com/data",
    extract_rules={"title": "//h1", "content": "//div[@class='article']"}
)
print(result)
```

### Form Automation

```python
from core.browser_agent import WebBrowserAgent

engine = WebBrowserAgent()
engine.run_workflow(
    "form_submission",
    form_data={
        "username": "test_user",
        "password": "secure_password",
        "captcha_selector": "#captcha_image"
    }
)
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| CAPTCHA not solved | Verify CAPTCHA solver configuration in settings.py and note placeholder status |
| Detection triggers | Enable stealth mode in WebBrowserAgent initialization |
| Browser crashes | Check memory settings in code or system resources |

## Folder Structure

- `web_automation/core`: Main automation logic
- `web_automation/captcha`: CAPTCHA solving modules
- `web_automation/config`: Settings and profiles
- `web_automation/utils`: Helper functions

## Contribution Guidelines

1. Fork the repository
2. Create feature branch (`git checkout -b feature/xyz`)
3. Commit changes (`git commit -am 'Add feature xyz'`)
4. Push to branch (`git push origin feature/xyz`)
5. Open pull request
