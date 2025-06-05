"""
Custom exceptions for the CAPTCHA solving system.
"""

class CaptchaError(Exception):
    """Base class for CAPTCHA related errors."""
    pass

class CaptchaNotFound(CaptchaError):
    """Raised when no solvable CAPTCHA is found on the page."""
    pass

class CaptchaSolveFailed(CaptchaError):
    """Raised when a CAPTCHA solving attempt fails."""
    pass

class VisionModelUnavailable(CaptchaError):
    """Raised when the vision model is not available or fails to load."""
    pass
