from enum import Enum

class AgentState(Enum):
    """
    Enum representing the possible states of the web automation agent.
    """
    IDLE = "IDLE"
    EXECUTING = "EXECUTING"
    AWAITING_NAVIGATION = "AWAITING_NAVIGATION"
    CAPTCHA_REQUIRED = "CAPTCHA_REQUIRED"
    UNEXPECTED_MODAL = "UNEXPECTED_MODAL"
    RECOVERING = "RECOVERING"
    FATAL_ERROR = "FATAL_ERROR"
