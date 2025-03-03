"""Pydantic validators for Home Generative Agent."""
try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError:
    raise ImportError(
        "The pydantic package is required for validation. "
        "Please install it with `pip install pydantic`"
    )
from typing import Any, Dict, Optional

class ConfigValidator(BaseModel):
    """Validator for configuration values."""
    thread_id: str
    user_id: str
    chat_model: Any
    prompt: str
    options: Dict[str, Any]
    vlm_model: Any
    summarization_model: Any
    langchain_tools: Dict[str, Any]
    ha_llm_api: Optional[Any] = None
    hass: Any

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure.

    Args:
        config: Configuration dictionary to validate

    Returns:
        bool: True if valid, raises ValidationError if invalid

    Example:
        >>> config = {
        ...     "configurable": {
        ...         "thread_id": "123",
        ...         "user_id": "user",
        ...         "chat_model": chat_model_instance,
        ...         "prompt": "Hello",
        ...         "options": {},
        ...         "vlm_model": vlm_model_instance,
        ...         "summarization_model": sum_model_instance,
        ...         "langchain_tools": {},
        ...         "hass": hass_instance
        ...     }
        ... }
        >>> validate_config(config)
        True
    """
    if "configurable" not in config:
        raise ValueError("Missing 'configurable' key in config")

    try:
        ConfigValidator(**config["configurable"])
        return True
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {str(e)}")
