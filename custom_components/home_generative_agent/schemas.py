"""Pydantic schemas for Home Generative Agent."""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

class ModelConfig(BaseModel):
    """Base configuration for AI models."""
    temperature: float = Field(default=0.7, ge=0, le=1)
    top_p: float = Field(default=1.0, ge=0, le=1)

class ChatModelConfig(ModelConfig):
    """Configuration for OpenAI chat model."""
    model_name: str
    timeout: int = Field(default=10, ge=1)
    api_key: str

class EdgeModelConfig(ModelConfig):
    """Configuration for edge chat model."""
    model: str
    format: Optional[str] = None
    num_predict: int = Field(default=1024, ge=1)
    num_ctx: int = Field(default=4096, ge=1)

class ToolCallResponse(BaseModel):
    """Response from a tool call."""
    success: bool
    result: Any
    error: Optional[str] = None

class ConversationState(BaseModel):
    """State of a conversation."""
    thread_id: str
    user_id: str
    messages: List[Union[HumanMessage, AIMessage]]

class HGAConfig(BaseModel):
    """Configuration for Home Generative Agent."""
    chat_model: ChatOpenAI
    edge_chat_model: ChatOllama
    vision_model: ChatOllama
    summarization_model: ChatOllama
    embedding_model: Any  # Type varies based on implementation

    class Config:
        arbitrary_types_allowed = True

class MemoryEntry(BaseModel):
    """Structure for memory entries."""
    content: str
    embedding: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str

    @validator('embedding')
    def validate_embedding_dimensions(cls, v):
        """Validate embedding dimensions."""
        if len(v) > 0 and len(v) != 384:  # Standard dimension for many embedding models
            raise ValueError('Embedding must have 384 dimensions')
        return v

class AppConfig(BaseModel):
    """Configuration for the application."""
    thread_id: str
    user_id: str
    chat_model: Any  # Type varies based on implementation
    prompt: str
    options: Dict[str, Any]
    vlm_model: Any
    summarization_model: Any
    langchain_tools: Dict[str, Any]
    ha_llm_api: Optional[Any] = None
    hass: Any

    class Config:
        arbitrary_types_allowed = True
