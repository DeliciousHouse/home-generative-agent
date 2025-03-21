"""Constants for Home Generative Agent."""

DOMAIN = "home_generative_agent"

### Configuration parameters that can be overridden in the integration's config UI. ###
# Name of the set of recommended options.
CONF_RECOMMENDED = "recommended"
# Name of system prompt.
CONF_PROMPT = "prompt"
# Run chat model in cloud or at edge.
CONF_CHAT_MODEL_LOCATION = "chat_model_location"
RECOMMENDED_CHAT_MODEL_LOCATION = "cloud"
### OpenAI chat model parameters.
# See https://platform.openai.com/docs/api-reference/chat/create.
CONF_CHAT_MODEL = "chat_model"
RECOMMENDED_CHAT_MODEL = "gpt-4o"
CONF_CHAT_MODEL_TEMPERATURE = "chat_model_temperature"
RECOMMENDED_CHAT_MODEL_TEMPERATURE = 1.0
### Ollama edge chat model parameters. ###
# See https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter
CONF_EDGE_CHAT_MODEL = "edge_chat_model"
RECOMMENDED_EDGE_CHAT_MODEL = "qwen2.5:32b"
CONF_EDGE_CHAT_MODEL_TEMPERATURE = "edge_chat_model_temperature"
RECOMMENDED_EDGE_CHAT_MODEL_TEMPERATURE = 0.7
CONF_EDGE_CHAT_MODEL_TOP_P = "edge_chat_model_top_p"
RECOMMENDED_EDGE_CHAT_MODEL_TOP_P = 0.8
### Ollama vision language model (VLM) parameters. ###
# The VLM is used for vision and summarization tasks.
# See https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter
CONF_VLM = "vlm_model"
RECOMMENDED_VLM = "llama3.2-vision"
CONF_VISION_MODEL_TEMPERATURE = "vision_model_temperature"
RECOMMENDED_VISION_MODEL_TEMPERATURE = 0.2
CONF_VISION_MODEL_TOP_P = "vision_model_top_p"
RECOMMENDED_VISION_MODEL_TOP_P = 0.5
CONF_SUMMARIZATION_MODEL = "summarization_model"
RECOMMENDED_SUMMARIZATION_MODEL = "qwen2.5:3b"
CONF_SUMMARIZATION_MODEL_TEMPERATURE = "summarization_model_temperature"
RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE = 0.6
CONF_SUMMARIZATION_MODEL_TOP_P = "summarization_model_top_p"
RECOMMENDED_SUMMARIZATION_MODEL_TOP_P = 0.9
### Ollama embedding model parameters. ###
# The embedding model is used for semantic search in long-term memory.
CONF_EMBEDDING_MODEL = "embedding_model"
RECOMMENDED_EMBEDDING_MODEL = "mxbai-embed-large"

### langchain logging level ###
# Options are "disable", "verbose" or "debug".
# See https://python.langchain.com/docs/how_to/debugging/
LANGCHAIN_LOGGING_LEVEL = "disable"

### Chat model parameters. ###
# Sets the size of the context window used to generate the next token.
CHAT_MODEL_NUM_CTX = 32768
# Next parameters manage chat model context length.
# CONTEXT_MANAGE_USE_TOKENS = True manages chat model context size via token
# counting, if False management is done via message counting.
CONTEXT_MANAGE_USE_TOKENS = True
# CONTEXT_MAX_MESSAGES should be set larger than CONTEXT_SUMMARIZE_THRESHOLD.
# CONTEXT_MAX_MESSAGES is messages to keep in context before deletion.
# Keep number of tokens below 30k otherwise rate limits may be triggered by OpenAI
# (Tokens Per Minute limit for Tier 1 pricing is 30k tokens/minute), or Ollama model
# context length limits will be reached.
# Assume worse case message is 300 tokens -> 100 messages in context will be 30k tokens.
# So, with 100 messages in context calls to OpenAI can be as frequent as 1 per minute.
CONTEXT_MAX_MESSAGES = 200
# If number of messages in context > CONTEXT_SUMMARIZE_THRESHOLD, generate a summary.
CONTEXT_SUMMARIZE_THRESHOLD = 40
# CONTEXT_MAX_TOKENS sets the limit on how large the context can grow. This should be
# no larger than CHAT_MODEL_NUM_CTX. Only used if context size is managed by tokens.
#
# Reduce by 2048 tokens because the token counter ignores tool schemas.
# Reduce by another 2048 because the token counter under counts by about 2k tokens.
# These offsets are for the qwen models and were empirically determined.
# TODO: fix the token counter to get an accurate count.
#
CONTEXT_MAX_TOKENS = (CHAT_MODEL_NUM_CTX - 2048 - 2048) # 28672
# Next two parameters are for chat model tool error handling.
TOOL_CALL_ERROR_SYSTEM_MESSAGE = """

Always call tools again with your mistakes corrected. Do not repeat mistakes.
"""
TOOL_CALL_ERROR_TEMPLATE = """
Error: {error}

Call the tool again with your mistake corrected.
"""

### Ollama edge chat model parameters. ###
# Edge chat model server URL.
EDGE_CHAT_MODEL_URL = "192.168.86.13:11434"
# Maximum number of tokens to predict when generating text.
EDGE_CHAT_MODEL_NUM_PREDICT = 4096

### Ollama VLM parameters. ###
# Ollama VLM server URL.
VLM_URL = "192.168.86.13:11434"
# Ollama VLM maximum number of output tokens to generate.
VLM_NUM_PREDICT = 4096
# Sets the size of the context window used to generate the next token.
VLM_NUM_CTX = 16384
# Ollama VLM model prompts for vision tasks.
VISION_MODEL_SYSTEM_PROMPT = """
You are a bot that generates scene analysis from camera images.
"""
VISION_MODEL_USER_PROMPT = "Task: Describe this image:"
VISION_MODEL_USER_KW_PROMPT =  "Task: Does this image contain"
VISION_MODEL_IMAGE_WIDTH = 1920
VISION_MODEL_IMAGE_HEIGHT = 1080

### Ollama summarization model parameters. ###
# Model server URL.
SUMMARIZATION_MODEL_URL = "192.168.86.13:11434"
# Maximum number of tokens to predict when generating text.
SUMMARIZATION_MODEL_PREDICT = 4096
# Sets the size of the context window used to generate the next token.
SUMMARIZATION_MODEL_CTX = 32768
# Model prompts for summary tasks.
SUMMARY_SYSTEM_PROMPT = "You are a bot that summarizes messages from a smart home AI."
SUMMARY_INITIAL_PROMPT = "Create a summary of the smart home messages above:"
SUMMARY_PROMPT_TEMPLATE = """
This is summary of the smart home messages so far: {summary}

Update the summary by taking into account the new smart home messages above:
"""

### Ollama embedding model parameters. ###
EMBEDDING_MODEL_URL = "192.168.86.13:11434"
EMBEDDING_MODEL_DIMS = 512
EMBEDDING_MODEL_PROMPT_TEMPLATE = """
Represent this sentence for searching relevant passages: {query}
"""

EVENT_AUTOMATION_REGISTERED = "automation_registered_via_home_generative_agent"

BLUEPRINT_NAME = "goruck/hga_scene_analysis.yaml"
