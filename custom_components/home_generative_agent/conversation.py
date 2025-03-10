"""Conversation support for Home Generative Agent using langgraph."""
from __future__ import annotations

import logging
import string
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import homeassistant.util.dt as dt_util
from homeassistant.components import assist_pipeline, conversation
from homeassistant.components.conversation import trace
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.exceptions import (
    HomeAssistantError,
    TemplateError,
)
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import intent, llm, template
from homeassistant.util import ulid
from langchain.globals import set_debug, set_verbose
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from .const import (
    CHAT_MODEL_MAX_TOKENS,
    CHAT_MODEL_NUM_CTX,
    CONF_CHAT_MODEL,
    CONF_CHAT_MODEL_LOCATION,
    CONF_CHAT_MODEL_TEMPERATURE,
    CONF_EDGE_CHAT_MODEL,
    CONF_EDGE_CHAT_MODEL_TEMPERATURE,
    CONF_EDGE_CHAT_MODEL_TOP_P,
    CONF_PROMPT,
    DOMAIN,
    EMBEDDING_MODEL_DIMS,
    LANGCHAIN_LOGGING_LEVEL,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_CHAT_MODEL_LOCATION,
    RECOMMENDED_CHAT_MODEL_TEMPERATURE,
    RECOMMENDED_EDGE_CHAT_MODEL,
    RECOMMENDED_EDGE_CHAT_MODEL_TEMPERATURE,
    RECOMMENDED_EDGE_CHAT_MODEL_TOP_P,
    TOOL_CALL_ERROR_SYSTEM_MESSAGE,
)
from .graph import workflow
from .tools import (
    add_automation,
    analyze_patterns,
    get_and_analyze_camera_image,
    get_current_device_state,
    get_entity_history,
    manage_scene,
    perform_location_action,
    reverse_geocode,
    run_diagnostics,
    suggest_contextual_automation,
    upsert_memory,
)
from .utilities import format_tool, generate_embeddings

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddEntitiesCallback

    from . import HGAConfigEntry

LOGGER = logging.getLogger(__name__)

if LANGCHAIN_LOGGING_LEVEL == "verbose":
    set_verbose(True)
    set_debug(False)
elif LANGCHAIN_LOGGING_LEVEL == "debug":
    set_verbose(False)
    set_debug(True)
else:
    set_verbose(False)
    set_debug(False)

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: HGAConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    agent = HGAConversationEntity(config_entry)
    async_add_entities([agent])

class HGAConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Home Generative Assistant conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self.app_config: dict[str, dict[str, str]] = {"configurable": {"thread_id": ""}}
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="LinTek",
            model="HGA",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

        if self.entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

        # Create database for thread-based (short-term) memory.
        # TODO: Use a DB-backed store in production use.
        memory = MemorySaver()

        # Create database for session-based (long-term) memory with semantic search.
        # TODO: Use a DB-backed store in production use.
        store = InMemoryStore(
            index={
                "embed": partial(generate_embeddings, model=entry.embedding_model),
                "dims": EMBEDDING_MODEL_DIMS,
                "fields": ["content"]
            }
        )

        # Complile graph into a LangChain Runnable.
        self.app = workflow.compile(
            store=store,
            checkpointer=memory,
            debug=LANGCHAIN_LOGGING_LEVEL=="debug"
        )

        # Use in-memory caching for langgraph calls to LLMs.
        set_llm_cache(InMemoryCache())

        self.tz = dt_util.get_default_time_zone()

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self.entry.entry_id, self.entity_id
        )
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def async_will_remove_from_hass() -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process the user input."""
        hass = self.hass
        options = self.entry.options
        intent_response = intent.IntentResponse(language=user_input.language)
        llm_api: llm.API | None = None
        tools: list[dict[str, Any]] | None = None
        user_name: str | None = None
        llm_context = llm.LLMContext(
            platform=DOMAIN,
            context=user_input.context,
            user_prompt=user_input.text,
            language=user_input.language,
            assistant=conversation.DOMAIN,
            device_id=user_input.device_id,
        )

        if options.get(CONF_LLM_HASS_API):
            try:
                llm_api = await llm.async_get_api(
                    hass,
                    options[CONF_LLM_HASS_API],
                    llm_context,
                )
            except HomeAssistantError as err:
                LOGGER.error("Error getting LLM API: %s", err)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Error preparing LLM API: {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=user_input.conversation_id
                )
            except Exception as err:  # Catch any unexpected errors during API retrieval
                LOGGER.error("Unexpected error getting LLM API: %s", err, exc_info=True)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    "Sorry, I encountered an unexpected error while setting up the LLM API.",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=user_input.conversation_id
                )
            tools = [
                format_tool(tool, llm_api.custom_serializer) for tool in llm_api.tools
            ]

        # Add langchain tools to the list of HA tools.
        langchain_tools = {
            "add_automation": add_automation,
            "analyze_patterns": analyze_patterns,
            "get_and_analyze_camera_image": get_and_analyze_camera_image,
            "get_current_device_state": get_current_device_state,
            "get_entity_history": get_entity_history,
            "manage_scene": manage_scene,
            "perform_location_action": perform_location_action,
            "reverse_geocode": reverse_geocode,
            "run_diagnostics": run_diagnostics,
            "suggest_contextual_automation": suggest_contextual_automation,
            "upsert_memory": upsert_memory,
        }
        tools.extend(langchain_tools.values())

        # Check if there's a recent conversation thread to continue
        if user_input.conversation_id is None:
            """
            # Look for recent conversations for this user
            try:
                user_name = "robot" if user_name is None else user_name
                store = self.app.executor.store
                recent_convos = await store.asearch(
                    (user_name, "conversations"),
                    query=None,
                    limit=1
                )

                # If found recent conversation, use that thread ID
                if recent_convos:
                    conversation_id = recent_convos[0].key[2]  # Extract thread ID
                    LOGGER.debug("Continuing conversation with ID: %s", conversation_id)
                else:
                    # Create a new thread if no recent ones
                    conversation_id = ulid.ulid_now()
            except Exception as err:
                # Fallback to new conversation ID
                LOGGER.warning("Error finding recent conversation: %s", err)
                conversation_id = ulid.ulid_now()
            """
            # Create a new conversation ID
            conversation_id = ulid.ulid_now()
        else:
            # Use provided conversation ID
            conversation_id = user_input.conversation_id

        if (
            user_input.context
            and user_input.context.user_id
            and (
                user := await hass.auth.async_get_user(user_input.context.user_id)
            )
        ):
            try:
                user_name = user.name
            except Exception as err:  # Catch any unexpected errors during user retrieval
                LOGGER.error("Unexpected error getting user name: %s", err, exc_info=True)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    "Sorry, I encountered an unexpected error while retrieving your user information.",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=conversation_id
                )

        try:
            prompt_parts = [
                template.Template(
                    (
                        llm.BASE_PROMPT
                        + options.get(CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT)
                        + f"\nYou are in the {self.tz} timezone."
                        + TOOL_CALL_ERROR_SYSTEM_MESSAGE if tools else ""
                    ),
                    self.hass,
                ).async_render(
                    {
                        "ha_name": self.hass.config.location_name,
                        "user_name": user_name,
                        "llm_context": llm_context,
                    },
                    parse_result=False,
                )
            ]

        except TemplateError as err:
            LOGGER.error("Error rendering prompt: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I had a problem with my template: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )
        except Exception as err:  # Catch any unexpected errors during prompt rendering
            LOGGER.error("Unexpected error rendering prompt: %s", err, exc_info=True)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "Sorry, I encountered an unexpected error while preparing the prompt.",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        if llm_api:
            prompt_parts.append(llm_api.api_prompt)

        prompt = "\n".join(prompt_parts)

        chat_model_location = self.entry.options.get(
            CONF_CHAT_MODEL_LOCATION,
            RECOMMENDED_CHAT_MODEL_LOCATION
        )
        if chat_model_location == "edge":
            chat_model = self.entry.edge_chat_model
            chat_model_with_config = chat_model.with_config(
                {"configurable":
                    {
                        "model": self.entry.options.get(
                            CONF_EDGE_CHAT_MODEL,
                            RECOMMENDED_EDGE_CHAT_MODEL
                        ),
                        "temperature": self.entry.options.get(
                            CONF_EDGE_CHAT_MODEL_TEMPERATURE,
                            RECOMMENDED_EDGE_CHAT_MODEL_TEMPERATURE
                        ),
                        "top_p": self.entry.options.get(
                            CONF_EDGE_CHAT_MODEL_TOP_P,
                            RECOMMENDED_EDGE_CHAT_MODEL_TOP_P,
                        ),
                        "num_predict": CHAT_MODEL_MAX_TOKENS,
                        "num_ctx": CHAT_MODEL_NUM_CTX,

                    }
                }
            )
        else:
            chat_model = self.entry.chat_model
            chat_model_with_config = chat_model.with_config(
                {"configurable":
                    {
                        "model_name": self.entry.options.get(
                            CONF_CHAT_MODEL,
                            RECOMMENDED_CHAT_MODEL
                        ),
                        "temperature": self.entry.options.get(
                            CONF_CHAT_MODEL_TEMPERATURE,
                            RECOMMENDED_CHAT_MODEL_TEMPERATURE
                        ),
                        "max_tokens": CHAT_MODEL_MAX_TOKENS,
                    }
                }
            )

        chat_model_with_tools = chat_model_with_config.bind_tools(tools)

        # A user name of None indicates an automation is being run.
        user_name = "robot" if user_name is None else user_name
        # Remove special characters since memory namespace labels cannot contain.
        user_name = user_name.translate(str.maketrans("", "", string.punctuation))
        LOGGER.debug("User name: %s", user_name)

        self.app_config = {
            "configurable": {
                "thread_id": conversation_id,
                "user_id": user_name,
                "chat_model": chat_model_with_tools,
                "prompt": prompt,
                "options": options,
                "vlm_model": self.entry.vision_model,
                "summarization_model": self.entry.summarization_model,
                "langchain_tools": langchain_tools,
                "ha_llm_api": llm_api or None,
                "hass": hass,
            },
            "recursion_limit": 10
        }

        # Interact with app.
        try:
            """
            # First try to retrieve conversation history
            previous_messages = []
            try:
                user_name = self.app_config["configurable"]["user_id"]
                store = self.app.executor.store

                # Get previous conversations for this user
                recent_contexts = await store.asearch(
                    (user_name, "conversations"),
                    query=None,
                    limit=7  # Last 7 conversation turns
                )

                # Process in chronological order (oldest first)
                sorted_contexts = sorted(
                    recent_contexts,
                    key=lambda x: x.value.get("timestamp", "")
                )

                # Build message history
                for context in sorted_contexts:
                    if isinstance(context.value, dict) and "messages" in context.value:
                        for msg in context.value["messages"]:
                            if msg.get("role") == "user":
                                previous_messages.append(HumanMessage(content=msg.get("content", "")))
                            elif msg.get("role") == "assistant":
                                previous_messages.append(AIMessage(content=msg.get("content", "")))

                LOGGER.debug("Retrieved %s previous messages for context", len(previous_messages))
            except Exception as err:
                LOGGER.warning("Failed to retrieve conversation history: %s", err)
            """
            # Use only current message
            previous_messages = []

            # Add current message
            previous_messages.append(HumanMessage(content=user_input.text))

            # Invoke with full conversation history
            response = await self.app.ainvoke(
                {"messages": previous_messages},
                config=self.app_config
            )

            """
            # Store the conversation for future context retrieval
            user_name = self.app_config["configurable"]["user_id"]
            store = self.app.executor.store

            # Format messages for storage
            conversation_data = {
                "messages": [
                    {"role": "user", "content": user_input.text},
                    {"role": "assistant", "content": response["messages"][-1].content}
                ],
                "timestamp": dt_util.now().isoformat(),
            }

            # Store in vector database for later retrieval
            try:
                await store.aput(
                    (user_name, "conversations", ulid.ulid_now()),
                    conversation_data,
                    content=f"User: {user_input.text}\nAssistant: {response['messages'][-1].content}"
                )
            except Exception as err:
                LOGGER.warning("Failed to store conversation for context: %s", err)
            """

        except Exception as err:  # Catch all exceptions here
            LOGGER.error("An unexpected error occurred: %s", err, exc_info=True)  # Log the full traceback
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "Sorry, I encountered an unexpected error while processing your request.",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        trace.async_conversation_trace_append(
            trace.ConversationTraceEventType.AGENT_DETAIL,
            {"messages": response["messages"], "tools": tools if tools else None},
        )

        LOGGER.debug("====== End of run ======")

        intent_response = intent.IntentResponse(language=user_input.language)
        try:
            intent_response.async_set_speech(response["messages"][-1].content)
        except IndexError:  # Handle potential IndexError if no messages are returned
            LOGGER.error("No response messages were generated.")
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "Sorry, I was unable to generate a response.",
            )
        except Exception as err:  # Catch any unexpected errors during response processing
            LOGGER.error("Unexpected error processing response: %s", err, exc_info=True)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "Sorry, I encountered an unexpected error while generating the response.",
            )
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features
        await hass.config_entries.async_reload(entry.entry_id)