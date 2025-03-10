"""Langgraph graphs for Home Generative Agent."""
from __future__ import annotations  # noqa: I001

import copy
import json
import logging
from functools import partial
from typing import Any, Literal

import voluptuous as vol
import homeassistant.util.dt as dt_util
from homeassistant.exceptions import (
    HomeAssistantError,
)
from homeassistant.helpers import llm
from langchain_core.messages import (
    AnyMessage,
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.runnables import RunnableConfig  # noqa: TCH002
from langgraph.store.base import BaseStore  # noqa: TCH002
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import ValidationError
from langgraph.store import SQLiteStore
from .const import (
    CONF_SUMMARIZATION_MODEL_TEMPERATURE,
    CONF_SUMMARIZATION_MODEL_TOP_P,
    CONTEXT_MANAGE_USE_TOKENS,
    CONTEXT_MAX_MESSAGES,
    CONTEXT_MAX_TOKENS,
    EMBEDDING_MODEL_PROMPT_TEMPLATE,
    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
    SUMMARY_INITIAL_PROMPT,
    SUMMARY_PROMPT_TEMPLATE,
    SUMMARY_SYSTEM_PROMPT,
    TOOL_CALL_ERROR_TEMPLATE,
    CONF_SUMMARIZATION_MODEL,
    RECOMMENDED_SUMMARIZATION_MODEL,
    SUMMARIZATION_MODEL_CTX,
    SUMMARIZATION_MODEL_PREDICT,
)

LOGGER = logging.getLogger(__name__)

async def debug_memory(store, user_id="robot"):
    """Debug helper to view memory contents."""
    try:
        # Debug available methods
        LOGGER.debug("Available store methods: %s", dir(store))

        # Try different methods to get data
        try:
            LOGGER.debug("Trying store.all() method")
            all_items = store.all()
            LOGGER.debug("All items: %d entries found", len(all_items))
        except Exception as e:
            LOGGER.debug("store.all() failed: %s", repr(e))

        try:
            # Try to filter just conversations for this user
            all_items = [item for item in store.all()
                        if isinstance(item.key, tuple) and
                        len(item.key) >= 2 and
                        item.key[0] == user_id and
                        item.key[1] == "conversations"]

            # Log what we found
            LOGGER.debug("MEMORY DEBUG: Found %d conversation entries for user %s",
                        len(all_items), user_id)

            for i, ctx in enumerate(all_items):
                LOGGER.debug("Entry %d: Key=%s", i, ctx.key)
                if hasattr(ctx, "value"):
                    LOGGER.debug("  Timestamp: %s", ctx.value.get("timestamp", "unknown"))
                    LOGGER.debug("  Messages: %s", len(ctx.value.get("messages", [])))
        except Exception as e:
            LOGGER.debug("Filtered all() failed: %s", repr(e))

    except Exception as e:
        LOGGER.debug("MEMORY DEBUG ERROR: %s", repr(e))

class State(MessagesState):
    """Extend MessagesState."""

    summary: str
    chat_model_usage_metadata: dict[str, Any]
    messages_to_remove: list[AnyMessage]

async def _call_model(
        state: State, config: RunnableConfig, *, store: BaseStore
    ) -> dict[str, list[BaseMessage]]:
    """Coroutine to call the model."""
    model = config["configurable"]["chat_model"]
    prompt = config["configurable"]["prompt"]
    user_id = config["configurable"]["user_id"]
    hass = config["configurable"]["hass"]

    # Retrieve most recent or search for most relevant memories for context.
    # Use semantic search if the last message was from the user.
    """
    last_message = state["messages"][-1]
    last_message_from_user = isinstance(last_message, HumanMessage)
    query_prompt = EMBEDDING_MODEL_PROMPT_TEMPLATE.format(
        query=last_message.content
    ) if last_message_from_user else None
    mems = await store.asearch(
        (user_id, "memories"),
        query=query_prompt,
        limit=10
    )
    formatted_mems = "\n".join(f"[{mem.key}]: {mem.value}" for mem in mems)
    """
    formatted_mems = ""

    # Form the System Message from the base prompt plus memories and past conversation
    # summaries, if they exist.
    system_message = prompt
    if formatted_mems:
        system_message += f"\n<memories>\n{formatted_mems}\n</memories>"
    summary = state.get("summary", "")
    if summary:
        system_message += (
            f"\n<past_conversation_summary>\n{summary}\n</past_conversation_summary>"
        )

    # Model input is the System Message plus current messages.
    messages = [SystemMessage(content=system_message)] + state["messages"]

    # Trim messages to manage context window length.
    # TODO - if using the token counter from the chat model API, the method
    # 'get_num_tokens_from_messages()' will be called which currently ignores
    # tool schemas and under counts message tokens for the qwen models.
    # Until this is fixed, 'max_tokens' should be set to a value less than
    # the maximum size of the model's context window. See const.py.
    num_tokens = await hass.async_add_executor_job(
        model.get_num_tokens_from_messages, messages
    )
    LOGGER.debug("Token count in messages from token counter: %s", num_tokens)
    if CONTEXT_MANAGE_USE_TOKENS:
        max_tokens = CONTEXT_MAX_TOKENS
        token_counter = config["configurable"]["chat_model"]
    else:
        max_tokens = CONTEXT_MAX_MESSAGES
        token_counter = len
    trimmed_messages = await hass.async_add_executor_job(
        partial(
            trim_messages,
            messages=messages,
            token_counter=token_counter,
            max_tokens=max_tokens,
            strategy="last",
            start_on="human",
            include_system=True,
        )
    )

    LOGGER.debug("Model call messages: %s", trimmed_messages)
    LOGGER.debug("Model call messages length: %s", len(trimmed_messages))

    response = await model.ainvoke(trimmed_messages)
    metadata = response.usage_metadata if hasattr(response, "usage_metadata") else {}
    # Clean up response, there is no need to include tool call metadata if there's none.
    if hasattr(response, "tool_calls"):
        response = AIMessage(content=response.content, tool_calls=response.tool_calls)
    else:
        response = AIMessage(content=response.content)
    LOGGER.debug("Model response: %s", response)
    LOGGER.debug("Token counts from metadata: %s", metadata)

    messages_to_remove = [m for m in state["messages"] if m not in trimmed_messages]
    LOGGER.debug("Messages to remove: %s", messages_to_remove)
    remove_messages = [RemoveMessage(id=m.id) for m in messages_to_remove]

    # Store conversation in memory for future retrieval
    try:
        user_id = config["configurable"]["user_id"]
        thread_id = config["configurable"].get("thread_id", "default")

        # Convert messages to storable format
        storable_messages = []
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                storable_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                storable_messages.append({"role": "assistant", "content": msg.content})

        # Add the new response
        storable_messages.append({"role": "assistant", "content": response.content})

        # Store the conversation
        store.put(
            key=(user_id, "conversations", thread_id),
            value={
                "timestamp": dt_util.now().isoformat(),
                "messages": storable_messages
            }
        )
        LOGGER.debug("Stored conversation for user %s, thread %s", user_id, thread_id)
    except Exception as e:
        LOGGER.error("Failed to store conversation: %s", repr(e))

    return {
        "messages": response,
        "chat_model_usage_metadata": metadata,
        "messages_to_remove": messages_to_remove,
    }

async def _summarize_and_remove_messages(
        state: State, config: RunnableConfig, *, store: BaseStore
    ) -> dict[str, str | list[AnyMessage]]:
    """Coroutine to summarize and remove messages."""
    summary = state.get("summary", "")
    msgs_to_remove = state.get("messages_to_remove", [])

    if not msgs_to_remove:
        return {"summary": summary}

    if summary:
        summary_message = SUMMARY_PROMPT_TEMPLATE.format(summary=summary)
    else:
        summary_message = SUMMARY_INITIAL_PROMPT

    # Form the messages that will be used by the summarization model.
    # The summary will be based on the messages that were trimmed away from the main
    # model call, ignoring those from tools since the AI message encapsulates them.
    messages = (
        [SystemMessage(content=SUMMARY_SYSTEM_PROMPT)] +
        [m.content for m in msgs_to_remove if isinstance(m, HumanMessage|AIMessage)] +
        [HumanMessage(content=summary_message)]
    )

    model = config["configurable"]["summarization_model"]
    options = config["configurable"]["options"]
    model_with_config = model.with_config(
        config={
            "model": options.get(
                CONF_SUMMARIZATION_MODEL,
                RECOMMENDED_SUMMARIZATION_MODEL,
            ),
            "temperature": options.get(
                CONF_SUMMARIZATION_MODEL_TEMPERATURE,
                RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
            ),
            "top_p": options.get(
                CONF_SUMMARIZATION_MODEL_TOP_P,
                RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
            ),
            "num_predict": SUMMARIZATION_MODEL_PREDICT,
            "num_ctx": SUMMARIZATION_MODEL_CTX,
        }
    )

    LOGGER.debug("Summary messages: %s", messages)
    response = await model_with_config.ainvoke(messages)
    LOGGER.debug("Summary response: %s", response)

    return {
        "summary": response.content,
        "messages": [RemoveMessage(id=m.id) for m in msgs_to_remove],
    }

async def _call_tools(
        state: State, config: RunnableConfig, *, store: BaseStore
    ) -> dict[str, list[ToolMessage]]:
    """Coroutine to call Home Assistant or langchain LLM tools."""
    # Tool calls will be the last message in state.
    tool_calls = state["messages"][-1].tool_calls

    langchain_tools = config["configurable"]["langchain_tools"]
    ha_llm_api = config["configurable"]["ha_llm_api"]

    tool_responses: list[ToolMessage] = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        LOGGER.debug(
            "Tool call: %s(%s)", tool_name, tool_args
        )

        def _handle_tool_error(err:str, name:str, tid:str) -> ToolMessage:
            return ToolMessage(
                content=TOOL_CALL_ERROR_TEMPLATE.format(error=err),
                name=name,
                tool_call_id=tid,
                status="error",
            )

        # A langchain tool was called.
        if tool_name in langchain_tools:
            lc_tool = langchain_tools[tool_name.lower()]

            # Provide hidden args to tool at runtime.
            tool_call_copy = copy.deepcopy(tool_call)
            tool_call_copy["args"].update(
                {
                    "store": store,
                    "config": config,
                }
            )

            try:
                tool_response = await lc_tool.ainvoke(tool_call_copy)
            except (HomeAssistantError, ValidationError) as e:
                tool_response = _handle_tool_error(repr(e), tool_name, tool_call["id"])
        # A Home Assistant tool was called.
        else:
            tool_input = llm.ToolInput(
                tool_name=tool_name,
                tool_args=tool_args,
            )

            try:
                response = await ha_llm_api.async_call_tool(tool_input)

                tool_response = ToolMessage(
                    content=json.dumps(response),
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            except (HomeAssistantError, vol.Invalid) as e:
                tool_response = _handle_tool_error(repr(e), tool_name, tool_call["id"])

        LOGGER.debug("Tool response: %s", tool_response)
        tool_responses.append(tool_response)
    return {"messages": tool_responses}

async def _initialize_context(
        state: State, config: RunnableConfig, *, store: BaseStore
    ) -> dict[str, State]:
    """Initialize agent state and load relevant previous context."""
    try:
        user_id = config["configurable"]["user_id"]

        # Call debug helper to inspect memory
        await debug_memory(store, user_id)

        # If there are already messages in the state, likely a continuation
        if state["messages"]:
            LOGGER.debug("Messages already in state, skipping initialization")
            return {}  # No changes needed

        # Retrieve recent context from memory store
        LOGGER.debug("Attempting to retrieve previous conversations for user: %s", user_id)
        try:
            # Get all items and filter locally
            all_items = store.all()
            recent_contexts = [item for item in all_items
                              if isinstance(item.key, tuple) and
                              len(item.key) >= 2 and
                              item.key[0] == user_id and
                              item.key[1] == "conversations"]

            LOGGER.debug("Retrieved %s context entries from store", len(recent_contexts))

            # Sort by timestamp (newest first)
            recent_contexts.sort(
                key=lambda x: x.value.get("timestamp", ""),
                reverse=True
            )

            # Limit to 10 most recent conversations
            recent_contexts = recent_contexts[:10]

            # Process contexts to extract messages
            # ... (rest of your processing code)
        except Exception as e:
            LOGGER.error("Error retrieving conversations: %s", repr(e))
            recent_contexts = []

        # Process contexts to extract messages
        context_messages = []
        for context in recent_contexts:
            try:
                if hasattr(context, "value") and isinstance(context.value, dict) and "messages" in context.value:
                    for msg in context.value["messages"]:
                        if isinstance(msg, dict) and "role" in msg and "content" in msg:
                            if msg["role"] == "user":
                                context_messages.append(HumanMessage(content=msg["content"]))
                            elif msg["role"] == "assistant":
                                context_messages.append(AIMessage(content=msg["content"]))
            except Exception as e:
                LOGGER.error("Error processing context entry: %s", e)
                continue

        LOGGER.debug("Retrieved %s previous context messages", len(context_messages))

        # Only return changes if we found context
        if context_messages:
            return {"messages": context_messages}

        return {}
    except Exception as e:
        LOGGER.error("Error in _initialize_context: %s", repr(e))
        # Don't modify state if there's an error
        return {}

async def _context_recovery(
        state: State, config: RunnableConfig, *, store: BaseStore
    ) -> dict[str, Any]:
    """Recover context if it appears to be lost."""
    """
    try:
        # Safely access messages
        messages = state.get("messages", [])
        if not messages or len(messages) == 0:
            LOGGER.debug("No messages in state, skipping recovery")
            return {}

        # Get the last message safely
        try:
            last_msg = messages[-1]
        except IndexError:
            LOGGER.error("Failed to get last message from state")
            return {}

        if not isinstance(last_msg, HumanMessage):
            LOGGER.debug("Last message not from human, skipping recovery")
            return {}

        # Look for indicators of lost context
        context_loss_indicators = [
            "you said earlier",
            "as i mentioned",
            "as you mentioned",
            "we were talking about",
            "going back to",
            "as we discussed",
            "you told me",
        ]

        content = last_msg.content.lower() if hasattr(last_msg, "content") else ""
        needs_recovery = any(indicator in content for indicator in context_loss_indicators)

        if not needs_recovery:
            LOGGER.debug("No context loss indicators detected")
            return {}

        LOGGER.debug("Detected potential context loss, attempting recovery")

        # Get user ID
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id:
            LOGGER.error("Missing user ID in config, cannot recover context")
            return {}

        # Search for relevant previous conversations
        search_query = last_msg.content
        try:
            related_contexts = await store.asearch(
                (user_id, "conversations"),
                query=search_query,
                limit=5
            )
        except Exception as e:
            LOGGER.error("Failed to search for related contexts: %s", repr(e))
            return {}

        # Add a system message with context summary
        context_summary = "Previous conversation context: "
        for idx, ctx in enumerate(related_contexts):
            if hasattr(ctx, "value") and isinstance(ctx.value, dict) and "messages" in ctx.value:
                for msg in ctx.value["messages"]:
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if content:
                        context_summary += f"\n- {role.capitalize()}: {content[:100]}..."

        # Insert recovery message at the beginning of the context
        recovery_msg = SystemMessage(content=context_summary)
        LOGGER.debug("Created recovery context message")

        # Return the recovery message to be inserted into state
        return {"messages": messages + [recovery_msg]}
    except Exception as e:
        LOGGER.error("Error in context recovery: %s", repr(e))
        return {}
    """
    return {}

def _should_continue(
        state: State
    ) -> Literal["action", "summarize_and_trim", "context_recovery", "__end__"]:
    """Return the next node in graph to execute."""
    messages = state["messages"]

    if messages[-1].tool_calls:
        return "action"

    if len(messages) > CONTEXT_SUMMARIZE_THRESHOLD:
        LOGGER.debug("Summarizing conversation")
        return "summarize_and_trim"

    """
    # Check if the last message is from a human and might need context recovery
    if isinstance(messages[-1], HumanMessage):
        content = messages[-1].content.lower()
        context_loss_indicators = [
            "you said earlier",
            "as i mentioned",
            "as you mentioned",
            "we were talking about",
            "going back to",
            "as we discussed",
            "you told me",
        ]
        if any(indicator in content for indicator in context_loss_indicators):
            LOGGER.debug("Potential context loss detected, routing to recovery")
            return "context_recovery"
    """

    return "__end__"

# Define a new graph
# workflow = StateGraph(State)
sqlite_path = "/config/ai_memory.sqlite"
store = SQLiteStore(uri=f"sqlite:///{sqlite_path}")
workflow = StateGraph(State, store=store)
# Define nodes.
workflow.add_node("initialize", _initialize_context)
workflow.add_node("agent", _call_model)
workflow.add_node("action", _call_tools)
workflow.add_node("summarize_and_trim", _summarize_messages)
workflow.add_node("context_recovery", _context_recovery)

# Define edges.
workflow.add_edge(START, "initialize")
workflow.add_edge("initialize", "agent")
workflow.add_conditional_edges("agent", _should_continue)
workflow.add_edge("action", "agent")
workflow.add_edge("context_recovery", "agent")
workflow.add_edge("summarize_and_trim", END)
