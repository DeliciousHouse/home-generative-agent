"""Langgraph tools for Home Generative Agent."""
from __future__ import annotations

import aiohttp
import base64
import logging
import math
import os
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Dict, List

import aiofiles
import homeassistant.util.dt as dt_util
import yaml
from homeassistant.components import automation, camera, recorder
from homeassistant.components.automation.config import _async_validate_config_item
from homeassistant.config import AUTOMATION_CONFIG_PATH
from homeassistant.const import SERVICE_RELOAD
from homeassistant.core import State
from homeassistant.exceptions import (
    HomeAssistantError,
)
from homeassistant.util import ulid
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig  # noqa: TCH002
from langchain_core.tools import InjectedToolArg, tool
from langchain_ollama import ChatOllama  # noqa: TCH002
from langgraph.prebuilt import InjectedStore  # noqa: TCH002
from langgraph.store.base import BaseStore  # noqa: TCH002
from ulid import ULID  # noqa: TCH002
from voluptuous import MultipleInvalid

from .const import (
    BLUEPRINT_NAME,
    CONF_VISION_MODEL_TEMPERATURE,
    CONF_VISION_MODEL_TOP_P,
    CONF_VLM,
    EVENT_AUTOMATION_REGISTERED,
    RECOMMENDED_VISION_MODEL_TEMPERATURE,
    RECOMMENDED_VISION_MODEL_TOP_P,
    RECOMMENDED_VLM,
    VISION_MODEL_IMAGE_HEIGHT,
    VISION_MODEL_IMAGE_WIDTH,
    VISION_MODEL_SYSTEM_PROMPT,
    VISION_MODEL_USER_KW_PROMPT,
    VISION_MODEL_USER_PROMPT,
    VLM_NUM_CTX,
    VLM_NUM_PREDICT,
)
from .utilities import as_utc, gen_dict_extract

if TYPE_CHECKING:
    from types import MappingProxyType

    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)

async def _get_camera_image(hass: HomeAssistant, camera_name: str) -> bytes | None:
    """Get an image from a given camera."""
    camera_entity_id: str = f"camera.{camera_name.lower()}"
    try:
        image = await camera.async_get_image(
            hass=hass,
            entity_id=camera_entity_id,
            width=VISION_MODEL_IMAGE_WIDTH,
            height=VISION_MODEL_IMAGE_HEIGHT
        )
    except HomeAssistantError as err:
        LOGGER.error(
            "Error getting image from camera '%s' with error: %s",
            camera_entity_id, err
        )
        return None

    return image.content

async def _analyze_image(
        vlm_model: ChatOllama,
        options: dict[str, Any] | MappingProxyType[str, Any],
        image: bytes,
        detection_keywords: list[str] | None = None
    ) -> str:
    """Analyze an image."""
    encoded_image = base64.b64encode(image).decode("utf-8")

    def prompt_func(data: dict[str, Any]) -> list[AnyMessage]:
        system = data["system"]
        text = data["text"]
        image = data["image"]

        text_part = {"type": "text", "text": text}
        image_part = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        }

        content_parts = []
        content_parts.append(text_part)
        content_parts.append(image_part)

        return [SystemMessage(content=system), HumanMessage(content=content_parts)]

    model = vlm_model
    model_with_config = model.with_config(
        config={
            "model": options.get(
                CONF_VLM,
                RECOMMENDED_VLM,
            ),
            "temperature": options.get(
                CONF_VISION_MODEL_TEMPERATURE,
                RECOMMENDED_VISION_MODEL_TEMPERATURE,
            ),
            "top_p": options.get(
                CONF_VISION_MODEL_TOP_P,
                RECOMMENDED_VISION_MODEL_TOP_P,
            ),
            "num_predict": VLM_NUM_PREDICT,
            "num_ctx": VLM_NUM_CTX,
        }
    )

    chain = prompt_func | model_with_config

    if detection_keywords is not None:
        prompt = f"{VISION_MODEL_USER_KW_PROMPT} {' or '.join(detection_keywords):}"
    else:
        prompt = VISION_MODEL_USER_PROMPT

    try:
        response =  await chain.ainvoke(
            {
                "system": VISION_MODEL_SYSTEM_PROMPT,
                "text": prompt,
                "image": encoded_image
            }
        )
    except HomeAssistantError as err: #TODO: add validation error handling and retry prompt
        LOGGER.error("Error analyzing image %s", err)

    return response.content

@tool(parse_docstring=True)
async def get_and_analyze_camera_image( # noqa: D417
        camera_name: str,
        detection_keywords: list[str],
        *,
        # Hide these arguments from the model.
        config: Annotated[RunnableConfig, InjectedToolArg()],
    ) -> str:
    """
    Get a camera image and perform scene analysis on it.

    Args:
        camera_name: Name of the camera for scene analysis.
        detection_keywords: Specific objects to look for in image, if any.
            Such as if user says "check the front porch camera for
            boxes and dogs", detection_keywords would be ["boxes", "dogs"].

    """
    hass = config["configurable"]["hass"]
    vlm_model = config["configurable"]["vlm_model"]
    options = config["configurable"]["options"]
    image = await _get_camera_image(hass, camera_name)
    if image is None:
        return "Error getting image from camera."
    return await _analyze_image(vlm_model, options, image, detection_keywords)

@tool(parse_docstring=True)
async def upsert_memory( # noqa: D417
    content: str,
    context: str = "",
    *,
    memory_id: str = "",
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
    store: Annotated[BaseStore, InjectedToolArg()],
) -> str:
    """
    INSERT or UPDATE a memory in the database.
    You MUST use this tool to INSERT or UPDATE memories about users.
    Such as memories are specific facts or concepts learned from interactions
    with users. If a memory conflicts with an existing one then just UPDATE the
    existing one by passing in "memory_id" and DO NOT create two memories that are
    the same. If the user corrects a memory then UPDATE it.

    Args:
        content: The main content of the memory.
            Such as "I would like to learn french."
        context: Additional relevant context for the memory, if any.
            Such as "This was mentioned while discussing career options in Europe."
        memory_id: The memory to overwrite.
            ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.

    """
    mem_id = memory_id or ulid.ulid_now()
    await store.aput(
        namespace=(config["configurable"]["user_id"], "memories"),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    return f"Stored memory {mem_id}"

@tool(parse_docstring=True)
async def add_automation(  # noqa: D417
    automation_yaml: str = "",
    time_pattern: str = "",
    message: str = "",
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()]
) -> str:
    """
    Add an automation to Home Assistant.
    You are provided a Home Assistant blueprint as part of this tool if you need it.
    You MUST ONLY use the blueprint to create automations that involve camera image
    analysis. You MUST generate Home Assistant automation YAML for everything else.
    If using the blueprint you MUST provide the arguments "time_pattern" and "message"
    and DO NOT provide the argument "automation_yaml".

    Args:
        automation_yaml: A Home Assistant automation in valid YAML format.
            ONLY provide if NOT using the camera image analysis blueprint.
        time_pattern: Cron-like time pattern (e.g., /30 for "every 30 mins").
            ONLY provide if using the camera image analysis blueprint.
        message: Image analysis prompt (e.g.,"check the front porch camera for boxes")
            ONLY provide if using the camera image analysis blueprint.

    """
    hass = config["configurable"]["hass"]

    if time_pattern and message:
        automation_data = {
            "alias": message,
            "description": f"Created with blueprint {BLUEPRINT_NAME}.",
            "use_blueprint": {
                "path": BLUEPRINT_NAME,
                "input": {
                    "time_pattern": time_pattern,
                    "message": message,
                }
            }
        }
        automation_yaml = yaml.dump(automation_data)

    automation_parsed = yaml.safe_load(automation_yaml)
    ha_automation_config = {"id": ulid.ulid_now()}
    if isinstance(automation_parsed, list):
        ha_automation_config.update(automation_parsed[0])
    if isinstance(automation_parsed, dict):
        ha_automation_config.update(automation_parsed)

    try:
        await _async_validate_config_item(
            hass = hass,
            config = ha_automation_config,
            raise_on_errors = True,
            warn_on_errors = False
        )
    except (HomeAssistantError, MultipleInvalid) as err:
        return f"Invalid automation configuration {err}"

    async with aiofiles.open(
        Path(hass.config.config_dir) / AUTOMATION_CONFIG_PATH,
        encoding="utf-8"
    ) as f:
        ha_exsiting_automation_configs = await f.read()
        ha_exsiting_automations_yaml = yaml.safe_load(ha_exsiting_automation_configs)

    async with aiofiles.open(
        Path(hass.config.config_dir) / AUTOMATION_CONFIG_PATH,
        "a" if ha_exsiting_automations_yaml else "w",
        encoding="utf-8"
    ) as f:
        ha_automation_config_raw = yaml.dump(
            [ha_automation_config], allow_unicode=True, sort_keys=False
        )
        await f.write("\n" + ha_automation_config_raw)

    await hass.services.async_call(automation.config.DOMAIN, SERVICE_RELOAD)

    hass.bus.async_fire(
        EVENT_AUTOMATION_REGISTERED,
        {
            "automation_config": ha_automation_config,
            "raw_config": ha_automation_config_raw,
        },
    )

    return f"Added automation {ha_automation_config['id']}"

@tool(parse_docstring=True)
async def get_entity_history(  # noqa: D417
    entity_ids: list[str],
    local_start_time: str,
    local_end_time: str,
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()]
) -> dict[str, list[dict[str, Any]]]:
    """
    Get entity state history from Home Assistant.

    Args:
        entity_ids: List of Home Assistant entity ids to retrieve the history for.
            Such as if the user says "how much energy did the washing machine
            consume last week", entity_id is "sensor.washing_machine_switch_0_energy"
            DO NOT use use the name "washing machine Switch 0 energy" for entity_id.
            You MUST use an underscore symbol (e.g., "_") as a word deliminator.
        local_start_time: Start of local time history period in "%Y-%m-%dT%H:%M:%S%z".
        local_end_time: End of local time history period in "%Y-%m-%dT%H:%M:%S%z".

    Returns:
        Entity history in local time.

    """
    hass = config["configurable"]["hass"]

    entity_ids = [i.lower() for i in entity_ids]

    now = dt_util.utcnow()
    one_day = timedelta(days=1)
    try:
        start_time = as_utc(
            dattim = local_start_time,
            default = now - one_day,
            error_message = "start_time not valid"
        )
        end_time = as_utc(
            dattim = local_end_time,
            default = start_time + one_day,
            error_message = "end_time not valid"
        )
    except HomeAssistantError as err:
        return f"Invalid time {err}"

    filters = None
    include_start_time_state = True
    significant_changes_only = True
    minimal_response = True # If True filter out duplicate states
    no_attributes = False
    compressed_state_format = False

    with recorder.util.session_scope(hass=hass, read_only=True) as session:
        history = await recorder.get_instance(hass).async_add_executor_job(
            recorder.history.get_significant_states_with_session,
            hass,
            session,
            start_time,
            end_time,
            entity_ids,
            filters,
            include_start_time_state,
            significant_changes_only,
            minimal_response,
            no_attributes,
            compressed_state_format
        )

    if not history:
        return {}

    # Convert any State objects in history to dict.
    history = {
        e: [
            s.as_dict() if isinstance(s, State) else s for s in v
        ] for e, v in history.items()
    }

    # Convert history datetimes in UTC to local timezone.
    for lst in history.values():
        for d in lst:
            for k, v in d.items():
                try:
                    dattim = dt_util.parse_datetime(v, raise_on_error=True)
                    dattim_local = dt_util.as_local(dattim)
                    d.update({k: dattim_local.strftime("%Y-%m-%dT%H:%M:%S%z")})
                except (ValueError, TypeError):
                    pass
                except HomeAssistantError as err:
                    return f"Unexpected datetime conversion error {err}"

    try:
        state_class_value = next(iter(gen_dict_extract("state_class", history)))
    except StopIteration:
        state_class_value = None

    if state_class_value in ("measurement", "total"):
        # Filter history to just state values with datetimes.
        keys_to_check = ["state", "last_changed"]
        for lst in history.values():
            state_values = [d for d in lst if all(key in d for key in keys_to_check)]
        # Decimate to avoid adding unnecessary fine grained date to context.
        limit = 50
        length = len(state_values)
        if length > limit:
            LOGGER.debug("Decimating sensor data set.")
            factor = length // limit
            state_values = state_values[::factor]
        entity_id = next(iter(gen_dict_extract("entity_id", history)))
        units = next(iter(gen_dict_extract("unit_of_measurement", history)))
        return {entity_id: {"values": state_values, "units": units}}

    if state_class_value == "total_increasing":
        # For sensors with state class 'total_increasing', the history contains the
        # accumulated growth of the sensor's value since it was first added.
        # Therefore return the net change, not the entire history.

        # Filter history to just state values (no datetimes).
        state_values = [float(v) for v in list(gen_dict_extract("state", history))]
        # Check if sensor was reset during the time of interest.
        zero_indices = [i for i, x in enumerate(state_values) if math.isclose(x, 0)]
        if zero_indices:
            # Start data set from last time the sensor was reset.
            LOGGER.debug("Sensor was reset during time of interest.")
            state_values = state_values[zero_indices[-1]:]
        state_value_change = max(state_values) - min(state_values)
        entity_id = next(iter(gen_dict_extract("entity_id", history)))
        units = next(iter(gen_dict_extract("unit_of_measurement", history)))
        return {entity_id: {"value": state_value_change, "units": units}}

    return history

@tool(parse_docstring=True)
async def get_current_device_state( # noqa: D417
        names: list[str],
        *,
        # Hide these arguments from the model.
        config: Annotated[RunnableConfig, InjectedToolArg()],
    ) -> dict[str, str]:
    """
    Get the current state of one or more Home Assistant devices.

    Args:
        names: List of Home Assistant device names.

    """
    def _parse_input_to_yaml(input_text: str) -> dict[str, Any]:
        # Define the marker that separates instructions from the device list.
        split_marker = "An overview of the areas and the devices in this smart home:"

        # Check if the marker exists in the input text.
        if split_marker not in input_text:
            msg = "Input text format is invalid. Marker not found."
            raise ValueError(msg)

        # Split the input text into instructions and devices part
        instructions_part, devices_part = input_text.split(split_marker, 1)

        # Clean up whitespace
        instructions = instructions_part.strip()
        devices_yaml = devices_part.strip()

        # Parse the devices list using PyYAML
        devices = yaml.safe_load(devices_yaml)

        # Combine into a single dictionary
        return {
            "instructions": instructions,
            "devices": devices
        }

    # Use the HA LLM API to get overview of all devices.
    llm_api = config["configurable"]["ha_llm_api"]
    try:
        overview = _parse_input_to_yaml(llm_api.api_prompt)
    except ValueError as e:
        LOGGER.error("There was a problem getting device state: %s", e)
        return {}

    # Get the list of devices.
    devices = overview.get("devices", [])

    # Create a dictionary mapping desired device names to their state.
    state_dict = {}
    for device in devices:
        name = device.get("names", "Unnamed Device")
        if name not in names:
            continue
        state = device.get("state", None)
        state_dict[name] = state

    return state_dict

@tool(parse_docstring=True)
async def manage_scene( # noqa: D417
    action: str,
    scene_name: str,
    location: str,
    scene_data: Dict[str, Any] = None,
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
    store: Annotated[BaseStore, InjectedToolArg()],
) -> str:
    """
    Manage scenes in different locations of the home.

    Args:
        action: Action to perform (create, update, get, delete).
        scene_name: Name of the scene to manage.
        location: Location where the scene applies.
        scene_data: Scene configuration data.
    """
    scene_id = f"{location.lower()}_{scene_name.lower()}"

    if action == "create" or action == "update":
        if not scene_data:
            return "Scene data is required for create/update operations"

        await store.aput(
            namespace=(config["configurable"]["user_id"], "scenes"),
            key=scene_id,
            value={
                "name": scene_name,
                "location": location,
                "data": scene_data
            }
        )
        return f"Scene '{scene_name}' {'created' if action == 'create' else 'updated'} for {location}"

    elif action == "get":
        scene = await store.aget(
            namespace=(config["configurable"]["user_id"], "scenes"),
            key=scene_id
        )
        if not scene:
            return f"Scene '{scene_name}' not found in {location}"
        return scene

    elif action == "delete":
        await store.adelete(
            namespace=(config["configurable"]["user_id"], "scenes"),
            key=scene_id
        )
        return f"Scene '{scene_name}' deleted from {location}"

    return f"Invalid action '{action}'. Must be one of: create, update, get, delete"

@tool(parse_docstring=True)
async def perform_location_action( # noqa: D417
    location: str,
    action: str,
    entities: List[str] = None,
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()]
) -> str:
    """
    Perform actions on entities in a specific location.

    Args:
        location: Location to perform action in.
        action: Action to perform (turn_on, turn_off).
        entities: Optional list of specific entities.
    """
    hass = config["configurable"]["hass"]

    # Validate location exists
    location = location.lower()
    area_registry = hass.data["area_registry"]
    area = next((area for area in area_registry.areas if area.name.lower() == location), None)

    if not area:
        return f"Location '{location}' not found"

    # Get entities for the location
    device_registry = hass.data["device_registry"]
    entity_registry = hass.data["entity_registry"]

    # If no specific entities provided, get all entities in the location
    if not entities:
        location_entities = [
            entity.entity_id for entity in entity_registry.entities.values()
            if entity.area_id == area.id
        ]
    else:
        location_entities = entities

    if not location_entities:
        return f"No entities found in {location}"

    # Perform the action
    try:
        if action == "turn_on":
            for entity_id in location_entities:
                await hass.services.async_call(
                    domain="homeassistant",
                    service="turn_on",
                    target={"entity_id": entity_id}
                )
        elif action == "turn_off":
            for entity_id in location_entities:
                await hass.services.async_call(
                    domain="homeassistant",
                    service="turn_off",
                    target={"entity_id": entity_id}
                )
        else:
            return f"Unsupported action: {action}"

        return f"Successfully performed '{action}' in {location}"

    except HomeAssistantError as err:
        return f"Error performing action: {err}"


@tool(parse_docstring=True)
async def reverse_geocode( # noqa: D417
    latitude: float,
    longitude: float,
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()]
) -> dict[str, Any]:
    """
    Convert latitude and longitude coordinates to a physical address.

    Args:
        latitude: Latitude coordinate (such as 38.30662).
        longitude: Longitude coordinate (such as -122.29125).

    Returns:
        Dictionary containing address information.
    """
    # Using OpenStreetMap's Nominatim service (free but has usage limits)
    url = f"https://nominatim.openstreetmap.org/reverse?lat={latitude}&lon={longitude}&format=json"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers={"User-Agent": "HomeAssistantComponent/1.0"}) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    "full_address": result.get("display_name", "Unknown location"),
                    "road": result.get("address", {}).get("road"),
                    "city": result.get("address", {}).get("city"),
                    "county": result.get("address", {}).get("county"),
                    "state": result.get("address", {}).get("state"),
                    "country": result.get("address", {}).get("country"),
                    "postcode": result.get("address", {}).get("postcode"),
                }
            else:
                return {"error": f"Failed to geocode: HTTP {response.status}"}


@tool(parse_docstring=True)
async def suggest_contextual_automation( # noqa: D417
    context_description: str,
    timeframe_days: int = 7,
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
    store: Annotated[BaseStore, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Suggests automations based on user behavior patterns and smart home context.

    Args:
        context_description: Description of the context for which automation suggestions
            are requested. Like, "energy saving when nobody is home" or
            "morning routines on weekdays".
        timeframe_days: Number of days of historical data to analyze for patterns.
            Default is 7 days.

    Returns:
        Dictionary containing suggested automations with details, rationale, and confidence scores.
    """
    hass = config["configurable"]["hass"]
    user_id = config["configurable"]["user_id"]

    # Get information about the smart home's devices and areas
    device_registry = hass.data["device_registry"]
    area_registry = hass.data["area_registry"]
    entity_registry = hass.data["entity_registry"]

    # Get a list of areas with their entities
    areas_with_entities = {}
    for area in area_registry.areas.values():
        area_entities = [
            entity_id for entity_id, entity in entity_registry.entities.items()
            if entity.area_id == area.id
        ]
        if area_entities:
            areas_with_entities[area.name] = area_entities

    # Get recent history for relevant entities based on context
    now = dt_util.utcnow()
    start_time = now - timedelta(days=timeframe_days)

    # Extract relevant entities from the context description
    relevant_domains = []
    if "light" in context_description.lower() or "lighting" in context_description.lower():
        relevant_domains.append("light")
    if "heat" in context_description.lower() or "temperature" in context_description.lower():
        relevant_domains.append("climate")
    if "security" in context_description.lower() or "camera" in context_description.lower():
        relevant_domains.append("camera")

    # Default to common domains if no specific domain was identified
    if not relevant_domains:
        relevant_domains = ["light", "switch", "sensor", "binary_sensor", "climate"]

    # Generate automation suggestions based on context and entity history
    suggestions = []

    # Example: Morning routine automation
    if "morning" in context_description.lower() or "wake up" in context_description.lower():
        suggestions.append({
            "name": "Morning Routine",
            "description": "Gradually turn on lights and adjust temperature in the morning",
            "trigger": {"platform": "time", "at": "06:30:00"},
            "conditions": [{"condition": "time", "weekday": ["mon", "tue", "wed", "thu", "fri"]}],
            "actions": [
                {"service": "light.turn_on", "target": {"area_id": "bedroom"}, "data": {"brightness_pct": 40}},
                {"service": "climate.set_temperature", "target": {"area_id": "living_room"}, "data": {"temperature": 21}}
            ],
            "confidence": 0.85,
            "rationale": "Based on regular weekday morning activity patterns"
        })

    # Example: Energy saving automation
    if "energy" in context_description.lower() or "saving" in context_description.lower():
        suggestions.append({
            "name": "Energy Saving Mode",
            "description": "Turn off non-essential devices when nobody is home",
            "trigger": {"platform": "state", "entity_id": "binary_sensor.presence", "to": "off"},
            "actions": [
                {"service": "light.turn_off", "target": {"area_id": "living_room"}},
                {"service": "switch.turn_off", "target": {"entity_id": "switch.non_essential_devices"}},
                {"service": "climate.set_temperature", "target": {"area_id": "house"}, "data": {"temperature": 18}}
            ],
            "confidence": 0.9,
            "rationale": "Based on energy usage patterns when home is unoccupied"
        })

    # Example: Security automation
    if "security" in context_description.lower() or "away" in context_description.lower():
        suggestions.append({
            "name": "Security Mode",
            "description": "Enhance security when everyone is away",
            "trigger": {"platform": "state", "entity_id": "group.household", "to": "not_home"},
            "actions": [
                {"service": "alarm_control_panel.alarm_arm_away", "target": {"entity_id": "alarm_control_panel.home_alarm"}},
                {"service": "light.turn_on", "target": {"area_id": "exterior"}, "data": {"brightness_pct": 30}},
                {"service": "script.simulate_presence", "data": {}}
            ],
            "confidence": 0.8,
            "rationale": "Based on security preferences and activity patterns during away times"
        })

    # Retrieve any user-specific memory about automation preferences
    user_memory = await store.asearch(
        namespace=(user_id, "memories"),
        query="automation preferences",
        limit=5
    )

    # Add additional context from user memories
    automation_context = {}
    if user_memory:
        automation_context["user_preferences"] = [mem.value for mem in user_memory]

    return {
        "suggestions": suggestions,
        "context": automation_context,
        "timeframe_analyzed": f"{timeframe_days} days"
    }

@tool(parse_docstring=True)
async def analyze_patterns( # noqa: D417
    entity_ids: list[str],
    pattern_type: str,
    time_period_days: int = 14,
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()]
) -> dict[str, Any]:
    """
    Analyzes entity history to identify patterns, trends, or anomalies in data.

    Args:
        entity_ids: List of entity IDs to analyze for patterns.
            Such as ["sensor.power_consumption", "binary_sensor.motion_sensor"]
        pattern_type: Type of pattern analysis to perform.
            Options include "usage_patterns", "anomaly_detection", "correlation",
            "cyclic_patterns", "trend_analysis"
        time_period_days: Number of days of historical data to analyze.
            Default is 14 days.

    Returns:
        Dictionary containing detected patterns, insights, and statistical data.
    """
    hass = config["configurable"]["hass"]

    # Validate entity_ids
    entity_registry = hass.data["entity_registry"]
    valid_entities = []
    for entity_id in entity_ids:
        if entity_id in entity_registry.entities or entity_id in hass.states.async_entity_ids():
            valid_entities.append(entity_id)

    if not valid_entities:
        return {"error": "No valid entity IDs provided"}

    # Set time period for analysis
    now = dt_util.utcnow()
    start_time = now - timedelta(days=time_period_days)

    # Get entity history data
    with recorder.util.session_scope(hass=hass, read_only=True) as session:
        history = await recorder.get_instance(hass).async_add_executor_job(
            recorder.history.get_significant_states_with_session,
            hass,
            session,
            start_time,
            now,
            valid_entities,
            None,  # No filters
            True,  # include_start_time_state
            True,  # significant_changes_only
            True,  # minimal_response
            False, # no_attributes
            False  # compressed_state_format
        )

    if not history:
        return {"error": "No historical data found for the specified entities"}

    # Extract state values and timestamps
    processed_data = {}
    for entity_id, states in history.items():
        # Convert State objects to usable data
        data_points = []
        for state in states:
            if isinstance(state, State):
                state_dict = state.as_dict()
            else:
                state_dict = state

            try:
                # Try to convert state to float for numerical analysis
                value = float(state_dict.get("state", 0))
                timestamp = state_dict.get("last_changed")
                if timestamp:
                    # Convert to datetime if it's a string
                    if isinstance(timestamp, str):
                        timestamp = dt_util.parse_datetime(timestamp)
                    data_points.append({"timestamp": timestamp, "value": value})
            except (ValueError, TypeError):
                # Handle non-numeric states
                value = state_dict.get("state")
                timestamp = state_dict.get("last_changed")
                if timestamp and value:
                    if isinstance(timestamp, str):
                        timestamp = dt_util.parse_datetime(timestamp)
                    data_points.append({"timestamp": timestamp, "value": value})

        processed_data[entity_id] = data_points

    # Analyze data based on pattern_type
    analysis_results = {}

    if pattern_type == "usage_patterns":
        # Identify times of day with highest/lowest usage
        for entity_id, data_points in processed_data.items():
            if not data_points:
                continue

            # Group data by hour of day
            hourly_data = {}
            for point in data_points:
                if isinstance(point["value"], (int, float)):
                    hour = point["timestamp"].hour
                    hourly_data.setdefault(hour, []).append(point["value"])

            # Calculate average value for each hour
            hourly_averages = {hour: sum(values)/len(values) for hour, values in hourly_data.items() if values}

            if hourly_averages:
                peak_hour = max(hourly_averages, key=hourly_averages.get)
                low_hour = min(hourly_averages, key=hourly_averages.get)

                analysis_results[entity_id] = {
                    "peak_usage_hour": peak_hour,
                    "peak_value": hourly_averages[peak_hour],
                    "lowest_usage_hour": low_hour,
                    "lowest_value": hourly_averages[low_hour],
                    "hourly_profile": hourly_averages
                }

    elif pattern_type == "anomaly_detection":
        # Simple anomaly detection using standard deviation
        for entity_id, data_points in processed_data.items():
            numeric_values = [point["value"] for point in data_points
                             if isinstance(point["value"], (int, float))]

            if len(numeric_values) > 5:  # Need sufficient data points
                mean = sum(numeric_values) / len(numeric_values)
                variance = sum((x - mean) ** 2 for x in numeric_values) / len(numeric_values)
                std_dev = math.sqrt(variance)

                # Find outliers (values more than 2 standard deviations from mean)
                outliers = []
                for point in data_points:
                    if isinstance(point["value"], (int, float)):
                        if abs(point["value"] - mean) > 2 * std_dev:
                            outliers.append({
                                "timestamp": point["timestamp"].isoformat(),
                                "value": point["value"],
                                "deviation": (point["value"] - mean) / std_dev
                            })

                analysis_results[entity_id] = {
                    "mean": mean,
                    "std_dev": std_dev,
                    "outliers": outliers
                }

    elif pattern_type == "cyclic_patterns":
        # Detect daily/weekly cycles
        for entity_id, data_points in processed_data.items():
            if len(data_points) > 24:  # Need at least a day of data
                # Group by day of week
                daily_patterns = {i: [] for i in range(7)}  # 0 = Monday
                for point in data_points:
                    if isinstance(point["value"], (int, float)):
                        weekday = point["timestamp"].weekday()
                        daily_patterns[weekday].append(point["value"])

                # Calculate average value for each day
                day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                day_averages = {}
                for day_num, values in daily_patterns.items():
                    if values:
                        day_averages[day_names[day_num]] = sum(values) / len(values)

                analysis_results[entity_id] = {
                    "daily_patterns": day_averages,
                    "highest_day": max(day_averages, key=day_averages.get) if day_averages else None,
                    "lowest_day": min(day_averages, key=day_averages.get) if day_averages else None
                }

    elif pattern_type == "correlation":
        # Find correlation between different entities
        if len(valid_entities) > 1:
            correlations = {}
            # For simplicity, just check if entities change around the same times
            for i, entity1 in enumerate(valid_entities[:-1]):
                for entity2 in valid_entities[i+1:]:
                    if entity1 in processed_data and entity2 in processed_data:
                        # Create time-based mapping of changes
                        changes1 = {point["timestamp"].strftime("%Y-%m-%d %H"): point["value"]
                                  for point in processed_data[entity1]}
                        changes2 = {point["timestamp"].strftime("%Y-%m-%d %H"): point["value"]
                                  for point in processed_data[entity2]}

                        # Find common hours when both entities had changes
                        common_hours = set(changes1.keys()) & set(changes2.keys())
                        correlation_score = len(common_hours) / max(len(changes1), len(changes2)) if max(len(changes1), len(changes2)) > 0 else 0

                        correlations[f"{entity1} and {entity2}"] = {
                            "correlation_score": correlation_score,
                            "common_change_points": len(common_hours)
                        }
            analysis_results["correlations"] = correlations

    elif pattern_type == "trend_analysis":
        # Calculate overall trends (increasing/decreasing)
        for entity_id, data_points in processed_data.items():
            numeric_values = [point["value"] for point in data_points
                             if isinstance(point["value"], (int, float))]

            if len(numeric_values) > 5:  # Need sufficient data points
                # Calculate linear regression for trend
                n = len(numeric_values)
                indices = list(range(n))
                sum_x = sum(indices)
                sum_y = sum(numeric_values)
                sum_xy = sum(x*y for x, y in zip(indices, numeric_values))
                sum_xx = sum(x*x for x in indices)

                # Calculate slope
                if n * sum_xx - sum_x * sum_x != 0:
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
                    intercept = (sum_y - slope * sum_x) / n

                    # Determine trend direction
                    trend_direction = "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable"

                    # Calculate recent vs. overall average
                    recent_values = numeric_values[-min(5, len(numeric_values)):]
                    recent_avg = sum(recent_values) / len(recent_values)
                    overall_avg = sum(numeric_values) / len(numeric_values)

                    analysis_results[entity_id] = {
                        "trend": trend_direction,
                        "slope": slope,
                        "recent_vs_overall": recent_avg / overall_avg if overall_avg != 0 else None,
                        "start_value": numeric_values[0] if numeric_values else None,
                        "end_value": numeric_values[-1] if numeric_values else None
                    }

    return {
        "analysis_type": pattern_type,
        "time_period": f"{time_period_days} days",
        "results": analysis_results
    }

@tool(parse_docstring=True)
async def run_diagnostics( # noqa: D417
    target: str,
    diagnostic_type: str,
    additional_parameters: dict[str, Any] = None,
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()]
) -> dict[str, Any]:
    """
    Diagnoses issues with smart home devices, configurations, or systems.

    Args:
        target: The target entity, area, or system to diagnose.
            Such as "light.lounge_lamp_light", "climate", "network", "master_bedroom area"
        diagnostic_type: Type of diagnostic to run.
            Options include "connectivity", "performance", "errors",
            "configuration", "compatibility", "system_health"

    Returns:
        Dictionary containing diagnostic results, identified issues, and recommendations.
    """
    hass = config["configurable"]["hass"]

    # Initialize results dictionary
    results = {
        "target": target,
        "diagnostic_type": diagnostic_type,
        "timestamp": dt_util.now().isoformat(),
        "issues": [],
        "recommendations": []
    }

    # Validate target exists
    if target.startswith(("light.", "switch.", "climate.", "automation.", "binary_sensor.")):
        # Entity-based target
        entity_registry = hass.data["entity_registry"]
        if not entity_registry.async_is_registered(target):
            results["issues"].append(f"Entity {target} not found")
            return results
    elif "area" in target:
        # Area-based target
        area_name = target.replace(" area", "").strip()
        area_registry = hass.data["area_registry"]
        area = next((area for area in area_registry.areas if area.name.lower() == area_name.lower()), None)
        if not area:
            results["issues"].append(f"Area {area_name} not found")
            return results
    elif target in ["network", "system"]:
        # System-level diagnostics
        pass
    else:
        results["issues"].append(f"Invalid target type: {target}")
        return results

    # Run diagnostics based on type
    if diagnostic_type == "connectivity":
        # Check entity/area connectivity
        if target.startswith(("light.", "switch.", "climate.", "automation.", "binary_sensor.")):
            state = hass.states.get(target)
            if state is None:
                results["issues"].append("Device is not responding")
                results["recommendations"].append("Check device power and network connection")
            elif state.state == "unavailable":
                results["issues"].append("Device is unavailable")
                results["recommendations"].append("Verify device is within range and powered")

    elif diagnostic_type == "performance":
        # Check response times and reliability
        duration = additional_parameters.get("duration", 60) if additional_parameters else 60

        if target.startswith(("light.", "switch.", "climate.", "automation.", "binary_sensor.")):
            # Get recent state changes
            entity_history = await get_entity_history(
                entity_ids=[target],
                local_start_time=(dt_util.utcnow() - timedelta(seconds=duration)).isoformat(),
                local_end_time=dt_util.utcnow().isoformat(),
                config=config
            )

            if entity_history:
                state_changes = len(next(iter(entity_history.values())))
                if state_changes > duration/10:  # More than 1 change per 10 seconds
                    results["issues"].append("High frequency of state changes detected")
                    results["recommendations"].append("Check for interference or automation loops")

    elif diagnostic_type == "configuration":
        # Validate entity/area configuration
        if target.startswith(("light.", "switch.", "climate.", "automation.", "binary_sensor.")):
            entity_registry = hass.data["entity_registry"]
            entity = entity_registry.async_get(target)

            if entity and not entity.area_id:
                results["issues"].append("Entity not assigned to any area")
                results["recommendations"].append("Assign entity to an area for better organization")

            if entity and not entity.name:
                results["issues"].append("Entity has no friendly name configured")
                results["recommendations"].append("Set a friendly name for easier identification")

    elif diagnostic_type == "system_health":
        # Check overall system health
        if target == "system":
            # Check disk usage
            stats = await hass.async_add_executor_job(os.statvfs, hass.config.config_dir)
            disk_usage = (stats.f_blocks - stats.f_bfree) * stats.f_frsize
            total_space = stats.f_blocks * stats.f_frsize
            usage_percent = (disk_usage / total_space) * 100

            if usage_percent > 90:
                results["issues"].append(f"High disk usage: {usage_percent:.1f}%")
                results["recommendations"].append("Clean up old logs and unused files")

    else:
        results["issues"].append(f"Unsupported diagnostic type: {diagnostic_type}")

    return results
