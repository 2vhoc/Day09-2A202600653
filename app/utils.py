from __future__ import annotations

from datetime import datetime, UTC
import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


def dump_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def extract_json_payload(text: str) -> dict[str, Any]:
    if not text:
        return {}
    candidate = text.strip()
    if candidate.startswith("```"):
        parts = [part for part in candidate.split("```") if part.strip()]
        candidate = parts[-1].replace("json", "", 1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def timestamp_utc() -> str:
    return datetime.now(UTC).isoformat()


def serialize_message(message: BaseMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": message.type,
        "content": message.content,
    }
    if isinstance(message, AIMessage):
        payload["tool_calls"] = message.tool_calls
    if isinstance(message, ToolMessage):
        payload["tool_name"] = message.name
        payload["tool_call_id"] = message.tool_call_id
    return payload


def list_worker_tools(messages: list[BaseMessage]) -> list[str]:
    tool_names: list[str] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                name = tool_call.get("name")
                if name and name not in tool_names:
                    tool_names.append(name)
    return tool_names


def get_last_ai_content(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return str(message.content)
    return ""
