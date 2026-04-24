from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, Mapping


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False

    if value is None:
        return fallback

    return bool(value)


@dataclass(frozen=True)
class ChatSettings:
    temperature: float
    max_tokens: int
    stream: bool

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any] | None,
        fallback: "ChatSettings",
    ) -> "ChatSettings":
        payload = payload or {}

        next_temperature = _coerce_float(payload.get("temperature"), fallback.temperature)
        next_max_tokens = _coerce_int(
            payload.get("max_tokens", payload.get("response_length")),
            fallback.max_tokens,
        )
        next_stream = _coerce_bool(payload.get("stream"), fallback.stream)

        return cls(
            temperature=max(0.0, min(2.0, next_temperature)),
            max_tokens=max(1, min(4096, next_max_tokens)),
            stream=next_stream,
        )

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_length": self.max_tokens,
            "stream": self.stream,
        }


class BaseChatModel(ABC):
    model_id: str
    label: str

    @abstractmethod
    def chat(self, prompt: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def stream_chat(self, prompt: str) -> Iterator[str]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_settings(self) -> ChatSettings:
        raise NotImplementedError

    @abstractmethod
    def update_settings(self, payload: Mapping[str, Any]) -> ChatSettings:
        raise NotImplementedError

    def describe(self) -> dict[str, str]:
        return {
            "id": self.model_id,
            "label": self.label,
        }
