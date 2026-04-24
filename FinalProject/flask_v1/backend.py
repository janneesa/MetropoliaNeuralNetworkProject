from __future__ import annotations

from typing import Any, Iterator

from .models import BaseChatModel, EmmaModel


class ChatBackend:
    def __init__(self) -> None:
        self._models = {
            "emma": EmmaModel(),
        }
        self._active_model_id = "emma"

    @property
    def active_model(self) -> BaseChatModel:
        return self._models[self._active_model_id]

    def _apply_payload_settings(self, payload: dict[str, Any]) -> None:
        if any(key in payload for key in ("temperature", "max_tokens", "response_length", "stream")):
            self.update_settings(payload)

    def should_stream(self, payload: dict[str, Any] | None = None) -> bool:
        payload = payload or {}
        if "stream" in payload:
            raw_value = payload["stream"]
            if isinstance(raw_value, bool):
                return raw_value
            if isinstance(raw_value, str):
                return raw_value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(raw_value)
        return bool(self.active_model.get_settings().stream)

    def _build_reset_payload(self) -> dict[str, Any]:
        return {
            "response": "Conversation reset.",
            "reset": True,
            "settings": self.get_settings_payload(),
            "model": self.active_model.describe(),
        }

    def get_settings_payload(self) -> dict[str, Any]:
        return {
            **self.active_model.get_settings().to_dict(),
            "model": self.active_model.describe(),
        }

    def update_settings(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        self.active_model.update_settings(payload or {})
        return self.get_settings_payload()

    def chat(self, prompt: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        normalized_prompt = str(prompt).strip()
        if not normalized_prompt:
            raise ValueError("Prompt must not be empty.")

        payload = payload or {}
        self._apply_payload_settings(payload)

        if normalized_prompt.lower() == "/reset":
            self.active_model.reset()
            return self._build_reset_payload()

        response = self.active_model.chat(normalized_prompt)
        return {
            "response": response,
            "reset": False,
            "settings": self.get_settings_payload(),
            "model": self.active_model.describe(),
        }

    def stream_chat(self, prompt: str, payload: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        normalized_prompt = str(prompt).strip()
        if not normalized_prompt:
            raise ValueError("Prompt must not be empty.")

        payload = payload or {}
        self._apply_payload_settings(payload)

        if normalized_prompt.lower() == "/reset":
            self.active_model.reset()
            yield {
                "type": "reset",
                **self._build_reset_payload(),
            }
            return

        yield {
            "type": "start",
            "settings": self.get_settings_payload(),
            "model": self.active_model.describe(),
        }

        response_parts: list[str] = []
        try:
            for chunk in self.active_model.stream_chat(normalized_prompt):
                response_parts.append(chunk)
                yield {
                    "type": "chunk",
                    "text": chunk,
                }
        except Exception as error:  # noqa: BLE001
            yield {
                "type": "error",
                "error": str(error),
                "settings": self.get_settings_payload(),
                "model": self.active_model.describe(),
            }
            return

        yield {
            "type": "done",
            "response": "".join(response_parts),
            "reset": False,
            "settings": self.get_settings_payload(),
            "model": self.active_model.describe(),
        }
