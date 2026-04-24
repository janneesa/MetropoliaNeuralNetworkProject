from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Mapping, cast

from .base import BaseChatModel, ChatSettings


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chatbots.emma.emma import EmmaChatService, GenerationConfig


class EmmaModel(BaseChatModel):
    model_id = "emma"
    label = "Emma"

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._lock = RLock()
        self._service: EmmaChatService | None = None
        self._session = None
        self._model_path = model_path or os.getenv("EMMA_MODEL_PATH")
        self._default_generation_config = GenerationConfig()
        self._settings = ChatSettings(
            temperature=self._default_generation_config.temperature,
            max_tokens=self._default_generation_config.max_new_tokens,
            stream=True,
        )

    def _build_generation_config(self) -> GenerationConfig:
        return replace(
            self._default_generation_config,
            temperature=self._settings.temperature,
            max_new_tokens=self._settings.max_tokens,
        )

    def _ensure_session(self):
        if self._service is None:
            self._service = EmmaChatService.build(
                model_path=self._model_path,
                prompt_for_model=False,
                default_generation_config=self._build_generation_config(),
                verbose=False,
            )

        if self._session is None:
            self._session = self._service.create_session(
                generation_config=self._build_generation_config(),
            )

        return self._session

    def get_settings(self) -> ChatSettings:
        with self._lock:
            return self._settings

    def update_settings(self, payload: Mapping[str, Any]) -> ChatSettings:
        with self._lock:
            next_settings = ChatSettings.from_payload(payload, fallback=self._settings)
            if next_settings == self._settings:
                return self._settings

            history = list(self._session.conversation_turns) if self._session is not None else []
            self._settings = next_settings

            if self._service is not None:
                self._session = self._service.create_session(
                    generation_config=self._build_generation_config(),
                    conversation_turns=history,
                )

            return self._settings

    def reset(self) -> None:
        with self._lock:
            if self._session is not None:
                self._session.reset()

    def chat(self, prompt: str) -> str:
        with self._lock:
            session = self._ensure_session()
            return str(session.chat(prompt, stream=False))

    def stream_chat(self, prompt: str) -> Iterator[str]:
        with self._lock:
            session = self._ensure_session()
            stream = cast(Iterator[str], session.chat(prompt, stream=True))
            for chunk in stream:
                yield str(chunk)
