"""Ollama /api/generate language model implementation."""

import asyncio
import logging
from typing import Any, TypeVar

import httpx
import json_repair
from pydantic import BaseModel, Field, InstanceOf, TypeAdapter

from memmachine_server.common.data_types import ExternalServiceAPIError
from memmachine_server.common.metrics_factory import MetricsFactory, OperationTracker

from .language_model import LanguageModel

T = TypeVar("T")

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE_URL = "http://host.docker.internal:11434"


class OllamaApiGenerateLanguageModelParams(BaseModel):
    """
    Parameters for OllamaApiGenerateLanguageModel.

    Attributes:
        client (httpx.AsyncClient):
            Async HTTP client for making requests to Ollama.
        model (str):
            Ollama model name (e.g. 'qwen3.5:9b').
        think (bool):
            Whether to enable Qwen think/reasoning mode (default: False).
        max_retry_interval_seconds (int):
            Maximal retry interval in seconds when retrying requests
            (default: 120).
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
    """

    client: InstanceOf[httpx.AsyncClient] = Field(
        ...,
        description="Async HTTP client for making requests to Ollama",
    )
    model: str = Field(
        ...,
        description="Ollama model name (e.g. 'qwen3.5:9b')",
    )
    think: bool = Field(
        False,
        description="Whether to enable Qwen think/reasoning mode",
    )
    max_retry_interval_seconds: int = Field(
        120,
        description="Maximal retry interval in seconds when retrying requests",
        gt=0,
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


class OllamaApiGenerateLanguageModel(LanguageModel):
    """Language model using Ollama's native /api/generate endpoint.

    This bypasses the OpenAI-compatibility layer in Ollama for significantly
    lower latency compared to /v1/chat/completions.
    """

    def __init__(self, params: OllamaApiGenerateLanguageModelParams) -> None:
        """
        Initialize the Ollama /api/generate language model.

        Args:
            params (OllamaApiGenerateLanguageModelParams):
                Parameters for the OllamaApiGenerateLanguageModel.

        """
        super().__init__()
        self._client = params.client
        self._model = params.model
        self._think = params.think
        self._max_retry_interval_seconds = params.max_retry_interval_seconds
        self._tracker = OperationTracker(
            params.metrics_factory, prefix="language_model_ollama_generate"
        )

    def _build_prompt(
        self,
        system_prompt: str | None,
        user_prompt: str | None,
    ) -> str:
        """Combine system and user prompts into a single prompt string."""
        parts = []
        if system_prompt:
            parts.append(system_prompt)
        if user_prompt:
            if parts:
                parts.append("\n")
            parts.append(user_prompt)
        return "".join(parts)

    async def _call_generate(self, prompt: str) -> str:
        """Call Ollama /api/generate and return the response text."""
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "think": self._think,
        }
        response = await self._client.post("/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,  # noqa: ARG002
        tool_choice: str | dict[str, str] | None = None,  # noqa: ARG002
        max_attempts: int = 1,
    ) -> tuple[str, Any]:
        """Generate a text response."""
        text, _, _, _ = await self._generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_attempts=max_attempts,
        )
        return text, []

    async def generate_response_with_token_usage(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,  # noqa: ARG002
        tool_choice: str | dict[str, str] | None = None,  # noqa: ARG002
        max_attempts: int = 1,
    ) -> tuple[str, Any, int, int]:
        """Generate a text response and return token usage."""
        return await self._generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_attempts=max_attempts,
        )

    async def generate_parsed_response(
        self,
        output_format: type[T],
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        max_attempts: int = 1,
    ) -> T | None:
        """Generate a structured response parsed into the given model."""
        async with self._tracker("generate_parsed_response"):
            if max_attempts <= 0:
                raise ValueError("max_attempts must be a positive integer")

            prompt = self._build_prompt(system_prompt, user_prompt)
            text, _, _, _ = await self._generate_response(
                system_prompt=None,
                user_prompt=prompt,
                max_attempts=max_attempts,
            )
            return TypeAdapter(output_format).validate_python(
                json_repair.loads(text)
            )

    async def _generate_response(
        self,
        system_prompt: str | None,
        user_prompt: str | None,
        max_attempts: int = 1,
    ) -> tuple[str, list[dict[str, Any]], int, int]:
        """Call /api/generate with retry logic."""
        async with self._tracker("generate_response"):
            if max_attempts <= 0:
                raise ValueError("max_attempts must be a positive integer")

            prompt = self._build_prompt(system_prompt, user_prompt)
            sleep_seconds = 1
            last_exc: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    text = await self._call_generate(prompt)
                except (
                    httpx.ConnectError,
                    httpx.TimeoutException,
                    httpx.RemoteProtocolError,
                ) as e:
                    last_exc = e
                    if attempt >= max_attempts:
                        error_message = (
                            f"Ollama /api/generate failed after {attempt} attempt(s) "
                            f"due to retryable {type(e).__name__}: "
                            f"max attempts {max_attempts} reached"
                        )
                        logger.exception(error_message)
                        raise ExternalServiceAPIError(error_message) from e

                    logger.info(
                        "Retrying Ollama /api/generate in %d seconds "
                        "after failed attempt %d due to retryable %s...",
                        sleep_seconds,
                        attempt,
                        type(e).__name__,
                    )
                    await asyncio.sleep(sleep_seconds)
                    sleep_seconds = min(
                        sleep_seconds * 2, self._max_retry_interval_seconds
                    )
                except httpx.HTTPStatusError as e:
                    error_message = (
                        f"Ollama /api/generate failed with HTTP {e.response.status_code} "
                        f"after attempt {attempt}"
                    )
                    logger.exception(error_message)
                    raise ExternalServiceAPIError(error_message) from e
                else:
                    return text, [], 0, 0

            raise ExternalServiceAPIError(
                f"Ollama /api/generate failed after {max_attempts} attempt(s)"
            ) from last_exc
