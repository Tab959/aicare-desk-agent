from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek

from aicare_agent_service.config import Environment, Settings
from aicare_agent_service.models.contracts import (
    ModelConfigurationError,
    ModelProfile,
    ModelPurpose,
)
from aicare_agent_service.models.deepseek import DeepSeekModelProvider

TEST_API_KEY = "deepseek-provider-unit-test-secret"


@pytest.fixture
def deepseek_settings() -> Settings:
    return Settings(
        environment=Environment.TEST,
        deepseek_api_key=TEST_API_KEY,
        deepseek_model="deepseek-v4-pro",
        deepseek_base_url="https://deepseek.example.com/v1",
        deepseek_max_retries=3,
        model_timeout_seconds=12.5,
        model_max_output_tokens=1024,
        specialist_timeout_seconds=40.5,
        specialist_max_output_tokens=2048,
        answer_timeout_seconds=45.5,
        answer_max_output_tokens=3072,
        _env_file=None,
    )


@pytest.mark.parametrize(
    ("purpose", "temperature", "timeout_seconds", "max_output_tokens"),
    [
        (ModelPurpose.ROUTING, 0, 12.5, 1024),
        (ModelPurpose.SUMMARY, 0, 12.5, 1024),
        (ModelPurpose.REVIEW, 0, 12.5, 1024),
        (ModelPurpose.SPECIALIST, 0.2, 40.5, 2048),
        (ModelPurpose.ANSWER, 0.2, 45.5, 3072),
    ],
)
def test_each_model_purpose_uses_its_expected_profile(
    deepseek_settings: Settings,
    purpose: ModelPurpose,
    temperature: float,
    timeout_seconds: float,
    max_output_tokens: int,
) -> None:
    model = DeepSeekModelProvider(deepseek_settings).create(purpose)

    assert model.temperature == temperature
    assert model.request_timeout == timeout_seconds
    assert model.max_tokens == max_output_tokens


def test_provider_passes_common_deepseek_settings(deepseek_settings: Settings) -> None:
    model = DeepSeekModelProvider(deepseek_settings).create(ModelPurpose.ROUTING)

    assert isinstance(model, BaseChatModel)
    assert isinstance(model, ChatDeepSeek)
    assert model.model_name == "deepseek-v4-pro"
    assert model.api_base == "https://deepseek.example.com/v1"
    assert model.max_retries == 3


@pytest.mark.parametrize("purpose", list(ModelPurpose))
def test_provider_explicitly_disables_thinking_mode(
    deepseek_settings: Settings,
    purpose: ModelPurpose,
) -> None:
    model = DeepSeekModelProvider(deepseek_settings).create(purpose)

    assert model.extra_body == {"thinking": {"type": "disabled"}}


def test_model_profile_is_immutable() -> None:
    profile = ModelProfile(temperature=0, timeout_seconds=10, max_output_tokens=512)

    with pytest.raises(FrozenInstanceError):
        profile.temperature = 0.5  # type: ignore[misc]


def test_provider_rejects_unknown_model_purpose(deepseek_settings: Settings) -> None:
    provider = DeepSeekModelProvider(deepseek_settings)

    with pytest.raises(ModelConfigurationError, match="不支持的模型用途") as exc_info:
        provider.create(cast(Any, "unknown-purpose"))

    assert TEST_API_KEY not in str(exc_info.value)


def test_provider_and_model_serialization_do_not_expose_key(
    deepseek_settings: Settings,
) -> None:
    provider = DeepSeekModelProvider(deepseek_settings)
    model = provider.create(ModelPurpose.ANSWER)

    rendered_values = (repr(provider), repr(model), model.model_dump_json())

    assert all(TEST_API_KEY not in rendered for rendered in rendered_values)
