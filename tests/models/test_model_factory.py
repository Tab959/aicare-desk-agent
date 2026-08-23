import pytest

from aicare_agent_service.config import Environment, ModelProviderName, Settings
from aicare_agent_service.models.contracts import (
    ChatModelProvider,
    ModelConfigurationError,
)
from aicare_agent_service.models.deepseek import DeepSeekModelProvider
from aicare_agent_service.models.factory import create_model_provider
from aicare_agent_service.models.fake import FakeModelProvider


def test_factory_creates_deepseek_provider() -> None:
    settings = Settings(
        environment=Environment.TEST,
        model_provider=ModelProviderName.DEEPSEEK,
        deepseek_api_key="factory-test-secret",
        _env_file=None,
    )

    provider = create_model_provider(settings)

    assert isinstance(provider, DeepSeekModelProvider)
    assert isinstance(provider, ChatModelProvider)


@pytest.mark.parametrize("api_key", [None, "", " \t "])
def test_factory_rejects_missing_or_blank_deepseek_key(api_key: str | None) -> None:
    settings = Settings(
        environment=Environment.TEST,
        model_provider=ModelProviderName.DEEPSEEK,
        deepseek_api_key=api_key,
        _env_file=None,
    )

    with pytest.raises(ModelConfigurationError, match="DEEPSEEK_API_KEY") as exc_info:
        create_model_provider(settings)

    assert "factory-test-secret" not in str(exc_info.value)


def test_factory_creates_fake_provider_only_in_test_environment() -> None:
    settings = Settings(
        environment=Environment.TEST,
        model_provider=ModelProviderName.FAKE,
        _env_file=None,
    )

    provider = create_model_provider(settings)

    assert isinstance(provider, FakeModelProvider)
    assert isinstance(provider, ChatModelProvider)


@pytest.mark.parametrize("environment", [Environment.DEVELOPMENT, Environment.PRODUCTION])
def test_factory_rejects_fake_provider_outside_test_environment(
    environment: Environment,
) -> None:
    settings = Settings(
        environment=environment,
        model_provider=ModelProviderName.FAKE,
        _env_file=None,
    )

    with pytest.raises(ModelConfigurationError, match="Fake模型Provider仅允许测试环境"):
        create_model_provider(settings)
