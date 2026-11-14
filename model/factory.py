from config.settings import AgentSettings
from model.client import ModelClient
from model.nvidia_client import NVIDIAClient
from model.ollama_client import OllamaClient


def create_model_client(
    settings: AgentSettings,
    timeout: int = 600,
) -> ModelClient:
    provider = settings.model_provider.strip().lower()

    if provider == "nvidia":
        return NVIDIAClient(
            model=settings.nvidia_model,
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_endpoint,
            timeout=timeout,
        )

    if provider == "ollama":
        client = OllamaClient(
            model=settings.model,
            base_url=settings.ollama_host,
            timeout=timeout,
        )
        client.ensure_model_available()
        return client

    raise ValueError("MODEL_PROVIDER must be 'nvidia' or 'ollama'.")
