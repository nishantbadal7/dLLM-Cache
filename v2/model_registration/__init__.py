import logging
from pathlib import Path

from transformers import AutoConfig, AutoModelForCausalLM

from .model_registration_helper import (
    get_model_config_path as get_model_config_path,
    load_model_registry,
    resolve_registrations,
)

logger = logging.getLogger(f"{__name__}.{Path(__file__).stem}")

REGISTRY_PATH = Path(__file__).parent / "model_registry.json"

def register_models():
    """Load the model registry and register all models with transformers AutoConfig/AutoModel."""
    loaded_registry = load_model_registry(REGISTRY_PATH)

    for model_type, config_class, model_class in resolve_registrations(loaded_registry):
        AutoConfig.register(model_type, config_class)
        AutoModelForCausalLM.register(config_class, model_class)
        logger.info("Registered model_type '%s' with AutoConfig/AutoModelForCausalLM.", model_type)

register_models()
