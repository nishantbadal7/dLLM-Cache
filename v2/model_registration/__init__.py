import logging
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from .configuration import Fast_dLLM_QwenConfig
from .modeling import Fast_dLLM_QwenForCausalLM

logger = logging.getLogger(f"{__name__}.{Path(__file__).stem}")

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
MODEL_CONFIG_PATHS = {
    "Efficient-Large-Model/Fast_dLLM_v2_7B": CONFIGS_DIR / "model_config_7b.json",
    "Efficient-Large-Model/Fast_dLLM_v2_1.5B": CONFIGS_DIR / "model_config_1_5b.json",
}

AutoConfig.register("Fast_dLLM_Qwen", Fast_dLLM_QwenConfig)
AutoModelForCausalLM.register(Fast_dLLM_QwenConfig, Fast_dLLM_QwenForCausalLM)

def get_model_config_path(model_name: str) -> Path:
    """Return config path for a model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Path to the local config JSON file.

    Raises:
        TypeError: If model_name is not a str.
        ValueError: If model_name is not in MODEL_CONFIG_PATHS.
    """

    logger.debug("Resolving config path for '%s'.", model_name)

    if not isinstance(model_name, str):
        logger.error("model_name must be a str, got %s.", type(model_name).__name__)
        raise TypeError(
            f"model_name must be a str, got {type(model_name).__name__}."
        )

    if model_name not in MODEL_CONFIG_PATHS:
        logger.error("'%s' not found in MODEL_CONFIG_PATHS.", model_name)
        raise ValueError(
            f"'{model_name}' not found in MODEL_CONFIG_PATHS. "
            f"Available: {list(MODEL_CONFIG_PATHS.keys())}."
        )

    config_path = MODEL_CONFIG_PATHS[model_name]
    logger.info("For model '%s' using config path '%s'.", model_name, config_path)
    
    return config_path
