import importlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(f"{__name__}.{Path(__file__).stem}")

REQUIRED_MODEL_FIELDS = {"config_path", "model_type", "config_class", "model_class"}

registry = {}

def get_model_config_path(model_name: str) -> Path:
    """
    Return config path for a model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Path to the local config JSON file.

    Raises:
        TypeError: If model_name is not a str.
        ValueError: If model_name is not in the registry.
        RuntimeError: If the registry hasn't been loaded yet.
    """
    logger.debug("Resolving config path for '%s'.", model_name)

    if not isinstance(model_name, str):
        logger.error("model_name must be a str, got %s.", type(model_name).__name__)
        raise TypeError(
            f"model_name must be a str, got {type(model_name).__name__}."
        )

    if len(registry) == 0:
        raise RuntimeError(
            "Model registry has not been loaded. "
            "Ensure model_registration is imported before calling get_model_config_path."
        )

    if model_name not in registry:
        logger.error("'%s' not found in model registry.", model_name)
        raise ValueError(
            f"'{model_name}' not found in model registry. "
            f"Available: {list(registry.keys())}."
        )

    config_path = registry[model_name]["config_path"]
    logger.info("For model '%s' using config path '%s'.", model_name, config_path)

    return config_path


def load_model_registry(registry_path: Path) -> dict[str, dict]:
    """
    Load the model registry JSON, validate fields, and resolve config paths.

    Args:
        registry_path: Path to the registry JSON file.

    Returns:
        Dict mapping model names to their registry entries.
        Each entry has: config_path (Path), model_type (str),
        config_class (str), model_class (str).

    Raises:
        TypeError: If registry_path is not a Path.
        FileNotFoundError: If registry_path does not exist.
        KeyError: If 'models' field is missing or model entries have missing fields.
        ValueError: If 'models' is empty.
    """
    logger.debug("Loading model registry from '%s'.", registry_path)

    if not isinstance(registry_path, Path):
        logger.error("registry_path must be a Path, got %s.", type(registry_path).__name__)
        raise TypeError(
            f"registry_path must be a Path, got {type(registry_path).__name__}."
        )

    if not registry_path.exists():
        logger.error("Model registry not found at '%s'.", registry_path)
        raise FileNotFoundError(
            f"Model registry not found at '{registry_path}'."
        )

    with open(registry_path) as registry_file:
        registry_file_content = json.load(registry_file)

    models = registry_file_content.get("models")
    if models is None:
        logger.error("Missing 'models' in registry.")
        raise KeyError(
            "model_registry.json must contain a 'models' field "
            "mapping model names to their configuration."
        )

    if len(models) == 0:
        logger.error("'models' in registry is empty.")
        raise ValueError(
            "'models' in model_registry.json is empty. "
            "Add at least one model entry."
        )

    resolved_models = {}
    for name, entry in models.items():
        missing_fields = REQUIRED_MODEL_FIELDS - set(entry.keys())
        if missing_fields:
            logger.error("Model '%s' is missing fields: %s.", name, missing_fields)
            raise KeyError(
                f"Model '{name}' is missing required fields: {missing_fields}. "
                f"Each model must have: {REQUIRED_MODEL_FIELDS}."
            )

        config_path = Path(entry["config_path"])
        if config_path.is_absolute():
            resolved_path = config_path
        else:
            resolved_path = (registry_path.parent / config_path).resolve()

        if not resolved_path.exists():
            logger.error("Config file not found for model '%s' at '%s'.", name, resolved_path)
            raise FileNotFoundError(
                f"Config file not found for model '{name}' at '{resolved_path}'."
            )

        resolved_models[name] = {
            "config_path": resolved_path,
            "model_type": entry["model_type"],
            "config_class": entry["config_class"],
            "model_class": entry["model_class"],
        }
        logger.debug("Registered model '%s' -> '%s'.", name, resolved_path)

    logger.info("Loaded %d model(s) from registry.", len(resolved_models))

    registry.clear()
    registry.update(resolved_models)

    return registry


def resolve_registrations(loaded_registry: dict) -> list[tuple]:
    """
    Resolve dotted class paths to actual Python classes for each unique model type.

    Args:
        loaded_registry: Output of load_model_registry.

    Returns:
        List of (model_type, config_cls, model_cls) tuples, one per unique model type.
    """
    registered_model_types = set()
    resolved_registrations = []

    for name, entry in loaded_registry.items():
        model_type = entry["model_type"]

        if model_type in registered_model_types:
            continue

        config_module_path, config_class_name = entry["config_class"].rsplit(".", 1)
        model_module_path, model_class_name = entry["model_class"].rsplit(".", 1)

        config_module = importlib.import_module(config_module_path)
        model_module = importlib.import_module(model_module_path)

        config_class = getattr(config_module, config_class_name)
        model_class = getattr(model_module, model_class_name)

        resolved_registrations.append((model_type, config_class, model_class))
        registered_model_types.add(model_type)
        logger.debug("Resolved classes for model_type '%s'.", model_type)

    return resolved_registrations
