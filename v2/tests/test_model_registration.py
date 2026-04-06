import json
from pathlib import Path
from unittest.mock import patch

import pytest

import model_registration.model_registration_helper as helper
from model_registration.model_registration_helper import (
    get_model_config_path,
    load_model_registry,
    resolve_registrations,
)

def create_registry(tmp_path, models):
    """
    Write a registry JSON and dummy config files under tmp_path.

    models should be a dict like:
        {"org/model-a": {"config_path": "config_a.json", "model_type": "MyModel", ...}}
    """
    registry = {"models": {}}
    for model_name, entry in models.items():
        registry["models"][model_name] = entry
        config_path = tmp_path / entry["config_path"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("{}")

    registry_path = tmp_path / "model_registry.json"
    registry_path.write_text(json.dumps(registry))

    return registry_path

def make_entry(config_path="config.json"):
    """Create a valid model entry with defaults."""
    return {
        "config_path": config_path,
        "model_type": "TestModel",
        "config_class": "some.module.TestConfig",
        "model_class": "some.module.TestModel",
    }

class TestModelRegistration:
    def test_getModelConfigPath_nonStringInput_raisesTypeError(self):
        with pytest.raises(TypeError, match="model_name must be a str"):
            get_model_config_path(123)

        with pytest.raises(TypeError, match="model_name must be a str"):
            get_model_config_path(None)

    def test_getModelConfigPath_registryNotLoaded_raisesRuntimeError(self):
        with patch.object(helper, "registry", {}):
            with pytest.raises(RuntimeError, match="registry has not been loaded"):
                get_model_config_path("any-model")

    def test_getModelConfigPath_invalidModelName_raisesValueError(self):
        fake_registry = {"org/my-model": make_entry()}
        with patch.object(helper, "registry", fake_registry):
            with pytest.raises(ValueError, match="not found in model registry"):
                get_model_config_path("org/nonexistent")

    def test_getModelConfigPath_validModelName_returnsPath(self, tmp_path):
        fake_registry = {
            "org/my-model": {
                "config_path": tmp_path / "my_config.json",
                "model_type": "TestModel",
                "config_class": "some.module.TestConfig",
                "model_class": "some.module.TestModel",
            }
        }
        with patch.object(helper, "registry", fake_registry):
            result = get_model_config_path("org/my-model")

        assert result == tmp_path / "my_config.json"

    def test_loadModelRegistry_nonPathInput_raisesTypeError(self):
        with pytest.raises(TypeError, match="registry_path must be a Path"):
            load_model_registry("not/a/path")

        with pytest.raises(TypeError, match="registry_path must be a Path"):
            load_model_registry(123)

    def test_loadModelRegistry_missingModelsKey_raisesKeyError(self, tmp_path):
        registry_path = tmp_path / "model_registry.json"
        registry_path.write_text(json.dumps({"not_models": {}}))

        with pytest.raises(KeyError, match="must contain a 'models' field"):
            load_model_registry(registry_path)

    def test_loadModelRegistry_missingRegistryFile_raisesFileNotFoundError(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Model registry not found"):
            load_model_registry(tmp_path / "nonexistent.json")

    def test_loadModelRegistry_emptyModels_raisesValueError(self, tmp_path):
        registry_path = create_registry(tmp_path, {})
        with pytest.raises(ValueError, match="'models' in model_registry.json is empty"):
            load_model_registry(registry_path)

    def test_loadModelRegistry_missingRequiredFields_raisesKeyError(self, tmp_path):
        incomplete_entry = {"config_path": "config.json"}
        registry_path = tmp_path / "model_registry.json"
        registry_path.write_text(json.dumps({
            "models": {"org/bad-model": incomplete_entry}
        }))
        (tmp_path / "config.json").write_text("{}")

        with pytest.raises(KeyError, match="missing required fields") as exc_info:
            load_model_registry(registry_path)

        error_message = str(exc_info.value)
        assert "model_type" in error_message
        assert "config_class" in error_message
        assert "model_class" in error_message

    def test_loadModelRegistry_missingConfigFile_raisesFileNotFoundError(self, tmp_path):
        entry = make_entry("nonexistent.json")
        registry_path = tmp_path / "model_registry.json"
        registry_path.write_text(json.dumps({
            "models": {"org/model": entry}
        }))

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_model_registry(registry_path)

    def test_loadModelRegistry_validRegistry_resolvesPathsAndFields(self, tmp_path):
        entry = make_entry("config_a.json")
        registry_path = create_registry(tmp_path, {"org/model-a": entry})
        result = load_model_registry(registry_path)

        assert result["org/model-a"]["config_path"] == (tmp_path / "config_a.json").resolve()
        assert result["org/model-a"]["model_type"] == "TestModel"
        assert result["org/model-a"]["config_class"] == "some.module.TestConfig"
        assert result["org/model-a"]["model_class"] == "some.module.TestModel"

    def test_resolveRegistrations_validClasses_resolvesCorrectly(self):
        import json as json_module

        fake_registry = {
            "org/model-a": {
                "config_path": "/fake/path",
                "model_type": "TestType",
                "config_class": "json.JSONEncoder",
                "model_class": "json.JSONDecoder",
            }
        }

        result = resolve_registrations(fake_registry)

        assert len(result) == 1
        model_type, config_cls, model_cls = result[0]
        assert model_type == "TestType"
        assert config_cls is json_module.JSONEncoder
        assert model_cls is json_module.JSONDecoder

    def test_resolveRegistrations_duplicateModelTypes_registersOnce(self):
        fake_registry = {
            "org/model-7B": {
                "config_path": "/fake/7b",
                "model_type": "SameType",
                "config_class": "json.JSONEncoder",
                "model_class": "json.JSONDecoder",
            },
            "org/model-1.5B": {
                "config_path": "/fake/1.5b",
                "model_type": "SameType",
                "config_class": "json.JSONEncoder",
                "model_class": "json.JSONDecoder",
            },
        }

        result = resolve_registrations(fake_registry)

        assert len(result) == 1
        assert result[0][0] == "SameType"

    def test_resolveRegistrations_multipleModelTypes_registersAll(self):
        fake_registry = {
            "org/model-a": {
                "config_path": "/fake/a",
                "model_type": "TypeA",
                "config_class": "json.JSONEncoder",
                "model_class": "json.JSONDecoder",
            },
            "org/model-b": {
                "config_path": "/fake/b",
                "model_type": "TypeB",
                "config_class": "pathlib.PurePath",
                "model_class": "pathlib.Path",
            },
        }

        result = resolve_registrations(fake_registry)

        assert len(result) == 2
        model_types = {r[0] for r in result}
        assert model_types == {"TypeA", "TypeB"}

    def test_resolveRegistrations_invalidModule_raisesModuleNotFoundError(self):
        fake_registry = {
            "org/model": {
                "config_path": "/fake/path",
                "model_type": "BadType",
                "config_class": "nonexistent.module.SomeClass",
                "model_class": "json.JSONDecoder",
            }
        }

        with pytest.raises(ModuleNotFoundError):
            resolve_registrations(fake_registry)

    def test_resolveRegistrations_invalidClassName_raisesAttributeError(self):
        fake_registry = {
            "org/model": {
                "config_path": "/fake/path",
                "model_type": "BadType",
                "config_class": "json.NonExistentClass",
                "model_class": "json.JSONDecoder",
            }
        }

        with pytest.raises(AttributeError):
            resolve_registrations(fake_registry)

    def test_resolveRegistrations_rsplitHandlesDeepPaths(self):
        import xml.etree.ElementTree as ET

        fake_registry = {
            "org/model": {
                "config_path": "/fake/path",
                "model_type": "DeepType",
                "config_class": "xml.etree.ElementTree.ElementTree",
                "model_class": "xml.etree.ElementTree.Element",
            }
        }

        result = resolve_registrations(fake_registry)

        assert len(result) == 1
        _, config_cls, model_cls = result[0]
        assert config_cls is ET.ElementTree
        assert model_cls is ET.Element
