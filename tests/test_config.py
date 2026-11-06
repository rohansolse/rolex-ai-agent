"""Config layering, coercion and error messages."""

from pathlib import Path

import pytest

from rolex.core.config import Config, load_config
from rolex.core.errors import ConfigError


def test_defaults_when_nothing_is_set(tmp_path: Path) -> None:
    config = load_config(root=tmp_path, env={})
    assert config.provider == "local"
    assert config.model == "qwen2.5:3b"
    assert config.ollama_models_dir == Path("/Volumes/DEVHAN/ollama-models")


def test_toml_overrides_defaults(tmp_path: Path) -> None:
    (tmp_path / "rolex.toml").write_text('[rolex]\nmodel = "llama3.2:3b"\n')
    assert load_config(root=tmp_path, env={}).model == "llama3.2:3b"


def test_env_overrides_toml(tmp_path: Path) -> None:
    (tmp_path / "rolex.toml").write_text('[rolex]\nmodel = "from-toml"\n')
    config = load_config(root=tmp_path, env={"ROLEX_MODEL": "from-env"})
    assert config.model == "from-env"


def test_dotenv_is_read_but_loses_to_real_env(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text('ROLEX_MODEL="from-dotenv"\nGEMINI_API_KEY=abc123\n')
    config = load_config(root=tmp_path, env={"ROLEX_MODEL": "from-env"})
    assert config.model == "from-env"
    assert config.gemini_api_key == "abc123"


def test_dotenv_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("\n# a comment\n\nROLEX_MODEL=quiet\n")
    assert load_config(root=tmp_path, env={}).model == "quiet"


def test_blank_env_value_does_not_override(tmp_path: Path) -> None:
    config = load_config(root=tmp_path, env={"ROLEX_MODEL": "   "})
    assert config.model == "qwen2.5:3b"


def test_models_dir_expands_user(tmp_path: Path) -> None:
    config = load_config(root=tmp_path, env={"OLLAMA_MODELS": "~/models"})
    assert config.ollama_models_dir == Path.home() / "models"


def test_timeout_is_coerced_to_float(tmp_path: Path) -> None:
    assert load_config(root=tmp_path, env={"ROLEX_REQUEST_TIMEOUT": "30"}).request_timeout == 30.0


@pytest.mark.parametrize("value", ["nope", "0", "-5"])
def test_bad_timeout_is_rejected(tmp_path: Path, value: str) -> None:
    with pytest.raises(ConfigError, match="request_timeout"):
        load_config(root=tmp_path, env={"ROLEX_REQUEST_TIMEOUT": value})


def test_unknown_provider_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="provider must be one of"):
        load_config(root=tmp_path, env={"ROLEX_PROVIDER": "chatgpt"})


def test_unknown_toml_key_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "rolex.toml").write_text('[rolex]\nmdoel = "typo"\n')
    with pytest.raises(ConfigError, match="unknown setting"):
        load_config(root=tmp_path, env={})


def test_malformed_toml_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "rolex.toml").write_text("[rolex\n")
    with pytest.raises(ConfigError):
        load_config(root=tmp_path, env={})


def test_malformed_dotenv_line_is_rejected(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("this line has no equals sign\n")
    with pytest.raises(ConfigError, match="expected KEY=value"):
        load_config(root=tmp_path, env={})


def test_config_is_frozen() -> None:
    with pytest.raises(AttributeError):
        Config().model = "nope"  # type: ignore[misc]


def test_gemini_key_is_demanded_with_a_useful_message() -> None:
    with pytest.raises(ConfigError, match=r"aistudio\.google\.com"):
        Config().require_gemini_key()


def test_gemini_key_is_returned_when_present() -> None:
    assert Config(gemini_api_key="k").require_gemini_key() == "k"


def test_env_values_are_stripped(tmp_path: Path) -> None:
    """Otherwise a trailing space becomes part of the model name, invisibly."""
    assert load_config(root=tmp_path, env={"ROLEX_MODEL": " qwen2.5:3b "}).model == "qwen2.5:3b"


def test_dotenv_strips_an_inline_comment(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("ROLEX_MODEL=qwen2.5:3b  # the small one\n")
    assert load_config(root=tmp_path, env={}).model == "qwen2.5:3b"


def test_dotenv_keeps_a_hash_inside_quotes(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text('GEMINI_API_KEY="abc#123"\n')
    assert load_config(root=tmp_path, env={}).gemini_api_key == "abc#123"


def test_dotenv_keeps_a_hash_with_no_space_before_it(tmp_path: Path) -> None:
    """`a#b` is a value, not a comment — only ` #` starts one."""
    (tmp_path / ".env").write_text("GEMINI_API_KEY=abc#123\n")
    assert load_config(root=tmp_path, env={}).gemini_api_key == "abc#123"


def test_temperature_defaults_low(tmp_path: Path) -> None:
    """Tool choice should not be a coin flip."""
    assert load_config(root=tmp_path, env={}).temperature == 0.2


def test_temperature_is_coerced(tmp_path: Path) -> None:
    assert load_config(root=tmp_path, env={"ROLEX_TEMPERATURE": "0"}).temperature == 0.0


@pytest.mark.parametrize("value", ["hot", "-1", "2.5"])
def test_bad_temperature_is_rejected(tmp_path: Path, value: str) -> None:
    with pytest.raises(ConfigError, match="temperature"):
        load_config(root=tmp_path, env={"ROLEX_TEMPERATURE": value})
