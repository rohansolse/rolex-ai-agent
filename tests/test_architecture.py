"""Guard the dependency rules written down in ARCHITECTURE.md.

`tools` must not import `llm`, and `llm` must not import `tools`. These are
cheap to state and expensive to rediscover once they have been broken, so a
test holds the line.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "rolex"

FORBIDDEN: dict[str, set[str]] = {
    "tools": {"rolex.llm", "rolex.agent", "rolex.interface"},
    "llm": {"rolex.tools", "rolex.agent", "rolex.interface"},
    "core": {"rolex.llm", "rolex.tools", "rolex.agent", "rolex.interface"},
}


def _source_files(package: str) -> list[Path]:
    """Every real .py file in a package.

    macOS writes AppleDouble sidecars ("._module.py") on this external volume.
    They match *.py but are binary, so they are skipped.
    """
    return sorted(p for p in (SRC / package).rglob("*.py") if not p.name.startswith("._"))


def _imports(path: Path) -> set[str]:
    """Absolute module names imported by `path`.

    Relative imports are resolved against the file's own package, because
    `from ..llm.base import X` is exactly how someone would break a layering
    rule without tripping a check that only looked at absolute imports.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = _resolve_relative(path, node.level) if node.level else ""
            parts = [part for part in (prefix, node.module) if part]
            if parts:
                found.add(".".join(parts))
            if node.level:
                found.update(".".join([*parts, alias.name]) for alias in node.names)
    return found


def _resolve_relative(path: Path, level: int) -> str:
    """The absolute package a relative import of `level` dots resolves to."""
    # parts[:-1] drops the filename, giving the containing package. For an
    # __init__.py that package is the module itself, which is what one dot
    # means there too, so both cases resolve the same way.
    package = ["rolex", *path.relative_to(SRC).parts[:-1]]
    up = level - 1
    return ".".join(package[: len(package) - up] if up else package)


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_layer_does_not_import_upward(package: str) -> None:
    banned = FORBIDDEN[package]
    files = _source_files(package)
    assert files, f"no source files found in rolex.{package}"
    for path in files:
        for imported in _imports(path):
            offending = {b for b in banned if imported == b or imported.startswith(f"{b}.")}
            assert not offending, f"{path.relative_to(SRC)} imports {imported}"


def test_guard_detects_absolute_and_relative_imports(tmp_path: Path) -> None:
    """Prove the detector fires. A guard nobody has seen fail proves nothing."""
    module = SRC / "tools" / "_probe.py"
    module.write_text(
        "import rolex.llm\nfrom rolex.llm.base import A\nfrom ..llm.base import B\n",
        encoding="utf-8",
    )
    try:
        found = _imports(module)
        assert "rolex.llm" in found
        assert "rolex.llm.base" in found
        with pytest.raises(AssertionError, match=r"rolex\.llm"):
            test_layer_does_not_import_upward("tools")
    finally:
        module.unlink()
        (module.parent / f"._{module.name}").unlink(missing_ok=True)


def test_relative_import_resolution() -> None:
    assert _resolve_relative(SRC / "tools" / "registry.py", 1) == "rolex.tools"
    assert _resolve_relative(SRC / "tools" / "registry.py", 2) == "rolex"
    assert _resolve_relative(SRC / "tools" / "__init__.py", 1) == "rolex.tools"
