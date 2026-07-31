"""The recognizer must not be able to read the generator's answers.

Every sample carries the exact cap height, contrast, background and text used to
draw it. That is deliberate — it is what makes failures diagnosable — but a model
handed the whole sample could read the label off the metadata instead of looking at
the pixels, and would score wonderfully while learning nothing.

Discipline alone will not hold that line over time, so it is asserted here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parent.parent / "src" / "fontaine"

#: What the recognizer side is allowed to import from the generator side. The data
#: contract and the stream reader, and nothing that produces or describes an item.
ALLOWED_GENERATOR_MODULES = {
    "fontaine.contracts",
    "fontaine.store",
    "fontaine.store.reader",
}
RECOGNIZER_PACKAGES = ("recognize", "evaluate")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _recognizer_files() -> list[Path]:
    return sorted(
        path
        for package in RECOGNIZER_PACKAGES
        for path in (SOURCE / package).rglob("*.py")
        if path.name != "__init__.py"
    )


def test_there_are_recognizer_modules_to_check() -> None:
    """Guard against the test passing because it found nothing."""
    assert _recognizer_files()


@pytest.mark.parametrize("path", _recognizer_files(), ids=lambda path: path.stem)
def test_the_recognizer_only_imports_the_data_contract(path: Path) -> None:
    forbidden = {
        module
        for module in _imports(path)
        if module.startswith("fontaine.")
        and module not in ALLOWED_GENERATOR_MODULES
        and not any(module.startswith(f"fontaine.{package}") for package in RECOGNIZER_PACKAGES)
    }

    assert not forbidden, (
        f"{path.name} imports {sorted(forbidden)} from the generator. The recognizer "
        f"should only need the data contract and the stream reader — anything else "
        f"risks it learning from how items were made rather than from the pixels."
    )


def test_the_featurizer_takes_an_image_not_a_sample() -> None:
    """A featurizer handed a Sample could read the answer out of its metadata."""
    tree = ast.parse((SOURCE / "recognize" / "features.py").read_text())
    describe = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "describe"
    )

    assert [argument.arg for argument in describe.args.args] == ["image"]
