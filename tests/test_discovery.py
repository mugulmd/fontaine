"""Adding a model must be adding a file, and nothing else."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from PIL import Image

from fontaine.contracts import Recognizer
from fontaine.recognize import discovery

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

#: These models ignore the pixels, so any crop will do.
A_CROP = Image.new("RGB", (8, 8))

A_MODEL = """
from fontaine.contracts import Recognizer


class Mine(Recognizer):
    \"\"\"Says the same thing every time.\"\"\"

    name = "mine"

    def predict(self, image):
        return "always-this"

    def learn(self, image, label):
        pass
"""


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "models"
    directory.mkdir()
    return directory


def _write(directory: Path, name: str, source: str) -> None:
    (directory / name).write_text(dedent(source).lstrip())


def test_the_shipped_baseline_is_found_like_any_other_model() -> None:
    """The baseline is not privileged: it lives in models/ and is found by scanning."""
    found = discovery.discover(MODEL_DIR)

    assert "baseline" in found
    assert issubclass(found["baseline"], Recognizer)


def test_a_new_file_is_all_it_takes(model_dir: Path) -> None:
    _write(model_dir, "mine.py", A_MODEL)

    model = discovery.load("mine", model_dir)

    assert isinstance(model, Recognizer)
    assert model.predict(A_CROP) == "always-this"


def test_a_model_can_be_a_package_with_its_own_helpers(model_dir: Path) -> None:
    """A real entry outgrows one file, so a directory works as well as a module."""
    package = model_dir / "big"
    package.mkdir()
    (package / "helpers.py").write_text("ANSWER = 'from-a-helper'\n")
    _write(
        package,
        "__init__.py",
        """
        from fontaine.contracts import Recognizer

        from big.helpers import ANSWER


        class Big(Recognizer):
            name = "big"

            def predict(self, image):
                return ANSWER

            def learn(self, image, label):
                pass
        """,
    )

    assert discovery.load("big", model_dir).predict(A_CROP) == "from-a-helper"


def test_files_starting_with_an_underscore_are_not_models(model_dir: Path) -> None:
    """The escape hatch for shared code that is not itself an entry."""
    _write(model_dir, "_shared.py", A_MODEL)

    assert discovery.discover(model_dir) == {}


def test_an_unknown_name_lists_what_is_available(model_dir: Path) -> None:
    _write(model_dir, "mine.py", A_MODEL)

    with pytest.raises(discovery.ModelNotFound, match="mine"):
        discovery.load("typo", model_dir)


def test_a_model_without_a_name_is_an_error_naming_the_file(model_dir: Path) -> None:
    """Silently skipping it would leave --model failing with no explanation."""
    _write(model_dir, "nameless.py", A_MODEL.replace('    name = "mine"\n', ""))

    with pytest.raises(discovery.ModelError, match="nameless"):
        discovery.discover(model_dir)


def test_two_models_claiming_one_name_is_an_error(model_dir: Path) -> None:
    """Otherwise scan order would decide which one --model scores."""
    _write(model_dir, "first.py", A_MODEL)
    _write(model_dir, "second.py", A_MODEL.replace("class Mine", "class Other"))

    with pytest.raises(discovery.ModelError, match="mine"):
        discovery.discover(model_dir)


def test_a_broken_model_file_names_itself(model_dir: Path) -> None:
    """The bare traceback points into importlib, which helps nobody."""
    _write(model_dir, "broken.py", "raise RuntimeError('nope')\n")

    with pytest.raises(discovery.ModelError, match="broken"):
        discovery.discover(model_dir)


def test_two_directories_with_the_same_filename_do_not_collide(tmp_path: Path) -> None:
    """Import caching would otherwise serve the first directory's class for both."""
    first, second = tmp_path / "a", tmp_path / "b"
    for directory, answer in ((first, "from-a"), (second, "from-b")):
        directory.mkdir()
        _write(directory, "mine.py", A_MODEL.replace("always-this", answer))

    assert discovery.load("mine", first).predict(A_CROP) == "from-a"
    assert discovery.load("mine", second).predict(A_CROP) == "from-b"
    assert discovery.load("mine", first).predict(A_CROP) == "from-a"


def test_a_missing_model_directory_says_so() -> None:
    with pytest.raises(discovery.ModelError, match="not found"):
        discovery.discover(Path("no/such/place"))
