"""Finding the recognizers a challenger wrote, without the framework knowing them.

Everything under ``models/`` is imported and searched for
:class:`~fontaine.contracts.Recognizer` subclasses. So adding a model is adding a
file — no registration call to remember, no import to add to a list, nothing
inside ``fontaine`` to touch. That is the property this repo is built around: a
new model is purely additive, and cannot perturb anyone else's.

The model directory goes on ``sys.path`` before anything is imported, so a model
that outgrows one file can sit in its own package (``models/my_cnn/__init__.py``)
and import its own helpers normally.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Iterator
from pathlib import Path

from fontaine.contracts import Recognizer

#: Where models live, relative to the repo root. Commands are run from there.
DEFAULT_MODEL_DIR = Path("models")


class ModelError(Exception):
    """A model directory that cannot be turned into a set of recognizers."""


class ModelNotFound(ModelError):
    """No recognizer answers to the requested name."""


def _module_names(model_dir: Path) -> Iterator[str]:
    """Importable top-level names in ``model_dir``: modules and packages alike.

    Leading underscores and dots are skipped, which is the escape hatch for
    shared helpers that are not models themselves.
    """
    for entry in sorted(model_dir.iterdir()):
        if entry.name.startswith(("_", ".")):
            continue
        if entry.is_file() and entry.suffix == ".py":
            yield entry.stem
        elif entry.is_dir() and (entry / "__init__.py").is_file():
            yield entry.name


def _evict_foreign(module_name: str, root: Path) -> None:
    """Forget a cached module of this name that came from somewhere else.

    Two model directories can hold a ``baseline.py`` each, and without this the
    second scan in a process would silently hand back the first one's class.
    """
    cached = sys.modules.get(module_name)
    if cached is None:
        return
    origin = getattr(cached, "__file__", None)
    if origin is not None and Path(origin).resolve().is_relative_to(root):
        return
    for name in [
        name for name in sys.modules if name == module_name or name.startswith(f"{module_name}.")
    ]:
        del sys.modules[name]


def discover(model_dir: Path = DEFAULT_MODEL_DIR) -> dict[str, type[Recognizer]]:
    """Map ``name`` → recognizer class for every model in ``model_dir``.

    Import errors are re-raised naming the file that failed, since the traceback
    otherwise points into importlib and leaves the reader guessing which model
    broke.
    """
    if not model_dir.is_dir():
        raise ModelError(f"model directory not found: {model_dir}")

    # Front of the path, every time: a second scan of a different directory must
    # not be shadowed by the first one still sitting ahead of it.
    root = model_dir.resolve()
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))

    found: dict[str, type[Recognizer]] = {}
    for module_name in _module_names(model_dir):
        _evict_foreign(module_name, root)
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            raise ModelError(f"{model_dir / module_name} failed to import: {error}") from error

        for _, candidate in inspect.getmembers(module, inspect.isclass):
            if not issubclass(candidate, Recognizer) or inspect.isabstract(candidate):
                continue
            # Classes imported *into* a model file, rather than defined in it,
            # would otherwise be registered once per file that imports them.
            if candidate.__module__ != module.__name__:
                continue
            name = getattr(candidate, "name", None)
            if not name:
                raise ModelError(
                    f"{model_dir / module_name}: {candidate.__qualname__} is a Recognizer "
                    f"but declares no `name`, so --model has nothing to match"
                )
            if name in found and found[name] is not candidate:
                raise ModelError(
                    f"two models both call themselves {name!r}: "
                    f"{found[name].__qualname__} and {candidate.__qualname__}"
                )
            found[name] = candidate
    return found


def load(name: str, model_dir: Path = DEFAULT_MODEL_DIR) -> Recognizer:
    """Instantiate the recognizer registered under ``name``.

    Models are constructed with no arguments: hyperparameters belong in the
    model's own file, where they are versioned with it, rather than in a CLI flag
    the framework would have to know the shape of.
    """
    available = discover(model_dir)
    if name not in available:
        known = ", ".join(sorted(available)) or "none"
        raise ModelNotFound(f"no model named {name!r} in {model_dir} — available: {known}")
    return available[name]()


def describe(model: type[Recognizer]) -> str:
    """The first line of a model's docstring, for the listing."""
    doc = inspect.getdoc(model) or ""
    return doc.splitlines()[0] if doc else ""
