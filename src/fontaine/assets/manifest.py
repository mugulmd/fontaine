"""The asset manifest: which files a stream is drawn from, pinned by checksum.

Two participants only get comparable scores if they generated from *byte-identical*
assets. Generation is already deterministic — item ``i`` derives its RNG from
``(seed, i)`` — so the assets are the last piece of unpinned input, and a font
silently updated in place changes the pixels without changing anything a diff
would show.

So the checksum, not the URL, is the thing this file exists for. The manifest is
text and lives in git; the bytes live wherever ``release`` points and never enter
the repository's history.

**The release is the only place bytes are fetched from.** Not a convenience: an
asset that could arrive from either the release or its upstream project is an asset
whose bytes depend on which one answered, and upstream re-generates its files —
every one of these twelve fonts already differs from what ``google/fonts`` serves at
``main``, which now ships variable fonts for most of those families. One source
means one possible outcome, and a fetch that succeeds is a fetch that got the same
thing everyone else got.

``source`` records where a face originally came from, for attribution, and is never
requested — the release redistributes the bytes, so the provenance has to be written
down somewhere, but nothing reads it over the network.

Paths are repo-relative, matching the config's ``font_dir`` and ``photo_dir``
defaults, since commands are run from the repo root.
"""

from __future__ import annotations

import fnmatch
import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BeforeValidator, Field, model_validator

from fontaine.config import ConfigBase

DEFAULT_MANIFEST_PATH = Path("assets/manifest.yaml")

#: Read in 1 MiB blocks: large enough that hashing 11 MB of assets is not
#: syscall-bound, small enough that a background photo never lands in memory whole.
_CHUNK = 1024 * 1024

#: Files that live in an asset directory without being assets.
_IGNORED_NAMES = frozenset({".gitkeep", ".DS_Store"})


def _normalized_digest(value: Any) -> Any:
    """Accept a checksum with stray whitespace or upper-case hex.

    A digest of nothing but decimal digits is valid hex, and YAML reads it as an
    integer — losing any leading zero on the way. Vanishingly rare for a real
    digest, but the shape a placeholder takes while a manifest is being drafted, so
    it earns a message that says what to do rather than a bare type error.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        raise ValueError(
            f"YAML read the digest {value} as a number — quote it, so a leading zero survives"
        )
    if isinstance(value, str):
        return value.strip().lower()
    return value


#: A SHA-256 hex digest. Validated on load rather than on first use, so a
#: truncated paste is caught by ``assets status`` instead of halfway through a fetch.
Sha256 = Annotated[
    str,
    BeforeValidator(_normalized_digest),
    Field(pattern=r"^[0-9a-f]{64}$", description="SHA-256 hex digest of the file"),
]


class ManifestError(Exception):
    """A manifest that cannot be turned into a set of fetchable assets."""


class Asset(ConfigBase):
    """One file the stream is generated from, pinned to an exact byte sequence."""

    #: Where the file belongs, relative to the repo root.
    path: Path
    #: SHA-256 of the exact bytes. This is what makes two participants' streams
    #: comparable, and what a fetch verifies before installing anything.
    sha256: Sha256
    #: SPDX-style licence identifier, or the terms in a few words. Recorded because
    #: the release redistributes these bytes, so the terms have to travel with them.
    license: str | None = None
    #: Where the file originally came from, for attribution. Never fetched — the
    #: release is the only source, so this is documentation, not an address.
    source: str | None = None

    @model_validator(mode="after")
    def _check_path_is_relative(self) -> Asset:
        if self.path.is_absolute():
            raise ValueError(f"asset paths are repo-relative, got an absolute one: {self.path}")
        if ".." in self.path.parts:
            raise ValueError(f"asset paths may not climb out of the repo: {self.path}")
        return self

    @property
    def name(self) -> str:
        """The file name, which is also its name on the release."""
        return self.path.name

    def resolve(self, root: Path) -> Path:
        """Absolute location of this asset under ``root``."""
        return root / self.path


class AssetManifest(ConfigBase):
    """Every asset a stream is generated from."""

    #: Base URL of the release holding every asset, named by file name. Required,
    #: and the only address a fetch will use. A GitHub release is the intended
    #: shape: its assets are served outside git history, so the bytes are versioned
    #: and downloadable without ever being committed. ``{name}`` is substituted if
    #: present, otherwise appended.
    release: str
    #: Seconds to wait on a stalled connection before giving up.
    timeout: Annotated[float, Field(gt=0)] = 30.0
    assets: tuple[Asset, ...] = ()

    @model_validator(mode="after")
    def _check_names_are_unique(self) -> AssetManifest:
        """Reject a manifest that pins one path twice or two paths under one name.

        File names must be unique even across directories: a release is a flat
        namespace — it cannot hold two assets of one name — so a collision there
        would have one asset silently overwrite the other's bytes.
        """
        seen_paths: set[Path] = set()
        seen_names: dict[str, Path] = {}
        for asset in self.assets:
            if asset.path in seen_paths:
                raise ValueError(f"{asset.path} is pinned twice")
            seen_paths.add(asset.path)

            clash = seen_names.get(asset.name)
            if clash is not None:
                raise ValueError(
                    f"{asset.path} and {clash} share the file name {asset.name!r}, which the "
                    f"release's flat namespace cannot hold — rename one of them"
                )
            seen_names[asset.name] = asset.path
        return self

    def url_for(self, asset: Asset) -> str:
        """The one URL this asset is fetched from."""
        return (
            self.release.replace("{name}", asset.name)
            if "{name}" in self.release
            else f"{self.release.rstrip('/')}/{asset.name}"
        )

    @property
    def directories(self) -> tuple[Path, ...]:
        """The asset directories this manifest covers, in first-seen order.

        Derived from the assets rather than configured, so adding the first asset
        under a new directory brings that directory under the untracked-file check
        without a second place to edit.
        """
        seen: dict[Path, None] = {}
        for asset in self.assets:
            seen.setdefault(asset.path.parent, None)
        return tuple(seen)

    def matching(self, pattern: str | None) -> tuple[Asset, ...]:
        """The assets whose path or file name matches a glob; all of them for ``None``."""
        if pattern is None:
            return self.assets
        return tuple(
            asset
            for asset in self.assets
            if fnmatch.fnmatch(str(asset.path), pattern) or fnmatch.fnmatch(asset.name, pattern)
        )


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> AssetManifest:
    """Load and validate an asset manifest."""
    if not path.is_file():
        raise ManifestError(f"asset manifest not found: {path}")
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as error:
        raise ManifestError(f"{path} is not valid YAML: {error}") from error
    try:
        return AssetManifest.model_validate(data)
    except ValueError as error:
        raise ManifestError(f"{path} is not a usable manifest: {error}") from error


def digest(path: Path) -> str:
    """SHA-256 of a file, read in blocks so a large photo is never held in memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()


class State(StrEnum):
    """What is on disk, relative to what the manifest pins."""

    #: Present with the pinned bytes. Nothing to do.
    OK = "ok"
    #: Not on disk at all — the state of a fresh clone.
    MISSING = "missing"
    #: Present with different bytes. Either a deliberate change whose checksum has
    #: not been updated yet, or the exact drift that makes two runs incomparable.
    CHANGED = "changed"


@dataclass(frozen=True, slots=True)
class Check:
    """The verdict on one asset. Returned to callers, never serialized."""

    asset: Asset
    state: State
    #: The digest actually found, when there was a file to hash. This is the value
    #: to paste into the manifest when the change was intended.
    actual: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the file on disk is the one the manifest pins."""
        return self.state is State.OK


def check(asset: Asset, root: Path = Path()) -> Check:
    """Compare one asset's bytes on disk against its pinned checksum."""
    location = asset.resolve(root)
    if not location.is_file():
        return Check(asset, State.MISSING)
    found = digest(location)
    state = State.OK if found == asset.sha256 else State.CHANGED
    return Check(asset, state, actual=found)


def check_all(manifest: AssetManifest, root: Path = Path()) -> tuple[Check, ...]:
    """Verify every asset in the manifest against disk. Reads files, not the network."""
    return tuple(check(asset, root) for asset in manifest.assets)


def untracked(manifest: AssetManifest, root: Path = Path()) -> tuple[Path, ...]:
    """Files sitting in an asset directory that the manifest does not pin.

    Worth reporting rather than ignoring: an extra font in ``assets/fonts/`` enters
    the label space, so it changes the classes and every score computed over them,
    while every checksum in the manifest still verifies. It is the one way to be out
    of sync that a checksum cannot catch.
    """
    pinned = {asset.path for asset in manifest.assets}
    found: list[Path] = []
    for directory in manifest.directories:
        location = root / directory
        if not location.is_dir():
            continue
        for entry in sorted(location.rglob("*")):
            if not entry.is_file() or entry.name in _IGNORED_NAMES or entry.name.startswith("."):
                continue
            relative = entry.relative_to(root)
            if relative not in pinned:
                found.append(relative)
    return tuple(found)


def entry_yaml(path: Path, *, root: Path = Path()) -> str:
    """A paste-ready manifest entry for a file already on disk.

    The licence and source are left as empty keys rather than guessed: the release
    redistributes these bytes, so those two lines are the ones a human has to fill
    in, and an entry that looks complete would not get filled in at all.
    """
    return yaml.safe_dump(
        [
            {
                "path": str(path),
                "sha256": digest(root / path),
                "license": None,
                "source": None,
            }
        ],
        sort_keys=False,
        default_flow_style=False,
    ).rstrip()
