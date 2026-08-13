"""Downloading pinned assets and refusing to install anything unexpected.

The order here is the whole point: hash while streaming, verify against the
manifest, and only then move the file into place. A truncated download or a release
serving the wrong bytes leaves the asset directory exactly as it was rather than
half-populated, so a failed fetch cannot be mistaken for a successful one — the
same reason ``store.writer`` writes the stream manifest last.
"""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fontaine.assets.manifest import Asset, AssetManifest, State, check

#: Streamed in 256 KiB blocks, so hashing overlaps the download instead of
#: following it and no background photo is ever held in memory whole.
_CHUNK = 256 * 1024

#: Some CDNs reject the default ``Python-urllib/3`` agent outright.
_HEADERS = {"User-Agent": "fontaine-assets/1 (+https://github.com/mugulmd/fontaine)"}

#: A manifest is repo data rather than user input, but it is data all the same, and
#: ``file://`` would turn a fetch into a local copy that verifies happily.
_ALLOWED_SCHEMES = frozenset({"http", "https"})


class FetchError(Exception):
    """One asset could not be fetched, or arrived with the wrong bytes."""


@dataclass(frozen=True, slots=True)
class Fetched:
    """An asset that is now on disk with the pinned bytes."""

    asset: Asset
    #: Where it came from, or ``None`` when it was already correct and left alone.
    url: str | None
    n_bytes: int

    @property
    def skipped(self) -> bool:
        """Whether this asset was already present and verified, so nothing was downloaded."""
        return self.url is None


@dataclass(frozen=True, slots=True)
class Failure:
    """An asset that could not be installed, and why."""

    asset: Asset
    url: str
    reason: str


class FetchFailed(FetchError):
    """One asset could not be installed. Carries the detail for the report."""

    def __init__(self, failure: Failure) -> None:
        super().__init__(f"{failure.asset.path}: {failure.reason}")
        self.failure = failure


@dataclass(frozen=True, slots=True)
class FetchReport:
    """What a fetch run did. Returned to the caller, never serialized."""

    downloaded: tuple[Fetched, ...]
    skipped: tuple[Fetched, ...]
    failed: tuple[Failure, ...]

    @property
    def n_bytes(self) -> int:
        """Total bytes actually transferred."""
        return sum(item.n_bytes for item in self.downloaded)


def _download(url: str, destination: Path, expected: str, *, timeout: float) -> int:
    """Stream ``url`` into ``destination``, verifying the digest before installing.

    The bytes land in a sibling ``.part`` file, which is replaced into position only
    once the digest matches. Same directory, so the final move is an atomic rename
    rather than a copy across filesystems.
    """
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise FetchError(f"refusing to fetch over {scheme or 'no'} scheme: {url}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    hasher = hashlib.sha256()
    written = 0

    try:
        # The scheme is checked above, so this cannot be turned into a local file read.
        request = urllib.request.Request(url, headers=_HEADERS)
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            partial.open("wb") as handle,
        ):
            while chunk := response.read(_CHUNK):
                hasher.update(chunk)
                handle.write(chunk)
                written += len(chunk)
    except urllib.error.HTTPError as error:
        partial.unlink(missing_ok=True)
        raise FetchError(f"HTTP {error.code}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        partial.unlink(missing_ok=True)
        raise FetchError(f"{type(error).__name__}: {error}") from error

    found = hasher.hexdigest()
    if found != expected:
        partial.unlink(missing_ok=True)
        raise FetchError(
            f"checksum mismatch — expected {expected[:12]}…, got {found[:12]}… "
            f"({written:,} bytes). The manifest and this URL disagree about the bytes"
        )

    partial.replace(destination)
    return written


def fetch_asset(
    asset: Asset,
    manifest: AssetManifest,
    root: Path = Path(),
    *,
    force: bool = False,
) -> Fetched:
    """Ensure one asset is on disk with its pinned bytes.

    A file already matching its checksum is left untouched — a fetch is safe to
    re-run and costs nothing on a warm asset directory. ``force`` re-downloads
    anyway, which is what to reach for when a file is corrupt in a way that
    happens to hash correctly, which is to say almost never.
    """
    if not force and check(asset, root).ok:
        return Fetched(asset, url=None, n_bytes=asset.resolve(root).stat().st_size)

    url = manifest.url_for(asset)
    try:
        written = _download(url, asset.resolve(root), asset.sha256, timeout=manifest.timeout)
    except FetchError as error:
        raise FetchFailed(Failure(asset, url, str(error))) from error
    return Fetched(asset, url=url, n_bytes=written)


def fetch_all(
    manifest: AssetManifest,
    root: Path = Path(),
    *,
    assets: tuple[Asset, ...] | None = None,
    force: bool = False,
    on_start: Callable[[Asset], None] | None = None,
    on_done: Callable[[Asset], None] | None = None,
) -> FetchReport:
    """Fetch every asset, collecting failures instead of stopping at the first.

    One dead URL should not hide the state of the other nineteen: a participant
    setting up wants the whole list of what is wrong, not the first line of it.
    """
    targets = manifest.assets if assets is None else assets
    downloaded: list[Fetched] = []
    skipped: list[Fetched] = []
    failed: list[Failure] = []

    for asset in targets:
        if on_start is not None:
            on_start(asset)
        try:
            result = fetch_asset(asset, manifest, root, force=force)
        except FetchFailed as error:
            failed.append(error.failure)
        else:
            (skipped if result.skipped else downloaded).append(result)
        if on_done is not None:
            on_done(asset)

    return FetchReport(tuple(downloaded), tuple(skipped), tuple(failed))


def summarize(report: FetchReport) -> str:
    """A one-line summary of a fetch run."""
    parts = [f"{len(report.downloaded)} downloaded"]
    if report.downloaded:
        parts[0] += f" ({report.n_bytes / 1_048_576:.1f} MB)"
    if report.skipped:
        parts.append(f"{len(report.skipped)} already verified")
    if report.failed:
        parts.append(f"{len(report.failed)} failed")
    return ", ".join(parts)


def describe_state(state: State) -> str:
    """What each on-disk state means, for the status table."""
    return {
        State.OK: "pinned bytes",
        State.MISSING: "not on disk — run `fontaine assets fetch`",
        State.CHANGED: "different bytes than pinned",
    }[state]
