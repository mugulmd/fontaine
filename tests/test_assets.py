"""The asset manifest, and the fetch that refuses to install the wrong bytes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from fontaine.assets import fetch as asset_fetch
from fontaine.assets import manifest as asset_manifest
from fontaine.assets.manifest import Asset, AssetManifest, ManifestError, State

PAYLOAD = b"not really a font, but bytes all the same"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def write_manifest(root: Path, data: dict) -> Path:
    """Write a manifest under ``root`` and return its path."""
    path = root / "assets" / "manifest.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def one_asset(**overrides) -> dict:
    entry = {"path": "assets/fonts/Fake-Regular.ttf", "sha256": DIGEST}
    entry.update(overrides)
    return {"release": "https://example.invalid/assets", "assets": [entry]}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A repo root with an empty asset directory."""
    (tmp_path / "assets" / "fonts").mkdir(parents=True)
    return tmp_path


class TestManifestSchema:
    def test_loads_paths_checksums_and_provenance(self, root: Path) -> None:
        path = write_manifest(root, one_asset(license="OFL-1.1", source="https://example.test"))
        manifest = asset_manifest.load_manifest(path)

        (asset,) = manifest.assets
        assert asset.path == Path("assets/fonts/Fake-Regular.ttf")
        assert asset.sha256 == DIGEST
        assert asset.license == "OFL-1.1"

    def test_upper_case_and_padded_digests_are_normalized(self, root: Path) -> None:
        path = write_manifest(root, one_asset(sha256=f"  {DIGEST.upper()}\n"))
        (asset,) = asset_manifest.load_manifest(path).assets
        assert asset.sha256 == DIGEST

    def test_a_truncated_digest_is_rejected_on_load(self, root: Path) -> None:
        path = write_manifest(root, one_asset(sha256=DIGEST[:40]))
        with pytest.raises(ManifestError):
            asset_manifest.load_manifest(path)

    def test_an_unknown_key_is_an_error(self, root: Path) -> None:
        # Same reasoning as the stream config: a mistyped option is the mistake
        # whose reasonable-looking default is hardest to notice.
        path = write_manifest(root, one_asset(licence="OFL-1.1"))
        with pytest.raises(ManifestError):
            asset_manifest.load_manifest(path)

    def test_the_same_path_pinned_twice_is_an_error(self, root: Path) -> None:
        data = one_asset()
        data["assets"] = [data["assets"][0], dict(data["assets"][0])]
        with pytest.raises(ManifestError, match="pinned twice"):
            asset_manifest.load_manifest(write_manifest(root, data))

    def test_two_directories_may_not_share_a_file_name(self, root: Path) -> None:
        # A release is a flat namespace, so a collision there would have one asset
        # silently overwrite the other's bytes.
        data = one_asset()
        data["assets"].append({"path": "assets/backgrounds/Fake-Regular.ttf", "sha256": DIGEST})
        with pytest.raises(ManifestError, match="share the file name"):
            asset_manifest.load_manifest(write_manifest(root, data))

    def test_a_manifest_with_no_release_is_an_error(self, root: Path) -> None:
        # The release is the only source, so a manifest without one pins bytes
        # that cannot be fetched from anywhere.
        data = one_asset()
        del data["release"]
        with pytest.raises(ManifestError):
            asset_manifest.load_manifest(write_manifest(root, data))

    def test_a_per_asset_url_is_not_an_accepted_key(self, root: Path) -> None:
        # Deliberately no override: an asset that could arrive from the release or
        # from upstream is an asset whose bytes depend on which one answered.
        data = one_asset(url="https://fonts.google.test/Fake-Regular.ttf")
        with pytest.raises(ManifestError):
            asset_manifest.load_manifest(write_manifest(root, data))

    def test_an_absolute_path_is_rejected(self, root: Path) -> None:
        with pytest.raises(ManifestError):
            asset_manifest.load_manifest(write_manifest(root, one_asset(path="/etc/passwd")))

    def test_a_path_climbing_out_of_the_repo_is_rejected(self, root: Path) -> None:
        with pytest.raises(ManifestError):
            asset_manifest.load_manifest(write_manifest(root, one_asset(path="../elsewhere.ttf")))

    def test_a_missing_manifest_names_the_path(self, root: Path) -> None:
        with pytest.raises(ManifestError, match="not found"):
            asset_manifest.load_manifest(root / "nope.yaml")


class TestUrlFor:
    def test_the_release_url_appends_the_file_name(self) -> None:
        manifest = AssetManifest(release="https://rel.test/v1/")
        asset = Asset(path=Path("assets/backgrounds/forest.png"), sha256=DIGEST)
        assert manifest.url_for(asset) == "https://rel.test/v1/forest.png"

    def test_a_template_is_substituted_rather_than_appended(self) -> None:
        manifest = AssetManifest(release="https://rel.test/get?f={name}")
        asset = Asset(path=Path("assets/fonts/A.ttf"), sha256=DIGEST)
        assert manifest.url_for(asset) == "https://rel.test/get?f=A.ttf"

    def test_the_directory_does_not_reach_the_url(self) -> None:
        # A release is flat: two assets under different directories would collide,
        # which is why the manifest refuses to load when their names match.
        manifest = AssetManifest(release="https://rel.test/v1")
        deep = Asset(path=Path("assets/fonts/vendor/A.ttf"), sha256=DIGEST)
        assert manifest.url_for(deep) == "https://rel.test/v1/A.ttf"


class TestCheck:
    def test_matching_bytes_are_ok(self, root: Path) -> None:
        manifest = asset_manifest.load_manifest(write_manifest(root, one_asset()))
        (root / "assets/fonts/Fake-Regular.ttf").write_bytes(PAYLOAD)

        (result,) = asset_manifest.check_all(manifest, root)
        assert result.ok
        assert result.state is State.OK

    def test_an_absent_file_is_missing_and_has_no_digest(self, root: Path) -> None:
        manifest = asset_manifest.load_manifest(write_manifest(root, one_asset()))
        (result,) = asset_manifest.check_all(manifest, root)
        assert result.state is State.MISSING
        assert result.actual is None

    def test_different_bytes_report_the_digest_found(self, root: Path) -> None:
        manifest = asset_manifest.load_manifest(write_manifest(root, one_asset()))
        (root / "assets/fonts/Fake-Regular.ttf").write_bytes(b"a different font")

        (result,) = asset_manifest.check_all(manifest, root)
        assert result.state is State.CHANGED
        assert result.actual == hashlib.sha256(b"a different font").hexdigest()


class TestUntracked:
    def test_an_unpinned_file_beside_a_pinned_one_is_reported(self, root: Path) -> None:
        manifest = asset_manifest.load_manifest(write_manifest(root, one_asset()))
        (root / "assets/fonts/Fake-Regular.ttf").write_bytes(PAYLOAD)
        (root / "assets/fonts/Sneaky-Regular.ttf").write_bytes(b"extra class")

        assert asset_manifest.untracked(manifest, root) == (
            Path("assets/fonts/Sneaky-Regular.ttf"),
        )

    def test_placeholders_and_dotfiles_are_not_untracked_assets(self, root: Path) -> None:
        manifest = asset_manifest.load_manifest(write_manifest(root, one_asset()))
        (root / "assets/fonts/Fake-Regular.ttf").write_bytes(PAYLOAD)
        (root / "assets/fonts/.gitkeep").touch()
        (root / "assets/fonts/.DS_Store").write_bytes(b"junk")

        assert asset_manifest.untracked(manifest, root) == ()

    def test_only_directories_the_manifest_mentions_are_scanned(self, root: Path) -> None:
        manifest = asset_manifest.load_manifest(write_manifest(root, one_asset()))
        (root / "models").mkdir()
        (root / "models" / "mine.py").write_text("# not an asset")

        assert asset_manifest.untracked(manifest, root) == ()


class TestEntryYaml:
    def test_the_entry_round_trips_and_leaves_the_licence_to_a_human(self, root: Path) -> None:
        target = Path("assets/fonts/New-Regular.ttf")
        (root / target).write_bytes(PAYLOAD)

        (entry,) = yaml.safe_load(asset_manifest.entry_yaml(target, root=root))
        assert entry == {"path": str(target), "sha256": DIGEST, "license": None, "source": None}

    def test_the_entry_is_accepted_by_the_manifest_schema(self, root: Path) -> None:
        target = Path("assets/fonts/New-Regular.ttf")
        (root / target).write_bytes(PAYLOAD)

        entries = yaml.safe_load(asset_manifest.entry_yaml(target, root=root))
        manifest = AssetManifest.model_validate({"release": "https://m.test", "assets": entries})
        assert manifest.assets[0].sha256 == DIGEST


class TestMatching:
    def test_no_pattern_selects_everything(self, root: Path) -> None:
        manifest = asset_manifest.load_manifest(write_manifest(root, one_asset()))
        assert manifest.matching(None) == manifest.assets

    def test_a_glob_matches_on_file_name_or_path(self, root: Path) -> None:
        manifest = asset_manifest.load_manifest(write_manifest(root, one_asset()))
        assert len(manifest.matching("*.ttf")) == 1
        assert len(manifest.matching("assets/fonts/*")) == 1
        assert manifest.matching("*.png") == ()


class TestFetch:
    """The fetch, driven against a local HTTP server rather than the real release."""

    def test_a_correct_download_is_installed(self, root: Path, http_assets) -> None:
        base = http_assets({"Fake-Regular.ttf": PAYLOAD})
        manifest = asset_manifest.load_manifest(
            write_manifest(
                root,
                {
                    "release": base,
                    "assets": [{"path": "assets/fonts/Fake-Regular.ttf", "sha256": DIGEST}],
                },
            )
        )

        report = asset_fetch.fetch_all(manifest, root)
        assert not report.failed
        assert len(report.downloaded) == 1
        assert (root / "assets/fonts/Fake-Regular.ttf").read_bytes() == PAYLOAD

    def test_wrong_bytes_are_not_installed(self, root: Path, http_assets) -> None:
        # The point of the whole mechanism: a release serving something else leaves
        # the asset directory untouched rather than half-populated.
        base = http_assets({"Fake-Regular.ttf": b"an entirely different font"})
        manifest = asset_manifest.load_manifest(
            write_manifest(
                root,
                {
                    "release": base,
                    "assets": [{"path": "assets/fonts/Fake-Regular.ttf", "sha256": DIGEST}],
                },
            )
        )

        report = asset_fetch.fetch_all(manifest, root)
        assert len(report.failed) == 1
        assert "checksum mismatch" in report.failed[0].reason
        assert not (root / "assets/fonts/Fake-Regular.ttf").exists()
        assert list((root / "assets/fonts").glob("*.part")) == []

    def test_an_already_correct_asset_is_not_downloaded_again(
        self, root: Path, http_assets
    ) -> None:
        base = http_assets({"Fake-Regular.ttf": PAYLOAD})
        manifest = asset_manifest.load_manifest(
            write_manifest(
                root,
                {
                    "release": base,
                    "assets": [{"path": "assets/fonts/Fake-Regular.ttf", "sha256": DIGEST}],
                },
            )
        )
        (root / "assets/fonts/Fake-Regular.ttf").write_bytes(PAYLOAD)

        report = asset_fetch.fetch_all(manifest, root)
        assert len(report.skipped) == 1
        assert report.downloaded == ()

    def test_force_downloads_an_asset_that_already_verifies(self, root: Path, http_assets) -> None:
        base = http_assets({"Fake-Regular.ttf": PAYLOAD})
        manifest = asset_manifest.load_manifest(
            write_manifest(
                root,
                {
                    "release": base,
                    "assets": [{"path": "assets/fonts/Fake-Regular.ttf", "sha256": DIGEST}],
                },
            )
        )
        (root / "assets/fonts/Fake-Regular.ttf").write_bytes(PAYLOAD)

        report = asset_fetch.fetch_all(manifest, root, force=True)
        assert len(report.downloaded) == 1

    def test_an_asset_the_release_does_not_hold_fails_naming_the_url(
        self, root: Path, http_assets
    ) -> None:
        base = http_assets({})
        manifest = asset_manifest.load_manifest(
            write_manifest(
                root,
                {
                    "release": base,
                    "assets": [{"path": "assets/fonts/Fake-Regular.ttf", "sha256": DIGEST}],
                },
            )
        )

        report = asset_fetch.fetch_all(manifest, root)
        assert report.failed[0].reason == "HTTP 404"
        assert report.failed[0].url == f"{base}/Fake-Regular.ttf"

    def test_one_failure_does_not_stop_the_others(self, root: Path, http_assets) -> None:
        other = b"a second asset"
        base = http_assets({"Good-Regular.ttf": other})
        manifest = asset_manifest.load_manifest(
            write_manifest(
                root,
                {
                    "release": base,
                    "assets": [
                        {"path": "assets/fonts/Fake-Regular.ttf", "sha256": DIGEST},
                        {
                            "path": "assets/fonts/Good-Regular.ttf",
                            "sha256": hashlib.sha256(other).hexdigest(),
                        },
                    ],
                },
            )
        )

        report = asset_fetch.fetch_all(manifest, root)
        assert len(report.failed) == 1
        assert len(report.downloaded) == 1

    def test_a_non_http_scheme_is_refused(self, root: Path, tmp_path: Path) -> None:
        # A manifest is repo data, but file:// would turn a fetch into a local
        # copy that verifies happily.
        (tmp_path / "local").mkdir()
        (tmp_path / "local" / "Fake-Regular.ttf").write_bytes(PAYLOAD)
        manifest = asset_manifest.load_manifest(
            write_manifest(
                root,
                {
                    "release": (tmp_path / "local").as_uri(),
                    "assets": [{"path": "assets/fonts/Fake-Regular.ttf", "sha256": DIGEST}],
                },
            )
        )

        report = asset_fetch.fetch_all(manifest, root)
        assert len(report.failed) == 1
        assert "scheme" in report.failed[0].reason
        assert not (root / "assets/fonts/Fake-Regular.ttf").exists()

    def test_only_the_selected_assets_are_fetched(self, root: Path, http_assets) -> None:
        other = b"a background"
        base = http_assets({"Fake-Regular.ttf": PAYLOAD, "bg.png": other})
        manifest = asset_manifest.load_manifest(
            write_manifest(
                root,
                {
                    "release": base,
                    "assets": [
                        {"path": "assets/fonts/Fake-Regular.ttf", "sha256": DIGEST},
                        {
                            "path": "assets/backgrounds/bg.png",
                            "sha256": hashlib.sha256(other).hexdigest(),
                        },
                    ],
                },
            )
        )

        report = asset_fetch.fetch_all(manifest, root, assets=manifest.matching("*.ttf"))
        assert len(report.downloaded) == 1
        assert not (root / "assets/backgrounds/bg.png").exists()


class TestRepoManifest:
    """The manifest actually checked in, which is the one participants fetch."""

    def test_it_loads(self) -> None:
        manifest = asset_manifest.load_manifest()
        assert manifest.assets
        assert manifest.release.startswith("https://")

    def test_every_asset_resolves_to_a_release_url(self) -> None:
        # No asset reaches outside the release, which is the property that makes a
        # fetch return the same bytes for everyone.
        manifest = asset_manifest.load_manifest()
        assert all(
            manifest.url_for(asset).startswith(manifest.release) for asset in manifest.assets
        )

    def test_every_asset_is_a_font_or_a_background(self) -> None:
        manifest = asset_manifest.load_manifest()
        assert set(manifest.directories) == {
            Path("assets/fonts"),
            Path("assets/backgrounds"),
        }

    def test_every_font_records_a_licence(self) -> None:
        # The release redistributes these bytes, so the terms have to travel with
        # them. Backgrounds are exempt until their provenance is settled.
        manifest = asset_manifest.load_manifest()
        fonts = [a for a in manifest.assets if a.path.parent == Path("assets/fonts")]
        assert fonts
        assert [a.path for a in fonts if a.license is None] == []
