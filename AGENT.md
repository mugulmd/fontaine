# AGENT.md

In-depth context on this repository, for coding agents. The `README.md` is
deliberately short — everything a human needs to bring in a new model. This file
is the rest: how each piece works and, more importantly, *why* it was built that
way, so a change does not quietly undo a decision.

**Read the "why" notes before changing behaviour.** Several choices here look
arbitrary and are not: cap-height normalization instead of em sizes, contrast
targeted at the hardest point of the background, stated arrival schedules instead
of a stochastic process, one confusion matrix as the source of every metric, assets
pinned by checksum and fetched rather than committed. Each is recorded below with
the failure it prevents.

Layout:

| Path | What it holds |
| --- | --- |
| `src/fontaine/contracts.py` | `FontFace`, `Sample`, `Recognizer` — the only module both sides import |
| `src/fontaine/assets/` | the asset manifest, and the checksum-verified fetch |
| `src/fontaine/fonts/` | registry: font dir → label space |
| `src/fontaine/render/`, `text/` | crop synthesis: metrics, backgrounds, corpus |
| `src/fontaine/stream/` | the arrival process and the generator |
| `src/fontaine/store/` | writing and replaying a stream directory |
| `src/fontaine/recognize/` | model discovery, and the feature extractors a model may reuse |
| `src/fontaine/evaluate/` | the prequential (test-then-train) scoring loop |
| `src/fontaine/viz/` | the annotated contact sheet |
| `src/fontaine/cli.py` | every command, one function each |
| `models/` | challenger models, discovered by scanning — nothing in `src/` knows they exist |
| `configs/stream.yaml` | the single config for a whole stream |
| `assets/manifest.yaml` | every font and background *photo*, pinned by SHA-256 |

## Setup

```sh
uv sync
uv run fontaine assets fetch
```

The second command downloads the fonts and backgrounds, verifying each against
`assets/manifest.yaml`. Two directories it fills in:

- `assets/fonts/` — the fonts that make up the label space (`.ttf` and `.otf`,
  scanned recursively). Nothing outside it is ever read, so the font universe is
  exactly what you put there.
- `assets/backgrounds/` — **photographs** to crop background patches from. Optional:
  with the directory empty the `photo` source drops out and the six synthetic
  patterns are renormalized over the remaining share. Note that they are not a
  fallback — see [Background sources](#background-sources).

`configs/stream.yaml` is the single config describing a whole stream, and every
option is commented there.

Your own work goes in `models/` — one file per recognizer, discovered by scanning.
See [Writing a recognizer](#writing-a-recognizer).

## Pinning the assets

Everyone competing has to be scored over the same stream, and the stream is not
distributed — it is regenerated. Generation is deterministic given
`(seed, config, assets)`, since every item derives its RNG from `(seed, index)`, so
the assets are the only input that was not already pinned by a file in git. A font
updated in place changes the pixels without changing anything a diff would show,
and two participants would be comparing scores over different data without either
noticing.

`assets/manifest.yaml` pins them by SHA-256. The manifest is text and lives in git;
the bytes live on a **GitHub release**, whose assets are served outside git history —
so 11 MB of fonts and photos that never change are versioned and downloadable
without being committed. That is the trade this design exists to make.

```sh
uv run fontaine assets fetch              # download and verify everything
uv run fontaine assets fetch --only '*.ttf'   # just the fonts
uv run fontaine assets status             # check disk against the manifest, no network
uv run fontaine assets status --verbose   # ... listing every asset, not just the bad ones
uv run fontaine assets hash <path>        # paste-ready entry for something you added
```

Three properties worth not undoing:

- **The release is the only source.** There is deliberately no per-asset URL and no
  upstream fallback: an asset that could arrive from either place is an asset whose
  bytes depend on which one answered, and a fallback silently converts a broken
  release into a stream that differs from everyone else's. `release` is a required
  field, and `url_for` is the one function that turns an asset into an address.
  `source` records where a font originally came from for attribution and is never
  requested — it could not be a download link even if we wanted one, since these are
  static instances from the `fonts.google.com` download service, which has no stable
  per-file URL. Every one of the twelve already differs from what `google/fonts`
  ships at `main`, which now carries variable fonts for most of those families.
- **Nothing is installed until its digest matches.** A download streams into a
  sibling `.part` file, is hashed as it arrives, and is renamed into place only on a
  match — same directory, so the move is an atomic rename. A truncated transfer or a
  release serving the wrong bytes leaves the asset directory exactly as it was, the
  same reason `store.writer` writes the stream manifest last.
- **Untracked files are reported.** An unpinned font in `assets/fonts/` enters the
  label space, so it changes the classes and every score computed over them, while
  every checksum in the manifest still verifies. It is the one way to be out of sync
  that a checksum cannot catch, so `assets status` fails on it.

`assets status` exits non-zero when the tree is not the pinned one, which is what
makes it usable as a check before a run whose numbers are meant to be comparable.

### Adding an asset

1. **Drop the file in** `assets/fonts/` or `assets/backgrounds/`. Only fonts and
   *photographs* belong there — a procedural pattern is a function in
   `render/background.py`, not an asset. See
   [Background sources](#background-sources).

2. **Generate its entry.** Given a directory, this emits one entry per file the
   manifest does not already pin, which is the shape of adding several fonts at once:

   ```sh
   uv run fontaine assets hash assets/fonts
   ```

   ```yaml
   - path: assets/fonts/Cormorant-Regular.ttf
     sha256: 4f21c0a95b0e77d3d6b2d9c8f0a1e4b7c3d5e8f9a0b1c2d3e4f5a6b7c8d9e0f1
     license: null
     source: null
   ```

3. **Paste it into `assets/manifest.yaml`** under `assets:`, and fill in `license`
   and `source`. Those two are left empty rather than guessed because the release
   redistributes the bytes, so the terms have to travel with them — and an entry that
   looked complete would never get filled in. `assets status` counts what is still
   unlicensed.

4. **Upload the file to the release** the `release` line points at, keeping the file
   name exactly as it is on disk — that name *is* the address, since `url_for` joins
   the release URL to the file name and drops the directory. A release is a flat
   namespace, so names must be unique across directories; the manifest refuses to
   load if two collide, because otherwise one asset would silently overwrite the
   other's bytes.

5. **Verify a cold start.** In a clone with no assets, `fontaine assets fetch`
   should report the new count with no failures.

Adding a font changes the label space, so the classes and every score over them
change with it. Cutting a new release tag (`assets-v2`) and bumping `release` is how
that gets a version, rather than a leaderboard whose rows were measured against
different pools.

### Changing or replacing an asset

Swapping bytes in place, or re-downloading a font that upstream has re-generated,
leaves the file no longer matching its pin:

```
$ uv run fontaine assets status
19/20 assets match assets/manifest.yaml

assets/fonts/Anton-Regular.ttf is not the pinned file. If that was intended:
    sha256: 97aae409210d255fb3b92f18c8af2ed14941e10a1ee134a35bb91f52086de1d0
otherwise `fontaine assets fetch --force` puts the pinned bytes back
```

The digest printed is the one on disk, on its own line, to paste over the old
`sha256` — then upload the new bytes to the release under the same file name. If
the change was *not* deliberate, `fontaine assets fetch --force` restores the pinned
bytes; `--force` is safe on a failure, since the existing file is only replaced once
a download verifies.

Take the second path more often than the first. Re-pinning is telling everyone their
scores are no longer comparable with yesterday's, so it belongs with a release tag
rather than a quiet commit.

### Licensing your own backgrounds

For a photo you took yourself, you already hold the copyright — nothing has to be
obtained. The `license` field is the licence you are **granting**, not one you had to
get, and it needs filling in anyway: the release redistributes those bytes to every
participant, and with no statement the default is all-rights-reserved, which formally
leaves them no right to the copy they just downloaded.

`CC0-1.0` is the least friction — it puts the file in the public domain, so nobody has
to think about attribution while fetching a texture. `CC-BY-4.0` if you want credit.

```yaml
- path: assets/backgrounds/kitchen-tile.png
  sha256: 8f0a…
  license: CC0-1.0
  source: own photograph
```

Three things to check before a phone photo goes on a public release, none of which
copyright ownership covers:

- **Strip the EXIF.** Phone photos carry GPS coordinates, timestamps and the device
  serial. A public release of a photo taken at home publishes your address. Strip it
  *before* hashing — metadata is bytes, so removing it afterwards breaks the pin.
- **No identifiable people.** Owning the copyright is not consent from the subject;
  in France `droit à l'image` is separate and stronger than in most jurisdictions, and
  GDPR treats a recognisable face as personal data.
- **Nothing else copyrighted in frame.** A poster, painting, album cover or sculpture
  carries its own rights, and France has no broad freedom-of-panorama exception. A
  logo is usually fine as incidental background, artwork is not.

Textures — fabric, concrete, paper, tile, wood, painted walls — sidestep all three,
and are the better choice for this repo anyway: no faces, no third-party artwork, and
no legible text of their own. That last one matters technically, not just legally. A
background with readable text in it puts a second set of letterforms in the crop, and
the label says only which font drew the foreground, so it is a mislabelled item rather
than a hard one.

None of the above is legal advice, and for a research challenge the practical risk is
low; the EXIF point is the one worth acting on regardless.

## The font registry

The registry resolves `assets/fonts/` into the label space. Inspect it before
generating anything:

```sh
uv run fontaine fonts scan
uv run fontaine fonts scan --font-dir /some/dir --json data/registry.json
```

Each face gets a stable slug label — `georgia:bold-italic` — derived from the
font's `name` table rather than its filename, so moving or renaming a file does
not rename the class. One file is one face: font collections (`.ttc`, `.otc`) are
not scanned, since a label would then have to carry an index into a file.

One face is one class: `georgia:regular` and `georgia:bold-italic` are distinct
labels, so the classes match what is actually visible in the pixels.

Faces that cannot render the configured `admission_charset`, and variable fonts
(v1 renders static instances only), are **reported** in the scan output rather
than dropped silently — you own the font dir, so you should hear about every
exclusion. Keep the admission charset to a core set: the corpus intersects its
alphabet with each face's real coverage, so a face missing a few rare glyphs
still contributes rather than being thrown out.

## Rendering crops

```sh
uv run fontaine preview -n 48 -o data/preview.png
```

Renders a batch of crops to an annotated contact sheet — the fastest way to see
what the config actually produces. Faces are taken in registry order and cycled,
so a large enough count shows every font at least once.

It is a render check, not a stream preview: the `arrival` config is ignored, so a
font made rare or scheduled to arrive late still appears. An earlier version had a
`--stream/--all-faces` pair that switched between the two, which was a distinction
too fine for a look-at-the-pixels command — use `fontaine arrival` to inspect a
schedule.

If the text comes out too small, `cap_height_px` is the knob, and `--cap-height`
overrides it without editing the config:

```sh
uv run fontaine preview --cap-height 18:64
```

Two things to know when reading the sheet. The range is **log-sampled**, so its
median is `sqrt(lo × hi)` and not the midpoint — `[8, 44]` puts half the crops
under 19px. And the sheet scales every crop to `--cell-height` (56px by default)
with nearest-neighbour, so small crops are shown blockier than they are; pass
`--cell-height 28` to see them closer to their real size.

Per item: sample a target cap height, resolve the em size giving it for this face,
sample text, build a background, pick a text colour at a sampled contrast ratio,
draw, then crop the tight ink box with independent per-side padding.

A few decisions worth knowing about:

- **Sizes are cap heights, not em sizes.** Two faces at the same em size can differ
  by nearly 2x in apparent size, which would let absolute scale stand in for the
  label. Cap height is measured from the rasterized glyph, not read from
  `OS/2.sCapHeight`, which fonts often omit or get wrong.
- **Contrast is targeted against the hardest point of the background**, not its
  mean: over a background with an edge crossing the text box, matching the average
  leaves the text invisible on one side. When the background's range is too wide
  for any single colour, a scrim is forced behind the text — the same thing a
  designer does — escalating opacity until the floor in `min_contrast` is met.
- **Crop jitter is not a degradation.** Boxes from a text detector are imprecise by
  nature, including tight enough to clip ascenders, so padding jitter stays on
  even in v1. Capture artefacts (blur, JPEG, noise, rotation) are not modelled at
  all: clean renders first, so the pipeline can be validated against an easy
  accuracy ceiling.
- **The text never correlates with the font.** Content and casing are sampled
  independently of the face, so the letterforms are the only signal.

### Background sources

`render.background.sources` is a **weighted mixture, not a fallback chain.** This is
the thing most easily misread: having photographs in `assets/backgrounds/` does not
switch the synthetic sources off, and the defaults put photos on 60% of items with
the six patterns sharing the other 40%.

```yaml
sources:
  photo: 6.0        # dropped and the rest renormalized if photo_dir is empty
  noise: 1.0        # grain at stroke scale, over a ramp
  blobs: 1.0        # soft multi-colour wash, the mesh-gradient look
  geometric: 0.75   # hard edges through the box — what forces a scrim
  gradient: 0.75    # linear ramp
  vignette: 0.25    # radial ramp
  solid: 0.25       # the easy floor, useful as a control
```

`photo` is the only conditional one, being the only one that needs files. When the
directory is empty it drops out and the remaining weights renormalize — so an empty
asset directory shifts the mixture rather than making every canvas flat.

Each pattern covers a regime the photographs do not reliably reach: `noise` puts
texture at the same spatial scale as the strokes, `geometric` puts a hard colour edge
through the text box (the case `min_contrast` and the forced scrim exist for), and
`blobs`/`gradient`/`vignette` are smooth but not flat. Set one to `1.0` and the rest
to `0` to isolate a regime and see what it costs a model.

**These were four PNGs until they became six functions.** A checksum on a
procedurally generated image pins its output where the function is the input, and
four fixed files got patch-cropped over and over where a function never repeats the
same canvas twice. Nothing is distributed, and there is no licence question for a
canvas nobody owns. Photographs are the opposite case — you cannot regenerate one
from a seed, so pinning the bytes is the only way to pin it at all, and
`assets/backgrounds/` is now photographs only.

Parameters are sampled from the item RNG with ranges fixed in `background.py` rather
than exposed in the config, following the existing `_gradient` and `_solid`. A knob
per pattern would roughly triple `BackgroundConfig` for choices no experiment has
needed to vary yet; the weights are the knob that matters. `noise` seeds a numpy
generator from the item RNG, since per-pixel grain through `random` is impractical —
item `i` still draws the same grain on every run, which the tests assert.

## The stream

Which font each item uses is **uniform over the whole label space** by default:
every font carries the same weight and is available from the first item. Two kinds
of override in `configs/stream.yaml` turn that into an experiment:

```yaml
arrival:
  default_weight: 1.0
  fonts:
    "roboto:regular":  { weight: 5 }     # five times its fair share
    "oswald:*":        { weight: 0.2 }   # whole family made rare
    "lobster:regular": { start: 3000 }   # a new class arriving mid-stream
    "anton:regular":   { stop: 2000 }    # an old class going away
    "jetbrains-mono:regular": { weight: 0 }   # excluded entirely
```

- **weights** create imbalance, so a recognizer has to learn one class from a
  handful of examples while another has thousands. A font's probability is its
  weight over the total of the fonts *active at that point*, so retiring a font
  redistributes its share over the rest automatically.
- **start / stop** create concept drift at points you choose. A class arriving
  mid-stream is what tests discovery; a class leaving tests whether the learner
  degrades gracefully on something it stops seeing.

Both are deliberately *stated* rather than emergent. An earlier version drew
popularity and arrival times from a Chinese-restaurant process, which meant the
imbalance and the novelty points were whatever a given seed produced — you could
only run it and see. Dictating them makes a stream a designed experiment, and has
the side benefit that adding a font to `assets/fonts/` perturbs only what you say
it perturbs instead of reshuffling the whole stream.

Keys are face ids (as printed by `fontaine fonts scan`) or globs over them. The
most specific match wins: an exact id beats a glob, a longer glob beats a shorter
one. **A pattern matching no font is an error** — otherwise renaming a font would
quietly turn a designed stream back into a uniform one.

Check a schedule without rendering anything:

```sh
uv run fontaine arrival -n 50000    # simulate only; well under a second
```

It prints the weight, window, realized share and first appearance per font, so you
can confirm the stream is the one you intended before spending minutes on images.

Then materialize a stream:

```sh
uv run fontaine generate -n 10000 -o data/streams/v1
```

```
data/streams/v1/manifest.json      config snapshot, font registry, discovery ground truth
data/streams/v1/annotations.jsonl  one line per item, in stream order
data/streams/v1/crops/00000/*.png  the crops, sharded by index
```

The manifest is written last, so a directory with a manifest is a complete stream
and an interrupted run cannot be mistaken for a finished one. It carries both halves
of the ground truth: the resolved `schedule` (the item each font was *meant* to
arrive at) and `arrival.face_first_seen` (the item it *actually* first appeared at). A rare
font allowed from item 3000 may not be drawn until 3200, and the two halves are what
tell those apart.

**The stream is an iterator, not a directory.** `StreamGenerator` yields
`Sample`s lazily and `store.reader.read_stream` yields the identical objects from
disk — verified pixel-for-pixel in the tests. So an online learner can train
against a live infinite generator or a frozen stream through the same code path,
with no branching on its side.

```python
from fontaine.store.reader import read_stream

for sample in read_stream(Path("data/streams/v1")):
    prediction = model.predict(sample.image)  # predict before seeing the label
    model.learn(sample.image, sample.label)  # then learn from it
```

## Writing a recognizer

A model is a class implementing `fontaine.contracts.Recognizer`, in a file under
`models/`. That is the whole extension point — nothing inside `src/fontaine/`
knows your model exists, so adding one cannot perturb anyone else's.

```python
# models/my_cnn.py
from fontaine.contracts import Recognizer


class MyCNN(Recognizer):
    """One line saying what this is — it shows up in `fontaine models list`."""

    name = "my-cnn"  # what --model matches

    def predict(self, image):  # a PIL image; None abstains
        return self._head(self._embed(image))

    def learn(self, image, label):  # label may be a font never seen before
        self._sgd_step(image, label)
```

```sh
uv run fontaine models list
uv run fontaine recognize --stream data/streams/v1 --model my-cnn
```

Three rules the evaluation loop enforces, and one guarantee it gives you:

- **You get the crop, never the `Sample`.** The metadata records the exact cap
  height, contrast and background used to draw the item, so a model handed the
  whole sample could read the answer off the generator instead of the pixels.
- **`predict` is always called before `learn`** for the same item, and the item is
  scored on that prediction. Test-then-train, so there is no way to be scored on
  something you have already learned.
- **The label space is not announced.** A font never seen before can arrive at any
  point, and the only notice you get is a `learn` call carrying its label.
  Discovering it is the task; anything with a fixed output layer cannot play.
- **The same image object goes to `predict` then `learn`**, so work done for the
  prediction can be cached across the pair instead of repeated. `models/baseline.py`
  keys a one-entry feature cache on exactly that.

Featurization is yours, not the framework's. `fontaine.recognize.features` is
available to reuse if the hand-crafted vector is a useful starting point, but
nothing obliges you to go through it — a model working straight off pixels is
expressible.

A model that outgrows one file can be a package instead: `models/my_cnn/__init__.py`,
with `models/` on `sys.path` so its own helpers import normally. Files starting with
`_` are skipped, which is the escape hatch for code shared between entries.

`models/last_seen.py` is the smallest possible implementation — it repeats the
previous label and scores 0.9x the majority baseline — if you want a template
shorter than the baseline to copy.

## How a run is scored

Every number `fontaine recognize` prints is read off a **confusion matrix**. Accuracy,
balanced accuracy, macro F1 and the per-font recall and precision are all functions of
the same counts, so there is one place for a number to be wrong rather than four, and a
metric added later is a function over the matrix rather than another counter in the loop.

There are two matrices, because they answer different questions and neither derives from
the other:

- the **lifetime** one, over every item — it never forgets, so it carries the cold start
  for the whole run;
- the **rolling** one, over the last 500 items — it has already forgotten it.

That is what the two columns of the headline table are, and reading across a row is the
point. A model still climbing shows a recent column well above its overall one; a model
that converged somewhere mediocre shows both the same. Averaged into a single lifetime
number, a fast learner and a slow one are hard to tell apart at 100k items.

One number cannot come from a matrix and is tracked beside them: the **majority
baseline**, what always answering with the commonest label so far would have scored. It
depends on the order the labels arrived in, not just the final counts.

The per-font table reports recall *and* precision, because either alone is easy to game.
A model that over-answers one font scores high recall there and precision on the floor —
on the twelve-font pool the baseline does exactly that with Bebas Neue, 77% recall against
42% precision, which the recall column alone would have shown as a strength.

## The baseline

A demonstration model with no neural network in it: hand-crafted features from
the ink, fed to an online k-nearest-neighbours classifier from
[river](https://riverml.xyz). It lives in `models/baseline.py` and is found by the
same scan as yours — it is the default for `--model`, and nothing else.

```sh
uv run fontaine recognize --stream data/streams/v1
uv run fontaine recognize -c configs/stream.yaml -n 5000     # generate live instead
```

Measured on the twelve-font pool with uniform weights, 3,000 items: **55.7%
accuracy against 8.3% chance**, 62% over the most recent 500 items. Nothing is fitted at all — the model keeps a bounded window
of recent feature vectors and answers by majority vote among the nearest few, which
makes it a fair floor for anything more sophisticated to be measured against. Three
alternatives were tried and dropped: Gaussian naive Bayes, a Hoeffding tree and an
adaptive forest all sat near 42% on the same stream.

The reason river suits this problem is that a label it has never seen can arrive
mid-stream and simply be learned — no output layer to resize, no retraining. Features
go in as a dict, wrapped in river's online `StandardScaler`, which the distance metric
needs: stroke width lives in the hundredths and slant in the tens, and unscaled the
vote would be decided by units alone.

**Read the balanced accuracy, not the accuracy.** On an imbalanced stream the two
diverge sharply — with one font weighted 5x, the same model scores 57.3% accuracy but
47.9% balanced, against a majority baseline of 37.0%. One of the discarded models
scored *exactly* the majority baseline with 10% balanced accuracy, having collapsed to
always answering the dominant font. Accuracy alone would have called that a decent
result. That is why the majority and chance baselines print beside every score, and
why a lift below 1x is called out.

The features are all scale-free and none of them reads the text: log aspect, ink
density, a 12-bin row profile (where the baseline and x-height sit), an 8-bin column
profile with its autocorrelation (set width, and the giveaway of a monospace),
stroke thickness mean and variance from a distance transform (weight, and the
thick-to-thin contrast that separates a Didone from a grotesque), estimated slant,
edge density and stroke orientation. 38 numbers, about 550 crops a second to
extract, and roughly 135 items a second end to end including the model.

Two things worth knowing. Polarity is normalized first, since crops are
light-on-dark as often as the reverse and every feature would otherwise flip. And
the mask is trimmed to the ink, so the crop jitter the generator introduces on
purpose cannot shift the measurements.

### What the baseline cannot do

The confusions are concentrated in the four plain sans faces — Roboto, Nunito,
Montserrat, Ubuntu — which differ in curve details that coarse statistics do not
reach. Casing is the other cost: an all-caps and an all-lowercase crop of one font
have genuinely different row profiles, and the corpus randomises casing on purpose.
Narrowing `corpus.casing` isolates the font signal if you want to measure that gap.

## Tests and checks

```sh
uv run pytest        # 241 tests, well under a second
uv run ruff format   # formatting
uv run ruff check    # linting
uv run ty check      # type checking
```

Registry tests build their own minimal fonts with `fontTools`, so they assert on
known glyph coverage and pass without depending on `assets/fonts/`.

The asset-fetch tests run against a throwaway HTTP server on localhost (the
`http_assets` fixture) rather than a mocked `urlopen`. The contract being tested is
what ends up on disk after a wrong or truncated response, and a stubbed transport is
exactly the thing that would paper that over. The suite stays offline either way.

Ruff runs with docstring checks on, since this codebase explains *why* it does
things and that only stays true if it is enforced. Two rule groups are switched
off deliberately, with the reasons recorded in `pyproject.toml`: `D401` (accessors
are documented as the noun they return) and `RUF001`-`RUF003` (the ambiguous
characters are the point — the charset presets need real curly quotes and dashes
as glyphs to test coverage against).
