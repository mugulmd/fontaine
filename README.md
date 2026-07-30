# fontaine

Online font recognition over a stream of text-box crops.

Two programs share one data contract:

- **the generator** — synthesises an endless stream of image crops around text boxes,
  each annotated with the font that drew it. Fonts are not all present at the start:
  new ones keep appearing as the stream advances.
- **the recognizer** (not built yet) — consumes that stream one item at a time,
  predicting before it is told the label, and discovering the font set as it goes.

## Setup

```sh
uv sync
```

Two directories you fill in:

- `assets/fonts/` — the fonts that make up the label space (`.ttf`, `.otf`, also
  `.ttc`; scanned recursively). Nothing outside it is ever read, so the font
  universe is exactly what you put there.
- `assets/backgrounds/` — PNGs to crop background patches from. Optional: with the
  directory empty, generation falls back to synthetic canvases.

`configs/stream.yaml` is the single config describing a whole stream, and every
option is commented there.

## The font registry

The registry resolves `assets/fonts/` into the label space. Inspect it before
generating anything:

```sh
uv run fontaine fonts scan
uv run fontaine fonts scan --font-dir /some/dir --json data/registry.json
```

Each face gets a stable slug label — `georgia:bold-italic` — derived from the
font's `name` table rather than its filename, so moving or renaming a file does
not rename the class. `.ttc` collections expand to one face per index.

Labels come at two granularities, set by `label_granularity` in the config:

- `face` — `georgia:regular` and `georgia:bold-italic` are distinct classes.
  Recommended: classes then match what is actually visible in the pixels.
- `family` — every weight and slant collapses into `georgia`.

Both are always stored per face, so a stream generated at one granularity can be
re-scored at the other.

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
  even in v1. Everything in the `degrade` section is off by default.
- **The text never correlates with the font.** Content and casing are sampled
  independently of the face, so the letterforms are the only signal.

## The stream

Which font each item uses is decided by a Chinese-restaurant process: at each step
it either introduces a font never seen before, or reuses one already seen with
probability proportional to that font's recent popularity. Three properties the
recognizer has to deal with fall out of that:

- **progressive discovery** — the font set is never declared up front,
- **a long tail** — popularity is rich-get-richer, so a few fonts dominate and many
  stay rare, which is far harder than a uniform draw,
- **drift** — popularity is counted with exponential forgetting, so a font that
  stops appearing fades and can come back later.

`half_life` is what produces the third one, and it matters more than it looks. With
forgetting off, the chance of a new font decays like 1/t and the stream ossifies —
on a 437-face pool, plain CRP had found 76 fonts after 50k items and effectively
stopped, while `half_life: 2000` reached 200 and was still discovering.

Tune it without rendering anything:

```sh
uv run fontaine arrival -n 50000        # simulate only; runs in well under a second
```

Then materialize a stream:

```sh
uv run fontaine generate -n 10000 -o data/streams/v1
uv run fontaine preview -n 36 --stream  # what the recognizer actually sees, in order
```

```
data/streams/v1/manifest.json      config snapshot, font registry, discovery ground truth
data/streams/v1/annotations.jsonl  one line per item, in stream order
data/streams/v1/crops/00000/*.png  the crops, sharded by index
```

The manifest is written last, so a directory with a manifest is a complete stream
and an interrupted run cannot be mistaken for a finished one. It records
`label_first_seen` per label, which is what you need to score discovery lag later.

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

## Tests and checks

```sh
uv run pytest        # 98 tests, well under a second
uv run ruff format   # formatting
uv run ruff check    # linting
uv run ty check      # type checking
```

Registry tests build their own minimal fonts with `fontTools`, so they assert on
known glyph coverage and pass without depending on `assets/fonts/`.

Ruff runs with docstring checks on, since this codebase explains *why* it does
things and that only stays true if it is enforced. Two rule groups are switched
off deliberately, with the reasons recorded in `pyproject.toml`: `D401` (accessors
are documented as the noun they return) and `RUF001`-`RUF003` (the ambiguous
characters are the point — the charset presets need real curly quotes and dashes
as glyphs to test coverage against).
