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
- `data/backgrounds/` — PNGs to crop background patches from. Optional: with the
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

## Tests

```sh
uv run pytest
```

Registry tests build their own minimal fonts with `fontTools`, so they assert on
known glyph coverage and pass without depending on `assets/fonts/`.
