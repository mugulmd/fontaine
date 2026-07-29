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

Drop the fonts you want in the label space into `assets/fonts/` (`.ttf`, `.otf`,
`.ttc`, `.otc`; scanned recursively). Nothing outside that directory is ever read,
so the font universe is exactly what you put there.

## The font registry

The registry resolves `assets/fonts/` into the label space. Inspect it before
generating anything:

```sh
uv run fontaine fonts scan                      # uses built-in defaults
uv run fontaine fonts scan -c configs/fonts.yaml
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

Faces that cannot render the configured `charset`, and variable fonts (v1 renders
static instances only), are **reported** in the scan output rather than dropped
silently — you own the font dir, so you should hear about every exclusion.

## Tests

```sh
uv run pytest
```

Registry tests build their own minimal fonts with `fontTools`, so they assert on
known glyph coverage and pass without depending on `assets/fonts/`.
