# fontaine

Online font recognition over a stream of text-box crops.

Two programs share one data contract:

- **the generator** — synthesises an endless stream of image crops around text boxes,
  each annotated with the font that drew it. Which fonts appear, how often, and from
  which item onwards are all things you state, so a stream is a designed experiment.
- **the recognizer** — consumes that stream one item at a time, predicting before it
  is told the label, and adding fonts to its vocabulary as they appear.

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
uv run fontaine preview -n 36 --stream  # what the recognizer actually sees, in order
```

```
data/streams/v1/manifest.json      config snapshot, font registry, discovery ground truth
data/streams/v1/annotations.jsonl  one line per item, in stream order
data/streams/v1/crops/00000/*.png  the crops, sharded by index
```

The manifest is written last, so a directory with a manifest is a complete stream
and an interrupted run cannot be mistaken for a finished one. It carries both halves
of the ground truth: the resolved `schedule` (the item each font was *meant* to
arrive at) and `label_first_seen` (the item it *actually* first appeared at). A rare
font allowed from item 3000 may not be drawn until 3200, so scoring detection lag
needs the former as the baseline.

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

## The recognizer

A demonstration baseline with no neural network in it: hand-crafted features from
the ink, fed to an online k-nearest-neighbours classifier from
[river](https://riverml.xyz).

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
mid-stream and simply be learned — no output layer to resize, no retraining. Anything
with a fixed label set cannot consume this dataset at all. Features go in as a dict,
wrapped in river's online `StandardScaler`, which the distance metric needs: stroke
width lives in the hundredths and slant in the tens, and unscaled the vote would be
decided by units alone.

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

`fontaine recognize` also reports, per font, the item it was **scheduled** to arrive
at, the item it first actually appeared at, and the item it was first predicted
correctly — the discovery lag the stream design exists to measure.

### What the baseline cannot do

The confusions are concentrated in the four plain sans faces — Roboto, Nunito,
Montserrat, Ubuntu — which differ in curve details that coarse statistics do not
reach. Casing is the other cost: an all-caps and an all-lowercase crop of one font
have genuinely different row profiles, and the corpus randomises casing on purpose.
Narrowing `corpus.casing` isolates the font signal if you want to measure that gap.

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
