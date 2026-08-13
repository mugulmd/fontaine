# AGENT.md

Repo context for coding agents; `README.md` covers the participant-facing side. What a
reader about to *change* the code needs: layout, the contract, the decisions not to
undo, the conventions to match.

## Commands

```sh
uv sync && uv run fontaine assets fetch   # setup: deps, then the pinned fonts/photos
uv run fontaine fonts scan                # label space from assets/fonts/
uv run fontaine preview -n 48             # contact sheet — fastest look at the pixels
uv run fontaine arrival -n 50000          # simulate the font schedule, nothing drawn
uv run fontaine generate -n 10000 -o data/streams/v1
uv run fontaine recognize --stream data/streams/v1 --model baseline
uv run fontaine assets status             # is the asset tree still the pinned one
uv run pytest && uv run ruff format && uv run ruff check && uv run ty check
```

## Layout

| Path (under `src/fontaine/` unless marked) | What it holds |
| --- | --- |
| `contracts.py` | `FontFace`, `Sample`, `Recognizer` — the only module both sides import |
| `config.py` | every config model: pydantic, `extra="forbid"`, `#:` on each field |
| `rng.py` | `item_rng(seed, index)` — the only source of randomness |
| `fonts/` | registry: font dir → label space, one face per class, slug labels |
| `render/`, `text/` | crop synthesis: metrics, backgrounds, contrast-driven colour, corpus |
| `stream/` | the arrival process (weights, start/stop) and the generator |
| `store/`, `assets/` | replaying a stream directory; the manifest and verified fetch |
| `recognize/` | model discovery by scanning, plus features a model *may* reuse |
| `evaluate/` | prequential (test-then-train) scoring off confusion matrices |
| `viz/`, `cli.py` | contact sheet; every command, one function each |
| `models/` (root) | challenger models — nothing in `src/` knows they exist |
| `configs/stream.yaml`, `assets/manifest.yaml` | one stream config; the SHA-256 pins |

## The one extension point

A model is a `contracts.Recognizer` subclass in a file under `models/`, found by
scanning — adding one touches nothing in `src/`.

```python
class MyCNN(Recognizer):
    name = "my-cnn"  # what --model matches; required ClassVar

    def predict(self, image): ...  # a PIL crop; None abstains
    def learn(self, image, label): ...  # label may be a font never seen before
```

Three rules the loop enforces: the model gets the crop and never the `Sample`, whose
metadata would let it read the answer off the generator; `predict` runs before `learn`
on an item and is scored on that prediction; the label space is never announced, so a
fixed output layer cannot play. In return the same image object reaches both calls, so
featurization can be cached across it.

## Decisions not to undo

Each looks arbitrary and is load-bearing; the failure it prevents is in brackets.

- **Sizes are cap heights measured from the raster**, not em sizes or
  `OS/2.sCapHeight`. [absolute scale would stand in for the label]
- **Contrast targets the hardest point of the background**, not the mean, forcing a
  scrim when no colour clears `min_contrast`. [text invisible one side of an edge]
- **Crop padding jitter stays on.** [detector boxes are imprecise; that is the task]
- **Text and casing never correlate with the font.** [letterforms are the only signal]
- **Arrival weights and start/stop are stated, never sampled**, and a pattern matching
  no font is an error. [a stream is a designed experiment, not what a seed produced]
- **Every metric is read off one confusion matrix**, lifetime beside rolling-500.
  [four counters were four places to be wrong; one average hides a slow learner]
- **The stream is an iterator.** `StreamGenerator` and `store.reader.read_stream` yield
  identical `Sample`s, asserted pixel-for-pixel. [no branch between live and frozen]
- **Assets come from the release only, verified before install.** [a second source
  means bytes depend on which host answered]
- **Synthetic backgrounds are functions, not files**, and `sources` is a weighted
  mixture, not a fallback. [a checksum on a generated image pins output, not input]
- **A stream manifest is written last.** [an interrupted run cannot pass for finished]

## Conventions

- Config is pydantic with `extra="forbid"`, so a mistyped option fails on load instead
  of keeping a plausible default.
- Errors name the file or key at fault. Each module raises its own exception type; the
  CLI catches it, prints one red line, exits 1.
- Docstrings are enforced (ruff `D`) and say *why*, not what. `D401` and `RUF001`-`003`
  are off deliberately — reasons in `pyproject.toml`.
- Randomness comes from the item RNG, never module state; numpy generators are seeded
  from it, so item `i` is reproducible without replaying `i-1`.
- Tests build their own fonts with `fontTools` and serve HTTP from localhost: the suite
  needs neither `assets/` nor a network.
- Prefer a function over a config knob until an experiment needs to vary it.

## Assets

`assets/manifest.yaml` pins every font and background photo by SHA-256; the bytes live
on the `assets-v1` GitHub release, never in git. To add one: drop the file in, run
`fontaine assets hash <path>`, paste the entry and fill in `license`/`source`, upload
to the release under the same file name, then `fontaine assets fetch` to confirm. Names
must be unique across directories — a release is flat. Strip EXIF from photographs
*before* hashing (phone GPS goes public; metadata is bytes). Re-pinning tells everyone
their scores no longer compare, so it belongs with a new release tag.
