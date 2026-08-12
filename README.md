# fontaine

Online font recognition over a stream of text-box crops.

```sh
uv sync                                             # 1. install
uv run fontaine fonts scan                          # 2. see your label space
uv run fontaine generate -n 5000 -o data/streams/v1 # 3. freeze a stream
uv run fontaine recognize --stream data/streams/v1  # 4. score the baseline
```

Drop fonts in `assets/fonts/` before step 2 (`.ttf`/`.otf`, scanned
recursively) — that directory *is* the label space. Backgrounds in
`assets/backgrounds/` are optional. Everything about the stream is one file,
`configs/stream.yaml`, fully commented.

Then write your model as **one file in `models/`** implementing two methods:

```python
# models/my_cnn.py
from fontaine.contracts import Recognizer


class MyCNN(Recognizer):
    """One line saying what this is — shows up in `fontaine models list`."""

    name = "my-cnn"  # what --model matches

    def predict(self, image):  # a PIL crop; return a face_id, or None to abstain
        ...

    def learn(self, image, label):  # label may be a font never seen before
        ...
```

```sh
uv run fontaine models list                                          # confirm it was found
uv run fontaine recognize --stream data/streams/v1 --model my-cnn    # score it
```

Nothing in `src/fontaine/` needs to change, and you can copy
`models/last_seen.py` (23 lines) as a template. Two rules that shape the design:
`predict` is always called before `learn` on the same item and scored on that
prediction, and the label space is never announced — a fixed output layer cannot
play.

Useful while iterating:

| Command | What it does |
| --- | --- |
| `fontaine preview -n 48 -o data/preview.png` | contact sheet of crops — the fastest look at your config |
| `fontaine arrival -n 50000` | simulate the font schedule with no rendering, under a second |
| `fontaine recognize -c configs/stream.yaml -n 5000` | score against a live stream instead of a saved one |
| `fontaine recognize --stream … --limit 500` | short run while debugging |
| `pytest` / `ruff check` / `ty check` | tests, lint, types |

Once your model works and you have some performance metrics to show, open up a PR!
