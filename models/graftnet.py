"""A classifier head that learns, online, on top of a frozen ImageNet backbone.

Two halves:
* The lower half is ResNet-18's first block with its ImageNet weights, frozen.
* The upper half is a small head that grows an output as each new font arrives
  and trains by replay.

Three problems the online setting poses, and where each is solved here:
* The label space is not known up front: `_OnlineHead.grow` appends an
  output row when a font first arrives, keeping every weight already learned.
* One item at a time is a terrible gradient: a single crop's gradient points
  almost anywhere, and following it leads to catastrophic forgetting, so every
  item is kept in a bounded per-font buffer and each step trains on a batch
  replayed from it.
* Fonts arrive in unequal numbers: batches are drawn per font rather than per
  item, so a font that shows up rarely still gets a say in every step.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from fontaine.contracts import Recognizer
from fontaine.recognize.preprocess import ink_mask

#: One window fed to the backbone, (height, width). The height is what the ink is
#: normalized to, so it fixes the scale at which strokes are seen; the width is a
#: few words' worth at that scale.
TILE = (64, 192)

#: Windows sampled from a wide crop. Averaging a few beats stretching a sentence to
#: fit one, which would make stroke width depend on how much text the crop holds.
MAX_WINDOWS = 3

#: Length of the vector the backbone yields: ``layer1``'s 64 channels, each
#: contributing a spatial mean and a spatial standard deviation.
DIMS = 128

#: Crops remembered per font. Bounded, so memory grows with the number of fonts
#: seen and not with the length of the stream.
PER_CLASS = 400

BATCH = 32
LEARNING_RATE = 0.001
HIDDEN = 512
SEED = 0

#: ImageNet channel statistics the pretrained weights were trained under. The ink
#: mask is greyscale, so the same plane is repeated into all three channels.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _build_trunk() -> nn.Module:
    """ResNet-18 up to ``layer1`` only, frozen and in eval mode.

    Why cut this early and not at a deeper layer?
    ``layer1`` still has stride 4, so a hairline is a few activations wide;
    by ``layer3`` a stroke is sub-pixel and the channels have moved on to object
    parts, which no font distinguishes itself by.
    """
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    trunk = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
    trunk.eval()
    trunk.requires_grad_(False)
    return trunk


def _windows(mask: np.ndarray) -> np.ndarray:
    """Fixed-size views on the ink, evenly spaced across its width.

    A crop holding one word and a crop holding a sentence arrive at the same ink
    height and wildly different widths. Sampling windows at that fixed height keeps
    the stroke geometry comparable between the two, which resizing to a common
    rectangle would not.
    """
    height, width = mask.shape
    _, tile_width = TILE
    if width < tile_width:
        # Centred rather than left-aligned: the convolutions see the same amount of
        # blank either side of a short word as they do inside a long one.
        padded = np.zeros((height, tile_width), dtype=mask.dtype)
        start = (tile_width - width) // 2
        padded[:, start : start + width] = mask
        return padded[None, ...]
    count = min(MAX_WINDOWS, max(1, round(width / tile_width)))
    starts = np.linspace(0, width - tile_width, count).round().astype(int)
    return np.stack([mask[:, start : start + tile_width] for start in starts])


@torch.inference_mode()
def embed(trunk: nn.Module, image: Image.Image) -> torch.Tensor:
    """Pool the trunk's activations over a crop's windows, then unit-normalize.

    A function of the trunk and the image and nothing else, so a whole stream can be
    embedded once and swept over: the trunk is frozen, so its output for an item
    never changes.

    Runs on the ink mask rather than the crop: polarity, colour and absolute scale
    are nuisance variables the generator varies on purpose, and
    ``preprocess.ink_mask`` already removes all three.

    Each channel contributes its spatial mean *and* its spatial spread. The mean
    alone says how much of a stroke shape the window holds; the spread says how
    unevenly it is distributed.

    Unit length because what follows then compares directions rather than activation
    magnitudes, which track how much ink a window happened to hold.
    """
    ink = ink_mask(image, ink_height=TILE[0])
    batch = torch.from_numpy(_windows(ink.mask)).float().unsqueeze(1).expand(-1, 3, -1, -1)
    maps = trunk((batch - _MEAN) / _STD)
    pooled = torch.cat([maps.mean(dim=(2, 3)), maps.std(dim=(2, 3))], dim=1).mean(dim=0)
    return pooled / pooled.norm().clamp(min=1e-6)


class _Running:
    """Running per-feature mean and a single global scale, updated by Welford."""

    def __init__(self, dims: int) -> None:
        self.count = 0
        self.mean = torch.zeros(dims)
        self.sum_squares = torch.zeros(dims)

    def update(self, vector: torch.Tensor) -> None:
        """Fold one embedding in."""
        self.count += 1
        delta = vector - self.mean
        self.mean += delta / self.count
        self.sum_squares += delta * (vector - self.mean)

    def normalize(self, vectors: torch.Tensor) -> torch.Tensor:
        """Centre, then divide by the mean spread over features."""
        if self.count < 2:
            return vectors - self.mean
        scale = (self.sum_squares / (self.count - 1)).sqrt().mean().clamp(min=1e-6)
        return (vectors - self.mean) / scale


class _OnlineHead:
    """A growable classifier over fixed-length vectors, trained by replay."""

    def __init__(
        self,
        dims: int = DIMS,
        hidden: int = HIDDEN,
        learning_rate: float = LEARNING_RATE,
        batch: int = BATCH,
        per_class: int = PER_CLASS,
        balanced: bool = True,
        seed: int = SEED,
    ) -> None:
        self.dims = dims
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.batch = batch
        self.per_class = per_class
        self.balanced = balanced
        self.generator = torch.Generator().manual_seed(seed)
        self.stats = _Running(dims)
        self.labels: list[str] = []
        self.index: dict[str, int] = {}
        self.buffer: list[deque[torch.Tensor]] = []
        self.trunk: nn.Module | None = None
        self.output: nn.Linear | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    def grow(self, label: str) -> None:
        """Give a newly seen font an output row, preserving the ones already there.

        The new row starts at zero rather than random: it contributes a logit of
        zero for every crop until it has been trained, which is a neutral opening
        bid against classes that have learned to be confident. Balanced replay then
        puts it in the very next batch, so it does not stay neutral for long.

        Adam's moments are dropped here, because they are shaped like the old
        parameter. It costs the existing classes their momentum, once per font.
        """
        self.index[label] = len(self.labels)
        self.labels.append(label)
        self.buffer.append(deque(maxlen=self.per_class))

        in_features = self.hidden or self.dims
        grown = nn.Linear(in_features, len(self.labels))
        with torch.no_grad():
            grown.weight.zero_()
            grown.bias.zero_()
            if self.output is not None:
                grown.weight[: self.output.out_features] = self.output.weight
                grown.bias[: self.output.out_features] = self.output.bias
        self.output = grown

        if self.trunk is None and self.hidden:
            # Random, unlike the output layer: zeros in a hidden layer would make
            # every unit identical and keep them that way.
            hidden_layer = nn.Linear(self.dims, self.hidden)
            for parameter in hidden_layer.parameters():
                with torch.no_grad():
                    bound = self.dims**-0.5
                    parameter.uniform_(-bound, bound, generator=self.generator)
            self.trunk = nn.Sequential(hidden_layer, nn.ReLU())

        self.optimizer = torch.optim.Adam(self._net().parameters(), lr=self.learning_rate)

    def _net(self) -> nn.Module:
        """The head as one module, whether or not there is a hidden layer."""
        assert self.output is not None
        return self.output if self.trunk is None else nn.Sequential(self.trunk, self.output)

    def predict_vector(self, vector: torch.Tensor) -> str | None:
        """The font whose row scores this embedding highest, or None before any."""
        if self.output is None:
            return None
        with torch.no_grad():
            logits = self._net()(self.stats.normalize(vector))
        return self.labels[int(logits.argmax())]

    def learn_vector(self, vector: torch.Tensor, label: str) -> None:
        """Remember the item, then take one step on a batch replayed around it."""
        if label not in self.index:
            self.grow(label)
        target = self.index[label]
        self.stats.update(vector)
        self.buffer[target].append(vector)
        self._step(vector, target)

    def _step(self, vector: torch.Tensor, target: int) -> None:
        """One Adam step on the current item plus a replayed batch."""
        assert self.optimizer is not None
        vectors, targets = self._replay(vector, target)
        loss = nn.functional.cross_entropy(self._net()(self.stats.normalize(vectors)), targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _replay(self, vector: torch.Tensor, target: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The current item, then ``batch - 1`` drawn from the buffer.

        Drawn per font when balanced: pick a font uniformly, then one of its
        remembered crops. A font seen fifty times and a font seen five thousand
        times then contribute equally to the step, which is what keeps the rare ones
        from being written off.
        """
        vectors = [vector]
        targets = [target]
        stocked = [index for index, items in enumerate(self.buffer) if items]
        for _ in range(self.batch - 1):
            if self.balanced:
                pick = stocked[int(torch.randint(len(stocked), (1,), generator=self.generator))]
            else:
                weights = torch.tensor([float(len(self.buffer[index])) for index in stocked])
                pick = stocked[int(torch.multinomial(weights, 1, generator=self.generator))]
            items = self.buffer[pick]
            offset = int(torch.randint(len(items), (1,), generator=self.generator))
            vectors.append(items[offset])
            targets.append(pick)
        return torch.stack(vectors), torch.tensor(targets)


class GraftNet(Recognizer):
    """A growable head trained by replay on frozen ResNet-18 embeddings."""

    name = "graftnet"

    def __init__(self) -> None:
        self.trunk = _build_trunk()
        self.head = _OnlineHead()
        # The loop hands predict() and learn() the same image object, so the forward
        # pass — the expensive half of this model — happens once per item rather than
        # twice.
        self._cached_for: Image.Image | None = None
        self._cached = torch.zeros(DIMS)

    def _vector(self, image: Image.Image) -> torch.Tensor:
        """The embedding of this crop, computed once and remembered."""
        if image is not self._cached_for:
            self._cached_for, self._cached = image, embed(self.trunk, image)
        return self._cached

    def predict(self, image: Image.Image) -> str | None:
        """Argmax over the head's logits for this crop's embedding."""
        return self.head.predict_vector(self._vector(image))

    def learn(self, image: Image.Image, label: str) -> None:
        """Buffer the crop's embedding and take one gradient step."""
        self.head.learn_vector(self._vector(image), label)
