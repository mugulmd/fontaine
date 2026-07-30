from __future__ import annotations

import random
import string
from typing import Any

import pytest

from fontaine.config import CorpusConfig, Range
from fontaine.text.corpus import CONTENT_KINDS, Corpus

ASCII_ALNUM = frozenset(ord(char) for char in string.ascii_letters + string.digits)
FULL_ASCII = frozenset(range(32, 127))


def _corpus(**overrides: Any) -> Corpus:
    return Corpus(CorpusConfig(**overrides))


@pytest.mark.parametrize("kind", CONTENT_KINDS)
def test_every_kind_produces_text(kind: str) -> None:
    corpus = _corpus(kinds={kind: 1.0})
    for seed in range(25):
        sample = corpus.sample(random.Random(seed), FULL_ASCII)
        assert sample.text.strip()
        assert sample.kind == kind


def test_output_is_restricted_to_covered_codepoints() -> None:
    corpus = _corpus()
    for seed in range(60):
        sample = corpus.sample(random.Random(seed), ASCII_ALNUM)
        for char in sample.text:
            assert char == " " or ord(char) in ASCII_ALNUM, sample


def test_dropped_characters_are_reported() -> None:
    # Digits only: any word or punctuation must be stripped and recorded.
    digits = frozenset(ord(char) for char in string.digits)
    corpus = _corpus(kinds={"word": 1.0})
    sample = corpus.sample(random.Random(3), digits)
    assert sample.dropped
    assert sample.kind == "fallback"
    assert set(sample.text) <= set(string.digits)


def test_falls_back_when_nothing_meaningful_survives_projection() -> None:
    corpus = _corpus(kinds={"sentence": 1.0})
    only_b = frozenset({ord("B")})
    sample = corpus.sample(random.Random(11), only_b)
    assert sample.kind == "fallback"
    assert set(sample.text) == {"B"}


def test_projection_closes_up_the_gaps_it_leaves() -> None:
    corpus = _corpus(kinds={"sentence": 1.0})
    vowels_only = frozenset(ord(char) for char in "aeiouAEIOU")
    for seed in range(20):
        sample = corpus.sample(random.Random(seed), vowels_only)
        assert "  " not in sample.text
        assert sample.text == sample.text.strip()


def test_same_seed_gives_same_text() -> None:
    corpus = _corpus()
    first = corpus.sample(random.Random(99), FULL_ASCII)
    second = corpus.sample(random.Random(99), FULL_ASCII)
    assert first == second


def test_uncovered_argument_leaves_text_untouched() -> None:
    corpus = _corpus(kinds={"price": 1.0})
    sample = corpus.sample(random.Random(5), None)
    assert sample.dropped == ""
    assert any(currency in sample.text for currency in "$€£¥")


def test_word_count_respects_config() -> None:
    corpus = _corpus(kinds={"phrase": 1.0}, words=Range(3, 3))
    for seed in range(15):
        sample = corpus.sample(random.Random(seed), FULL_ASCII)
        assert len(sample.text.split()) == 3


def test_numeric_kinds_are_not_recased() -> None:
    # Casing an SKU or a price would change what the content *is*.
    corpus = _corpus(kinds={"token": 1.0}, casing={"lower": 1.0})
    for seed in range(15):
        sample = corpus.sample(random.Random(seed), FULL_ASCII)
        assert sample.text.upper() == sample.text


def test_unknown_kinds_and_casings_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown content kinds"):
        _corpus(kinds={"haiku": 1.0})
    with pytest.raises(ValueError, match="unknown casing"):
        _corpus(casing={"sarcastic": 1.0})
