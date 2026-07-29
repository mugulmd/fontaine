"""Charset resolution and glyph-coverage reporting.

Coverage is a *report*, not a silent filter: you control ``assets/fonts``, so a
face that cannot render the corpus charset should be something you hear about.
"""

from __future__ import annotations

import string

#: Named charsets usable in ``configs/fonts.yaml``. Anything not matching a
#: preset name is taken as a literal set of characters.
CHARSET_PRESETS: dict[str, str] = {
    "ascii_alnum": string.ascii_letters + string.digits,
    "ascii_printable": string.ascii_letters + string.digits + string.punctuation,
    "latin1_printable": (
        string.ascii_letters
        + string.digits
        + string.punctuation
        + "àâäçèéêëîïôöùûüÿœæÀÂÄÇÈÉÊËÎÏÔÖÙÛÜŸŒÆ"
        + "áíóúñÁÍÓÚÑ¿¡"
        + "äöüßÄÖÜ"
        + "“”‘’–—…€£"
    ),
}


def resolve_charset(spec: str) -> str:
    """Resolve a charset spec to the literal characters a face must cover.

    ``spec`` is either a preset name from :data:`CHARSET_PRESETS` or a literal
    string of characters. The space character is always excluded — no font needs
    a visible glyph for it.
    """
    charset = CHARSET_PRESETS.get(spec, spec)
    return "".join(dict.fromkeys(char for char in charset if not char.isspace()))
