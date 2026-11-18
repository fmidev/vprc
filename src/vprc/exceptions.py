"""Exceptions raised by vprc processing."""


class NoUsableProfileError(ValueError):
    """A valid VVP file contains no profile levels that can be processed.

    Raised for expected data conditions rather than invalid input, e.g. a
    header-only product or a profile without levels at or above the lowest
    profile level. Callers can catch it to fall back to climatology without
    treating the file as a processing failure. It subclasses ``ValueError``,
    so existing ``ValueError`` handlers still apply.
    """
