"""Drawing shared by everything that writes a video or an interactive view.

The three renderers each grew their own copy of the same palette, the same
"darken a strip so text stays readable" trick, and the same fading-trail loop.
They had already drifted: the roster picker carried a comment saying its
colours matched the demo's, which is the sort of thing that stops being true.
One copy here, so a player is the same colour wherever he is drawn.
"""

from __future__ import annotations

import cv2
import numpy as np

# Track 0 amber, track 1 blue.  Chosen to stay apart on a pale court and to
# survive being faded toward the background in a trail.
TRACK_COLOURS = [(90, 220, 255), (255, 150, 60)]        # BGR
INK = (245, 245, 245)
DIM = (150, 150, 155)
GO = (120, 240, 140)
WARN = (90, 90, 250)

FONT = cv2.FONT_HERSHEY_SIMPLEX
COURT_TINT = 210        # trails fade toward the court, not toward black


def fourcc() -> int:
    """mp4v FOURCC, under whichever spelling this OpenCV exposes."""
    factory = getattr(cv2.VideoWriter, "fourcc", None) or cv2.VideoWriter_fourcc
    return int(factory(*"mp4v"))


def writer(path: str, fps: float, size) -> cv2.VideoWriter:
    """An mp4 writer, so no caller has to remember the codec dance."""
    return cv2.VideoWriter(path, fourcc(), fps, size)


def labels() -> list:
    """The two players' names -- from the roster if one is loaded.

    Imported late: this module is drawing, and the tracker imports it.
    """
    from .detect import players
    if all(s is not None for s in players.PLAYER_SHIRTS):
        return list(players.PLAYER_NAMES)
    return ["Player 0", "Player 1"]


def named() -> bool:
    """Did a person say who these two are, or is the tracker guessing?"""
    from .detect import players
    return all(s is not None for s in players.PLAYER_SHIRTS)


def panel(canvas: np.ndarray, x: int, y: int, w: int, h: int,
          alpha: float = 0.55) -> None:
    """Darken a region in place, so overlaid text stays legible over bright court."""
    patch = canvas[y:y + h, x:x + w]
    if patch.size:
        canvas[y:y + h, x:x + w] = (patch * (1 - alpha)).astype(np.uint8)


def swatch(canvas: np.ndarray, x: int, y: int, rgb, size: int = 16) -> None:
    """A small filled square of an RGB colour, outlined so it reads on any ground."""
    r, g, b = rgb
    cv2.rectangle(canvas, (x, y), (x + size, y + size - 2), (int(b), int(g), int(r)), -1)
    cv2.rectangle(canvas, (x, y), (x + size, y + size - 2), (90, 90, 90), 1)


def trail(canvas: np.ndarray, points, colour, max_gap: int = 3,
          thickness: int = 2) -> None:
    """Draw a path whose older samples fade out.

    `points` may contain None for samples with no position; a run of more than
    `max_gap` missing samples is left unbridged rather than drawn across, so
    the picture cannot imply movement that was never observed.
    """
    seen = [(k, p) for k, p in enumerate(points) if p is not None]
    for (k0, p0), (k1, p1) in zip(seen, seen[1:]):
        if k1 - k0 > max_gap:
            continue
        weight = 0.35 + 0.65 * (k1 + 1) / max(len(points), 1)
        shade = tuple(int(c * weight + COURT_TINT * (1 - weight)) for c in colour)
        cv2.line(canvas, p0, p1, shade,
                 1 if weight < 0.7 else thickness, cv2.LINE_AA)


def text(canvas: np.ndarray, message: str, at, scale: float = 0.45,
         colour=INK, thickness: int = 1) -> None:
    """One line of antialiased text, with the font choice made once."""
    cv2.putText(canvas, message, at, FONT, scale, colour, thickness, cv2.LINE_AA)
