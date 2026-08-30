"""Read rally ends off the burnt-in scoreboard.

A broadcast scoreboard changes exactly once per point, which makes it a far
better rally-boundary signal than anything in the player movement.  This does
not OCR the digits: it watches the score box for a change that *persists*,
which is enough to timestamp a point and needs no font model.

The operator presses the button a beat after the rally actually ends, so treat
these times as "a point ended shortly before here", not as the instant of the
final shot.  Measured on the Bates capture the lag is about 1-2 s.

    python -m squashvision scoreboard VIDEO --start 240 --duration 180
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .. import cli

# Score cells for the Bates/Amherst capture, as fractions of frame size: *all*
# the per-game digit cells for both players, not just the first.  Covering only
# the first cell reads game 1 and then goes blind, because the live score moves
# one cell right when a new game starts.  Fractions so this survives a different
# decode resolution; a different broadcast needs a new box.
# Measured off a 4x grid of the 1920x1080 encode: the digit cells occupy
# x 1011..1207 and y 973..1067, which is a little tighter than the box first
# used.  Getting this wrong shifts every cell boundary and mixes two games
# together in one reading.
SCORE_BOX = (0.52656, 0.90093, 0.62865, 0.98796)   # x0, y0, x1, y1

SAMPLE_HZ = 2.0             # the score is static for tens of seconds
GAME_CELLS = 5              # per-game score cells the box is split into
CHANGE_LEVEL = 2.5          # floor for the change threshold, in grey levels
NOISE_SIGMAS = 4.0          # how far above an encode's own noise a change sits
PERSIST_S = 3.0             # a real change stays changed; a glitch does not
SETTLE_S = 2.0              # ignore the graphic fading in at the start of a clip
MIN_GAP_S = 6.0             # two points cannot be closer than this; the board
                            # redrawing itself after a game can be, so it is
                            # absorbed into the change that preceded it


@dataclass
class ScoreChange:
    """A point ended at (a little before) this time."""

    time_s: float
    magnitude: float


def _box_pixels(frame: np.ndarray, box=SCORE_BOX) -> np.ndarray:
    """The score box as 8-bit grey.

    Kept as uint8 rather than float32 because every sampled patch is held in
    memory until the pass ends: a half-hour match at 2 Hz is ~3600 of them, and
    on a 3016x1696 source that is the difference between ~160 MB and ~650 MB.
    The subtraction below widens to a signed type where it needs to.
    """
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    patch = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)


def _cell_difference(patch, reference, lo: int, hi: int) -> float:
    """Mean absolute difference over one column band, in grey levels."""
    return float(cv2.absdiff(patch[:, lo:hi], reference[:, lo:hi]).mean())


def _difference(patch, reference) -> float:
    """The largest change in any single game cell.

    Per cell rather than over the whole box: one digit flipping is a big change
    inside its own cell but a small one averaged across a row of five, so a
    whole-box mean goes blind exactly as more games are covered.
    """
    width = patch.shape[1] // GAME_CELLS
    if width < 2:
        return _cell_difference(patch, reference, 0, patch.shape[1])
    return max(_cell_difference(patch, reference, c * width, (c + 1) * width)
               for c in range(GAME_CELLS))


def score_changes(video: str, start_s: float = 0.0, duration_s: float | None = None,
                  box=SCORE_BOX, progress=None, report=None) -> list[ScoreChange]:
    """Times at which the score box changed and stayed changed."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit("cannot open " + video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(round(start_s * fps))
    last = total if duration_s is None else min(total, first + int(round(duration_s * fps)))
    step = max(1, int(round(fps / SAMPLE_HZ)))

    times, patches = [], []
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    index = first
    while index < last:
        ok, frame = cap.read()
        if not ok:
            break
        if (index - first) % step == 0:
            times.append(index / fps)
            patches.append(_box_pixels(frame, box))
            if progress and len(patches) % 200 == 0:
                progress(index - first, last - first)
        index += 1
    cap.release()
    if len(patches) < 3:
        return []

    # Set the threshold from this encode's own noise rather than a constant.
    # A lower-bitrate encode of the same broadcast wobbles the score cells by
    # two or three grey levels all on its own, which a fixed threshold reads as
    # a point every few seconds.  Score changes are rare, so the bulk of the
    # frame-to-frame differences *is* the noise: median plus a wide multiple of
    # its MAD sits above it without needing to know the encode.
    steps = np.array([_difference(b, a) for a, b in zip(patches, patches[1:])])
    middle = float(np.median(steps))
    spread = float(np.median(np.abs(steps - middle))) + 1e-6
    level = max(CHANGE_LEVEL, middle + NOISE_SIGMAS * spread)

    # Compare each sample against the last *settled* score rather than against
    # its neighbour: a one-frame flicker then reverts, a real point does not.
    if report:
        report(level, middle, spread)
    persist = max(1, int(round(PERSIST_S * SAMPLE_HZ)))
    changes: list[ScoreChange] = []
    # Seed from a settled sample, not the first one: at the head of a clip the
    # scoreboard graphic is often still fading in, which reads as a huge change.
    seed = min(len(patches) - 1, int(round(SETTLE_S * SAMPLE_HZ)))
    settled = patches[seed]
    k = seed + 1
    while k < len(patches):
        delta = _difference(patches[k], settled)
        if delta > level:
            ahead = patches[k:k + persist]
            if all(_difference(p, settled) > level for p in ahead):
                if changes and times[k] - changes[-1].time_s < MIN_GAP_S:
                    settled = patches[min(k + persist - 1, len(patches) - 1)]
                    k += persist
                    continue
                changes.append(ScoreChange(times[k], delta))
                settled = patches[min(k + persist - 1, len(patches) - 1)]
                k += persist
                continue
        k += 1
    return changes


def main(argv=None) -> None:
    p = cli.parser("Find rally ends from the scoreboard.", __doc__)
    p.add_argument("video")
    p.add_argument("--start", type=float, default=0.0,
                   help="start time in seconds")
    p.add_argument("--duration", type=float, default=None,
                   help="seconds to read (default: to the end)")
    args = p.parse_args(argv)
    progress = cli.progress()

    def report(level, middle, spread):
        print("change threshold %.2f (encode noise: median %.2f, MAD %.2f)"
              % (level, middle, spread), flush=True)
        if level > 2 * CHANGE_LEVEL:
            print("WARNING: the score box is noisy in this encode, so the "
                  "threshold had to be raised well\n"
                  "         above the floor. Points will be both missed and "
                  "invented -- on a 1920x1080\n"
                  "         re-encode of this broadcast that cost about a "
                  "quarter of the detections.\n"
                  "         Spot-check the times before using them as labels.",
                  flush=True)

    changes = score_changes(args.video, args.start, args.duration,
                            progress=progress, report=report)
    print("%d score changes (points) found" % len(changes))
    previous = None
    for c in changes:
        gap = "" if previous is None else "   (+%.1f s)" % (c.time_s - previous)
        print("   t=%8.2f  cell change %5.2f%s" % (c.time_s, c.magnitude, gap))
        previous = c.time_s


if __name__ == "__main__":
    main()
