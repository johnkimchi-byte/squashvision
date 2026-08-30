"""Read the scoreboard by watching each digit cell settle, not by OCR.

Detecting *change* in the score box is fragile: on a low-bitrate encode the
cells wobble by two or three grey levels on their own, which reads as a point
every few seconds.  Recognising the digits fixes that, but a font model is more
machinery than this needs.

The trick is that a score cell is not an arbitrary picture.  Within a game it
counts 0, 1, 2, 3 ... in order and never goes back, so the *k*-th distinct
appearance a cell settles into **is** the number k.  That gives a real reading
with no templates, and it turns every detection into a claim that can be
checked: the final count in each cell must equal the final score on the board.

It also removes the failure that motivated this.  A flicker is not a point,
because a point requires the new appearance to hold; and a knock-up at 0-0
scores nothing at all, because the cell never settles anywhere new.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .scoreboard import SCORE_BOX

CELL_COLUMNS = 5            # per-game cells across the box
CELL_ROWS = 2               # one row per player
CELL_SIZE = (24, 32)        # each cell is normalised to this before comparing
DIGIT_LEVEL = 0.55          # digits are light on a dark cell
BLANK_CONTRAST = 22.0       # grey range below which a cell holds no digit
MIN_INK = 0.05              # a digit covers at least this much of its cell
MAX_INK = 0.40              # ...and at most this much; more is a flash
# Comparisons are always one cell against itself over time, never cell against
# cell.  Within a cell the same digit re-renders almost identically (frame to
# frame difference: median 0.000, 90th percentile 0.008), while the closest
# pair of *different* digits measured on this board -- 8 against 0 -- differ by
# 0.077.  So the boundary belongs near the bottom of that gap.  It emphatically
# does not generalise across cells: the same "0" drawn in two different cells
# differs by up to 0.142, more than some genuinely different digits do.
CELL_DISTANCE = 0.04        # fraction of pixels differing to be a new digit
STABLE_S = 1.5              # how long a new reading must hold to be believed
SAMPLE_HZ = 2.0


@dataclass
class Point:
    """One point: a cell settled on the next number up."""

    time_s: float
    row: int                # 0 = top player on the board, 1 = bottom
    game: int               # which per-game cell moved, 0-based
    score: int              # the cell's new value, counted from the clip start

    def __str__(self) -> str:
        return ("t=%8.2f  player %d, game %d -> %d"
                % (self.time_s, self.row, self.game + 1, self.score))


@dataclass
class Reading:
    """Everything read off the board for one clip."""

    points: list = field(default_factory=list)
    final: dict = field(default_factory=dict)      # (row, game) -> value
    sampled: int = 0

    def board(self) -> str:
        """The final tally, laid out like the scoreboard itself."""
        rows = []
        for r in range(CELL_ROWS):
            cells = [self.final.get((r, c), 0) for c in range(CELL_COLUMNS)]
            rows.append("   player %d: %s" % (r, "  ".join("%2d" % v for v in cells)))
        return "\n".join(rows)


def _digit(cell: np.ndarray):
    """Binarise one cell, or None when it does not hold a readable number.

    Two rejections matter as much as the binarising itself:

    A cell with almost no contrast is an unplayed game, printed dim.  Stretching
    it to full range would amplify its noise into a different random pattern
    every frame, which reads as a blizzard of points in a game nobody played.

    A cell that comes out mostly white is the board flashing as it registers a
    point.  A flash holds still long enough to look like a settled reading, so
    without this it gets counted as a number in its own right.
    """
    # Trim the borders before measuring anything: the dividing lines between
    # cells are the darkest thing in the crop, and letting them set the scale
    # made the fainter of the two player rows binarise to half the ink of the
    # brighter one for the very same glyph.
    h, w = cell.shape[:2]
    inset = cell[int(h * 0.14):int(h * 0.86), int(w * 0.14):int(w * 0.86)]
    if inset.size == 0 or float(inset.max() - inset.min()) < BLANK_CONTRAST:
        return None
    # Otsu rather than a fixed level, so a dim row and a highlighted one are
    # read the same way.
    _, binary = cv2.threshold(np.clip(inset, 0, 255).astype(np.uint8), 0, 1,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    small = cv2.resize(binary.astype(np.float32), CELL_SIZE,
                       interpolation=cv2.INTER_AREA)
    mask = (small > 0.5).astype(np.uint8)
    ink = float(mask.mean())
    if not MIN_INK <= ink <= MAX_INK:
        return None
    return mask


def _differs(a, b) -> bool:
    """Do two cell readings show different numbers?"""
    if a is None or b is None:
        return (a is None) != (b is None)
    return float(np.mean(a != b)) > CELL_DISTANCE


def cells(patch: np.ndarray):
    """Split the score box into its per-game, per-player cells."""
    h, w = patch.shape[:2]
    ch, cw = h // CELL_ROWS, w // CELL_COLUMNS
    for r in range(CELL_ROWS):
        for c in range(CELL_COLUMNS):
            yield r, c, patch[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]


def sample(video: str, start_s: float = 0.0, duration_s=None, box=SCORE_BOX,
           progress=None):
    """Binarised cell readings over time: (times, {(row, col): [masks]})."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit("cannot open " + video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(round(start_s * fps))
    last = total if duration_s is None else min(total, first + int(round(duration_s * fps)))
    step = max(1, int(round(fps / SAMPLE_HZ)))

    times: list[float] = []
    series: dict = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    index = first
    while index < last:
        ok, frame = cap.read()
        if not ok:
            break
        if (index - first) % step == 0:
            h, w = frame.shape[:2]
            x0, y0, x1, y1 = box
            patch = cv2.cvtColor(frame[int(y0 * h):int(y1 * h),
                                       int(x0 * w):int(x1 * w)],
                                 cv2.COLOR_BGR2GRAY).astype(np.float32)
            times.append(index / fps)
            for r, c, cell in cells(patch):
                series.setdefault((r, c), []).append(_digit(cell))
            if progress and len(times) % 300 == 0:
                progress(index - first, last - first)
        index += 1
    cap.release()
    return times, series


def read(video: str, start_s: float = 0.0, duration_s=None, box=SCORE_BOX,
         progress=None) -> Reading:
    """Read every point off the board."""
    times, series = sample(video, start_s, duration_s, box, progress)
    if len(times) < 3:
        return Reading()

    hold = max(2, int(round(STABLE_S * SAMPLE_HZ)))
    reading = Reading(sampled=len(times))

    for (row, col), masks in series.items():
        runs = _stable_runs(masks, hold)
        value = 0
        previous = None
        for start, digit in runs:
            if digit is None:
                # The board flashes as it registers a point, and the flash is
                # not a readable number.  Skip over it rather than treating it
                # as a reading: comparing across the gap is the whole point,
                # since every real change looks like digit -> flash -> digit.
                continue
            if previous is not None and _differs(digit, previous):
                value += 1
                reading.points.append(Point(times[start], row, col, value))
            previous = digit
        reading.final[(row, col)] = value

    reading.points.sort(key=lambda p: p.time_s)
    return reading


def _stable_runs(masks, hold: int):
    """Stretches where a cell holds still, as (first index, reading).

    Comparing frame to frame does not work: the board takes a second or two to
    settle after a point, and during that the cell differs from everything,
    including itself a frame ago.  So instead find the stretches where it is
    *not* changing -- those are the readings -- and ignore whatever happens in
    between.  A reading has to hold for `hold` samples to count, which is what
    makes a flicker unable to score.
    """
    runs = []
    start = 0
    for k in range(1, len(masks) + 1):
        if k < len(masks) and not _differs(masks[k], masks[start]):
            continue
        if k - start >= hold:
            runs.append((start, masks[start]))
        start = k
    return runs


def point_times(video: str, start_s: float = 0.0, duration_s=None,
                progress=None) -> list[float]:
    """Just the times, for use as rally-end labels."""
    return [p.time_s for p in read(video, start_s, duration_s, progress=progress).points]


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Read points off a burnt-in scoreboard.")
    p.add_argument("video")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--quiet", action="store_true", help="totals only")
    args = p.parse_args(argv)

    def progress(done, total):
        print("  %d/%d frames" % (done, total), flush=True)

    result = read(args.video, args.start, args.duration, progress=progress)
    print("%d points read from %d sampled frames" % (len(result.points), result.sampled))
    print("final tally, counted from the start of the clip:")
    print(result.board())
    if not args.quiet:
        print()
        previous = None
        for point in result.points:
            gap = "" if previous is None else "   (+%.1f s)" % (point.time_s - previous)
            print("   %s%s" % (point, gap))
            previous = point.time_s


if __name__ == "__main__":
    main()
