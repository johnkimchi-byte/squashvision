"""Fit the detection settings to a particular camera and encode.

The court calibration transfers between recordings of the same court, because
it is stored as fractions of the frame.  The *detection* settings do not: they
depend on how strongly the players differ from the background, which changes
with the encode, the lighting and what the players are wearing.  A white shirt
against a pale court is a much weaker signal than a navy one.

This sweeps the settings over several windows spread through the match and
keeps whichever lets the tracker *observe both players* most often.

It scores on tracking, not on how many blobs the detector finds, because those
two disagree.  A cheap "how often are exactly two bodies visible" proxy was
tried first and picked settings that were measurably worse to track with
(69% both-players against 83%): loosening the size limits produces more
two-blob frames while quietly feeding the tracker junk to choose between.

    python -m squashvision autotune VIDEO --out profile.json
"""

from __future__ import annotations

import itertools
import json

import cv2
import numpy as np

from .. import cli
from ..detect import players as P

# Candidate settings.  Kept coarse on purpose: each combination costs a pass
# over every sampled window, and the score is not smooth enough for a fine grid
# to mean anything.
GRID = {
    "DIFF_THRESHOLD": (12, 15, 18, 22, 25, 30),
    "PLAYER_MIN_AREA": (300.0, 400.0, 550.0),
    "PLAYER_MIN_HEIGHT": (35, 45),
}
WINDOWS = 5                 # sample points spread through the match
WINDOW_S = 12.0             # seconds analysed at each


def _sample_windows(video: str, count: int, seconds: float, stride: int):
    """Grab greyscale ROI frames from several places in the match."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit("cannot open " + video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    span = int(seconds * fps)
    windows = []
    # Spread the samples across the middle 80%: the head and tail of a
    # recording are usually warm-up or an empty court.
    for i in range(count):
        start = int(total * (0.1 + 0.8 * (i + 0.5) / count))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for k in range(span):
            ok, frame = cap.read()
            if not ok:
                break
            if k % stride == 0:
                frames.append(P._prepare(frame))
        if len(frames) > 5:
            windows.append(frames)
    cap.release()
    if not windows:
        raise SystemExit("could not read any frames from " + video)
    return windows


def _score(windows, settings, stride: int = 2) -> float:
    """Fraction of frames in which the tracker observes both players.

    Observed, not merely filled in: a position the motion model coasted to is
    not evidence the settings found anybody.
    """
    saved = P.current_profile()
    P.apply_profile(settings)
    try:
        seen = total = 0
        for frames in windows:
            stack = np.stack([g for g, _ in frames])
            background = np.median(stack[::5], axis=0).astype(np.uint8)
            tracker = P.Tracker()
            for i, (gray, lab) in enumerate(frames):
                sample = tracker.step(i * stride, i * stride / 30.0, gray, lab,
                                      background, stride)
                seen += all(d is not None and d.source != "predicted"
                            for d in sample.slots)
                total += 1
        return seen / max(total, 1)
    finally:
        P.apply_profile(saved)


def tune(video: str, windows: int = WINDOWS, seconds: float = WINDOW_S,
         stride: int = 2, grid=GRID, progress=None):
    """Search the grid; returns the best settings and the full ranking."""
    frames = _sample_windows(video, windows, seconds, stride)
    names = list(grid)
    combos = list(itertools.product(*(grid[n] for n in names)))
    results = []
    for i, values in enumerate(combos):
        settings = dict(zip(names, values))
        results.append((_score(frames, settings, stride), settings))
        if progress:
            progress(i + 1, len(combos))
    results.sort(key=lambda r: -r[0])
    return results[0][1], results


def save(path: str, video: str, settings: dict, score: float,
         baseline: float) -> None:
    # Every key a profile can set, not only the searched ones.  A file that
    # omits a key inherits whatever the code's default happens to be when it is
    # next loaded, so an old profile silently changes meaning the day a default
    # does -- PLAYER_MAX_AREA, which the grid does not search, is the one this
    # actually bites.
    blob = {
        "video": video,
        "detection": dict(P.current_profile(), **settings),
        "searched": sorted(settings),
        "fit": {"both_players_observed": round(score, 3),
                "both_players_observed_with_defaults": round(baseline, 3)},
        "note": "Detection settings only. The court calibration is separate "
                "and transfers between recordings of the same camera. Keys "
                "outside `searched` were not fitted; they record the defaults "
                "this fit ran against, so the profile means the same thing "
                "later as it did here.",
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)


def main(argv=None) -> None:
    p = cli.parser("Fit detection settings to a camera and encode.", __doc__)
    p.add_argument("video")
    p.add_argument("--out", default="detection_profile.json")
    p.add_argument("--windows", type=int, default=WINDOWS,
                   help="sample points spread through the match")
    p.add_argument("--seconds", type=float, default=WINDOW_S,
                   help="seconds analysed at each")
    p.add_argument("--stride", type=int, default=2,
                   help="analyse every Nth frame")
    args = p.parse_args(argv)

    def progress(done, total):
        if done % 10 == 0 or done == total:
            print("  tried %d/%d settings" % (done, total), flush=True)

    print("sampling %d windows of %.0f s..." % (args.windows, args.seconds), flush=True)
    best, results = tune(args.video, args.windows, args.seconds, args.stride,
                         progress=progress)
    baseline = next((s for s, settings in results
                     if settings == P.current_profile()), None)

    print("\ntop settings by how often the tracker observes both players:")
    for score, settings in results[:6]:
        print("   %5.1f%%  %s" % (100 * score, settings))
    if baseline is not None:
        print("\ncurrent defaults score %.1f%%; best is %.1f%%"
              % (100 * baseline, 100 * results[0][0]))
    save(args.out, args.video, best, results[0][0], baseline or 0.0)
    print("\nprofile -> %s   (pass it with --profile %s)" % (args.out, args.out))


if __name__ == "__main__":
    main()
