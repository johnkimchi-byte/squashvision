"""Fit the rally-end detector using the scoreboard as free training labels.

Every point played writes itself onto the broadcast scoreboard, so a match
carries its own labels: no hand annotation is needed to tune the detector.

The catch is that it is trivially easy to fool yourself here.  Sweeping
parameters and reporting the best score measures how well the settings fit
*those* points, not how well they will do on a match you have not seen.  So
this splits the match into contiguous time blocks and, for each fold, fits on
some blocks and scores on the ones it did not see.  The number worth quoting is
the held-out one, and it is normally the lower of the two.

    python -m squashvision train VIDEO --out rally_config.json
"""

from __future__ import annotations

import itertools
import json
import statistics
from dataclasses import dataclass

from .. import cli
from ..detect.play import mark_play, suppress_breaks
from ..rally import rallies as R
from ..score.scoreboard import score_changes

# The grid searched.  Deliberately small: with ~20 points of reference, a finer
# grid buys nothing but a better-looking number.
GRID = {
    "quiet_speed": (0.5, 0.7, 0.9, 1.1, 1.3),
    "min_break": (1.5, 2.5, 4.0),
    "serve_bonus": (1.0, 1.6, 2.4),
}
FOLDS = 3


@dataclass
class Fitted:
    parameters: dict
    train_f1: float
    heldout_f1: float
    heldout_precision: float
    heldout_recall: float
    reference_points: int


def f1(detected, reference, tolerance: float) -> float:
    result = R.evaluate(detected, reference, tolerance)
    tp = len(result["matched"])
    fp, fn = len(result["false_alarms"]), len(result["missed"])
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def _window(samples, tracks, lo: float, hi: float):
    """The slice of samples and tracks inside a time range."""
    keep = [k for k, s in enumerate(samples) if lo <= s.time_s < hi]
    if not keep:
        return None, None
    a, b = keep[0], keep[-1] + 1
    return samples[a:b], [t[a:b] for t in tracks]


def search(segments, court, dt, reference, tolerance: float, grid=GRID):
    """Best parameters over one or more contiguous pieces of the match.

    `segments` is a list of (samples, tracks) pieces rather than a single
    span, so a fold can fit on exactly the blocks it is allowed to see: fitting
    on the whole match and filtering the reference afterwards would let a
    held-out block push the parameters around through its own false alarms,
    since a detection with no matching reference point there still scores as
    one.  Their rally-end times are pooled before scoring, not scored piece by
    piece and averaged, so precision and recall over the pieces come out the
    same as over one contiguous span.
    """
    best, best_score = None, -1.0
    names = list(grid)
    for values in itertools.product(*(grid[n] for n in names)):
        params = dict(zip(names, values))
        ends = []
        for seg_samples, seg_tracks in segments:
            found, _ = R.find_rallies(seg_samples, seg_tracks, court, dt, **params)
            ends += R.rally_end_times(found)
        score = f1(sorted(ends), reference, tolerance)
        if score > best_score:
            best, best_score = params, score
    return best, best_score


def _excluding(samples, tracks, lo: float, hi: float):
    """The parts of the match outside [lo, hi), as `search` segments.

    Splitting the match this way is not free: `find_rallies` treats the first
    and last sample of each piece as a play boundary, so cutting one span into
    two manufactures up to two rally edges that a single contiguous fit would
    not have seen.  With ~20 reference points that is noise next to the false
    alarms this avoids, but it is not nothing.
    """
    pieces = []
    for a, b in ((samples[0].time_s, lo), (hi, samples[-1].time_s + 1.0)):
        if b <= a:
            continue
        seg_samples, seg_tracks = _window(samples, tracks, a, b)
        if seg_samples:
            pieces.append((seg_samples, seg_tracks))
    return pieces


def cross_validate(samples, tracks, court, dt, reference, tolerance: float,
                   folds: int = FOLDS, grid=GRID):
    """Fit on some blocks of the match, score on the blocks left out."""
    span_lo, span_hi = samples[0].time_s, samples[-1].time_s
    edges = [span_lo + (span_hi - span_lo) * i / folds for i in range(folds + 1)]
    scores = []
    for i in range(folds):
        lo, hi = edges[i], edges[i + 1]
        test_ref = [t for t in reference if lo <= t < hi]
        train_ref = [t for t in reference if not (lo <= t < hi)]
        if not test_ref or not train_ref:
            continue
        # Fit on everything except this block -- see _excluding and search.
        params, _ = search(_excluding(samples, tracks, lo, hi), court, dt,
                           train_ref, tolerance, grid)
        test_samples, test_tracks = _window(samples, tracks, lo, hi)
        if test_samples is None:
            continue
        found, _ = R.find_rallies(test_samples, test_tracks, court, dt, **params)
        result = R.evaluate(R.rally_end_times(found), test_ref, tolerance)
        tp = len(result["matched"])
        fp, fn = len(result["false_alarms"]), len(result["missed"])
        scores.append({
            "fold": i, "range": (lo, hi), "parameters": params,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
            "precision": result["precision"], "recall": result["recall"],
            "points": len(test_ref),
        })
    return scores


def fit(args, labels_from: str = "digits", folds: int = FOLDS,
        show_progress: bool = False) -> tuple[Fitted, list]:
    """Fit on scoreboard-derived labels, and score on blocks left out.

    Takes the parsed arguments rather than a dozen parameters: every one of
    them was being threaded through unchanged from the command line.
    """
    video, tolerance = args.video, args.tolerance
    start, duration = args.start, args.duration
    court = cli.load_court(args)

    if labels_from == "digits":
        # Reading the digits is preferred because it is checkable: the tally it
        # ends on must equal the score on the board.  Change detection cannot
        # be checked at all -- it only ever reports that *something* moved.
        from ..score.scoredigits import point_times
        reference = point_times(video, start, duration)
    else:
        reference = [c.time_s for c in score_changes(video, start, duration)]
    if len(reference) < 4:
        raise SystemExit("only %d scoreboard points found -- too few to fit on. "
                         "Check the score box with `squashvision scoreboard`."
                         % len(reference))

    run = cli.analyse_in_court(args, court, show_progress=show_progress)
    samples, tracks, dt = run.samples, run.tracks, run.dt
    mark_play(samples, dt)
    suppress_breaks(samples)

    scores = cross_validate(samples, tracks, court, dt, reference, tolerance, folds)
    params, train_score = search([(samples, tracks)], court, dt, reference, tolerance)
    fitted = Fitted(
        parameters=params,
        train_f1=train_score,
        heldout_f1=statistics.fmean(s["f1"] for s in scores) if scores else 0.0,
        heldout_precision=statistics.fmean(s["precision"] for s in scores) if scores else 0.0,
        heldout_recall=statistics.fmean(s["recall"] for s in scores) if scores else 0.0,
        reference_points=len(reference),
    )
    return fitted, scores


def save(path: str, video: str, fitted: Fitted) -> None:
    blob = {
        "video": video,
        "parameters": fitted.parameters,
        "fit": {
            "reference_points": fitted.reference_points,
            "train_f1": round(fitted.train_f1, 3),
            "heldout_f1": round(fitted.heldout_f1, 3),
            "heldout_precision": round(fitted.heldout_precision, 3),
            "heldout_recall": round(fitted.heldout_recall, 3),
        },
        "note": "train_f1 is fitted and scored on the same points; heldout_* "
                "is the honest figure.",
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)


def main(argv=None) -> None:
    p = cli.parser("Fit the rally-end detector on scoreboard-derived labels.",
                   __doc__)
    cli.add_span_arguments(p, stride=3)
    cli.add_tracking_arguments(p)
    cli.add_court_argument(p)
    p.add_argument("--tolerance", type=float, default=5.0,
                   help="seconds a detection may be off and still count")
    p.add_argument("--folds", type=int, default=FOLDS,
                   help="time blocks to hold out, one at a time")
    p.add_argument("--out", default="rally_config.json")
    p.add_argument("--labels-from", choices=("digits", "change"), default="digits",
                   help="how to read points off the scoreboard")
    args = p.parse_args(argv)
    cli.apply_tracking(args)

    fitted, scores = fit(args, args.labels_from, args.folds, show_progress=True)

    print("\n%d scoreboard points used as labels" % fitted.reference_points)
    print("\nper fold (fitted without that block, scored on it):")
    for s in scores:
        print("   fold %d  t=%.0f..%.0f  %d points  F1 %.2f  (P %.2f R %.2f)  %s"
              % (s["fold"], s["range"][0], s["range"][1], s["points"],
                 s["f1"], s["precision"], s["recall"], s["parameters"]))
    print("\nheld-out F1 %.2f  (precision %.2f, recall %.2f)   <- quote this one"
          % (fitted.heldout_f1, fitted.heldout_precision, fitted.heldout_recall))
    print("fitted-on-everything F1 %.2f  <- optimistic, do not quote"
          % fitted.train_f1)
    print("\nfinal parameters: %s" % fitted.parameters)
    save(args.out, args.video, fitted)
    print("saved -> %s   (use with: squashvision rallies --config %s)"
          % (args.out, args.out))


if __name__ == "__main__":
    main()
