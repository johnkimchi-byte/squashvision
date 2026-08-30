"""Fit a volley classifier on hand labels, held-out scored by rally.

A volley -- struck before the bounce -- is not directly observable either,
but three things about a shot correlate with it and none of them need a
ball: how far forward it was played, how little time passed since the shot
before it, and whether the player was still moving forward when they hit it.
Silhouette height was tried and set aside: measured in the brief at the
0.2-0.3s reach timescale, the noise a segmentation wobble adds (p75 6-21%)
swamps the ~15% a raised arm contributes, and `_components()` in
`detect/players.py` discards the tallest reach frames outright before
tracking ever sees them (observed maxima 103/109px against a 110 ceiling).
So it is not used as a feature at all, rather than included and expected to
carry weight it cannot.

Three features, all derived from `shots.py` output and court geometry:

  depth_short   -- shot depth relative to the short line (metres; +back)
  depth_median  -- shot depth relative to *this rally's* median shot depth,
                   which separates a genuinely short ball from a player whose
                   whole rally was played deep
  interval_norm -- time since the previous shot in the rally, normalised by
                   that rally's median inter-shot interval; a volley takes
                   time away from the opponent, so this runs low
  forward_vy    -- the player's court-y velocity at the shot, metres/second
                   toward the front wall; positive means still closing in

A serve (a rally's first shot) has no previous interval and is never a
volley by definition, so it is excluded from training rather than given a
fabricated interval.

Logistic regression, ridge-regularised: with as few labels as hand-marking
can realistically produce, an unregularised fit is one separable fold away
from diverging, and there is not enough data to justify anything past a
linear model.  L2 strength is the standard weak default (1.0 in standardized
feature units) chosen to keep the fit well-posed on a small label set, not a
value measured from this data -- there is no data yet to measure it from.

Held out **by rally**, not by time block like `train.py`: a rally's shots
share a court position and pace, so leaving one in both train and test folds
would let the model partly memorise it rather than generalise.

    python -m squashvision volleys VIDEO --labels labels.json --out volley_config.json
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass

import numpy as np

from .. import cli
from ..detect.play import mark_play, suppress_breaks
from ..rally import label as L
from ..rally import rallies as R
from ..rally import shots as S

L2 = 1.0                     # ridge strength in standardized feature units
NEWTON_ITERS = 25
FOLDS = 3
MIN_LABELS = 8               # too few to hold out even one fold sensibly
MARK_TOLERANCE = 1.0         # seconds a hand mark may be off a detected shot


@dataclass
class Fitted:
    weights: list            # [intercept, w_depth_short, w_depth_median, w_interval, w_vy]
    mean: list
    std: list
    train_precision: float
    train_recall: float
    heldout_precision: float
    heldout_recall: float
    labelled: int
    positive: int


def _velocity_y(track, k: int, dt: float):
    n = len(track)
    if 0 < k < n - 1 and track[k - 1] is not None and track[k + 1] is not None:
        return (track[k + 1][1] - track[k - 1][1]) / (2 * dt)
    if k + 1 < n and track[k] is not None and track[k + 1] is not None:
        return (track[k + 1][1] - track[k][1]) / dt
    if k - 1 >= 0 and track[k] is not None and track[k - 1] is not None:
        return (track[k][1] - track[k - 1][1]) / dt
    return None


def _nearest_index(samples, k0: int, k1: int, t: float) -> int:
    return min(range(k0, k1), key=lambda k: abs(samples[k].time_s - t))


def build_features(samples, tracks, court, rallies_found, shots_found, dt):
    """One feature row per non-serve shot: (rally_index, shot, features) or
    (rally_index, shot, None) where a feature could not be computed."""
    by_rally: dict[int, list] = {}
    for shot in shots_found:
        by_rally.setdefault(shot.rally_index, []).append(shot)
    for lst in by_rally.values():
        lst.sort(key=lambda s: s.time_s)

    rows = []
    for ridx, rally in enumerate(rallies_found):
        group = by_rally.get(ridx, [])
        if len(group) < 2:
            continue
        idxs = [k for k, s in enumerate(samples) if rally.start_s <= s.time_s < rally.end_s]
        if not idxs:
            continue
        k0, k1 = idxs[0], idxs[-1] + 1
        depths = [s.position_m[1] for s in group]
        median_depth = statistics.median(depths)
        intervals = [b.time_s - a.time_s for a, b in zip(group, group[1:])]
        median_interval = statistics.median(intervals) if intervals else None
        for pos, shot in enumerate(group):
            if pos == 0 or not median_interval or median_interval <= 1e-6:
                rows.append((ridx, shot, None))
                continue
            interval = shot.time_s - group[pos - 1].time_s
            k = _nearest_index(samples, k0, k1, shot.time_s)
            vy = _velocity_y(tracks[shot.player], k, dt)
            if vy is None:
                rows.append((ridx, shot, None))
                continue
            features = [
                shot.position_m[1] - court.spec.short_line,
                shot.position_m[1] - median_depth,
                interval / median_interval,
                -vy,             # y grows toward the back wall; flip so + = forward
            ]
            rows.append((ridx, shot, features))
    return rows


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = L2, iters: int = NEWTON_ITERS):
    """Ridge logistic regression by Newton-Raphson (IRLS).

    X excludes the intercept column; standardization and the intercept are
    handled by the caller so the penalty can skip the intercept correctly.
    """
    n, d = X.shape
    design = np.hstack([np.ones((n, 1)), X])
    beta = np.zeros(d + 1)
    penalty = np.eye(d + 1) * l2
    penalty[0, 0] = 0.0          # never regularise the intercept
    for _ in range(iters):
        p = _sigmoid(design @ beta)
        w = np.clip(p * (1 - p), 1e-6, None)
        gradient = design.T @ (p - y) + penalty @ beta
        hessian = design.T @ (design * w[:, None]) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        beta = beta - step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def predict(beta, X: np.ndarray) -> np.ndarray:
    design = np.hstack([np.ones((X.shape[0], 1)), X])
    return _sigmoid(design @ beta)


def _prf(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def cross_validate_by_rally(rally_ids, X, y, folds: int = FOLDS, l2: float = L2):
    """Fit on some rallies, score on the rallies left out."""
    unique = sorted(set(rally_ids))
    if len(unique) < 2:
        return []
    fold_of = {r: i % folds for i, r in enumerate(unique)}
    scores = []
    rally_ids = np.array(rally_ids)
    for f in range(folds):
        test_mask = np.array([fold_of[r] == f for r in rally_ids])
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        if len(set(y[train_mask].tolist())) < 2:
            continue          # a fold with only one class cannot fit a boundary
        mean, std = X[train_mask].mean(axis=0), X[train_mask].std(axis=0)
        std[std < 1e-9] = 1.0
        beta = fit_logistic((X[train_mask] - mean) / std, y[train_mask], l2)
        pred = (predict(beta, (X[test_mask] - mean) / std) >= 0.5).astype(int)
        precision, recall = _prf(y[test_mask], pred)
        scores.append({"fold": f, "precision": precision, "recall": recall,
                       "n": int(test_mask.sum())})
    return scores


def fit(rows, folds: int = FOLDS, l2: float = L2) -> tuple[Fitted, list] | None:
    """rows: [(rally_index, is_volley, features), ...] with no Nones."""
    rally_ids = [r[0] for r in rows]
    y = np.array([r[1] for r in rows], dtype=float)
    X = np.array([r[2] for r in rows], dtype=float)

    scores = cross_validate_by_rally(rally_ids, X, y, folds, l2)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-9] = 1.0
    beta = fit_logistic((X - mean) / std, y, l2)
    pred = (predict(beta, (X - mean) / std) >= 0.5).astype(int)
    train_p, train_r = _prf(y, pred)

    fitted = Fitted(
        weights=beta.tolist(), mean=mean.tolist(), std=std.tolist(),
        train_precision=train_p, train_recall=train_r,
        heldout_precision=statistics.fmean(s["precision"] for s in scores) if scores else 0.0,
        heldout_recall=statistics.fmean(s["recall"] for s in scores) if scores else 0.0,
        labelled=len(rows), positive=int(y.sum()),
    )
    return fitted, scores


def save(path: str, video: str, fitted: Fitted) -> None:
    blob = {
        "video": video,
        "features": ["depth_short", "depth_median", "interval_norm", "forward_vy"],
        "weights": fitted.weights, "mean": fitted.mean, "std": fitted.std,
        "fit": {
            "labelled_shots": fitted.labelled, "positive": fitted.positive,
            "train_precision": round(fitted.train_precision, 3),
            "train_recall": round(fitted.train_recall, 3),
            "heldout_precision": round(fitted.heldout_precision, 3),
            "heldout_recall": round(fitted.heldout_recall, 3),
        },
        "note": "train_* is fitted and scored on the same shots; heldout_* "
                "(held out by rally) is the honest figure to quote.",
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)


def main(argv=None) -> None:
    p = cli.parser("Fit the volley classifier on hand labels, held-out by rally.",
                   __doc__)
    cli.add_span_arguments(p, stride=2)
    cli.add_tracking_arguments(p)
    cli.add_court_argument(p)
    p.add_argument("--labels", required=True, metavar="FILE",
                   help="hand labels from `squashvision label` "
                        "(kind=volley are positive; kind=shot/winner negative)")
    p.add_argument("--tolerance", type=float, default=MARK_TOLERANCE,
                   help="seconds a hand mark may be off a detected shot")
    p.add_argument("--folds", type=int, default=FOLDS)
    p.add_argument("--l2", type=float, default=L2)
    p.add_argument("--out", default="volley_config.json")
    args = p.parse_args(argv)
    cli.apply_tracking(args)

    court = cli.load_court(args)
    run = cli.analyse_in_court(args, court, show_progress=True)
    samples, tracks, dt = run.samples, run.tracks, run.dt
    mark_play(samples, dt)
    suppress_breaks(samples)
    rallies_found, _breaks = R.find_rallies(samples, tracks, court, dt)
    shots_found, _stats = S.find_shots(samples, tracks, court, rallies_found)
    print("%d rallies, %d shots" % (len(rallies_found), len(shots_found)))

    marks = L.load(args.labels)
    volley_times = [m.time_s for m in marks if m.kind == "volley"]
    non_volley_times = [m.time_s for m in marks if m.kind in ("shot", "winner")]
    print("%d volley marks, %d non-volley shot marks in %s"
          % (len(volley_times), len(non_volley_times), args.labels))

    rows = build_features(samples, tracks, court, rallies_found, shots_found, dt)
    labelled = []
    for ridx, shot, features in rows:
        if features is None:
            continue
        near_v = min((abs(shot.time_s - t) for t in volley_times), default=1e9)
        near_s = min((abs(shot.time_s - t) for t in non_volley_times), default=1e9)
        if near_v <= args.tolerance and near_v <= near_s:
            labelled.append((ridx, 1, features))
        elif near_s <= args.tolerance:
            labelled.append((ridx, 0, features))

    print("%d detected shots matched to a hand label (%d volley, %d not)"
          % (len(labelled), sum(r[1] for r in labelled),
             sum(1 for r in labelled if r[1] == 0)))
    if len(labelled) < MIN_LABELS or len({r[1] for r in labelled}) < 2:
        raise SystemExit(
            "only %d usable labelled shots (need >= %d, both classes) -- "
            "not enough to fit on.  Label some volleys and non-volley shots "
            "with `squashvision label --volley SECONDS --shot SECONDS`, or "
            "interactively with the v/s keys." % (len(labelled), MIN_LABELS))

    fitted, scores = fit(labelled, args.folds, args.l2)
    print("\nper fold (fitted without that rally block, scored on it):")
    for s in scores:
        print("   fold %d  n=%d  precision %.2f  recall %.2f"
              % (s["fold"], s["n"], s["precision"], s["recall"]))
    print("\nheld-out precision %.2f, recall %.2f   <- quote this one"
          % (fitted.heldout_precision, fitted.heldout_recall))
    print("fitted-on-everything precision %.2f, recall %.2f  <- optimistic, do not quote"
          % (fitted.train_precision, fitted.train_recall))

    save(args.out, args.video, fitted)
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()
