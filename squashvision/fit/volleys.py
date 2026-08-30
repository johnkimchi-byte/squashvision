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

Four features, from the shot instant plus the striker's recent track:

  depth_short   -- shot depth relative to the short line (metres; +back)
  interval_norm -- time since the previous shot in the rally, normalised by
                   that rally's median inter-shot interval; a volley takes
                   time away from the opponent, so this runs low.
                   **Contributes nothing on the shipped path** -- see below
  radial_v      -- rate of change of the striker's distance from the T over
                   REACH_WINDOW of history; negative means they were not
                   travelling outward when they struck
  out_reach     -- how far above their recent minimum that distance sits: how
                   far out they came to reach this ball

Point-biserial against the volley label, measured twice: on 117 hand-marked
shots (14 volleys), and on the detected shots this command actually runs on.
Quote the right column -- they are not the same feature set in practice.

                     hand instants   detected instants
    radial_v            -0.391            -0.239
    out_reach           -0.369            -0.215
    depth_short         -0.245            -0.274
    interval_norm       -0.256            +0.038

`radial_v` and `out_reach` describe the *shape* of the excursion -- the
quantity `shots.MIN_PROMINENCE` spends on rejecting candidates rather than on
describing them -- and are the strongest predictors available.  They come from
the continuous tracks, so no missing shot can break them; they still lose
~40% of their signal to an instant landing up to MARK_TOLERANCE away from the
real strike, which is why the two columns differ for them at all.

**`interval_norm` earns nothing here and is kept deliberately.**  It is the
one feature computed across two *detected* shots, and `shots.py` misses 43%
of them, so a missed predecessor roughly doubles it: the detected series sits
at mean 1.396, sd 1.900 where a median-normalised interval must centre on 1.0
(hand marks: 1.052, sd 0.328).  It is retained because the signal is real and
recoverable -- level with `depth_short` on clean instants -- and returns as
soon as shot timing improves.  Do not read its hand-instant figure as a
current contribution, and do not drop it as dead without re-measuring both
columns.  See `HANDOFF_4_shot_type.md` S3.

Two features were dropped on measurement, not taste.  `depth_median` (depth
against the rally's own median) correlates with `depth_short` at r = 0.97,
carries no separate information, and its sign flips in the multivariate fit.
`forward_vy` (court-y velocity at the shot) is noise: +0.063 on detected
instants, -0.015 on hand-marked ones -- not even a stable sign.

Also measured and rejected, so they are not retried: every opponent-derived
feature (opponent depth, distance between players, who is further forward,
opponent distance from the T) scores |r| <= 0.17, and a model built from them
alone reaches AP 0.165 against a 0.12 floor.  So does the incoming ball path
approximated by the opponent's last radial apex before the shot: |r| <= 0.15,
AP 0.202 alone.  Squash is a two-body game but volleys, measured here, are
not a two-body signal.

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

Scored by **average precision on pooled held-out predictions**, not by
precision and recall averaged over folds.  Two failures that avoids.  A fold
predicting no positives scores precision 0.0 through `_prf` -- "abstained"
read as "wrong" -- and averaging it in understates the model: measured here,
0.03 averaged against ~0.09 pooled.  And a 0.5 threshold is close to
meaningless on an 11%-positive class, where ridge logistic predicts
all-negative by construction; ranking is what a searchable shot directory
consumes anyway.  Average precision must always be quoted against the
positive rate, which is its no-skill floor.

    python -m squashvision volleys VIDEO --labels labels.json --out volley_config.json
"""

from __future__ import annotations

import json
import math
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

# Reporting points for precision@k, not a tuned threshold: a person scanning a
# ranked clip list looks at a screenful.  With ~100 held-out shots and ~11%
# positives, k=5/10/20 spans "the top of the list" to "a fifth of it".
PRECISION_AT = (5, 10, 20)

# History used by radial_v and out_reach.  Set to the median hand-marked
# inter-shot interval (1.3 s) rounded down, so the window spans one excursion
# without reaching back into the previous shot's.  Not swept: at 14 positives
# the held-out score cannot separate nearby values, so a sweep would be
# fitting noise.  Sweep it when more volleys are labelled.
REACH_WINDOW = 1.0           # seconds


@dataclass
class Fitted:
    weights: list            # [intercept, w_depth_short, w_interval, w_radial_v, w_out_reach]
    mean: list
    std: list
    train_precision: float
    train_recall: float
    train_ap: float
    heldout_precision: float
    heldout_recall: float
    heldout_ap: float        # the figure to quote; compare against base_rate
    heldout_at_k: dict       # k -> precision among the top k ranked shots
    base_rate: float         # positive share = what average precision must beat
    labelled: int
    positive: int


def _reach(track, tee, k: int, k_start: int, dt: float):
    """(radial_v, out_reach) from REACH_WINDOW of the striker's history, or None.

    Both describe the excursion the striker was on when they struck.  The
    window is clipped at `k_start` -- the rally's first sample -- so a shot
    early in a rally never reads its history out of the break before it, where
    the players are wandering and the radial series means nothing.
    """
    span = max(1, int(round(REACH_WINDOW / dt)))
    lo = max(k_start, k - span)
    history = [math.dist(p, tee) for p in track[lo:k + 1] if p is not None]
    if len(history) < 2:
        return None
    return (history[-1] - history[0]) / REACH_WINDOW, history[-1] - min(history)


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
        tee = (court.spec.half, court.spec.short_line)
        intervals = [b.time_s - a.time_s for a, b in zip(group, group[1:])]
        median_interval = statistics.median(intervals) if intervals else None
        for pos, shot in enumerate(group):
            if pos == 0 or not median_interval or median_interval <= 1e-6:
                rows.append((ridx, shot, None))
                continue
            interval = shot.time_s - group[pos - 1].time_s
            k = _nearest_index(samples, k0, k1, shot.time_s)
            reach = _reach(tracks[shot.player], tee, k, k0, dt)
            if reach is None:
                rows.append((ridx, shot, None))
                continue
            radial_v, out_reach = reach
            features = [
                shot.position_m[1] - court.spec.short_line,
                interval / median_interval,
                radial_v,
                out_reach,
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


def average_precision(y_true, score) -> float:
    """Mean precision at the rank of each positive -- area under the PR curve.

    The headline metric, in place of precision/recall at a 0.5 threshold.
    That threshold is close to meaningless on a class this rare: ridge
    logistic on an 11%-positive set parks the intercept low and predicts
    all-negative, which `_prf` reports as precision 0.0 -- indistinguishable
    from a model that predicted positives and got them all wrong.  Measured
    on the full label set, the fit scored 0.00/0.00 at 0.5 while still
    ranking well above chance.

    Ranking is also what the tool actually consumes: a searchable directory
    of shots wants "most likely first", never a hard yes/no.

    The no-skill value is the positive rate, so never quote this without it.
    """
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y = np.asarray(y_true, dtype=float)[order]
    total = y.sum()
    if total == 0:
        return 0.0
    hits = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((hits * y).sum() / total)


def precision_at_k(y_true, score, k: int) -> float:
    """Share of the top k ranked shots that are really positive.

    What a person scanning a ranked clip list actually experiences.
    """
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")[:k]
    y = np.asarray(y_true, dtype=float)[order]
    return float(y.mean()) if len(y) else 0.0


def cross_validate_by_rally(rally_ids, X, y, folds: int = FOLDS, l2: float = L2):
    """Fit on some rallies, predict the rallies left out, and **pool** the result.

    Returns (held_out_y, held_out_score, per_fold).  The pooling is the point.
    The first version scored each fold with `_prf` and averaged, which is a
    broken estimator here: `_prf` returns precision 0.0 when a fold predicts
    no positives at all, so an abstaining fold is scored identically to a
    wrong one.  Measured on the 174-label set, two of three folds predicted
    nothing and the averaged precision came out **0.03** where pooling the
    same predictions gives ~0.09 -- a threefold difference produced entirely
    by the estimator, not the model.

    Pooling also puts every held-out shot into one ranking, which is what
    `average_precision` and `precision_at_k` need; a per-fold ranking of
    twenty-odd shots is too short to mean anything.
    """
    unique = sorted(set(rally_ids))
    if len(unique) < 2:
        return np.array([]), np.array([]), []
    fold_of = {r: i % folds for i, r in enumerate(unique)}
    rally_ids = np.array(rally_ids)
    pooled_y, pooled_score, per_fold = [], [], []
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
        score = predict(beta, (X[test_mask] - mean) / std)
        pooled_y.append(y[test_mask])
        pooled_score.append(score)
        per_fold.append({"fold": f, "n": int(test_mask.sum()),
                         "positive": int(y[test_mask].sum()),
                         "predicted": int((score >= 0.5).sum())})
    if not pooled_y:
        return np.array([]), np.array([]), []
    return np.concatenate(pooled_y), np.concatenate(pooled_score), per_fold


def fit(rows, folds: int = FOLDS, l2: float = L2) -> tuple[Fitted, list] | None:
    """rows: [(rally_index, is_volley, features), ...] with no Nones."""
    rally_ids = [r[0] for r in rows]
    y = np.array([r[1] for r in rows], dtype=float)
    X = np.array([r[2] for r in rows], dtype=float)

    ho_y, ho_score, per_fold = cross_validate_by_rally(rally_ids, X, y, folds, l2)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-9] = 1.0
    beta = fit_logistic((X - mean) / std, y, l2)
    train_score = predict(beta, (X - mean) / std)
    train_p, train_r = _prf(y, (train_score >= 0.5).astype(int))

    # Pooled across folds, then scored once -- see cross_validate_by_rally.
    if len(ho_y):
        ho_p, ho_r = _prf(ho_y, (ho_score >= 0.5).astype(int))
        ho_ap = average_precision(ho_y, ho_score)
        at_k = {k: precision_at_k(ho_y, ho_score, k) for k in PRECISION_AT}
    else:
        ho_p = ho_r = ho_ap = 0.0
        at_k = {k: 0.0 for k in PRECISION_AT}

    fitted = Fitted(
        weights=beta.tolist(), mean=mean.tolist(), std=std.tolist(),
        train_precision=train_p, train_recall=train_r,
        train_ap=average_precision(y, train_score),
        heldout_precision=ho_p, heldout_recall=ho_r, heldout_ap=ho_ap,
        heldout_at_k=at_k,
        base_rate=float(ho_y.mean()) if len(ho_y) else float(y.mean()),
        labelled=len(rows), positive=int(y.sum()),
    )
    return fitted, per_fold


def save(path: str, video: str, fitted: Fitted) -> None:
    blob = {
        "video": video,
        "features": ["depth_short", "interval_norm", "radial_v", "out_reach"],
        "weights": fitted.weights, "mean": fitted.mean, "std": fitted.std,
        "fit": {
            "labelled_shots": fitted.labelled, "positive": fitted.positive,
            "base_rate": round(fitted.base_rate, 3),
            "train_precision": round(fitted.train_precision, 3),
            "train_recall": round(fitted.train_recall, 3),
            "train_average_precision": round(fitted.train_ap, 3),
            "heldout_precision": round(fitted.heldout_precision, 3),
            "heldout_recall": round(fitted.heldout_recall, 3),
            "heldout_average_precision": round(fitted.heldout_ap, 3),
            "heldout_precision_at_k": {str(k): round(v, 3)
                                       for k, v in fitted.heldout_at_k.items()},
        },
        "note": "quote heldout_average_precision, against base_rate as the "
                "no-skill floor.  train_* is fitted and scored on the same "
                "shots and is optimistic.  heldout_precision/recall are at a "
                "0.5 threshold, which is near-meaningless on a class this "
                "rare -- kept only for comparison with older runs.",
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

    fitted, per_fold = fit(labelled, args.folds, args.l2)
    print("\nper fold (fitted without that rally block, predicted on it):")
    for s in per_fold:
        print("   fold %d  n=%-3d  positives %-3d  predicted positive at 0.5: %d"
              % (s["fold"], s["n"], s["positive"], s["predicted"]))

    print("\nheld out by rally, predictions pooled across folds and scored once:")
    print("   average precision %.3f   <- quote this one (no-skill floor %.3f)"
          % (fitted.heldout_ap, fitted.base_rate))
    for k in sorted(fitted.heldout_at_k):
        print("   precision@%-3d      %.3f" % (k, fitted.heldout_at_k[k]))
    print("   precision %.2f, recall %.2f at a 0.5 threshold  -- near-meaningless "
          "on a %.0f%% class, kept for comparison only"
          % (fitted.heldout_precision, fitted.heldout_recall, 100 * fitted.base_rate))
    print("\nfitted on everything: average precision %.3f  <- optimistic, do not quote"
          % fitted.train_ap)

    save(args.out, args.video, fitted)
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()
