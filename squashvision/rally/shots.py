"""Detect shot instants within each rally, and attribute each to a player.

No ball tracking exists (0/14 impacts resolved on real footage -- see S2 in
the brief) and no usable audio, so shots are found from player movement
alone.  Squash play returns to a stable reference between shots: a player
leaves the T to reach the ball and comes back, so a shot is one excursion
from the T and back.  That makes the shot instant the peak of the excursion
-- equivalently, where radial velocity about the T reverses from outbound to
inbound -- which is what this detects, rather than a plain speed minimum.
Speed alone was rejected as the primary signal: a player's speed also dips
while recovering balance or waiting out an opponent, neither of which is a
shot, whereas a genuine reversal of direction relative to the T is specific
to the reach-and-return pattern a shot requires.

Measured on squashvisiontest.mp4 t=300-400 (module defaults, stride 2): the
already-smoothed court tracks from `cli.analyse_in_court` still carry enough
residual jitter that unfiltered local maxima in radial distance run 7-52 per
player per rally -- far above the 5-20 shots per *rally* sanity bound in the
brief.  Smoothing the radial series further and requiring each peak to stand
above its neighbouring troughs by a minimum prominence, with a minimum gap
between accepted peaks, brings that down to a per-player peak-to-peak
interval with median 2.7-3.1 s across a `SMOOTH_WINDOW` in [0.2, 0.45] s, a
`MIN_SHOT_GAP` in [0.3, 0.5] s and a `MIN_PROMINENCE` in [0.25, 0.6] m --
each player's own successive shots are two rally exchanges apart, so half
that (about 1.4-1.6 s) is the pooled, alternating shot-to-shot interval the
brief's sanity bounds are stated in, and that lands inside 0.8-2.0 s.  The
values below sit in the middle of that stable plateau, not at an edge of it.

Attribution needs no ball either.  Shots strictly alternate between the two
players within a rally, and the server is identifiable from the pre-rally
formation (`rallies.serve_ready` / `rallies.in_service_box`), so server plus
alternation labels every shot without needing to trust which player's track
produced a given candidate -- which matters because a tracker identity swap
during a merge would otherwise misattribute silently.  Candidates from both
players are pooled in time order and walked once, alternating from the
server; a candidate that arrives out of turn is dropped as noise rather than
used to guess a missed shot, which trades recall for never fabricating an
attribution.  Anchoring (finding the server) is re-done independently at
each rally start, so one missed shot cannot flip parity for the rest of the
match.

    python -m squashvision shots VIDEO --start 300 --duration 100 --labels labels.json
"""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass

from .. import cli
from ..detect.play import mark_play, suppress_breaks
from . import rallies as R

# Radial distance from the T is smoothed over this window before peak-picking
# -- on top of the alpha-beta smoothing `cli.analyse_in_court` already
# applies, which damps position noise but not the resulting jitter in a
# *derived* quantity like distance-from-T.  See the module docstring for the
# sweep this and the two constants below were chosen from.
SMOOTH_WINDOW = 0.3          # seconds
MIN_PROMINENCE = 0.4         # metres a peak must stand above its flanks
MIN_SHOT_GAP = 0.4           # seconds between accepted peaks, same player

# How far into a rally to search for the pre-serve formation before giving
# up and calling it unanchored.  Generous: the formation holds until the
# serve is struck, and rallies as short as 2.5 s exist (rallies.MIN_RALLY).
ANCHOR_SEARCH = 2.0           # seconds from rally start


@dataclass
class Shot:
    """One shot: an excursion from the T, attributed by server + alternation."""

    time_s: float
    player: int                  # 0 or 1, from server-at-rally-start + alternation
    position_m: tuple[float, float]
    rally_index: int
    source: str = "radial_peak"  # how the instant was found; one method exists


def _radial(track, tee) -> list[float | None]:
    return [None if p is None else math.dist(p, tee) for p in track]


def _smooth(times, values, window_s: float) -> list[float | None]:
    """Centred moving average over present values within +-window_s/2.

    O(n * window) rather than a running sum: a rally is at most a few hundred
    samples and this runs once per rally per player, so the simpler version
    is not worth the bug surface of a windowed running-sum with gaps in it.
    """
    n = len(values)
    out: list[float | None] = [None] * n
    for k in range(n):
        if values[k] is None:
            continue
        window = [values[j] for j in range(n)
                  if values[j] is not None
                  and abs(times[j] - times[k]) <= window_s / 2]
        out[k] = statistics.fmean(window)
    return out


def _peaks(times, values, min_gap_s: float, min_prominence: float):
    """Local maxima with a prominence floor, non-max-suppressed by time.

    Prominence is the drop to the nearest larger value on each side (or to
    the end of the present run) -- a peak sitting on a plateau of otherwise
    high values is not a real excursion, however tall the plateau.
    """
    idx = [k for k, v in enumerate(values) if v is not None]
    candidates = []
    for pos in range(1, len(idx) - 1):
        k = idx[pos]
        v = values[k]
        if v < values[idx[pos - 1]] or v < values[idx[pos + 1]]:
            continue
        left_min = v
        for j in range(pos - 1, -1, -1):
            left_min = min(left_min, values[idx[j]])
            if values[idx[j]] > v:
                break
        right_min = v
        for j in range(pos + 1, len(idx)):
            right_min = min(right_min, values[idx[j]])
            if values[idx[j]] > v:
                break
        prominence = v - max(left_min, right_min)
        if prominence >= min_prominence:
            candidates.append((times[k], prominence))
    candidates.sort(key=lambda c: -c[1])
    kept = []
    for t, p in candidates:
        if all(abs(t - kt) >= min_gap_s for kt, _ in kept):
            kept.append((t, p))
    kept.sort()
    return kept


def _identify_server(tracks, samples, k0: int, k1: int, court,
                     search_s: float = ANCHOR_SEARCH):
    """Index of the serving player at rally start, or None if unclear.

    Same formation `rallies.serve_ready` checks, but returning *which* side
    is in the box rather than just whether the formation holds -- mirrored
    rather than refactored out of `rallies.py`, since that module is settled
    (see the brief) and this is the only caller that needs the extra value.
    """
    search_end = samples[k0].time_s + search_s
    for k in range(k0, k1):
        if samples[k].time_s > search_end:
            break
        p0, p1 = tracks[0][k], tracks[1][k]
        for server, receiver, idx in ((p0, p1, 0), (p1, p0, 1)):
            box = R.in_service_box(server, court)
            if box is None:
                continue
            if receiver is None or receiver[1] < court.spec.short_line:
                continue
            on_left = receiver[0] < court.spec.half
            if (box == "L") != on_left:
                return idx
    return None


def find_shots(samples, tracks, court, rallies_found,
              smooth_window: float = SMOOTH_WINDOW,
              min_prominence: float = MIN_PROMINENCE,
              min_gap: float = MIN_SHOT_GAP):
    """Shots in every rally, attributed by server + strict alternation.

    Returns (shots, stats) where stats carries the per-rally diagnostics the
    CLI prints: candidates found, how many were dropped as out-of-turn, and
    whether the rally anchored.
    """
    tee = court.spec.half, court.spec.short_line
    shots: list[Shot] = []
    stats = []
    for ridx, rally in enumerate(rallies_found):
        idxs = [k for k, s in enumerate(samples) if rally.start_s <= s.time_s < rally.end_s]
        if not idxs:
            continue
        k0, k1 = idxs[0], idxs[-1] + 1

        pooled = []
        for i in (0, 1):
            times = [samples[k].time_s for k in range(k0, k1)]
            radial = _radial(tracks[i][k0:k1], tee)
            smoothed = _smooth(times, radial, smooth_window)
            for t, prominence in _peaks(times, smoothed, min_gap, min_prominence):
                pooled.append((t, i, prominence))
        pooled.sort(key=lambda c: c[0])

        server = _identify_server(tracks, samples, k0, k1, court)
        expected = server if server is not None else (pooled[0][1] if pooled else 0)
        dropped = 0
        for t, player, _prominence in pooled:
            if player != expected:
                dropped += 1
                continue
            # Position at the shot: the nearest sample's own point for the
            # attributed player, not necessarily the one whose excursion
            # produced this candidate -- attribution comes from alternation,
            # not from which track the peak was measured on.
            nearest = min(range(k0, k1), key=lambda k: abs(samples[k].time_s - t))
            pos = tracks[expected][nearest]
            if pos is not None:
                shots.append(Shot(round(t, 2), expected, pos, ridx, "radial_peak"))
            expected = 1 - expected

        stats.append({
            "rally_index": ridx, "duration": rally.duration,
            "candidates": len(pooled), "dropped": dropped,
            "anchored": server is not None,
        })
    return shots, stats


def save(path: str, shots: list[Shot]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_s", "player", "position_x_m", "position_y_m",
                         "rally_index", "source"])
        for s in shots:
            writer.writerow([s.time_s, s.player, round(s.position_m[0], 3),
                            round(s.position_m[1], 3), s.rally_index, s.source])


def main(argv=None) -> None:
    p = cli.parser("Detect shot instants within each rally, attributed to a player.",
                   __doc__)
    cli.add_span_arguments(p, stride=2)
    cli.add_tracking_arguments(p)
    cli.add_court_argument(p)
    p.add_argument("--out", default="shots.csv")
    p.add_argument("--labels", metavar="FILE",
                   help="hand shot labels from `squashvision label` (kind=shot)")
    p.add_argument("--tolerance", type=float, default=1.0,
                   help="seconds a detected shot may be off and still count")
    p.add_argument("--smooth-window", type=float, default=SMOOTH_WINDOW)
    p.add_argument("--min-prominence", type=float, default=MIN_PROMINENCE)
    p.add_argument("--min-gap", type=float, default=MIN_SHOT_GAP)
    args = p.parse_args(argv)
    cli.apply_tracking(args)

    court = cli.load_court(args)
    run = cli.analyse_in_court(args, court, show_progress=True)
    samples, tracks, dt = run.samples, run.tracks, run.dt
    mark_play(samples, dt)
    suppress_breaks(samples)
    rallies_found, _breaks = R.find_rallies(samples, tracks, court, dt)
    print("%d rallies found" % len(rallies_found))

    shots, stats = find_shots(samples, tracks, court, rallies_found,
                              args.smooth_window, args.min_prominence, args.min_gap)

    unanchored = sum(1 for s in stats if not s["anchored"])
    for s in stats:
        per_shot = [x for x in shots if x.rally_index == s["rally_index"]]
        print("   rally %d (%.1fs): %d candidates, %d dropped out-of-turn, "
              "%d shots kept, %s"
              % (s["rally_index"], s["duration"], s["candidates"], s["dropped"],
                 len(per_shot), "anchored" if s["anchored"] else "UNANCHORED (guessed)"))
    print("%d rallies failed to anchor" % unanchored)

    counts = [sum(1 for x in shots if x.rally_index == s["rally_index"]) for s in stats]
    if counts:
        print("shots per rally: min %d, median %.1f, max %d"
              % (min(counts), statistics.median(counts), max(counts)))
    intervals = []
    for s in stats:
        times = sorted(x.time_s for x in shots if x.rally_index == s["rally_index"])
        intervals += [b - a for a, b in zip(times, times[1:])]
    if intervals:
        print("inter-shot interval: median %.2fs (n=%d)  -- sanity bounds 0.8-2.0s"
              % (statistics.median(intervals), len(intervals)))

    save(args.out, shots)
    print("%d shots -> %s" % (len(shots), args.out))

    if args.labels:
        from . import label as L
        # "shot", "volley" and "winner" are all shots in substance -- see
        # Mark's docstring in label.py -- so all three count as ground truth
        # here.  This module only scores *timing*, not shot type.
        marks = [m for m in L.load(args.labels) if m.kind in ("shot", "volley", "winner")]
        reference = sorted(m.time_s for m in marks)
        if reference:
            result = R.evaluate(sorted(x.time_s for x in shots), reference, args.tolerance)
            print("\nagainst %d hand shot labels (tolerance %.1fs):" % (len(reference), args.tolerance))
            print("   precision %.2f, recall %.2f  (%d matched, %d false, %d missed)"
                  % (result["precision"], result["recall"], len(result["matched"]),
                     len(result["false_alarms"]), len(result["missed"])))
        else:
            print("\nno shot/volley/winner marks in %s" % args.labels)


if __name__ == "__main__":
    main()
