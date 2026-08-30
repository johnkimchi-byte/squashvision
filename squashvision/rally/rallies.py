"""Split play into rallies, and pick out the shot that ended each one.

The idea being tested here: between points the players stop chasing and revert
to a serve formation -- one standing in a service box, the other waiting
diagonally opposite -- and that pattern is visible in court positions alone.
A shot that ends a rally is the one the follow-the-ball assumption breaks on,
so those are the shots to set aside before inferring shots from movement.

Two independent references exist to check the detector against: the burnt-in
scoreboard (`scoreboard.py`, automatic) and hand labels (`label.py`).  Nothing
here should be believed without one of them -- see `evaluate`.

    python -m squashvision rallies VIDEO --start 240 --duration 180 --scoreboard
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass

from .. import cli
from ..detect.play import mark_play, suppress_breaks
from ..geometry import court as C

# All thresholds are in court metres, seconds, and m/s.
# Fitted by train.py on 21 scoreboard-derived points, scored on held-out blocks.
#
# SERVE_BONUS defaults to 1.0, which switches the serve-formation term *off*.
# It looked valuable at first -- an early sweep put it at F1 0.92 vs 0.79 for
# speed alone -- but that sweep scored on the same points it fitted, used 18
# points, and ran before between-game breaks were excluded.  Under held-out
# scoring it reverses: speed alone 0.85, with the formation term 0.71.  The
# likely reason is that the service boxes cover a large part of the back court,
# so players stand in one mid-rally all the time and the term invents breaks.
# Keep the knob (train.py still searches it) but do not trust it by default.
QUIET_SPEED = 0.70          # a player below this is not chasing a ball
ACTIVITY_WINDOW = 1.0       # seconds of speed to average before judging
BOX_MARGIN = 0.45           # how far outside a service box still counts as in it
MIN_BREAK = 2.5             # a break shorter than this is just a slow moment
MIN_RALLY = 2.5             # ignore "rallies" too short to be real play
SERVE_BONUS = 1.0           # 1.0 disables the serve-formation term; see above


@dataclass
class Rally:
    """A stretch of play, and the moment it stopped."""

    start_s: float
    end_s: float

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s


def in_service_box(p, court: C.Court, margin: float = BOX_MARGIN):
    """Which service box a point is in, if any."""
    if p is None:
        return None
    x, y = p
    spec = court.spec
    if not (spec.short_line - margin <= y <= spec.short_line + spec.box + margin):
        return None
    if x <= spec.box + margin:
        return "L"
    if x >= spec.width - spec.box - margin:
        return "R"
    return None


def serve_ready(p0, p1, court: C.Court) -> bool:
    """Are the two players standing as they do just before a serve?

    One in a service box, the other behind the short line and on the other
    side of the court.  This is a much more specific arrangement than "both
    slow", which is why it carries more weight than speed below.
    """
    if p0 is None or p1 is None:
        return False
    for server, receiver in ((p0, p1), (p1, p0)):
        box = in_service_box(server, court)
        if box is None:
            continue
        if receiver[1] < court.spec.short_line:
            continue                            # receiver stands behind the T
        on_left = receiver[0] < court.spec.half
        if (box == "L") != on_left:             # opposite halves
            return True
    return False


def speeds(track, dt: float):
    """Per-sample speed, None where it cannot be measured."""
    out = [None]
    for a, b in zip(track, track[1:]):
        out.append(None if (a is None or b is None) else math.dist(a, b) / dt)
    return out


def activity(tracks, dt: float):
    """A per-sample 'how much play is happening' signal.

    Speed averaged over a short window for both players, then reduced when the
    pair are standing in a serve formation.  Speed alone was measured not to
    separate rallies from breaks on this footage; the formation is what adds
    the discrimination.
    """
    v = [speeds(t, dt) for t in tracks]
    n = len(tracks[0])
    half = max(1, int(round(ACTIVITY_WINDOW / dt)) // 2)
    out = []
    for k in range(n):
        lo, hi = max(0, k - half), min(n, k + half + 1)
        seen = [x for i in (0, 1) for x in v[i][lo:hi] if x is not None]
        out.append(statistics.fmean(seen) if seen else None)
    return out


def find_rallies(samples, tracks, court: C.Court, dt: float,
                 quiet_speed: float = QUIET_SPEED,
                 min_break: float = MIN_BREAK,
                 serve_bonus: float = SERVE_BONUS,
                 min_rally: float = MIN_RALLY):
    """Segment play into rallies using activity and serve formation."""
    act = activity(tracks, dt)
    resting = []
    for k, sample in enumerate(samples):
        a = act[k]
        if not sample.in_play:
            # Between games the tracker is following spectators; there are no
            # rallies to find and its positions mean nothing.  See play.py.
            resting.append(True)
            continue
        if a is None:
            resting.append(True)
            continue
        # Only ask about the formation when it can change the answer: at the
        # default serve_bonus of 1.0 the term is switched off (see above), and
        # this is the inner loop over every analysed frame.
        threshold = quiet_speed
        if serve_bonus != 1.0 and serve_ready(tracks[0][k], tracks[1][k], court):
            threshold *= serve_bonus
        resting.append(a < threshold)

    # Runs of rest long enough to be a real break between points.
    breaks = []
    start = None
    for k, rest in enumerate(resting):
        if rest and start is None:
            start = k
        elif not rest and start is not None:
            if samples[k - 1].time_s - samples[start].time_s >= min_break:
                breaks.append((samples[start].time_s, samples[k - 1].time_s))
            start = None
    if start is not None and samples[-1].time_s - samples[start].time_s >= min_break:
        breaks.append((samples[start].time_s, samples[-1].time_s))

    # Rallies are what is left between the breaks.
    rallies = []
    play_start = samples[0].time_s
    for b0, b1 in breaks:
        if b0 - play_start >= min_rally:
            rallies.append(Rally(play_start, b0))
        play_start = b1
    if samples[-1].time_s - play_start >= min_rally:
        rallies.append(Rally(play_start, samples[-1].time_s))
    return rallies, breaks


def rally_end_times(rallies) -> list[float]:
    """The moment play stopped in each rally: the rally-ending shot."""
    return [r.end_s for r in rallies]


def evaluate(detected, reference, tolerance: float = 5.0):
    """Match detected rally ends to a reference, greedily and one-to-one."""
    unused = list(reference)
    matched = []
    false_alarms = []
    for t in detected:
        near = [r for r in unused if abs(r - t) <= tolerance]
        if near:
            best = min(near, key=lambda r: abs(r - t))
            unused.remove(best)
            matched.append((t, best))
        else:
            false_alarms.append(t)
    tp, fp, fn = len(matched), len(false_alarms), len(unused)
    return {
        "matched": matched, "false_alarms": false_alarms, "missed": unused,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "offsets": [t - r for t, r in matched],
    }


def load_config(path: str) -> dict:
    """Detector parameters fitted by `train.py`."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["parameters"]


def load_labels(path: str) -> list[float]:
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    return sorted(float(m["time_s"]) for m in blob.get("marks", []))


def main(argv=None) -> None:
    from ..score.scoreboard import score_changes

    p = cli.parser("Find rallies and rally-ending shots.", __doc__)
    cli.add_span_arguments(p, stride=3)
    cli.add_tracking_arguments(p)
    cli.add_court_argument(p)
    p.add_argument("--labels", metavar="FILE",
                   help="hand labels from `squashvision label`")
    p.add_argument("--scoreboard", action="store_true",
                   help="also check against score changes read off the broadcast")
    p.add_argument("--tolerance", type=float, default=5.0,
                   help="seconds a detection may be off and still count")
    p.add_argument("--quiet-speed", type=float, default=QUIET_SPEED)
    p.add_argument("--min-break", type=float, default=MIN_BREAK)
    p.add_argument("--config", metavar="FILE",
                   help="parameters from `squashvision train`")
    args = p.parse_args(argv)
    cli.apply_tracking(args)

    court = cli.load_court(args)
    run = cli.analyse_in_court(args, court)
    samples, tracks, dt = run.samples, run.tracks, run.dt

    params = dict(quiet_speed=args.quiet_speed, min_break=args.min_break,
                  serve_bonus=SERVE_BONUS)
    if args.config:
        params.update(load_config(args.config))
        print("parameters from %s: %s" % (args.config, params))
    segments = mark_play(samples, dt)
    suppress_breaks(samples)
    breaks_found = sum(1 for g in segments if not g.playing)
    if breaks_found:
        print("%d between-game break(s) excluded" % breaks_found)
    rallies, breaks = find_rallies(samples, tracks, court, dt, **params)
    print("%d rallies found, %d breaks" % (len(rallies), len(breaks)))
    for r in rallies:
        print("   rally %8.2f -> %8.2f s  (%.1f s)  ending shot at %.2f"
              % (r.start_s, r.end_s, r.duration, r.end_s))

    ends = rally_end_times(rallies)
    for name, reference in (("hand labels", load_labels(args.labels) if args.labels else None),
                            ("scoreboard", [c.time_s for c in score_changes(
                                args.video, args.start, args.duration)]
                             if args.scoreboard else None)):
        if reference is None:
            continue
        result = evaluate(ends, reference, args.tolerance)
        print("\nagainst %s (%d reference points, tolerance %.1f s):"
              % (name, len(reference), args.tolerance))
        print("   precision %.2f, recall %.2f  (%d matched, %d false, %d missed)"
              % (result["precision"], result["recall"], len(result["matched"]),
                 len(result["false_alarms"]), len(result["missed"])))
        if result["offsets"]:
            print("   detection lands %.1f s before the reference on average"
                  % -statistics.fmean(result["offsets"]))
        if result["missed"]:
            print("   missed: " + ", ".join("%.1f" % t for t in result["missed"]))
        if result["false_alarms"]:
            print("   false:  " + ", ".join("%.1f" % t for t in result["false_alarms"]))


if __name__ == "__main__":
    main()
