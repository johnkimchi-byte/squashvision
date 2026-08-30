"""Classify each rally as a winner, an error, or a let -- from the scoreboard.

Neither is directly observable without a ball, but both follow from two facts
that are: who hit a rally's last shot (`shots.py`), and which side the score
moved to (`scoredigits.py`).  If the side that scored also hit last, the shot
was a winner -- unreturnable.  If the other side scored, the last hitter's
shot was an error -- tin or out.  A rally end with no score change nearby is
a let: a replay, or a false alarm from `rallies.find_rallies`.

The one piece this needs that neither module hands over directly is *which*
scoreboard row belongs to *which* tracked player.  PAR scoring (games to 11,
winner serves next) makes that self-fixing: the side that just scored serves
the next rally, and `shots.py` already says who that server is from the
pre-serve formation.  Correlating "row that moved" against "who served next"
over many rallies pins the mapping down, and the agreement rate is a free
check that the mapping -- and everything upstream of it -- is not confused.
This assumes PAR scoring; checked here by the games running to 11 rather
than 9 (game 1 on squashvisiontest.mp4 ends 11-6).  English scoring would
break the serves-next inference, hence `--board-rows` as a manual escape.

A **stroke** -- a point awarded by the referee for obstruction, with no
winning shot at all -- looks identical to a winner or an error in this data:
it is just a score change near a rally end.  There is no movement signal
that tells a stroke apart from either, so strokes are not separated out;
they land in whichever bucket the timing puts them in, silently.

    python -m squashvision winners VIDEO --start 300 --duration 290
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass

from .. import cli
from ..detect.play import mark_play, suppress_breaks
from ..score.scoredigits import read as read_score
from . import rallies as R
from . import shots as S

# How far from a rally's end to look for the matching score change, in
# either direction.  Measured on squashvisiontest.mp4 t=300-590: the three
# rallies whose end plausibly matched a real point landed within +-0.14s of
# it -- not several seconds after, which is what scoreboard.py's operator-lag
# figure (1-2s) would suggest, and one of the three led rather than lagged.
# `find_rallies`'s own end-of-play estimate apparently already absorbs
# whatever lag exists.  3.0s is >20x that observed offset, generous margin
# while still well short of the next-nearest wrong candidates, which were
# 5.5s+ away in the same data.
MATCH_WINDOW = 3.0          # seconds, either side of rally.end_s


@dataclass
class Outcome:
    """What happened at the end of one rally."""

    rally_index: int
    end_s: float
    kind: str                    # "winner", "error", or "let"
    scorer: int | None = None    # track slot that won the rally, if known
    game: int | None = None      # 0-based, from the scoreboard cell


def _server_of(shots_in_rally) -> int | None:
    return shots_in_rally[0].player if shots_in_rally else None


def _last_hitter_of(shots_in_rally) -> int | None:
    return shots_in_rally[-1].player if shots_in_rally else None


def _nearest_point(end_s: float, points, window: float = MATCH_WINDOW):
    """The scoreboard point that plausibly belongs to this rally, if any.

    Searches both directions from rally.end_s -- see MATCH_WINDOW.
    """
    candidates = [p for p in points if abs(p.time_s - end_s) <= window]
    return min(candidates, key=lambda p: abs(p.time_s - end_s)) if candidates else None


def fit_row_to_slot(rallies_found, shots_by_rally, points, window: float = MATCH_WINDOW):
    """The row->slot mapping, and how well it held.

    Votes come from consecutive rallies: whichever row scored to end rally N
    should match the slot serving rally N+1, under PAR scoring.  Returns
    (mapping, agreement, votes) where mapping is {0: slot, 1: slot} and
    agreement is the fraction of votes the winning mapping explains.
    """
    tally = {(0, 0): 0, (1, 1): 0, (0, 1): 0, (1, 0): 0}
    for i in range(len(rallies_found) - 1):
        point = _nearest_point(rallies_found[i].end_s, points, window)
        server_next = _server_of(shots_by_rally.get(i + 1, []))
        if point is None or server_next is None:
            continue
        tally[(point.row, server_next)] += 1
    straight = tally[(0, 0)] + tally[(1, 1)]
    crossed = tally[(0, 1)] + tally[(1, 0)]
    votes = straight + crossed
    if votes == 0:
        return {0: 0, 1: 1}, 0.0, 0            # nothing to go on; identity default
    if straight >= crossed:
        return {0: 0, 1: 1}, straight / votes, votes
    return {0: 1, 1: 0}, crossed / votes, votes


def classify(rallies_found, shots_found, points, row_to_slot, window: float = MATCH_WINDOW):
    """One Outcome per rally: winner, error, or let."""
    by_rally: dict[int, list] = {}
    for shot in shots_found:
        by_rally.setdefault(shot.rally_index, []).append(shot)
    for lst in by_rally.values():
        lst.sort(key=lambda s: s.time_s)

    outcomes = []
    current_game = 0
    for i, rally in enumerate(rallies_found):
        point = _nearest_point(rally.end_s, points, window)
        if point is None:
            outcomes.append(Outcome(i, rally.end_s, "let", game=current_game))
            continue
        current_game = max(current_game, point.game)
        scorer = row_to_slot[point.row]
        last_hitter = _last_hitter_of(by_rally.get(i, []))
        if last_hitter is None:
            # No shots at all for this rally (e.g. it failed to anchor and
            # detected zero candidates) -- a score change happened, so it is
            # not a let, but there is nothing to judge winner-vs-error from.
            outcomes.append(Outcome(i, rally.end_s, "let", scorer, point.game))
            continue
        kind = "winner" if last_hitter == scorer else "error"
        outcomes.append(Outcome(i, rally.end_s, kind, scorer, point.game))
    return outcomes


def save(path: str, outcomes: list[Outcome]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rally_index", "end_s", "kind", "scorer", "game"])
        for o in outcomes:
            writer.writerow([o.rally_index, o.end_s, o.kind, o.scorer, o.game])


def main(argv=None) -> None:
    p = cli.parser("Classify rallies as winners, errors or lets, from the scoreboard.",
                   __doc__)
    cli.add_span_arguments(p, stride=2)
    cli.add_tracking_arguments(p)
    cli.add_court_argument(p)
    p.add_argument("--out", default="winners.csv")
    p.add_argument("--window", type=float, default=MATCH_WINDOW,
                   help="seconds either side of a rally end to look for a score change")
    p.add_argument("--board-rows", choices=("0:0,1:1", "0:1,1:0"), default=None,
                   help="override the fitted row->slot mapping (English scoring "
                        "or a fit that failed to agree)")
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

    print("reading the scoreboard...")
    reading = read_score(args.video, args.start, args.duration, progress=cli.progress())
    print("%d scoreboard points read" % len(reading.points))

    by_rally: dict[int, list] = {}
    for shot in shots_found:
        by_rally.setdefault(shot.rally_index, []).append(shot)
    for lst in by_rally.values():
        lst.sort(key=lambda s: s.time_s)

    if args.board_rows:
        row_to_slot = {int(a): int(b) for pair in args.board_rows.split(",")
                       for a, b in [pair.split(":")]}
        print("row->slot mapping: %s (manual override)" % row_to_slot)
        agreement, votes = None, 0
    else:
        row_to_slot, agreement, votes = fit_row_to_slot(rallies_found, by_rally,
                                                         reading.points, args.window)
        print("row->slot mapping: %s (agreement %.0f%% over %d votes)"
              % (row_to_slot, 100 * agreement, votes))
        if votes < 5:
            print("WARNING: fewer than 5 votes -- this mapping is not well "
                  "checked; consider --board-rows to set it by hand")

    outcomes = classify(rallies_found, shots_found, reading.points, row_to_slot, args.window)

    games = sorted({o.game for o in outcomes if o.game is not None})
    for g in games:
        in_game = [o for o in outcomes if o.game == g]
        winners = sum(1 for o in in_game if o.kind == "winner")
        errors = sum(1 for o in in_game if o.kind == "error")
        lets = sum(1 for o in in_game if o.kind == "let")
        print("game %d: %d winners, %d errors, %d lets (%d rallies)"
              % (g, winners, errors, lets, len(in_game)))

    matched = sum(1 for o in outcomes if o.kind in ("winner", "error"))
    lets_total = sum(1 for o in outcomes if o.kind == "let")
    print("\n%d rallies matched to a score change, %d lets, %d rallies total"
          % (matched, lets_total, len(outcomes)))
    print("%d scoreboard points read; %d unmatched (strokes, or a rally the "
          "detector missed entirely)" % (len(reading.points), len(reading.points) - matched))
    for slot in (0, 1):
        counted = sum(1 for o in outcomes if o.kind == "winner" and o.scorer == slot) \
            + sum(1 for o in outcomes if o.kind == "error" and o.scorer == slot)
        board = sum(reading.final.get((row, game), 0)
                    for row, s in row_to_slot.items() if s == slot
                    for game in games)
        print("   slot %d: %d rallies attributed vs %d points on the board "
              "across the games seen here" % (slot, counted, board))

    save(args.out, outcomes)
    print("%d outcomes -> %s" % (len(outcomes), args.out))


if __name__ == "__main__":
    main()
