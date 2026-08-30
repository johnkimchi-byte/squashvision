"""Tell play apart from the breaks between games.

Between games the two players walk off, and coaches, officials and team-mates
wander on.  The tracker has no idea those are not players -- it happily follows
a stranger in a warm-up jacket -- so court positions during a break are
meaningless, and anything measured across one is worse than nothing.

The signal is occupancy: how often two player-sized bodies are on court at the
same time.  Measured on the Bates capture, over 100-second windows:

    play   (t=300-400)   62% of frames show two or more
    break  (t=405-505)   15%
    play   (t=505-590)   73%

A single frame does not separate them -- one blob is common during play when
the pair merge or one stands still -- so the test is over a window of frames,
not an instant.
"""

from __future__ import annotations

from dataclasses import dataclass

PLAY_WINDOW = 15.0          # seconds of history the occupancy test looks at
PLAY_FRACTION = 0.35        # two-body frames needed within it to call it play
MIN_SEGMENT = 8.0           # ignore play or break stretches shorter than this


@dataclass
class Segment:
    """A stretch of video that is either play or a break."""

    start_s: float
    end_s: float
    playing: bool

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s

    def __str__(self) -> str:
        return "%-5s %8.2f -> %8.2f s  (%5.1f s)" % (
            "play" if self.playing else "BREAK", self.start_s, self.end_s,
            self.duration)


def two_body_fraction(samples, dt: float, window: float = PLAY_WINDOW):
    """Per sample, the fraction of a centred window holding two-plus bodies."""
    half = max(1, int(round(window / dt)) // 2)
    flags = [1.0 if s.blobs >= 2 else 0.0 for s in samples]
    out = []
    running = sum(flags[:half + 1])
    lo, hi = 0, min(len(flags), half + 1)
    for k in range(len(flags)):
        new_lo, new_hi = max(0, k - half), min(len(flags), k + half + 1)
        while hi < new_hi:
            running += flags[hi]; hi += 1
        while lo < new_lo:
            running -= flags[lo]; lo += 1
        out.append(running / max(1, hi - lo))
    return out


def mark_play(samples, dt: float, fraction: float = PLAY_FRACTION,
              window: float = PLAY_WINDOW, min_segment: float = MIN_SEGMENT):
    """Set `in_play` on every sample, and return the segments found."""
    occupancy = two_body_fraction(samples, dt, window)
    raw = [o >= fraction for o in occupancy]

    # Drop segments too short to be a real break or a real rally, so a couple
    # of merged frames cannot manufacture a "break" in the middle of a rally.
    runs = []
    start = 0
    for k in range(1, len(raw) + 1):
        if k == len(raw) or raw[k] != raw[start]:
            runs.append([start, k - 1, raw[start]])
            start = k
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, (a, b, state) in enumerate(runs):
            span = samples[b].time_s - samples[a].time_s
            if span < min_segment:
                runs[i][2] = not state          # absorb into its neighbours
                changed = True
        merged = [runs[0]]
        for run in runs[1:]:
            if run[2] == merged[-1][2]:
                merged[-1][1] = run[1]
            else:
                merged.append(run)
        if len(merged) != len(runs):
            changed = True
        runs = merged

    for a, b, state in runs:
        for k in range(a, b + 1):
            samples[k].in_play = state
    return [Segment(samples[a].time_s, samples[b].time_s, state)
            for a, b, state in runs]


def suppress_breaks(samples) -> int:
    """Discard tracked positions outside play; returns how many were dropped.

    During a break the tracker is following spectators, so its output is not
    just noisy, it is about the wrong people.  Dropping it is the adjustment:
    a gap is honest, a position on a passing coach is not.
    """
    dropped = 0
    for s in samples:
        if s.in_play:
            continue
        for i in (0, 1):
            if s.slots[i] is not None:
                s.slots[i] = None
                dropped += 1
    return dropped


def main(argv=None) -> None:
    from .. import cli
    from .players import analyse

    p = cli.parser("Split a match into play and the breaks between games.", __doc__)
    cli.add_span_arguments(p, stride=3)
    cli.add_tracking_arguments(p)
    p.add_argument("--fraction", type=float, default=PLAY_FRACTION,
                   help="two-body frames needed in a window to call it play")
    args = p.parse_args(argv)
    cli.apply_tracking(args)

    video = cli.video_info(args.video)
    samples, _ = analyse(args.video, args.start, args.duration, args.stride,
                         cli.progress())
    dt = video.dt(args.stride)
    segments = mark_play(samples, dt, args.fraction)
    dropped = suppress_breaks(samples)
    playing = sum(s.duration for s in segments if s.playing)
    total = sum(s.duration for s in segments)
    print("%d segments; %.0f s of play out of %.0f s (%.0f%%); "
          "%d tracked positions discarded as not-play"
          % (len(segments), playing, total, 100 * playing / max(total, 1e-9), dropped))
    for s in segments:
        print("   " + str(s))


if __name__ == "__main__":
    main()
