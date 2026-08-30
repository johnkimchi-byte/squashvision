"""Player paths in court metres: projecting them, and cleaning them up.

Detections live in image pixels; everything downstream -- distance covered,
speed, whether two players are standing in a serve formation -- wants metres on
the floor.  This is the crossing point, and it is deliberately *not* in the
rendering layer: the rally detector and the parameter fitter both need these
paths and neither of them draws anything.

Positions come from each player's foot contact point rather than the blob
centroid; `court.foot_fraction` explains why that distinction matters.
"""

from __future__ import annotations

import math
import statistics

from .. import overlay
from . import court as C

SMOOTH = 5                  # median window, run before the momentum filter
MAX_SPEED = 7.0             # m/s: a squash player's sprint, so a step above
                            # this is a measurement fault, not movement
MAX_REJECTS = 4             # consecutive outliers before the filter re-seeds

GAP_COAST = 6               # samples the path may be carried across a gap

# Momentum of the alpha-beta filter.  Alpha is how much of each measurement is
# believed: lower means more momentum and a smoother path, at the cost of
# lagging behind a genuine sharp change of direction.
SMOOTHING = {"none": None, "light": 0.50, "normal": 0.32, "strong": 0.18}


def project(samples, small, court: C.Court):
    """Court position per track per sample, or None where unavailable.

    Positions off the court by more than the tolerance are dropped rather than
    reported: they mean the feet were mis-measured (usually a merge, or a box
    whose bottom was clipped by the scoreboard overlay), not that a player left
    the building.
    """
    tracks = [[], []]
    dropped = 0
    for s in samples:
        for i, det in enumerate(s.slots):
            if det is None:
                tracks[i].append(None)
                continue
            ground = C.foot_fraction(det, small)
            if ground is None:          # coasted: no observed feet to project
                tracks[i].append(None)
                continue
            fx, fy = ground
            cx, cy = court.to_court(fx, fy)
            if math.isnan(cx) or not court.inside(cx, cy):
                tracks[i].append(None)
                dropped += 1
                continue
            tracks[i].append((cx, cy))
    return tracks, dropped


def median_filter(track, window: int = SMOOTH):
    """Median-filter a court track, preserving gaps.

    Runs before the momentum filter: a median deletes an isolated spike
    outright, where a smoother would only average it in.
    """
    out = []
    for k, p in enumerate(track):
        if p is None:
            out.append(None)
            continue
        lo, hi = max(0, k - window // 2), min(len(track), k + window // 2 + 1)
        seen = [q for q in track[lo:hi] if q is not None]
        out.append((statistics.median(q[0] for q in seen),
                    statistics.median(q[1] for q in seen)))
    return out


def smooth_track(track, dt: float, alpha: float, max_speed: float = MAX_SPEED):
    """Give a court path momentum, and drop steps no player could have made.

    An alpha-beta filter: the state carries a velocity, so the reported
    position carries on through a noisy sample instead of snapping to it --
    that is the momentum.  `beta = alpha^2 / (2 - alpha)` is the critically
    damped choice, which stops the path ringing after a sharp change of
    direction and avoids inventing a second tuning constant.

    Momentum alone only *attenuates* an outlier though: it smears one bad
    point across the samples that follow.  So a sample that would need more
    than `max_speed` is rejected first and the filter coasts on its velocity
    instead.  If several in a row are rejected the measurement is probably
    right and the state stale -- after a re-acquisition, say -- so the filter
    gives up and re-seeds rather than sailing off in the wrong direction.
    """
    beta = alpha * alpha / (2.0 - alpha)
    gate = max_speed * dt
    out, solid = [], []
    position, velocity, rejected, coasted = None, (0.0, 0.0), 0, 0
    accepted = None                             # last measurement believed

    def emit(point, measured):
        out.append(point)
        solid.append(measured)

    for measurement in track:
        if measurement is None:
            # A short gap is not a reason to break the path -- coast on the
            # velocity, which is exactly what momentum is for.  A long one is:
            # past a few samples the extrapolation is fiction.
            if position is not None and coasted < GAP_COAST:
                coasted += 1
                position = (position[0] + velocity[0] * dt,
                            position[1] + velocity[1] * dt)
                emit(position, False)
            else:
                emit(None, False)
                position, accepted, velocity, rejected = None, None, (0.0, 0.0), 0
            continue
        coasted = 0
        if position is None:
            position, accepted, velocity, rejected = \
                measurement, measurement, (0.0, 0.0), 0
            emit(position, True)
            continue

        # Judge plausibility measurement-to-measurement, never against the
        # filter's own estimate: a heavily smoothed path lags on purpose, and
        # gating on that lag would make stronger smoothing reject more.
        if math.dist(measurement, accepted) > gate:
            rejected += 1
            if rejected <= MAX_REJECTS:
                position = (position[0] + velocity[0] * dt,
                            position[1] + velocity[1] * dt)     # coast
                emit(position, False)
                continue
            # Persistently far away: this is a real relocation, not noise.
            # Re-seed, but report nothing for this sample -- snapping across
            # the gap would manufacture a stride faster than any real one.
            position, accepted, velocity, rejected = \
                measurement, measurement, (0.0, 0.0), 0
            emit(None, False)
            continue

        rejected = 0
        accepted = measurement
        predicted = (position[0] + velocity[0] * dt, position[1] + velocity[1] * dt)
        rx, ry = measurement[0] - predicted[0], measurement[1] - predicted[1]
        position = (predicted[0] + alpha * rx, predicted[1] + alpha * ry)
        vx, vy = velocity[0] + beta * rx / dt, velocity[1] + beta * ry / dt
        speed = math.hypot(vx, vy)
        if speed > max_speed:                   # never let momentum run away
            vx, vy = vx * max_speed / speed, vy * max_speed / speed
        velocity = (vx, vy)
        emit(position, True)
    return out, solid


def prepare(track, dt: float, level: str):
    """The full smoothing chain for one court track."""
    alpha = SMOOTHING[level]
    if alpha is None:
        return list(track), [p is not None for p in track]
    return smooth_track(median_filter(track), dt, alpha)


def distance_covered(track, dt: float, max_speed: float = MAX_SPEED) -> float:
    """Metres travelled, ignoring steps that would break the speed limit."""
    total, previous = 0.0, None
    for p in track:
        if p is None:
            previous = None
            continue
        if previous is not None:
            step = math.dist(previous, p)
            if step <= max_speed * dt:
                total += step
        previous = p
    return total


def peak_speed(track, dt: float) -> float:
    """Fastest implied speed in a track, m/s -- a check on the smoothing."""
    fastest, previous = 0.0, None
    for p in track:
        if p is None:
            previous = None
            continue
        if previous is not None:
            fastest = max(fastest, math.dist(previous, p) / dt)
        previous = p
    return fastest


def summarise_court(raw_tracks, tracks, dropped: int, court, dt: float) -> str:
    short_line = court.spec.short_line
    names = overlay.labels()
    lines = ["court positions: %d dropped as off-court" % dropped]
    for i, track in enumerate(tracks):
        seen = [p for p in track if p is not None]
        if not seen:
            lines.append("  %s: no positions" % names[i])
            continue
        front = sum(1 for p in seen if p[1] < short_line)
        lines.append(
            "  %s: %d positions, %.1f m covered, mean (%.2f, %.2f) m, "
            "%.0f%% of time in the front half"
            % (names[i], len(seen), distance_covered(track, dt),
               statistics.fmean(p[0] for p in seen),
               statistics.fmean(p[1] for p in seen),
               100.0 * front / len(seen)))
        lines.append(
            "            range x %.2f..%.2f m (of 0..%.2f), y %.2f..%.2f m (of 0..%.2f)"
            % (min(p[0] for p in seen), max(p[0] for p in seen), court.spec.width,
               min(p[1] for p in seen), max(p[1] for p in seen), court.spec.length))
        # Peak speed is the honest check on smoothing: a foot point that jumps
        # implies a sprinter, so if this still reads absurd the path is noise.
        lines.append(
            "            peak speed %.1f m/s (was %.1f m/s unsmoothed; a sprint is ~7)"
            % (peak_speed(track, dt), peak_speed(raw_tracks[i], dt)))
    return "\n".join(lines)


