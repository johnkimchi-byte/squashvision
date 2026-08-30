"""Track the two players on a squash court from a fixed-camera match video.

Detection is background-subtraction based rather than a deep model: the machine
has no CUDA GPU, and player-sized blobs are separable from the clutter that
defeats ball tracking (compression specks, scoreboard burn-in, spectators
behind the glass) by a simple size floor.

Detections are fed to a pair of constant-velocity Kalman filters.  Players move
~4.5 px between samples against a body 20-60 px wide, so the motion model is
not there to chase fast movement -- it is there to (a) reject associations that
are physically impossible, which is what produced identity swaps, and (b) say
where each player is inside a blob when the two of them merge into one.

Who the two tracks *are* is a separate question from following them, and the
footage cannot always answer it: an arena camera sees spectators at the near
glass who are player-sized, and two players whose shirts differ only in
lightness.  So identity can be supplied by hand -- see `squashvision roster`,
whose file names the two shirt colours and fences off the people who are not
playing.  Everything here works without one; it just has to guess instead.

The parameters below are the ones measured on the Bates/Amherst arena capture.
"""

from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

# Fraction-of-frame crop (x0, y0, x1, y1) that keeps the court and drops the
# adjacent court at frame-left, the crowd above, and the burnt-in overlays.
COURT_ROI = (0.134, 0.138, 0.944, 0.992)

PLAYER_WIDTH = 480          # analysis width in px; all sizes below assume it
PLAYER_MIN_AREA = 400.0
PLAYER_MAX_AREA = 1900.0    # above this, _detections() tries to split a merge
PLAYER_MIN_HEIGHT = 45
# There used to be a PLAYER_MAX_HEIGHT ceiling here (110px) meant to reject
# merged blobs.  Removed 2026-08-30 (T6, HANDOFF_2_winners_volleys.md):
# merges are already caught by PLAYER_MAX_AREA above, and the ceiling
# additionally discarded genuine reaching players (observed reach maxima
# 103/109px, right against 110).  Measured on both match videos before
# keeping this, per the brief's own warning that the change is video-wide:
# neutral on squashvisiontest.mp4 (73/77/78% both-players-resolved at
# stride 1/2/4, t=300-400, identical with or without the ceiling -- it
# turns out to essentially never trigger on this video's analysis scale) and
# a genuine improvement on the Connecticut video (63.3% -> 65.1%
# both-players-observed via autotune).  A first attempt at this measurement
# wrongly compared squashvisiontest's raw per-frame Tracker.source rate
# (autotune's metric) against the documented baseline, which was measured
# through cli.analyse_in_court's smoothed/coasted rate instead -- different
# pipeline stages, not comparable; re-measuring both states with the same
# metric as the baseline is what showed the true (neutral) result.
DIFF_THRESHOLD = 25         # abs-diff level against the local median background
CHUNK_FRAMES = 300          # frames per background-model chunk
BG_SAMPLES = 21             # frames sampled within a chunk to build the median

# Motion model, in ROI pixels and units of one analysed sample.
ACCEL_STD = 5.0             # players reverse direction abruptly; keep this loose
MEAS_STD = 4.0              # centroid of a body blob is itself noisy
N_SIGMA = 3.5               # gate width in predicted standard deviations
GATE_MIN = 18.0             # never gate tighter than this
GATE_MAX = 100.0            # never trust the model further than this
MAX_COAST = 6               # samples a track may be reported from prediction alone
MAX_MISSES = 6              # samples without a measurement before a track dies
COLOUR_WEIGHT = 0.8         # px of association cost per unit of shirt-colour distance
COLOUR_LIGHTNESS_W = 0.4    # lightness counts less than hue: it moves with shadow
COLOUR_MEMORY = 0.95        # shirt colour is fixed; adapt only slowly to lighting
# A shirt colour the *user* named is a different kind of evidence from one the
# tracker taught itself, so it is compared differently.  Lightness is discounted
# above because it swings with shadow, but on the Bates/Connecticut capture it is
# the whole of the difference between the two players: the light shirt measures
# L 146-174 around the court and the dark one 98-123, never overlapping, while
# hue is all but identical.  Against a fixed reference, lightness counts fully.
ANCHOR_LIGHTNESS_W = 1.0
ANCHOR_WEIGHT = 0.6         # px of association cost per unit of distance from it
# One frame's torso colour cannot decide who somebody is.  A differencing
# detector only sees the parts of a player that moved, so a man standing still
# in the back corner contributes an arm and a racket rather than a shirt: over
# ten consecutive frames of the *same* stationary player the colour measured
# here swings from 3 units away from his own reference to 42.  Averaged over a
# second or two it does separate the two players, and that average -- not any
# single frame -- is what identity is settled on.
IDENTITY_MEMORY = 0.88      # weight kept from the evidence so far
IDENTITY_MIN_SAMPLES = 8    # measurements before a track's evidence is trusted
IDENTITY_MARGIN = 4.0       # Lab units of daylight needed to reorder the slots
# Seeding is held to a higher standard than everything after it, because it is
# the one decision made from a single frame and the only one the tracker cannot
# revise cheaply.  Two tests, and a candidate pair must pass both.
#
# SEED_MARGIN is relative: each of the two must lean this far toward its own
# player, measured as (distance from the other man's shirt - distance from its
# own).  It stops a frame where both bodies read as the same colour from
# seeding at all.
#
# SEED_MAX_DISTANCE is absolute, and it is what the relative test cannot do:
# leaning is satisfied by anyone who merely resembles one player *more* than
# the other, which on the Connecticut warm-up included a teammate in a maroon
# hoodie standing outside the court -- 64 units from one shirt and 42 from the
# other, so he "leaned dark" by 22 while looking like nobody at all.  Measured
# over 1669 match-play frames, a player sits this far from his own shirt:
#     light shirt  p10 4.6  median 13.6  p75 22.1  p90 28.2
#     dark shirt   p10 2.8  median  7.3  p75 11.2  p90 15.2
# So the cutoff has to clear the light shirt's upper quartile or it starts
# refusing ordinary play.  Swept on this footage, counting positions actually
# observed (a stricter cutoff costs re-seeding time after both tracks die):
#     off  224 / 3173      28  224 / 3171      36  224 / 3173
#     24   209 / 3169      32  224 / 3173
# over t=300-320 s and t=600-900 s.  24 is measurably too tight; 28 costs
# nothing against no cutoff at all, still refuses the warm-up frame this was
# written for, and leaves 14 units of daylight to that hoodie.
SEED_MARGIN = 6.0
SEED_MAX_DISTANCE = 28.0
# Dropping a track that has settled on furniture.
#
# The tempting rule is "a track that stops moving is not on a player", and on
# its own it is wrong: a differencing detector does not stop seeing a player who
# stands still, and players stand still often.  Over the Connecticut knock-up
# the *real* dark-shirted player held one spot for six seconds -- 47 consecutive
# measurements with 1.8 px of wander between them -- while his partner hit.
# Stillness alone would have deleted him.
#
# What separates the two is stillness *plus* not looking like the man the track
# is called.  Measured over 1459 windows of three seconds, 227 of them nearly
# stationary, the mean distance from the track's own shirt inside a stationary
# window comes out bimodal:
#     p10 2.8   median 6.8   p75 8.7   |   p90 35.1   max 43.1
# The lower mode is players pausing between rallies; the upper is a track sitting
# on somebody at the near glass.  In 1459 windows only two places produced both
# conditions at once, and both were spectators.  Inert without a roster, which is
# the only thing that supplies the "looks like nobody" half.
STILL_WINDOW_S = 3.0        # seconds of measurements the test looks back over
STILL_MIN_SAMPLES = 10      # measurements needed before it will judge
STILL_WANDER = 25.0         # ROI px the positions may span and still count as still
STILL_ANCHOR = 28.0         # ...while sitting this far from its own player's shirt
# A candidate this many times further from a track's own player than from his
# opponent is refused outright, and the track coasts instead.  Refusing stops a
# track whose player has gone still -- and so faded into the background -- from
# adopting the nearest stranger: on this footage that is a spectator at the near
# glass, close enough to the camera to be player-sized.
#
# It applies *only to a track that is already coasting* (see _Track.cost), and
# that restriction is the whole difference between the test helping and hurting.
# Applied on every frame it vetoes the tracked player himself, because a single
# frame's torso colour is unreliable: the light-shirted player regularly reads
# darker than his opponent, trips the ratio, and is refused.  Measured on the
# Connecticut match as (light-shirt measurements / both-players / implausible
# jumps), over t=0-60, 300-420, 600-900 and 900-1200 s:
#     always on     273/91.6/7   594/91.9/20  1481/89.5/47  1205/71.6/31
#     off entirely  325/91.3/7   706/89.7/26  1681/91.5/63  1344/71.6/32
#     coasting only 313/92.2/6   680/92.1/23  1637/90.4/54  1339/70.4/35
# Always-on costs a fifth of one player's data -- the symptom being "it ignores
# the white-shirted player".  Off entirely gives that back but lets a coasting
# track sit on a spectator for about a second.  Restricting it to coasting
# tracks keeps nearly all the measurements *and* the fewest jumps.
ASSOCIATION_RATIO = 2.0
# Nothing this far from a named shirt is that player, whatever the motion model
# thinks.  This is the test that keeps a track off a spectator in a colour
# neither player is wearing -- the relative test above cannot, because such a
# stranger is far from *both* anchors and so leans neither way, and the motion
# gate cannot, because he may be standing exactly where the player was.
#
# Without it the light-shirt slot held a detection more than 35 units from its
# own shirt in 9-15% of measured frames (against 3-6% for the dark slot, which
# is easier to detect against a pale court), with lock-ons of up to 18
# consecutive frames at a mean distance of 46-59.  Swept over 18 minutes,
# counting light-shirt measurements kept and the frames still held beyond 35:
#     off  6189 kept / 281 beyond      40  6065 kept / 135 beyond
#     55   6185 kept / 273 beyond      35  5866 kept /   0 beyond
#     45   6122 kept / 190 beyond
# 40 removes every association beyond 40 for 2% of the light player's
# measurements and 0.07 points of both-player coverage.  Going to 35 costs 5%
# for a band that is genuinely mixed -- at mid-court it is usually the player
# himself, badly measured.
ANCHOR_MAX_DISTANCE = 40.0

# Re-acquiring a lost track is a *relative* test, not an absolute one.  Measured
# on this footage the two players are only 22.3 apart in shirt colour (5th pct
# 9.7), while one player varies by 1.9 (95th pct 8.3) frame to frame -- so any
# absolute cutoff loose enough to re-find a player also admits his opponent.
REVIVE_MAX = 18.0           # candidate must at least look like this player
REVIVE_RATIO = 0.6          # ...and look markedly less like the other one
COLLAPSE_RADIUS = 16.0      # two tracks this close are on one body, not two

# Prediction-guided merge splitting.
SPLIT_PAD = 12.0            # how far outside a blob a prediction may still sit
SPLIT_MIN_SEPARATION = 8.0  # predictions closer than this cannot direct a cut
SPLIT_SEARCH = 7            # px either side of the midpoint to seek a valley
SPLIT_MIN_FRACTION = 0.55   # each half must reach this fraction of the area floor

# Fractions of a player box holding the shirt: below the head, above the shorts.
TORSO_TOP, TORSO_BOTTOM = 0.15, 0.55

# Detection settings that must be re-fitted per camera and per broadcast.  The
# defaults above suit the 3016x1696 Bates capture; a 1920x1080 encode of the
# same court needs DIFF_THRESHOLD 18 rather than 25, because a player in a white
# shirt against a pale court barely differs from the background.  Fit these with
# `squashvision autotune` and pass the result with --profile.
PROFILE_KEYS = ("DIFF_THRESHOLD", "PLAYER_MIN_AREA", "PLAYER_MAX_AREA",
                "PLAYER_MIN_HEIGHT")


def current_profile() -> dict:
    """The detection settings in force right now."""
    return {k: globals()[k] for k in PROFILE_KEYS}


def apply_profile(profile) -> dict:
    """Override detection settings; accepts a dict or a path to one."""
    if profile is None:
        return current_profile()
    if isinstance(profile, str):
        import json
        with open(profile, encoding="utf-8") as fh:
            profile = json.load(fh).get("detection", {})
    unknown = [k for k in profile if k not in PROFILE_KEYS]
    if unknown:
        raise SystemExit("unknown detection setting(s): " + ", ".join(unknown))
    globals().update(profile)
    return current_profile()


# --- who the players are ----------------------------------------------------
# Optional, and supplied by a person rather than fitted: see detect/roster.py.
# Without it the tracker bootstraps identity from the footage -- the two largest
# blobs, brighter one first -- which is only as good as that first frame.

PLAYER_NAMES = ["Player 0", "Player 1"]
PLAYER_SHIRTS = [None, None]    # reference torso Lab per slot, or None
IGNORE_ZONES = ()               # (x0, y0, x1, y1, height) in ROI fractions
CANDIDATE_BLOBS = 4             # blobs carried into association each frame
IGNORE_HEIGHT_TOLERANCE = 0.30  # how far from the fenced blob's height still counts


def apply_roster(roster) -> None:
    """Adopt a roster: reference shirt colours and no-player regions.

    Accepts a path, a dict in the form detect/roster.py writes, or None to
    clear.  Deliberately parses the file itself rather than importing that
    module, which imports this one.
    """
    global PLAYER_NAMES, PLAYER_SHIRTS, IGNORE_ZONES
    if roster is None:
        PLAYER_NAMES = ["Player 0", "Player 1"]
        PLAYER_SHIRTS = [None, None]
        IGNORE_ZONES = ()
        return
    if isinstance(roster, str):
        import json
        with open(roster, encoding="utf-8") as fh:
            roster = json.load(fh)
    entries = roster.get("players", [])
    if len(entries) != 2:
        raise SystemExit("a roster must name exactly two players")
    PLAYER_NAMES = [e.get("name") or "Player %d" % i for i, e in enumerate(entries)]
    PLAYER_SHIRTS = [tuple(e["shirt"]) if e.get("shirt") else None for e in entries]
    zones = []
    for entry in roster.get("ignore", ()):
        if isinstance(entry, dict):
            zones.append(tuple(entry["box"]) + (entry.get("height"),))
        else:
            zones.append(tuple(entry) + (None,))     # raw box: no height recorded
    IGNORE_ZONES = tuple(zones)


def in_zone(zones, x: int, y: int, w: int, h: int, roi_w: int, roi_h: int) -> bool:
    """Is this blob caught by one of these fences?  See _ignored for the rule."""
    fx, fy, fh = (x + w / 2.0) / roi_w, (y + h / 2.0) / roi_h, h / roi_h
    for x0, y0, x1, y1, height in zones:
        if not (x0 <= fx <= x1 and y0 <= fy <= y1):
            continue
        if height is None or abs(fh - height) <= IGNORE_HEIGHT_TOLERANCE * height:
            return True
    return False


def _ignored(x: int, y: int, w: int, h: int, roi_w: int, roi_h: int) -> bool:
    """Is this blob the non-player somebody fenced off?

    Position alone is not enough on a camera that looks down the court from
    behind: the spectators standing at the near glass occupy the same rows as a
    player at the back of the court, and a box drawn round one of them deletes
    the other.  What separates them is size -- everything is bigger near the
    camera, so a 40 px body in a row where a player measures 65 px is not a
    player -- so a fence remembers the height of the blob it was drawn around
    and only catches blobs of about that height.  A zone with no height
    recorded (one passed in as raw coordinates) catches everything inside it.
    """
    return bool(IGNORE_ZONES) and in_zone(IGNORE_ZONES, x, y, w, h, roi_w, roi_h)


# Ground contact point: the silhouette's *centroid* x paired with its lowest
# row.  Both halves were picked by measurement, and both are counter-intuitive:
#  - Taking x from the feet themselves is markedly worse (|dx| p90 0.33 m vs
#    0.11 m), because the feet swing back and forth with every stride while the
#    body's centroid does not.  Lateral position wants the whole silhouette.
#  - A high percentile of the rows, rather than the lowest, biases every mapped
#    position ~0.7 m toward the front wall; requiring a minimum row occupancy
#    is unbiased but noisier in depth (|dy| p90 0.50 m vs 0.39 m).


@dataclass
class Detection:
    """One player-sized blob in one frame, in ROI pixel coordinates.

    `source` records how the position was arrived at: a plain connected
    component ("measured"), one half of a split merge ("split"), or the motion
    model coasting through a gap ("predicted").
    """

    cx: float
    cy: float
    x: int
    y: int
    w: int
    h: int
    area: int
    brightness: float
    shirt: tuple | None = None      # median Lab of the torso, or None if unseen
    foot: tuple | None = None       # ground contact point, or None if not observed
    source: str = "measured"

    @property
    def rgb(self) -> tuple:
        """Shirt colour as 8-bit RGB, for display and for the CSV."""
        if self.shirt is None:
            return (0, 0, 0)
        lab = np.array([[list(self.shirt)]], dtype=np.uint8)
        b, g, r = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0][0]
        return int(r), int(g), int(b)

    def norm(self, roi_w: int, roi_h: int) -> tuple[float, float]:
        """Centre as a fraction of the court ROI."""
        return self.cx / roi_w, self.cy / roi_h


@dataclass
class Sample:
    """What one analysed frame yielded, as two track slots."""

    frame_index: int
    time_s: float
    slots: list = field(default_factory=lambda: [None, None])
    merged: bool = False
    blobs: int = 0              # player-sized components seen, before tracking
    in_play: bool = True        # set by play.mark_play; False between games
    waiting: bool = False       # declined as too ambiguous to start tracks from
    # Per track, the (x, y, radius) the motion model expected before it saw
    # this frame, or None for a track that was not alive.  Kept for display.
    gates: list = field(default_factory=lambda: [None, None])


def _roi_box(width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = COURT_ROI
    return (
        int(round(x0 * width)),
        int(round(y0 * height)),
        int(round(x1 * width)),
        int(round(y1 * height)),
    )


def _prepare(frame: np.ndarray):
    """Downscale, crop to the court, and return (grayscale, Lab) views.

    Differencing runs on the grayscale copy; shirt identity is read from the
    Lab copy, where hue survives the shadow that swings plain brightness.
    """
    h, w = frame.shape[:2]
    scale = PLAYER_WIDTH / w
    small = cv2.resize(frame, (PLAYER_WIDTH, int(round(h * scale))),
                       interpolation=cv2.INTER_AREA)
    x0, y0, x1, y1 = _roi_box(small.shape[1], small.shape[0])
    roi = small[y0:y1, x0:x1]
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)


def _shirt_distance(a, b, lightness_w: float | None = None) -> float:
    """Distance between two shirt colours, discounting lightness by default."""
    if a is None or b is None:
        return float("nan")
    weight = COLOUR_LIGHTNESS_W if lightness_w is None else lightness_w
    dl, da, db = float(a[0] - b[0]), float(a[1] - b[1]), float(a[2] - b[2])
    return math.sqrt(weight * dl * dl + da * da + db * db)


def _anchor_distance(slot: int, shirt) -> float:
    """How far a detection sits from the shirt colour named for a slot."""
    if shirt is None or PLAYER_SHIRTS[slot] is None:
        return float("nan")
    return _shirt_distance(PLAYER_SHIRTS[slot], shirt, ANCHOR_LIGHTNESS_W)


# --- Kalman filter ----------------------------------------------------------

class _Kalman:
    """Constant-velocity filter over (x, y) with an isotropic gate."""

    def __init__(self, x: float, y: float):
        self.state = np.array([x, y, 0.0, 0.0], dtype=float)
        self.cov = np.diag([MEAS_STD ** 2, MEAS_STD ** 2, 100.0, 100.0])

    def predict(self, dt: float) -> None:
        f = np.eye(4)
        f[0, 2] = f[1, 3] = dt
        q = np.array([
            [dt ** 4 / 4, 0.0, dt ** 3 / 2, 0.0],
            [0.0, dt ** 4 / 4, 0.0, dt ** 3 / 2],
            [dt ** 3 / 2, 0.0, dt ** 2, 0.0],
            [0.0, dt ** 3 / 2, 0.0, dt ** 2],
        ]) * ACCEL_STD ** 2
        self.state = f @ self.state
        self.cov = f @ self.cov @ f.T + q

    def update(self, z: tuple[float, float]) -> None:
        h = np.zeros((2, 4))
        h[0, 0] = h[1, 1] = 1.0
        r = np.eye(2) * MEAS_STD ** 2
        innovation = np.asarray(z, dtype=float) - h @ self.state
        s = h @ self.cov @ h.T + r
        gain = self.cov @ h.T @ np.linalg.inv(s)
        self.state = self.state + gain @ innovation
        self.cov = (np.eye(4) - gain @ h) @ self.cov

    @property
    def position(self) -> tuple[float, float]:
        return float(self.state[0]), float(self.state[1])

    @property
    def gate(self) -> float:
        """Radius of the region the player could plausibly be in."""
        spread = math.sqrt(max(0.0, (self.cov[0, 0] + self.cov[1, 1]) / 2.0))
        radius = N_SIGMA * math.sqrt(spread ** 2 + MEAS_STD ** 2)
        return min(GATE_MAX, max(GATE_MIN, radius))


class _Track:
    """One player: a motion filter plus the appearance used to break ties."""

    def __init__(self, det: Detection, slot: int | None = None):
        self.kf = _Kalman(det.cx, det.cy)
        self.brightness = det.brightness
        self.shirt = det.shirt
        self.slot = slot            # which roster entry this track answers to
        self.box = (det.w, det.h)
        self.misses = 0
        self.alive = True
        # Running evidence that the body being followed is roster player 0
        # rather than player 1: positive means it looks like the first.
        self.evidence = 0.0
        self.evidence_n = 0
        # (time, x, y, distance from own shirt) for recent measurements, so the
        # track can be asked whether it has settled on furniture.
        self.seen: list = []
        self._note_identity(det)

    def predict(self, dt: float) -> None:
        self.kf.predict(dt)

    def _note_identity(self, det: Detection) -> None:
        """Fold one measurement into the running who-is-this evidence."""
        score = _anchor_distance(1, det.shirt) - _anchor_distance(0, det.shirt)
        if math.isnan(score):
            return
        self.evidence = score if self.evidence_n == 0 else (
            IDENTITY_MEMORY * self.evidence + (1 - IDENTITY_MEMORY) * score)
        self.evidence_n += 1

    def loitering(self, time_s: float) -> bool:
        """Has this track settled on somebody who is not its player?

        Two conditions together, because neither is sufficient alone -- see
        STILL_WANDER for the measurements.  Needs a roster: without reference
        shirts there is no way to ask the second question, and asking only the
        first would delete a player who is standing still.
        """
        if self.slot is None:
            return False
        recent = [seen for seen in self.seen if seen[0] > time_s - STILL_WINDOW_S]
        if len(recent) < STILL_MIN_SAMPLES:
            return False
        if recent[0][0] > time_s - STILL_WINDOW_S * 0.8:
            return False                     # window not yet full of history
        xs = [seen[1] for seen in recent]
        ys = [seen[2] for seen in recent]
        if math.hypot(max(xs) - min(xs), max(ys) - min(ys)) >= STILL_WANDER:
            return False                     # it is moving; nothing to answer
        anchors = [seen[3] for seen in recent if not math.isnan(seen[3])]
        if not anchors:
            return False
        return sum(anchors) / len(anchors) > STILL_ANCHOR

    def absorb(self, det: Detection, time_s: float) -> None:
        self.kf.update((det.cx, det.cy))
        self._note_identity(det)
        self.seen.append((time_s, det.cx, det.cy,
                          _anchor_distance(self.slot, det.shirt)
                          if self.slot is not None else float("nan")))
        # Only the window matters; anything older is dead weight over a match.
        if len(self.seen) > 2 * STILL_MIN_SAMPLES + 40:
            self.seen = [s for s in self.seen if s[0] > time_s - STILL_WINDOW_S]
        if not math.isnan(det.brightness):
            self.brightness = det.brightness if math.isnan(self.brightness) else (
                0.9 * self.brightness + 0.1 * det.brightness)
        if det.shirt is not None:
            if self.shirt is None:
                self.shirt = det.shirt
            else:
                # Shirts do not change colour; only the light on them does, so
                # adapt slowly.  A fast average would let a track that briefly
                # latched onto the wrong player keep him.
                self.shirt = tuple(COLOUR_MEMORY * old + (1 - COLOUR_MEMORY) * new
                                   for old, new in zip(self.shirt, det.shirt))
        self.box = (det.w, det.h)
        self.misses = 0

    def cost(self, det: Detection) -> float | None:
        """Gated association cost, or None if the detection is out of reach.

        Motion decides what is reachable; shirt colour decides which of the
        reachable candidates is actually this player.  At a crossing the two
        players are both within a few px of the prediction, so colour is what
        breaks the tie.

        Two colours are consulted, and they answer different questions.  The
        track's own running colour asks "is this the body I was following a
        frame ago", which tolerates the light changing across the court.  The
        roster colour asks "is this the man the user pointed at", and it is the
        only one of the two that cannot drift: a track that latched onto the
        wrong player carries its mistake forward in the running colour, but not
        in the anchor.
        """
        distance = math.dist(self.kf.position, (det.cx, det.cy))
        if distance > self.kf.gate:
            return None
        penalty = 0.0
        gap = _shirt_distance(self.shirt, det.shirt)
        if not math.isnan(gap):
            penalty += COLOUR_WEIGHT * gap
        if self.slot is not None:
            mine = _anchor_distance(self.slot, det.shirt)
            theirs = _anchor_distance(1 - self.slot, det.shirt)
            if not math.isnan(mine):
                # Refusing a candidate outright is reserved for a track that has
                # already lost its player, which is the only time adopting a
                # stranger is the risk.  Applied to a healthy track it does far
                # more harm than good: one frame's torso colour is unreliable
                # enough that the light-shirted player regularly reads darker
                # than his opponent, and vetoing on it cost 22% of his
                # measurements -- the tracker refusing the very man it was
                # following.  See ASSOCIATION_RATIO.
                # Absolute: nothing this far from a named shirt is that player,
                # however close it happens to be.  The relative test below
                # cannot catch a stranger in a colour neither player is wearing,
                # because he is far from *both* anchors and so leans neither
                # way.  See ANCHOR_MAX_DISTANCE.
                if mine > ANCHOR_MAX_DISTANCE:
                    return None
                if (self.misses > 0 and not math.isnan(theirs)
                        and mine > ASSOCIATION_RATIO * theirs):
                    return None      # far more like the other player than mine
                penalty += ANCHOR_WEIGHT * mine
        return distance + penalty


# --- detection --------------------------------------------------------------

def _foreground(gray: np.ndarray, background: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(gray, background)
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def _components(mask: np.ndarray):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    roi_h, roi_w = mask.shape[:2]
    boxes = []
    for i in range(1, count):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < PLAYER_MIN_AREA or h < PLAYER_MIN_HEIGHT:
            continue
        if _ignored(x, y, w, h, roi_w, roi_h):
            continue
        boxes.append((x, y, w, h, area))
    boxes.sort(key=lambda b: b[4], reverse=True)
    return boxes


def _measure(gray: np.ndarray, lab: np.ndarray, mask: np.ndarray, box, source: str):
    """Turn a mask sub-region into a Detection, or None if too little of it."""
    px, py, pw, ph = box
    sub = mask[py:py + ph, px:px + pw]
    area = int(sub.sum() // 255)
    floor = PLAYER_MIN_AREA * (SPLIT_MIN_FRACTION if source == "split" else 1.0)
    if area < floor:
        return None
    ys, xs = np.nonzero(sub)
    if not len(xs):
        return None

    # Shirt colour: the torso band only -- head and legs are skin and shorts,
    # which are shared between players and would wash the difference out.
    top = int(ph * TORSO_TOP)
    bottom = max(top + 1, int(ph * TORSO_BOTTOM))
    band_mask = sub[top:bottom, :] > 0
    brightness = float("nan")
    shirt = None
    if band_mask.any():
        grey_band = gray[py + top:py + bottom, px:px + pw]
        brightness = float(np.median(grey_band[band_mask]))
        lab_band = lab[py + top:py + bottom, px:px + pw]
        shirt = tuple(int(v) for v in np.median(lab_band[band_mask], axis=0))

    # Ground contact point, for projecting onto the floor plane: the
    # silhouette's centroid x paired with its lowest row.  Both halves are
    # counter-intuitive and both were settled by measurement -- see the note
    # above `Detection`, which carries the numbers.
    foot = (px + float(xs.mean()), py + float(ys.max()) + 1.0)

    return Detection(cx=px + float(xs.mean()), cy=py + float(ys.mean()),
                     x=px, y=py, w=pw, h=ph, area=area,
                     brightness=brightness, shirt=shirt, foot=foot, source=source)


def _column_profile(mask: np.ndarray, box) -> np.ndarray:
    x, y, w, h = box
    return mask[y:y + h, x:x + w].sum(axis=0) / 255.0


def _valley_split(mask: np.ndarray, box):
    """Blind split: cut an oversized blob at its deepest interior valley.

    Used only when the motion model has nothing to say.  Deliberately strict --
    without external evidence, a wrong cut invents two players out of one.
    """
    x, y, w, h = box
    if w < 20:
        return None
    column = _column_profile(mask, box)
    margin = max(4, w // 5)
    interior = column[margin:w - margin]
    if interior.size == 0:
        return None
    cut = int(np.argmin(interior)) + margin
    shoulders = (column[:cut].max(), column[cut:].max())
    if column[cut] > 0.55 * min(shoulders):
        return None                      # no real gap
    left, right = column[:cut].sum(), column[cut:].sum()
    if min(left, right) < 0.30 * max(left, right):
        return None                      # lopsided: probably one player
    return [(x, y, cut, h), (x + cut, y, w - cut, h)]


def _guided_split(mask: np.ndarray, box, predictions):
    """Split a merged blob using where the two tracks are predicted to be.

    The predictions say roughly where each player is inside the blob; the
    column profile is then searched near their midpoint for the actual seam.
    Knowing two players are in there is what lets this accept a shallower
    valley than the blind split would.
    """
    x, y, w, h = box
    inside = [p for p in predictions
              if x - SPLIT_PAD <= p[0] <= x + w + SPLIT_PAD
              and y - SPLIT_PAD <= p[1] <= y + h + SPLIT_PAD]
    if len(inside) != 2:
        return None
    left_pred, right_pred = sorted(inside, key=lambda p: p[0])
    if right_pred[0] - left_pred[0] < SPLIT_MIN_SEPARATION:
        return None                      # predicted to be on top of each other
    column = _column_profile(mask, box)
    margin = max(3, w // 8)
    if w - margin <= margin:
        return None
    midpoint = (left_pred[0] + right_pred[0]) / 2.0 - x
    low = int(max(margin, midpoint - SPLIT_SEARCH))
    high = int(min(w - margin, midpoint + SPLIT_SEARCH + 1))
    if high <= low:
        return None
    cut = low + int(np.argmin(column[low:high]))
    shoulders = (column[:cut].max(), column[cut:].max())
    if column[cut] > 0.80 * min(shoulders):
        return None                      # solid all the way through: truly merged
    return [(x, y, cut, h), (x + cut, y, w - cut, h)]


# --- tracking ---------------------------------------------------------------

class Tracker:
    """Two player tracks, associated frame to frame under a motion gate."""

    def __init__(self):
        self.tracks: list = [None, None]
        self.last_index: int | None = None
        # Frames declined as too ambiguous to start from.  Counted rather than
        # discarded silently: a long wait means the roster does not describe
        # what is on screen, which is worth knowing about.
        self.seed_refusals = 0
        # Tracks given up because they had settled on somebody who is not a
        # player.  Also counted: it is the tell that a bystander needs fencing.
        self.loiter_drops = 0

    def _detections(self, gray, lab, mask, boxes, predictions):
        """Components for this frame, with merges split where possible."""
        detections = []
        merged = False
        # Keeping only the largest blobs is what let a spectator displace a
        # player: on the Connecticut capture a man sitting below the near wall
        # is a bigger blob than the player at the back of the court, so with a
        # cap of three the real player was cut before tracking saw him.  Fencing
        # off the spectators (a roster) is the fix; a slightly larger cap is the
        # belt-and-braces for whoever has not made one.
        for x, y, w, h, area in boxes[:CANDIDATE_BLOBS]:
            parts, source = [(x, y, w, h)], "measured"
            if area > PLAYER_MAX_AREA:
                split = (_guided_split(mask, (x, y, w, h), predictions)
                         or _valley_split(mask, (x, y, w, h)))
                if split is None:
                    merged = True        # oversized and unsplittable
                    continue             # claim no position for this blob
                parts, source = split, "split"
            for part in parts:
                det = _measure(gray, lab, mask, part, source)
                if det is not None:
                    detections.append(det)
        detections.sort(key=lambda d: d.area, reverse=True)
        return detections[:CANDIDATE_BLOBS], merged

    def _assign(self, detections):
        """Best gated assignment of detections to live tracks."""
        live = [i for i in (0, 1) if self.tracks[i] is not None and self.tracks[i].alive]
        best, best_cost = None, None
        for size in range(min(len(live), len(detections)), 0, -1):
            for tracks in itertools.combinations(live, size):
                for dets in itertools.permutations(range(len(detections)), size):
                    total = 0.0
                    for t, d in zip(tracks, dets):
                        cost = self.tracks[t].cost(detections[d])
                        if cost is None:
                            total = None
                            break
                        total += cost
                    if total is None:
                        continue
                    # Prefer more matches; among equal counts, prefer lower cost.
                    if best is None or total < best_cost:
                        best, best_cost = dict(zip(tracks, dets)), total
            if best is not None:
                break                    # a larger assignment always wins
        return best or {}

    def _seed(self, detections):
        """Start both tracks, but only from a frame that can tell them apart.

        With a roster, each candidate is scored by how far it *leans* toward one
        player: its distance from the other man's shirt minus the distance from
        this one's.  A frame is only fit to seed from when one candidate leans
        clearly toward player 0 and another leans clearly toward player 1.

        Refusing matters because seeding is the one decision made from a single
        frame, and the frames at the head of a recording are the worst ones to
        make it from.  During a knock-up the players move slowly, so background
        subtraction recovers about half of each body -- limbs and edges rather
        than the shirt itself -- and both men can read as the same colour.  On
        the Connecticut capture at t=0 the white-shirted player measured L=127
        against the dark-shirted player's L=132: *darker* than his opponent.
        Taking the cheapest pairing there hands slot 0 to the wrong man, and
        every label is inverted until the accumulated-evidence check has seen
        enough measurements to swap them back.

        So the tracker waits instead.  Waiting costs nothing but the frames it
        waits through -- it is retried on every frame, and a clean one arrives
        as soon as the two are separated and moving.
        """
        usable = [d for d in detections
                  if d.shirt is not None and not math.isnan(d.brightness)]
        if len(usable) < 2:
            return None
        if all(s is not None for s in PLAYER_SHIRTS):
            best, best_cost = None, None
            for candidate_0, candidate_1 in itertools.permutations(usable, 2):
                mine = _anchor_distance(0, candidate_0.shirt)
                theirs = _anchor_distance(1, candidate_1.shirt)
                if math.isnan(mine) or math.isnan(theirs):
                    continue
                # Absolute: each must actually look like the man he is being
                # called, not merely look like him more than like the other.
                if mine > SEED_MAX_DISTANCE or theirs > SEED_MAX_DISTANCE:
                    continue
                # Relative: each must lean toward his own shirt, so a frame in
                # which both bodies read the same colour seeds nothing.
                if (_anchor_distance(1, candidate_0.shirt) - mine < SEED_MARGIN
                        or _anchor_distance(0, candidate_1.shirt) - theirs
                        < SEED_MARGIN):
                    continue
                if best_cost is None or mine + theirs < best_cost:
                    best, best_cost = (candidate_0, candidate_1), mine + theirs
            if best is None:
                # Nothing in this frame is recognisably both players.  Claim no
                # position and try again on the next one.
                self.seed_refusals += 1
                return None
            first, second = best
        else:
            first, second = sorted(usable, key=lambda d: -d.brightness)[:2]
        self.tracks = [_Track(first, 0), _Track(second, 1)]
        return first, second

    def step(self, frame_index: int, time_s: float, gray, lab, background,
             stride: int) -> Sample:
        mask = _foreground(gray, background)
        boxes = _components(mask)

        dt = 1.0
        if self.last_index is not None:
            dt = max(1.0, (frame_index - self.last_index) / stride)
        self.last_index = frame_index

        predictions = []
        gates = [None, None]
        for i, track in enumerate(self.tracks):
            if track is not None and track.alive:
                track.predict(dt)
                predictions.append(track.kf.position)
                gates[i] = track.kf.position + (track.kf.gate,)

        detections, merged = self._detections(gray, lab, mask, boxes, predictions)
        sample = Sample(frame_index, time_s, [None, None], merged, gates=gates)
        sample.blobs = len(boxes)

        if all(t is None or not t.alive for t in self.tracks):
            refused_before = self.seed_refusals
            seeded = self._seed(detections)
            if seeded is None:
                sample.waiting = self.seed_refusals > refused_before
                return sample
            # Report the seeding frame from the measurements themselves.
            sample.slots[0], sample.slots[1] = seeded
            return sample

        assignment = self._assign(detections)
        for i in (0, 1):
            track = self.tracks[i]
            if track is None or not track.alive:
                continue
            if i in assignment:
                det = detections[assignment[i]]
                track.absorb(det, time_s)
                sample.slots[i] = det
                if track.loitering(time_s):
                    # Measuring steadily, in one spot, and looking like neither
                    # player: this is somebody at the glass, not the man this
                    # slot is for.  Give the slot up so it can be re-acquired.
                    track.alive = False
                    sample.slots[i] = None
                    self.loiter_drops += 1
            else:
                track.misses += 1
                if track.misses > MAX_MISSES:
                    track.alive = False
                elif track.misses <= MAX_COAST:
                    # No measurement.  For a differencing detector this often
                    # means the player stopped moving, so coasting on the filter
                    # is reasonable -- but it is flagged, never passed off as an
                    # observation.
                    cx, cy = track.kf.position
                    bw, bh = track.box
                    sample.slots[i] = Detection(
                        cx=cx, cy=cy, x=int(cx - bw / 2), y=int(cy - bh / 2),
                        w=bw, h=bh, area=0, brightness=track.brightness,
                        shirt=track.shirt, source="predicted")

        self._revive(detections, assignment, sample)
        self._reconcile(sample)
        self._separate(sample)
        return sample

    def _reconcile(self, sample: Sample) -> None:
        """Put whichever track looks like roster player 0 into slot 0.

        Following a body frame to frame and knowing *which* body it is are two
        different problems, and this solves the second one on a longer clock
        than the first.  Motion keeps each track on its own player; the shirt
        evidence each has accumulated then says whether the pair are the right
        way round, and swaps them if not.  So a bad seeding frame -- or a swap
        through a merge that motion could not prevent -- is corrected a second
        later instead of poisoning the rest of the clip.

        Inert without a roster: with no reference colours there is no evidence,
        and the slots keep whatever order the tracker gave them.
        """
        first, second = self.tracks
        if first is None or second is None or not (first.alive and second.alive):
            return
        if min(first.evidence_n, second.evidence_n) < IDENTITY_MIN_SAMPLES:
            return
        if second.evidence - first.evidence <= IDENTITY_MARGIN:
            return
        self.tracks = [second, first]
        first.slot, second.slot = 1, 0
        sample.slots = [sample.slots[1], sample.slots[0]]
        sample.gates = [sample.gates[1], sample.gates[0]]

    @staticmethod
    def _separate(sample: Sample) -> None:
        """Drop a coasted slot that has drifted onto the other player.

        Two tracks within a body-width of each other are reporting one person
        twice.  Whichever of them is only a prediction has no pixels behind it,
        so it is the one to give up -- reporting nothing is honest, reporting a
        duplicate is not.
        """
        first, second = sample.slots
        if first is None or second is None:
            return
        if math.dist((first.cx, first.cy), (second.cx, second.cy)) > COLLAPSE_RADIUS:
            return
        for i, (own, other) in enumerate(((first, second), (second, first))):
            if own.source == "predicted" and other.source != "predicted":
                sample.slots[i] = None
                return
        if first.source == "predicted" and second.source == "predicted":
            sample.slots[0] = sample.slots[1] = None

    def _revive(self, detections, assignment, sample) -> None:
        """Re-acquire a dead track from a detection nothing else claimed.

        A track dies when the player reappears outside the motion gate -- after
        an occlusion, or a long merge.  Motion cannot re-link across that, so
        this falls back to the shirt colour, which is the only evidence that
        survives the gap.  Requiring a colour match is what stops the revived
        track from picking up the *other* player.
        """
        spare = [d for i, d in enumerate(detections)
                 if i not in assignment.values() and d.shirt is not None]
        for i in (0, 1):
            track = self.tracks[i]
            if track is not None and track.alive:
                continue
            if not spare:
                return
            if PLAYER_SHIRTS[i] is not None:
                # The user named this shirt, so re-acquisition is a direct
                # question -- which spare blob looks most like him, and does it
                # look more like him than like his opponent.
                best = min(spare, key=lambda d: _anchor_distance(i, d.shirt))
                mine = _anchor_distance(i, best.shirt)
                theirs = _anchor_distance(1 - i, best.shirt)
                if math.isnan(mine) or mine > REVIVE_MAX:
                    continue
                if not math.isnan(theirs) and mine > REVIVE_RATIO * theirs:
                    continue
                spare.remove(best)
                self.tracks[i] = _Track(best, i)
                sample.slots[i] = best
                continue
            other = self.tracks[1 - i]
            rival = None if other is None else other.shirt
            reference = None if track is None else track.shirt
            if reference is None:
                # Never tracked this slot: fall back to the ordering convention,
                # but only against a live sibling, so the two cannot coincide.
                candidates = [d for d in spare
                              if rival is None
                              or _shirt_distance(rival, d.shirt) > REVIVE_MAX]
                if not candidates:
                    continue
                best = max(candidates, key=lambda d: d.brightness) if i == 0 else \
                    min(candidates, key=lambda d: d.brightness)
            else:
                best = min(spare, key=lambda d: _shirt_distance(reference, d.shirt))
                mine = _shirt_distance(reference, best.shirt)
                if mine > REVIVE_MAX:
                    continue             # does not look like this player at all
                theirs = _shirt_distance(rival, best.shirt) if rival else float("inf")
                if not math.isnan(theirs) and mine > REVIVE_RATIO * theirs:
                    continue             # too close a call: wait rather than guess
            spare.remove(best)
            self.tracks[i] = _Track(best, i)
            self.tracks[i].shirt = reference or best.shirt
            sample.slots[i] = best


def analyse(path: str, start_s: float = 0.0, duration_s: float | None = None,
            stride: int = 2, progress=None):
    """Run detection and tracking over a span of the video.

    Returns the per-frame samples and the (width, height) of the court ROI.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("cannot open " + path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(round(start_s * fps))
    last = total if duration_s is None else min(total, first + int(round(duration_s * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    tracker = Tracker()
    samples = []
    roi_size = (0, 0)
    index = first
    while index < last:
        chunk = []
        while index < last and len(chunk) < CHUNK_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            if (index - first) % stride == 0:
                gray, lab = _prepare(frame)
                chunk.append((index, gray, lab))
            index += 1
        if not chunk:
            break
        stack = np.stack([g for _, g, _ in chunk])
        pick = np.linspace(0, len(chunk) - 1, min(BG_SAMPLES, len(chunk))).astype(int)
        background = np.median(stack[pick], axis=0).astype(np.uint8)
        roi_size = (background.shape[1], background.shape[0])
        for frame_index, gray, lab in chunk:
            samples.append(tracker.step(frame_index, frame_index / fps,
                                        gray, lab, background, stride))
        if progress:
            progress(index - first, last - first)
    cap.release()
    return samples, roi_size


def write_csv(samples, roi_size, out: str) -> None:
    w, h = roi_size
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "time_s",
                         "p0_x", "p0_y", "p0_area", "p0_shirt", "p0_src",
                         "p1_x", "p1_y", "p1_area", "p1_shirt", "p1_src",
                         "merged"])
        for s in samples:
            row = [s.frame_index, "%.3f" % s.time_s]
            for d in s.slots:
                if d is None:
                    row += ["", "", "", "", ""]
                else:
                    nx, ny = d.norm(w, h)
                    row += ["%.4f" % nx, "%.4f" % ny, d.area,
                            "#%02x%02x%02x" % d.rgb, d.source]
            row.append(int(s.merged))
            writer.writerow(row)


def summarise(samples) -> str:
    total = len(samples) or 1
    both = sum(1 for s in samples if all(d is not None for d in s.slots))
    one = sum(1 for s in samples if sum(d is not None for d in s.slots) == 1)
    none = len(samples) - both - one
    observed = sum(1 for s in samples for d in s.slots
                   if d is not None and d.source != "predicted")
    split = sum(1 for s in samples for d in s.slots
                if d is not None and d.source == "split")
    coasted = sum(1 for s in samples for d in s.slots
                  if d is not None and d.source == "predicted")
    merged = sum(1 for s in samples if s.merged)
    waiting = sum(1 for s in samples if s.waiting)
    lines = [
        "%d frames analysed: both players %d (%.0f%%), one %d (%.0f%%), "
        "none %d (%.0f%%)"
        % (len(samples), both, 100 * both / total, one, 100 * one / total,
           none, 100 * none / total),
        "positions: %d observed (%d from split merges), %d coasted; "
        "%d frames still flagged merged"
        % (observed, split, coasted, merged),
    ]
    if waiting:
        lines.append(
            "seeding declined %d frame%s in which no two bodies looked like the "
            "two named\nplayers, and waited for one that could tell them apart"
            % (waiting, "" if waiting == 1 else "s"))
    return "\n".join(lines)


def main(argv=None) -> None:
    from .. import cli

    p = cli.parser("Track the two players in a squash match video.", __doc__)
    cli.add_span_arguments(p, stride=2)
    cli.add_tracking_arguments(p)
    p.add_argument("--csv", default=None, help="write per-frame positions here")
    p.add_argument("--preview", default=None,
                   help="write an annotated mp4 here (the same one "
                        "`squashvision demo` renders)")
    args = p.parse_args(argv)
    cli.apply_tracking(args)

    samples, roi = analyse(args.video, args.start, args.duration, args.stride,
                           cli.progress())
    print(summarise(samples))
    if args.csv:
        write_csv(samples, roi, args.csv)
        print("positions -> " + args.csv)
    if args.preview:
        from ..view.demo import render_demo
        render_demo(args.video, samples, args.preview, args.stride)
        print("preview   -> " + args.preview)


if __name__ == "__main__":
    main()
