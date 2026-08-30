"""Map tracked players from the camera view onto a bird's-eye court plan.

The floor of a squash court is a plane, so the camera view of it is related to
a top-down view by a homography, which eight numbers fix completely.  What
matters is *which* image point you push through it: a homography of the floor
is only valid for points that are on the floor.  A player's blob centroid sits
around mid-torso, roughly a metre up, and mapping that would place everyone
systematically further from the camera than they are -- so positions come from
the foot contact point, the bottom-centre of the player's box.

Court coordinates are metres with the origin at the front-left floor corner:
x runs across the court, y from the front wall to the back wall.  "Left" is
image-left, i.e. as seen from a camera behind the back wall.

Calibration for a new court or camera angle is done with `calibrate.py`, which
writes a JSON file that `Court.load` reads.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from ..detect.players import PLAYER_WIDTH, _roi_box


@dataclass(frozen=True)
class CourtSpec:
    """Court dimensions in metres.  Defaults are the WSF singles standard."""

    width: float = 6.40             # side wall to side wall
    length: float = 9.75            # front wall to back wall
    short_line: float = 4.26        # front wall to short line
    box: float = 1.60               # service boxes are square

    @property
    def half(self) -> float:
        return self.width / 2.0


SINGLES = CourtSpec()
DOUBLES = CourtSpec(width=7.62)     # untested: no doubles footage to check against


def landmarks(spec: CourtSpec = SINGLES) -> dict:
    """Named floor landmarks a person can point at, in court metres.

    Only the four service-box corners, the court corners and the lines they
    define are used: everything here is a marking painted on the floor or a
    corner of it, so it can be identified unambiguously from any angle.
    """
    back = spec.length
    short = spec.short_line
    box_back = short + spec.box
    return {
        "front_left_corner": (0.0, 0.0),
        "front_right_corner": (spec.width, 0.0),
        "back_left_corner": (0.0, back),
        "back_right_corner": (spec.width, back),
        "short_left": (0.0, short),
        "short_right": (spec.width, short),
        "tee": (spec.half, short),
        "half_court_back": (spec.half, back),
        "box_left_inner_front": (spec.box, short),
        "box_left_outer_back": (0.0, box_back),
        "box_left_inner_back": (spec.box, box_back),
        "box_right_inner_front": (spec.width - spec.box, short),
        "box_right_outer_back": (spec.width, box_back),
        "box_right_inner_back": (spec.width - spec.box, box_back),
    }


DESCRIPTIONS = {
    "front_left_corner": "front wall meets LEFT side wall, at floor level",
    "front_right_corner": "front wall meets RIGHT side wall, at floor level",
    "back_left_corner": "back wall meets LEFT side wall, at floor level",
    "back_right_corner": "back wall meets RIGHT side wall, at floor level",
    "short_left": "short line meets the LEFT side wall",
    "short_right": "short line meets the RIGHT side wall",
    "tee": "the T: short line meets the half-court line",
    "half_court_back": "half-court line meets the back wall",
    "box_left_inner_front": "LEFT box, corner nearest the T",
    "box_left_outer_back": "LEFT box, back corner at the side wall",
    "box_left_inner_back": "LEFT box, back corner nearest the middle",
    "box_right_inner_front": "RIGHT box, corner nearest the T",
    "box_right_outer_back": "RIGHT box, back corner at the side wall",
    "box_right_inner_back": "RIGHT box, back corner nearest the middle",
}

# Built-in calibration for the Bates/Amherst capture, kept as a worked example
# and as the default when no calibration file is given.  Fractions of frame
# size, so it survives decoding at another resolution -- but not a camera move.
BATES_CALIBRATION = [
    ("front_left_corner", (0.32394, 0.53892)),
    ("front_right_corner", (0.73441, 0.53774)),
    ("short_left", (0.24934, 0.71816)),
    ("short_right", (0.80338, 0.72170)),
    ("box_left_inner_front", (0.38461, 0.72052)),
    ("box_right_inner_front", (0.66412, 0.72170)),
    ("box_left_outer_back", (0.21054, 0.80012)),
    ("box_left_inner_back", (0.36240, 0.80307)),
    ("box_right_inner_back", (0.67706, 0.80602)),
    ("box_right_outer_back", (0.82925, 0.80602)),
]


class Court:
    """The floor plane: converts between image fractions and court metres."""

    def __init__(self, points=None, spec: CourtSpec = SINGLES, source: str = "built-in"):
        self.spec = spec
        self.source = source
        self.points = list(points if points is not None else BATES_CALIBRATION)
        known = landmarks(spec)
        missing = [n for n, _ in self.points if n not in known]
        if missing:
            raise SystemExit("unknown landmark(s): " + ", ".join(missing))
        if len(self.points) < 4:
            raise SystemExit("need at least 4 landmarks to fit a court, got %d"
                             % len(self.points))
        image = np.array([p for _, p in self.points], dtype=np.float64)
        world = np.array([known[n] for n, _ in self.points], dtype=np.float64)
        self.to_image_matrix, _ = cv2.findHomography(world, image, 0)
        if self.to_image_matrix is None:
            raise SystemExit("calibration is degenerate -- are the points collinear?")
        self.to_court_matrix = np.linalg.inv(self.to_image_matrix)

    # --- persistence -------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "Court":
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        spec = CourtSpec(**blob.get("court", {}))
        points = [(p["name"], tuple(p["image"])) for p in blob["points"]]
        return cls(points, spec, source=path)

    def save(self, path: str, video: str = "") -> None:
        known = landmarks(self.spec)
        blob = {
            "video": video,
            "court": asdict(self.spec),
            "points": [{"name": n, "image": [round(x, 5), round(y, 5)],
                        "court": list(known[n])} for n, (x, y) in self.points],
            "residuals_cm": [round(r, 1) for r in self.residuals()],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2)

    # --- geometry ----------------------------------------------------------

    @staticmethod
    def _apply(matrix, x: float, y: float) -> tuple[float, float]:
        v = matrix @ np.array([x, y, 1.0])
        if abs(v[2]) < 1e-12:
            return float("nan"), float("nan")
        return float(v[0] / v[2]), float(v[1] / v[2])

    def to_court(self, fx: float, fy: float) -> tuple[float, float]:
        """Image fraction -> court metres."""
        return self._apply(self.to_court_matrix, fx, fy)

    def to_image(self, cx: float, cy: float) -> tuple[float, float]:
        """Court metres -> image fraction."""
        return self._apply(self.to_image_matrix, cx, cy)

    def residuals(self) -> list[float]:
        """Reprojection error of each calibration point, in centimetres.

        Measured in court units rather than pixels, because a pixel near the
        front wall is worth several times one near the camera.
        """
        known = landmarks(self.spec)
        out = []
        for name, (fx, fy) in self.points:
            out.append(100.0 * math.dist(self.to_court(fx, fy), known[name]))
        return out

    def report(self) -> str:
        r = self.residuals()
        return ("court calibration (%s): %d points, reprojection error "
                "mean %.1f cm, max %.1f cm"
                % (self.source, len(self.points), sum(r) / len(r), max(r)))

    def inside(self, cx: float, cy: float, margin: float = 0.5) -> bool:
        """Is a court position physically possible (allowing for foot slop)?"""
        return (-margin <= cx <= self.spec.width + margin
                and -margin <= cy <= self.spec.length + margin)

    def visible_depth(self, frame_fraction: float = 1.0) -> float:
        """How far back the camera actually sees, in metres from the front wall.

        The back of a court is often below the bottom of frame when the camera
        sits behind the back wall; positions beyond this are unobservable
        rather than merely absent.
        """
        lo, hi = 0.0, self.spec.length
        for _ in range(40):
            mid = (lo + hi) / 2.0
            edge = max(self.to_image(x, mid)[1] for x in (0.0, self.spec.half,
                                                          self.spec.width))
            if edge > frame_fraction:
                hi = mid
            else:
                lo = mid
        return lo

    # --- bird's-eye plan ---------------------------------------------------

    PLAN_SCALE = 46             # px per metre
    PLAN_MARGIN = 26

    def plan_size(self) -> tuple[int, int]:
        return (int(self.spec.width * self.PLAN_SCALE) + 2 * self.PLAN_MARGIN,
                int(self.spec.length * self.PLAN_SCALE) + 2 * self.PLAN_MARGIN)

    def plan_point(self, cx: float, cy: float) -> tuple[int, int]:
        return (int(round(self.PLAN_MARGIN + cx * self.PLAN_SCALE)),
                int(round(self.PLAN_MARGIN + cy * self.PLAN_SCALE)))

    def draw_plan(self) -> np.ndarray:
        """An empty court plan, front wall at the top."""
        spec = self.spec
        w, h = self.plan_size()
        canvas = np.full((h, w, 3), 30, dtype=np.uint8)
        line, floor, edge = (120, 120, 130), (222, 232, 240), (70, 75, 85)
        cv2.rectangle(canvas, self.plan_point(0, 0),
                      self.plan_point(spec.width, spec.length), floor, -1)
        cv2.rectangle(canvas, self.plan_point(0, 0),
                      self.plan_point(spec.width, spec.length), edge, 2)
        cv2.line(canvas, self.plan_point(0, spec.short_line),
                 self.plan_point(spec.width, spec.short_line), line, 2)
        cv2.line(canvas, self.plan_point(spec.half, spec.short_line),
                 self.plan_point(spec.half, spec.length), line, 2)
        for x0 in (0.0, spec.width - spec.box):
            cv2.rectangle(canvas, self.plan_point(x0, spec.short_line),
                          self.plan_point(x0 + spec.box, spec.short_line + spec.box),
                          line, 2)
        cv2.putText(canvas, "FRONT WALL", (self.PLAN_MARGIN, self.PLAN_MARGIN - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 175), 1, cv2.LINE_AA)
        return canvas

    def draw_overlay(self, image: np.ndarray) -> np.ndarray:
        """Reproject the court markings onto a camera frame, to check the fit."""
        out = image.copy()
        h, w = out.shape[:2]
        spec = self.spec

        def pt(x, y):
            fx, fy = self.to_image(x, y)
            return int(round(fx * w)), int(round(fy * h))

        def seg(a, b, colour, thickness=2):
            cv2.line(out, pt(*a), pt(*b), colour, thickness, cv2.LINE_AA)

        for a, b in (((0, 0), (spec.width, 0)),
                     ((0, 0), (0, spec.length)),
                     ((spec.width, 0), (spec.width, spec.length)),
                     ((0, spec.length), (spec.width, spec.length))):
            seg(a, b, (0, 255, 255))
        seg((0, spec.short_line), (spec.width, spec.short_line), (0, 0, 255))
        seg((spec.half, spec.short_line), (spec.half, spec.length), (0, 0, 255))
        for x0 in (0.0, spec.width - spec.box):
            back = spec.short_line + spec.box
            seg((x0, spec.short_line), (x0 + spec.box, spec.short_line), (255, 0, 255))
            seg((x0, back), (x0 + spec.box, back), (255, 0, 255))
            seg((x0, spec.short_line), (x0, back), (255, 0, 255))
            seg((x0 + spec.box, spec.short_line), (x0 + spec.box, back), (255, 0, 255))
        for _, (fx, fy) in self.points:
            cv2.drawMarker(out, (int(fx * w), int(fy * h)), (0, 255, 0),
                           cv2.MARKER_CROSS, 18, 2)
        return out


def foot_fraction(det, small_size):
    """Fraction-of-frame position of a detection's feet, or None.

    Detections are in ROI pixels at the analysis width, so this undoes the crop
    and the downscale.  Returns None for a detection the tracker coasted: its
    box is synthesised from the last known size, so its bottom edge is a guess,
    and projecting a guess onto the floor produces a confident-looking position
    that no pixels support.
    """
    if det.foot is None:
        return None
    small_w, small_h = small_size
    x0, y0, _, _ = _roi_box(small_w, small_h)
    fx, fy = det.foot
    return (x0 + fx) / small_w, (y0 + fy) / small_h


def small_size(frame_width: int, frame_height: int) -> tuple[int, int]:
    """The analysis-resolution frame size that detections are expressed in."""
    return PLAYER_WIDTH, int(round(frame_height * PLAYER_WIDTH / frame_width))


def clean_frame(video: str, at_s: float = 280.0, samples: int = 60,
                spacing: int = 40) -> np.ndarray:
    """A median frame with the players removed, for calibrating on.

    Markings are much easier to point at when nobody is standing on them.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit("cannot open " + video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = min(int(at_s * fps), max(0, total - samples * spacing))
    frames = []
    for k in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + k * spacing)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit("could not read any frames from " + video)
    return np.median(np.stack(frames), axis=0).astype(np.uint8)
