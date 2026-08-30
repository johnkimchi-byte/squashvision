"""Calibrate the court geometry for a new court or camera angle.

Interactive picker -- click each floor landmark it asks for:

    python -m squashvision calibrate VIDEO --out mycourt.json

Clicking is two-stage: the first click zooms in around that spot, the second
sets the point at full resolution.  This matters, because the fit is only as
good as the pixel you pointed at.  Once four landmarks are placed the court is
re-fitted after every click and drawn back over the frame, so a bad point shows
up immediately as markings that do not sit on the paint.

Keys:  u  undo      s  skip this landmark      Esc  cancel a zoom
       w  write and quit        q  quit without saving

Without a display -- over SSH, or in CI -- pass the points directly:

    python -m squashvision calibrate VIDEO --out mycourt.json \\
        --point tee=0.512,0.718 --point short_left=0.249,0.718 ...

and check the result with:

    python -m squashvision calibrate VIDEO --verify mycourt.json --out-image fit.png
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import cli, overlay
from .court import (DESCRIPTIONS, DOUBLES, SINGLES, Court, clean_frame,
                    landmarks)

# Landmarks are offered in this order: the outer ones first, because a
# homography is best conditioned by points that are far apart, and the box
# corners then refine it.
ORDER = [
    "front_left_corner", "front_right_corner",
    "short_left", "short_right",
    "box_left_inner_front", "box_right_inner_front",
    "box_left_outer_back", "box_right_outer_back",
    "box_left_inner_back", "box_right_inner_back",
    "tee", "half_court_back",
    "back_left_corner", "back_right_corner",
]

WINDOW = "squashvision calibration"
ZOOM = 4                        # magnification of the second-stage click
ZOOM_SPAN = 140                 # px of the source frame shown when zoomed
INK, DIM, GO, WARN = overlay.INK, overlay.DIM, overlay.GO, overlay.WARN


class Picker:
    """Interactive landmark picker over a single frame."""

    def __init__(self, frame: np.ndarray, spec=SINGLES, display_width: int = 1500):
        self.frame = frame
        self.spec = spec
        self.height, self.width = frame.shape[:2]
        self.scale = min(1.0, display_width / self.width)
        self.picked: dict = {}
        self.index = 0
        self.zoom_at = None         # full-res (x, y) the zoom is centred on
        self.cursor = (0, 0)
        self.click = None

    # --- state ---------------------------------------------------------

    @property
    def current(self):
        while self.index < len(ORDER) and ORDER[self.index] in self.picked:
            self.index += 1
        return ORDER[self.index] if self.index < len(ORDER) else None

    def court(self):
        if len(self.picked) < 4:
            return None
        try:
            return Court([(n, p) for n, p in self.picked.items()], self.spec,
                         source="in progress")
        except SystemExit:
            return None         # collinear so far; keep collecting

    def place(self, x: float, y: float) -> None:
        name = self.current
        if name:
            self.picked[name] = (x / self.width, y / self.height)

    def undo(self) -> None:
        if self.picked:
            self.picked.pop(list(self.picked)[-1])
            self.index = 0

    # --- drawing -------------------------------------------------------

    def _overview(self) -> np.ndarray:
        court = self.court()
        base = court.draw_overlay(self.frame) if court else self.frame
        view = cv2.resize(base, None, fx=self.scale, fy=self.scale,
                          interpolation=cv2.INTER_AREA)
        for name, (fx, fy) in self.picked.items():
            cv2.drawMarker(view, (int(fx * view.shape[1]), int(fy * view.shape[0])),
                           GO, cv2.MARKER_CROSS, 16, 2)
        return view

    def _zoomed(self) -> np.ndarray:
        cx, cy = self.zoom_at
        half = ZOOM_SPAN // 2
        x0 = int(max(0, min(self.width - ZOOM_SPAN, cx - half)))
        y0 = int(max(0, min(self.height - ZOOM_SPAN, cy - half)))
        patch = self.frame[y0:y0 + ZOOM_SPAN, x0:x0 + ZOOM_SPAN]
        view = cv2.resize(patch, None, fx=ZOOM, fy=ZOOM,
                          interpolation=cv2.INTER_NEAREST)
        centre = (view.shape[1] // 2, view.shape[0] // 2)
        cv2.line(view, (centre[0], 0), (centre[0], view.shape[0]), (0, 200, 255), 1)
        cv2.line(view, (0, centre[1]), (view.shape[1], centre[1]), (0, 200, 255), 1)
        self._zoom_origin = (x0, y0)
        return view

    def _banner(self, view: np.ndarray) -> np.ndarray:
        name = self.current
        pad = np.full((84, view.shape[1], 3), 26, dtype=np.uint8)
        if name is None:
            headline = "all landmarks placed -- press 'w' to write"
        else:
            headline = "click: %s" % name.replace("_", " ")
        cv2.putText(pad, headline, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    INK, 1, cv2.LINE_AA)
        if name:
            cv2.putText(pad, DESCRIPTIONS[name], (14, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, DIM, 1, cv2.LINE_AA)
        court = self.court()
        if court:
            r = court.residuals()
            colour = GO if max(r) < 30 else WARN
            status = "%d points   fit: mean %.1f cm, worst %.1f cm" % (
                len(self.picked), sum(r) / len(r), max(r))
        else:
            colour = DIM
            status = "%d points   (4 needed before the fit is shown)" % len(self.picked)
        cv2.putText(pad, status, (14, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    colour, 1, cv2.LINE_AA)
        hint = "zoomed: click to place, Esc to back out" if self.zoom_at else \
            "click near the landmark to zoom in"
        cv2.putText(pad, hint + "    u undo   s skip   w write   q quit",
                    (14, 84 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, DIM, 1, cv2.LINE_AA)
        return np.vstack([pad, view])

    # --- loop ----------------------------------------------------------

    def _on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.cursor = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)

    def run(self) -> Court | None:
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW, self._on_mouse)
        except cv2.error as exc:
            raise SystemExit(
                "no display available for the interactive picker (%s).\n"
                "Use --point NAME=x,y instead; see --help." % exc)

        while True:
            view = self._zoomed() if self.zoom_at is not None else self._overview()
            cv2.imshow(WINDOW, self._banner(view))
            key = cv2.waitKey(20) & 0xFF

            if self.click is not None:
                cx, cy = self.click
                self.click = None
                cy -= 84                        # the banner sits above the image
                if cy >= 0:
                    if self.zoom_at is None:
                        self.zoom_at = (cx / self.scale, cy / self.scale)
                    else:
                        ox, oy = self._zoom_origin
                        self.place(ox + cx / ZOOM, oy + cy / ZOOM)
                        self.zoom_at = None

            if key == ord('q'):
                cv2.destroyWindow(WINDOW)
                return None
            if key == ord('w'):
                cv2.destroyWindow(WINDOW)
                return self.court()
            if key == ord('u'):
                self.undo()
                self.zoom_at = None
            if key == ord('s'):
                self.index += 1
                self.zoom_at = None
            if key == 27:
                self.zoom_at = None


def parse_point(text: str) -> tuple[str, tuple[float, float]]:
    """Parse a NAME=x,y argument into a landmark and a frame fraction."""
    if "=" not in text:
        raise SystemExit("expected NAME=x,y, got %r" % text)
    name, _, coords = text.partition("=")
    parts = coords.split(",")
    if len(parts) != 2:
        raise SystemExit("expected NAME=x,y, got %r" % text)
    try:
        x, y = float(parts[0]), float(parts[1])
    except ValueError:
        raise SystemExit("coordinates must be numbers, got %r" % coords)
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise SystemExit("coordinates are fractions of frame size, so 0..1; got %r"
                         % coords)
    return name.strip(), (x, y)


def main(argv=None) -> None:
    p = cli.parser("Calibrate court geometry for a new court or camera angle.",
                   __doc__)
    p.add_argument("video", nargs="?")
    p.add_argument("--out", default="court.json", help="calibration file to write")
    p.add_argument("--point", action="append", default=[],
                   metavar="NAME=x,y", help="place a landmark without the GUI")
    p.add_argument("--verify", metavar="FILE",
                   help="check an existing calibration instead of making one")
    p.add_argument("--out-image", metavar="PNG",
                   help="write the fit drawn over the frame, for checking")
    p.add_argument("--at", type=float, default=280.0,
                   help="seconds into the video to calibrate on")
    p.add_argument("--dirty", action="store_true",
                   help="use a single frame rather than a players-removed median")
    p.add_argument("--doubles", action="store_true", help="doubles court dimensions")
    p.add_argument("--list", action="store_true", help="list landmark names and exit")
    args = p.parse_args(argv)

    spec = DOUBLES if args.doubles else SINGLES
    if args.list:
        for name in ORDER:
            x, y = landmarks(spec)[name]
            print("  %-24s (%.2f, %.2f) m   %s" % (name, x, y, DESCRIPTIONS[name]))
        return
    if not args.video:
        p.error("a video is required")

    if args.dirty:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.at * fps))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("could not read a frame from " + args.video)
    else:
        print("building a players-removed frame to calibrate on...", flush=True)
        frame = clean_frame(args.video, args.at)

    if args.verify:
        court = Court.load(args.verify)
    elif args.point:
        court = Court([parse_point(t) for t in args.point], spec, source=args.out)
    else:
        court = Picker(frame, spec).run()
        if court is None:
            print("cancelled; nothing written")
            return

    print(court.report())
    for (name, _), residual in zip(court.points, court.residuals()):
        print("   %-24s %5.1f cm" % (name, residual))
    depth = court.visible_depth()
    if depth < court.spec.length - 0.05:
        print("note: the camera only sees %.1f m of the %.1f m court; positions "
              "beyond that are unobservable, not absent." % (depth, court.spec.length))

    if args.out_image:
        cv2.imwrite(args.out_image, court.draw_overlay(frame))
        print("fit overlay -> " + args.out_image)
    if not args.verify:
        court.save(args.out, args.video)
        print("calibration -> " + args.out)


if __name__ == "__main__":
    main()
