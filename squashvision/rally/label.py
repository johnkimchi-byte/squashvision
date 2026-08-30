"""Mark rally-ending shots by hand, to check the detector against.

    python -m squashvision label VIDEO --out labels.json

Scrub through the match and press `m` at the moment play stops.  The score box
is shown magnified in the corner: the score flipping is the easiest confirmation
that a point really ended, and it flips a beat *after* the last shot, so mark
the shot, not the scoreboard.

Keys:  space  play / pause          . , step one frame
       -> <-  jump 2 s              ] [ jump 15 s
       m      mark a rally end      k  mark a rally end that ended the game
       s      mark a shot           v  mark a volley
       n      mark a winner (a rally-ending shot that was not reached)
       u      undo the last mark    w  write and quit        q  quit

Without a display, marks can be passed straight in:

    python -m squashvision label VIDEO --out labels.json --mark 267.5 --shot 271.0

and an existing file can be reviewed with `--show labels.json`.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .. import cli, overlay
from ..score.scoreboard import SCORE_BOX

WINDOW = "squashvision labelling"
INK, DIM, GO, WARN = overlay.INK, overlay.DIM, overlay.GO, overlay.WARN
DISPLAY_WIDTH = 1400
BANNER = 92

# Ribbon colour per mark kind; anything not listed (just "rally") falls back to GO.
MARK_COLOUR = {
    "game": WARN,
    "shot": overlay.TRACK_COLOURS[0],
    "volley": overlay.TRACK_COLOURS[1],
    "winner": INK,
}


@dataclass
class Mark:
    """One hand-marked moment: a rally end, or (for brief 2) a shot."""

    time_s: float
    # "rally"/"game" mark where a point ended; "shot"/"volley"/"winner" mark a
    # single shot -- "volley" and "winner" are also "shot" in substance, kept
    # as distinct kinds so the training data for brief 2 needs no relabelling.
    kind: str = "rally"

    def as_dict(self) -> dict:
        return asdict(self)


def save(path: str, video: str, marks) -> None:
    blob = {
        "video": video,
        "marks": [m.as_dict() for m in sorted(marks, key=lambda m: m.time_s)],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)


def load(path: str):
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    return [Mark(float(m["time_s"]), m.get("kind", "rally")) for m in blob["marks"]]


class Labeller:
    """A small scrubbing UI over one video."""

    def __init__(self, video: str, start_s: float = 0.0):
        self.cap = cv2.VideoCapture(video)
        if not self.cap.isOpened():
            raise SystemExit("cannot open " + video)
        self.video = video
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.index = min(self.total - 1, max(0, int(start_s * self.fps)))
        self.playing = False
        self.marks: list[Mark] = []
        self.frame = None
        self.shown = -1

    @property
    def time_s(self) -> float:
        return self.index / self.fps

    def seek(self, frames: int) -> None:
        self.index = max(0, min(self.total - 1, self.index + frames))

    def read(self):
        if self.shown != self.index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.index)
            ok, frame = self.cap.read()
            if ok:
                self.frame = frame
                self.shown = self.index
        return self.frame

    def mark(self, kind: str = "rally") -> None:
        # Replace rather than duplicate if the same moment is marked twice.
        self.marks = [m for m in self.marks if abs(m.time_s - self.time_s) > 0.4]
        self.marks.append(Mark(round(self.time_s, 2), kind))

    def undo(self) -> None:
        if self.marks:
            self.marks.pop()

    def render(self):
        frame = self.read()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        scale = DISPLAY_WIDTH / w
        view = cv2.resize(frame, (DISPLAY_WIDTH, int(round(h * scale))),
                          interpolation=cv2.INTER_AREA)

        # The score box, magnified: the ground truth for "did a point end".
        x0, y0, x1, y1 = SCORE_BOX
        patch = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        if patch.size:
            patch = cv2.resize(patch, None, fx=2.0, fy=2.0,
                               interpolation=cv2.INTER_CUBIC)
            ph, pw = patch.shape[:2]
            view[10:10 + ph, view.shape[1] - pw - 10:view.shape[1] - 10] = patch
            cv2.rectangle(view, (view.shape[1] - pw - 10, 10),
                          (view.shape[1] - 10, 10 + ph), (90, 90, 90), 1)

        pad = np.full((BANNER, view.shape[1], 3), 26, dtype=np.uint8)
        cv2.putText(pad, "t = %8.2f s   frame %d / %d   %s"
                    % (self.time_s, self.index, self.total,
                       "PLAYING" if self.playing else "paused"),
                    (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, INK, 1, cv2.LINE_AA)
        cv2.putText(pad, "%d marks" % len(self.marks) +
                    ("   last: %.2f s (%s)" % (self.marks[-1].time_s, self.marks[-1].kind)
                     if self.marks else ""),
                    (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GO if self.marks else DIM,
                    1, cv2.LINE_AA)
        cv2.putText(pad, "space play   . , frame   arrows 2s   ] [ 15s   "
                        "m mark   k game end   s shot   v volley   n winner   "
                        "u undo   w write   q quit",
                    (14, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.42, DIM, 1, cv2.LINE_AA)

        # A ribbon of where the marks sit, so the coverage is visible at a glance.
        ribbon = np.full((16, view.shape[1], 3), 40, dtype=np.uint8)
        for m in self.marks:
            x = int(view.shape[1] * m.time_s * self.fps / max(1, self.total))
            cv2.line(ribbon, (x, 0), (x, 16), MARK_COLOUR.get(m.kind, GO), 2)
        cursor = int(view.shape[1] * self.index / max(1, self.total))
        cv2.line(ribbon, (cursor, 0), (cursor, 16), INK, 1)
        return np.vstack([pad, view, ribbon])

    def run(self):
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        except cv2.error as exc:
            raise SystemExit(
                "no display available for the labelling UI (%s).\n"
                "Use --mark SECONDS instead; see --help." % exc)
        while True:
            view = self.render()
            if view is None:
                break
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(int(1000 / self.fps) if self.playing else 20) & 0xFF

            if self.playing:
                self.seek(1)
                if self.index >= self.total - 1:
                    self.playing = False
            if key == ord(' '):
                self.playing = not self.playing
            elif key == ord('.'):
                self.playing = False; self.seek(1)
            elif key == ord(','):
                self.playing = False; self.seek(-1)
            elif key in (83, ord('l')):
                self.seek(int(2 * self.fps))
            elif key in (81, ord('h')):
                self.seek(-int(2 * self.fps))
            elif key == ord(']'):
                self.seek(int(15 * self.fps))
            elif key == ord('['):
                self.seek(-int(15 * self.fps))
            elif key == ord('m'):
                self.mark("rally")
            elif key == ord('k'):
                self.mark("game")
            elif key == ord('s'):
                self.mark("shot")
            elif key == ord('v'):
                self.mark("volley")
            elif key == ord('n'):
                self.mark("winner")
            elif key == ord('u'):
                self.undo()
            elif key == ord('w'):
                cv2.destroyWindow(WINDOW)
                return self.marks
            elif key == ord('q'):
                cv2.destroyWindow(WINDOW)
                return None
        cv2.destroyWindow(WINDOW)
        return self.marks


def main(argv=None) -> None:
    p = cli.parser("Hand-mark rally-ending shots.", __doc__)
    p.add_argument("video", nargs="?")
    p.add_argument("--out", default="labels.json")
    p.add_argument("--start", type=float, default=0.0, help="start scrubbing here")
    p.add_argument("--mark", action="append", type=float, default=[],
                   metavar="SECONDS", help="add a mark without the GUI")
    p.add_argument("--game-mark", action="append", type=float, default=[],
                   metavar="SECONDS", help="add a game-ending mark without the GUI")
    p.add_argument("--shot", action="append", type=float, default=[],
                   metavar="SECONDS", help="add a shot mark without the GUI")
    p.add_argument("--volley", action="append", type=float, default=[],
                   metavar="SECONDS", help="add a volley mark without the GUI")
    p.add_argument("--winner", action="append", type=float, default=[],
                   metavar="SECONDS", help="add a winner mark without the GUI")
    p.add_argument("--show", metavar="FILE", help="print an existing label file")
    p.add_argument("--append", action="store_true",
                   help="add to --out rather than replacing it")
    args = p.parse_args(argv)

    if args.show:
        marks = load(args.show)
        print("%d marks in %s" % (len(marks), args.show))
        previous = None
        for m in marks:
            gap = "" if previous is None else "   (+%.1f s)" % (m.time_s - previous)
            print("   t=%8.2f  %s%s" % (m.time_s, m.kind, gap))
            previous = m.time_s
        return
    if not args.video:
        p.error("a video is required")

    existing = load(args.out) if args.append and os.path.exists(args.out) else []

    headless = (args.mark, args.game_mark, args.shot, args.volley, args.winner)
    if any(headless):
        kinds = ("rally", "game", "shot", "volley", "winner")
        marks = existing + [Mark(round(t, 2), kind)
                            for kind, times in zip(kinds, headless) for t in times]
    else:
        picked = Labeller(args.video, args.start).run()
        if picked is None:
            print("cancelled; nothing written")
            return
        marks = existing + picked

    save(args.out, args.video, marks)
    print("%d marks -> %s" % (len(marks), args.out))


if __name__ == "__main__":
    main()
