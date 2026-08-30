"""Say who the two players are, by pointing at them.

The tracker bootstraps identity from the footage itself: it takes the two
biggest player-sized blobs and calls the brighter one player 0.  On the
Bates/Connecticut capture that is wrong in both halves.  A spectator sitting
below the near wall and another at the right edge are *bigger* blobs than the
Connecticut player when he is at the back of the court, so the real player is
dropped before tracking ever sees him; and the two shirts here differ mainly in
lightness, which the association cost deliberately discounts, so nothing keeps
the identities from trading places at a crossing.

Both are things a person can settle in a few seconds and no heuristic can:

    python -m squashvision roster VIDEO --out conn_roster.json --profile conn.json

Pass the same --profile the tracker will run with: the blobs offered for
clicking are the blobs detection actually produces, and they change with it.

Scrub to a frame where both players are clear, click a player, and press `1` or
`2` to say which one he is.  Do that at several points around the court -- the
shirt colour is stored as the median over the samples, so a few clicks from
front and back cover the light there.  Click anyone who is *not* a player and
press `x` to fence that patch of the frame off.  A fence remembers how big the
person in it was and only catches blobs of about that size, because the
spectators at the near glass stand in the same rows as a player at the back of
the court; if you fence something as tall as a player the banner says so, and
`u` takes it back.

Keys:  space play/pause    . , step a frame    -> <- 2 s    ] [ 15 s
       click  select a blob      1 / 2  this is that player
       x      not a player: never detect anything here again
       u      undo the last thing      w  write and quit      q  quit

Without a display, pass the picks in as fractions of the frame:

    python -m squashvision roster VIDEO --out r.json \\
        --player 1=0.21,0.30@152.5 --player 2=0.78,0.28@152.5 \\
        --not-player 0.32,0.92@152.5

and review an existing file with `--show r.json`.
"""

from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np

from .. import cli, overlay
from . import players as P

WINDOW = "squashvision roster"
DISPLAY_WIDTH = 1400
BANNER = 96
INK, DIM, GO, WARN = overlay.INK, overlay.DIM, overlay.GO, overlay.WARN
SLOT_COLOURS = overlay.TRACK_COLOURS

IGNORE_PAD = 8          # ROI px grown around a rejected blob, since a spectator
                        # shifts in his seat between frames
PICK_RADIUS = 40        # px in the displayed view: how near a click must land
RISKY_RELATIVE = 0.75   # a fenced blob this near the height of the others in
                        # its frame is probably a player, not a spectator
BG_STRIDE = 2           # the tracker's default --stride: a background chunk
                        # spans CHUNK_FRAMES *sampled* frames, so this many
                        # times as many source frames.  Matching it matters --
                        # a median over a narrower window bakes a player who
                        # stood still into the background and he stops being a
                        # blob you can click on.


def _roi_of(frame: np.ndarray) -> tuple[int, int]:
    """Size of the analysis ROI for this frame, in ROI pixels."""
    h, w = frame.shape[:2]
    small_h = int(round(h * P.PLAYER_WIDTH / w))
    x0, y0, x1, y1 = P._roi_box(P.PLAYER_WIDTH, small_h)
    return x1 - x0, y1 - y0


def _frame_to_roi(fx: float, fy: float, frame: np.ndarray) -> tuple[float, float]:
    """A point given as a fraction of the whole frame, in ROI pixels."""
    h, w = frame.shape[:2]
    small_h = int(round(h * P.PLAYER_WIDTH / w))
    x0, y0, _, _ = P._roi_box(P.PLAYER_WIDTH, small_h)
    return fx * P.PLAYER_WIDTH - x0, fy * small_h - y0


class Roster:
    """Who the players are: a reference shirt colour each, plus dead zones."""

    def __init__(self, video: str = "", names=None):
        self.video = video
        self.names = list(names or ["Player 1", "Player 2"])
        self.samples: list[list] = [[], []]      # Lab triples clicked per slot
        self.heights: list[list] = [[], []]      # box heights, in ROI fractions
        self.ignore: list[dict] = []             # ROI-fraction boxes
        self.history: list[tuple] = []           # for undo

    # --- content -------------------------------------------------------

    def shirt(self, slot: int):
        """Reference torso colour for a slot: the median of its samples."""
        if not self.samples[slot]:
            return None
        return tuple(int(v) for v in np.median(np.array(self.samples[slot]), axis=0))

    def spread(self, slot: int) -> float:
        """Typical distance of a sample from the slot's reference colour.

        This is the number that says whether enough has been clicked: if it is
        as large as the gap between the two players, the colours cannot tell
        them apart and more samples will not help -- the shirts are too alike.
        """
        reference = self.shirt(slot)
        if reference is None or len(self.samples[slot]) < 2:
            return float("nan")
        gaps = [P._shirt_distance(reference, s, P.ANCHOR_LIGHTNESS_W)
                for s in self.samples[slot]]
        return float(np.median(gaps))

    def separation(self) -> float:
        """Distance between the two reference colours."""
        return P._shirt_distance(self.shirt(0), self.shirt(1), P.ANCHOR_LIGHTNESS_W)

    def add_sample(self, slot: int, shirt, height: float | None = None) -> None:
        self.samples[slot].append([int(v) for v in shirt])
        if height is not None:
            self.heights[slot].append(height)
        self.history.append(("sample", slot))

    def zones(self) -> tuple:
        """The fences in the shape players.in_zone wants."""
        return tuple(tuple(z["box"]) + (z.get("height"),) for z in self.ignore)

    def player_height(self) -> float:
        """Typical height of a blob somebody called a player, in ROI fractions."""
        seen = self.heights[0] + self.heights[1]
        return float(np.median(seen)) if seen else float("nan")

    def risky(self, zone) -> bool:
        """Would this fence also catch a player?

        A fence only catches blobs of the height it was drawn around, so one
        drawn around a spectator is harmless.  One drawn around something as
        tall as a player is not: the spectators at the near glass stand in the
        same rows as a player at the back of the court, and only size tells
        them apart.

        The comparison is against the tallest *other* blob in the frame the
        fence was drawn in, which is the only like-for-like evidence available:
        apparent height doubles from the front of the court to the back, so a
        fence cannot be judged against players measured somewhere else.  On
        this footage a spectator comes out at around 0.45 of the players beside
        him, and anything above 0.75 is more likely to be a player.
        """
        relative = zone.get("relative")
        return relative is not None and relative >= RISKY_RELATIVE

    def add_ignore(self, box, roi_w: int, roi_h: int, tallest=None) -> None:
        """Fence off a padded box around a blob, in ROI fractions.

        The blob's height goes in with it: a spectator at the near glass sits in
        the same rows as a player at the back of the court, and only size tells
        them apart.  See players._ignored.  `tallest` is the height of the
        biggest other blob in the same frame, kept so the fence can be checked
        for catching a player -- see `risky`.
        """
        x, y, w, h = box
        self.ignore.append({
            "box": [max(0.0, (x - IGNORE_PAD) / roi_w),
                    max(0.0, (y - IGNORE_PAD) / roi_h),
                    min(1.0, (x + w + IGNORE_PAD) / roi_w),
                    min(1.0, (y + h + IGNORE_PAD) / roi_h)],
            "height": h / roi_h,
            "relative": None if not tallest else round(h / float(tallest), 3),
        })
        self.history.append(("ignore", None))

    def undo(self) -> None:
        if not self.history:
            return
        kind, slot = self.history.pop()
        if kind == "sample" and self.samples[slot]:
            self.samples[slot].pop()
            if self.heights[slot]:
                self.heights[slot].pop()
        elif kind == "ignore" and self.ignore:
            self.ignore.pop()

    def complete(self) -> bool:
        return all(self.shirt(i) is not None for i in (0, 1))

    # --- storage -------------------------------------------------------

    def as_dict(self) -> dict:
        return {
            "video": self.video,
            "roi": list(P.COURT_ROI),
            "players": [
                {
                    "slot": i,
                    "name": self.names[i],
                    "shirt": list(self.shirt(i)) if self.shirt(i) else None,
                    "rgb": list(_lab_to_rgb(self.shirt(i))) if self.shirt(i) else None,
                    "samples": self.samples[i],
                    "heights": [round(v, 4) for v in self.heights[i]],
                    "spread": round(self.spread(i), 1) if self.samples[i] else None,
                }
                for i in (0, 1)
            ],
            "ignore": [{"box": [round(v, 4) for v in z["box"]],
                        "height": round(z["height"], 4) if z["height"] else None,
                        "relative": z.get("relative")}
                       for z in self.ignore],
            "separation": (round(self.separation(), 1)
                           if self.complete() else None),
            "note": "Shirt colours are median Lab over the clicked samples, in the "
                    "analysis ROI. Ignore boxes are ROI fractions: nothing inside "
                    "one is ever treated as a player.",
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, indent=2)


def _lab_to_rgb(lab) -> tuple:
    if lab is None:
        return (0, 0, 0)
    patch = np.array([[list(lab)]], dtype=np.uint8)
    b, g, r = cv2.cvtColor(patch, cv2.COLOR_LAB2BGR)[0][0]
    return int(r), int(g), int(b)


def load(path: str) -> Roster:
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    roster = Roster(blob.get("video", ""),
                    [p.get("name") or "Player %d" % (i + 1)
                     for i, p in enumerate(blob["players"])])
    for i, entry in enumerate(blob["players"]):
        roster.samples[i] = [list(s) for s in entry.get("samples", [])]
        roster.heights[i] = [float(v) for v in entry.get("heights", [])]
        if not roster.samples[i] and entry.get("shirt"):
            roster.samples[i] = [list(entry["shirt"])]
    roster.ignore = [z if isinstance(z, dict) else {"box": list(z), "height": None}
                     for z in blob.get("ignore", [])]
    return roster


class Picker:
    """Scrub the match and point at the two players."""

    def __init__(self, video: str, start_s: float = 0.0, roster: Roster | None = None):
        self.cap = cv2.VideoCapture(video)
        if not self.cap.isOpened():
            raise SystemExit("cannot open " + video)
        self.video = video
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.index = min(self.total - 1, max(0, int(start_s * self.fps)))
        self.playing = False
        self.roster = roster or Roster(video)
        self.frame = None
        self.shown = -1
        self.background = None
        self.bg_chunk = None
        self.blobs: list = []            # (box, Detection) for the shown frame
        self.selected: int | None = None
        self.click = None
        self.scale = 1.0

    # --- frames --------------------------------------------------------

    @property
    def time_s(self) -> float:
        return self.index / self.fps

    def seek(self, frames: int) -> None:
        self.index = max(0, min(self.total - 1, self.index + frames))

    def _background(self, gray_shape) -> np.ndarray:
        """Median background for the chunk the playhead is in.

        Rebuilt only when the playhead crosses into a new chunk -- it costs 21
        seeks, which is too slow to do while scrubbing.
        """
        window = P.CHUNK_FRAMES * BG_STRIDE
        chunk = self.index // window
        if chunk == self.bg_chunk and self.background is not None:
            return self.background
        first = chunk * window
        last = min(self.total - 1, first + window - 1)
        stack = []
        for at in np.linspace(first, last, P.BG_SAMPLES):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(at))
            ok, frame = self.cap.read()
            if ok:
                stack.append(P._prepare(frame)[0])
        self.shown = -1                  # the capture position has moved
        if not stack:
            return np.zeros(gray_shape, dtype=np.uint8)
        self.background = np.median(np.stack(stack), axis=0).astype(np.uint8)
        self.bg_chunk = chunk
        return self.background

    def read(self):
        if self.shown != self.index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.index)
            ok, frame = self.cap.read()
            if ok:
                self.frame = frame
                self.shown = self.index
                self.blobs = []
        return self.frame

    def detect(self):
        """Player-sized blobs in the shown frame, with their torso colour."""
        if self.blobs or self.frame is None:
            return self.blobs
        gray, lab = P._prepare(self.frame)
        # Held in a local rather than read back from self.background: a chunk
        # from which no frame could be read returns a stand-in without caching
        # one, and the retry below would then difference against None.
        background = self._background(gray.shape)
        mask = P._foreground(gray, background)
        if self.shown != self.index:     # background build moved the capture
            self.read()
            gray, lab = P._prepare(self.frame)
            mask = P._foreground(gray, background)
        roi_h, roi_w = mask.shape[:2]
        zones = self.roster.zones()
        found = []
        for x, y, w, h, _area in P._components(mask):
            # A blob already fenced off is gone from here too, so pressing `x`
            # visibly removes it and the remaining picks cannot land on it.
            if P.in_zone(zones, x, y, w, h, roi_w, roi_h):
                continue
            det = P._measure(gray, lab, mask, (x, y, w, h), "measured")
            if det is not None:
                found.append(((x, y, w, h), det))
        self.blobs = found
        self.selected = None
        return found

    # --- picking -------------------------------------------------------

    def pick(self, vx: float, vy: float) -> None:
        """Select the blob under a click in the displayed view."""
        x, y = vx / self.scale, vy / self.scale
        best, best_gap = None, None
        for i, ((bx, by, bw, bh), _det) in enumerate(self.blobs):
            if bx <= x <= bx + bw and by <= y <= by + bh:
                gap = 0.0
            else:
                gap = math.dist((x, y), (bx + bw / 2, by + bh / 2))
            if best_gap is None or gap < best_gap:
                best, best_gap = i, gap
        if best is not None and best_gap <= PICK_RADIUS / self.scale:
            self.selected = best

    def assign(self, slot: int) -> None:
        if self.selected is None:
            return
        box, det = self.blobs[self.selected]
        if det.shirt is not None:
            self.roster.add_sample(slot, det.shirt, box[3] / _roi_of(self.frame)[1])

    def reject(self) -> None:
        if self.selected is None:
            return
        roi_w, roi_h = _roi_of(self.frame)
        others = [b[3] for i, (b, _d) in enumerate(self.blobs) if i != self.selected]
        self.roster.add_ignore(self.blobs[self.selected][0], roi_w, roi_h,
                               max(others) if others else None)
        self.selected = None
        self.blobs = []                  # re-detect: the box is now fenced off

    # --- drawing -------------------------------------------------------

    def _view(self) -> np.ndarray:
        frame = self.read()
        gray, _ = P._prepare(frame)
        roi_h, roi_w = gray.shape[:2]
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (P.PLAYER_WIDTH, int(round(h * P.PLAYER_WIDTH / w))),
                           interpolation=cv2.INTER_AREA)
        x0, y0, x1, y1 = P._roi_box(small.shape[1], small.shape[0])
        self.scale = DISPLAY_WIDTH / roi_w
        view = cv2.resize(small[y0:y1, x0:x1], None, fx=self.scale, fy=self.scale,
                          interpolation=cv2.INTER_LINEAR)

        for zone in self.roster.ignore:
            fx0, fy0, fx1, fy1 = zone["box"]
            a = (int(fx0 * view.shape[1]), int(fy0 * view.shape[0]))
            b = (int(fx1 * view.shape[1]), int(fy1 * view.shape[0]))
            patch = view[a[1]:b[1], a[0]:b[0]]
            if patch.size:
                view[a[1]:b[1], a[0]:b[0]] = (patch * 0.45).astype(np.uint8)
            cv2.rectangle(view, a, b, (70, 70, 90), 1)
            cv2.putText(view, "ignored", (a[0] + 4, a[1] + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 130), 1, cv2.LINE_AA)

        for i, ((bx, by, bw, bh), det) in enumerate(self.detect()):
            p0 = (int(bx * self.scale), int(by * self.scale))
            p1 = (int((bx + bw) * self.scale), int((by + bh) * self.scale))
            chosen = i == self.selected
            # Which slot this blob looks like, so a mis-click is visible before
            # it is committed.
            gaps = [P._shirt_distance(self.roster.shirt(s), det.shirt,
                                      P.ANCHOR_LIGHTNESS_W) for s in (0, 1)]
            near = None
            if not all(math.isnan(g) for g in gaps):
                near = int(np.nanargmin(gaps))
            colour = SLOT_COLOURS[near] if near is not None else DIM
            cv2.rectangle(view, p0, p1, INK if chosen else colour, 2 if chosen else 1)
            if det.shirt is not None:
                r, g, b = det.rgb
                cv2.rectangle(view, (p0[0], p0[1] - 16), (p0[0] + 16, p0[1] - 2),
                              (int(b), int(g), int(r)), -1)
                cv2.rectangle(view, (p0[0], p0[1] - 16), (p0[0] + 16, p0[1] - 2),
                              (90, 90, 90), 1)
            if near is not None and not math.isnan(gaps[near]):
                cv2.putText(view, "~%s %.0f" % (self.roster.names[near], gaps[near]),
                            (p0[0] + 20, p0[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, colour, 1, cv2.LINE_AA)
        return view

    def _banner(self, width: int) -> np.ndarray:
        pad = np.full((BANNER, width, 3), 26, dtype=np.uint8)
        cv2.putText(pad, "t = %8.2f s   frame %d / %d   %d blobs   %s"
                    % (self.time_s, self.index, self.total, len(self.blobs),
                       "PLAYING" if self.playing else "paused"),
                    (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, INK, 1, cv2.LINE_AA)
        for i in (0, 1):
            shirt = self.roster.shirt(i)
            x = 14 + 470 * i
            cv2.rectangle(pad, (x, 40), (x + 16, 56), SLOT_COLOURS[i], -1)
            if shirt is not None:
                r, g, b = _lab_to_rgb(shirt)
                cv2.rectangle(pad, (x + 22, 40), (x + 38, 56), (int(b), int(g), int(r)), -1)
                cv2.rectangle(pad, (x + 22, 40), (x + 38, 56), (90, 90, 90), 1)
            spread = self.roster.spread(i)
            text = "%d: %s   %d clicks" % (i + 1, self.roster.names[i],
                                           len(self.roster.samples[i]))
            if not math.isnan(spread):
                text += "   varies +-%.0f" % spread
            cv2.putText(pad, text, (x + 46, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        INK if shirt is not None else DIM, 1, cv2.LINE_AA)
        risky = sum(1 for z in self.roster.ignore if self.roster.risky(z))
        if risky:
            cv2.putText(pad, "%d fence(s) are player-sized and will delete him -- "
                             "press u" % risky, (width - 470, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, WARN, 1, cv2.LINE_AA)
        if self.roster.complete():
            gap = self.roster.separation()
            worst = max(v for v in (self.roster.spread(0), self.roster.spread(1))
                        if not math.isnan(v)) if any(
                len(s) > 1 for s in self.roster.samples) else float("nan")
            ok = math.isnan(worst) or gap > 2 * worst
            cv2.putText(pad, "shirts %.0f apart%s   %d ignored zones" % (
                gap,
                "" if math.isnan(worst) else
                (" vs +-%.0f within a player -- usable" % worst if ok else
                 " vs +-%.0f within a player -- too alike to separate" % worst),
                len(self.roster.ignore)),
                (14, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GO if ok else WARN,
                1, cv2.LINE_AA)
        else:
            cv2.putText(pad, "click a player, then press 1 or 2 to say which he is"
                             "     x = not a player     %d ignored zones"
                        % len(self.roster.ignore),
                        (14, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, DIM, 1, cv2.LINE_AA)
        return pad

    def _on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)

    def run(self) -> Roster | None:
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW, self._on_mouse)
        except cv2.error as exc:
            raise SystemExit(
                "no display available for the roster picker (%s).\n"
                "Use --player SLOT=x,y@SECONDS instead; see --help." % exc)

        while True:
            view = self._view()
            if view is None:
                break
            cv2.imshow(WINDOW, np.vstack([self._banner(view.shape[1]), view]))
            key = cv2.waitKey(int(1000 / self.fps) if self.playing else 20) & 0xFF

            if self.click is not None:
                cx, cy = self.click
                self.click = None
                if cy >= BANNER:
                    self.pick(cx, cy - BANNER)

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
            elif key in (ord('1'), ord('2')):
                self.assign(key - ord('1'))
            elif key == ord('x'):
                self.reject()
            elif key == ord('u'):
                self.roster.undo()
            elif key == ord('w'):
                cv2.destroyWindow(WINDOW)
                return self.roster
            elif key == ord('q'):
                cv2.destroyWindow(WINDOW)
                return None
        cv2.destroyWindow(WINDOW)
        return self.roster


def point_at(video: str, fx: float, fy: float, time_s: float, roster: Roster,
             slot: int | None) -> bool:
    """Headless equivalent of a click: take the blob nearest (fx, fy) at time_s.

    `slot` names the player it belongs to, or None to fence the blob off as
    somebody who is not playing.
    """
    picker = Picker(video, time_s, roster)
    picker.read()
    if picker.frame is None:
        return False
    roi_w, roi_h = _roi_of(picker.frame)
    x, y = _frame_to_roi(fx, fy, picker.frame)
    picker.scale = 1.0
    picker.detect()
    picker.pick(x, y)
    if picker.selected is None:
        return False
    if slot is None:
        picker.reject()
    else:
        picker.assign(slot)
    return True


def box_to_roi(box, video: str) -> tuple:
    """A box given as fractions of the frame, as fractions of the analysis ROI."""
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("cannot read " + video)
    roi_w, roi_h = _roi_of(frame)
    x0, y0 = _frame_to_roi(box[0], box[1], frame)
    x1, y1 = _frame_to_roi(box[2], box[3], frame)
    return (max(0.0, x0 / roi_w), max(0.0, y0 / roi_h),
            min(1.0, x1 / roi_w), min(1.0, y1 / roi_h))


def parse_player(text: str):
    """`SLOT=x,y@SECONDS` -> (slot, x, y, seconds)."""
    try:
        slot, rest = text.split("=", 1)
        where, when = rest.split("@", 1)
        fx, fy = (float(v) for v in where.split(","))
        return int(slot) - 1, fx, fy, float(when)
    except ValueError:
        raise SystemExit("bad --player %r; expected SLOT=x,y@SECONDS" % text)


def parse_point(text: str):
    """`x,y@SECONDS` -> (x, y, seconds)."""
    try:
        where, when = text.split("@", 1)
        fx, fy = (float(v) for v in where.split(","))
        return fx, fy, float(when)
    except ValueError:
        raise SystemExit("bad --not-player %r; expected x,y@SECONDS" % text)


def parse_box(text: str):
    try:
        values = [float(v) for v in text.split(",")]
        if len(values) != 4:
            raise ValueError
        return tuple(values)
    except ValueError:
        raise SystemExit("bad --ignore %r; expected x0,y0,x1,y1" % text)


def describe(roster: Roster) -> str:
    lines = []
    for i in (0, 1):
        shirt = roster.shirt(i)
        lines.append("  %d %-16s shirt %s  rgb %s  from %d clicks%s" % (
            i + 1, roster.names[i], shirt, _lab_to_rgb(shirt),
            len(roster.samples[i]),
            "" if math.isnan(roster.spread(i)) else
            "  varying +-%.1f" % roster.spread(i)))
    if roster.complete():
        lines.append("  shirts are %.1f apart" % roster.separation())
    for zone in roster.ignore:
        lines.append("  ignored: %.3f,%.3f - %.3f,%.3f (ROI fractions)%s%s%s" % (
            tuple(zone["box"]) +
            ("" if not zone.get("height") else
             ", only blobs about %.3f tall" % zone["height"],
             "" if zone.get("relative") is None else
             " (%.0f%% of the others in frame)" % (100 * zone["relative"]),
             "   <-- player-sized: this fence can delete a player"
             if roster.risky(zone) else "")))
    return "\n".join(lines)


def main(argv=None) -> None:
    p = cli.parser("Say who the two players are.", __doc__)
    p.add_argument("video", nargs="?")
    p.add_argument("--out", default="roster.json")
    p.add_argument("--start", type=float, default=0.0, help="start scrubbing here")
    p.add_argument("--names", help="the two players, comma separated")
    p.add_argument("--player", action="append", default=[], metavar="SLOT=x,y@S",
                   help="sample a player without the GUI, at a fraction of the frame")
    p.add_argument("--not-player", action="append", default=[], metavar="x,y@S",
                   dest="not_player",
                   help="fence off whoever is at this point, without the GUI")
    p.add_argument("--ignore", action="append", default=[], metavar="x0,y0,x1,y1",
                   help="fence off a region outright, as fractions of the frame")
    p.add_argument("--profile", metavar="FILE",
                   help="detection settings from `squashvision autotune` -- pass the "
                        "same one the tracker will use, or the blobs shown here are "
                        "not the blobs it will see")
    p.add_argument("--edit", metavar="FILE", help="open an existing roster to add to")
    p.add_argument("--show", metavar="FILE", help="print an existing roster")
    args = p.parse_args(argv)

    P.apply_profile(args.profile)

    if args.show:
        roster = load(args.show)
        print("roster %s\n%s" % (args.show, describe(roster)))
        return
    if not args.video:
        p.error("a video is required")

    roster = load(args.edit) if args.edit and os.path.exists(args.edit) \
        else Roster(args.video)
    roster.video = args.video
    if args.names:
        names = [n.strip() for n in args.names.split(",")]
        if len(names) != 2:
            raise SystemExit("--names takes exactly two, comma separated")
        roster.names = names

    if args.player or args.ignore or args.not_player:
        for box in args.ignore:
            roster.ignore.append({"box": list(box_to_roi(parse_box(box), args.video)),
                                  "height": None})
        # Fence off the non-players first, so a spectator cannot then be the
        # blob a --player point lands on.
        for spec in args.not_player:
            fx, fy, when = parse_point(spec)
            if not point_at(args.video, fx, fy, when, roster, None):
                print("no blob near %.3f,%.3f at %.2f s -- nothing fenced off"
                      % (fx, fy, when))
        for spec in args.player:
            slot, fx, fy, when = parse_player(spec)
            if slot not in (0, 1):
                raise SystemExit("--player slot must be 1 or 2")
            if not point_at(args.video, fx, fy, when, roster, slot):
                print("no blob near %.3f,%.3f at %.2f s -- not sampled" % (fx, fy, when))
    else:
        picked = Picker(args.video, args.start, roster).run()
        if picked is None:
            print("cancelled; nothing written")
            return
        roster = picked

    if not roster.complete():
        print("warning: a player has no shirt colour; the tracker will fall back "
              "to its own guess for that slot")
    risky = sum(1 for z in roster.ignore if roster.risky(z))
    if risky:
        print("warning: %d fence(s) were drawn around something as tall as a "
              "player, and will delete him if he stands there -- see below" % risky)
    roster.save(args.out)
    print("roster -> %s\n%s" % (args.out, describe(roster)))


if __name__ == "__main__":
    main()
