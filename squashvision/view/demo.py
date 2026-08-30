"""Render an annotated demo clip of the player tracker.

Draws what the tracker actually believes, not just where it ended up: the
measurement boxes, the recent path of each player, and the Kalman gate -- the
region the motion model expected the player to be in before it saw the frame.
Positions that came from the model coasting rather than from pixels are drawn
differently and labelled, so the clip cannot overstate what was observed.

    python -m squashvision demo VIDEO --start 300 --duration 20 --out demo.mp4
"""

from __future__ import annotations

import cv2

from .. import cli, overlay
from ..detect.players import (PLAYER_WIDTH, _anchor_distance, _roi_box,
                              analyse, summarise)

DEMO_WIDTH = 1280           # output width; annotations scale to match
TRAIL_LENGTH = 24           # samples of path history to draw behind a player

COLOURS = overlay.TRACK_COLOURS
INK, DIM, WARN = overlay.INK, overlay.DIM, overlay.WARN
# Used only when nobody has said who these two are; with a roster the names
# come from it.  See overlay.labels.
GUESSED = ["Player 0 (brighter shirt)", "Player 1 (darker shirt)"]


def _dashed_circle(canvas, centre, radius, colour, dashes=18):
    """A dashed ring, to read as 'expected region' rather than 'measurement'."""
    step = 360.0 / dashes
    for k in range(0, dashes, 2):
        cv2.ellipse(canvas, centre, (radius, radius), 0.0,
                    k * step, (k + 1) * step, colour, 1, cv2.LINE_AA)


def render_demo(video: str, samples, out: str, stride: int,
                width: int = DEMO_WIDTH) -> None:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit("cannot open " + video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    by_index = {s.frame_index: s for s in samples}
    order = sorted(by_index)
    if not order:
        raise SystemExit("nothing to render")

    scale = width / PLAYER_WIDTH          # annotations are in 480-wide ROI px
    labelled = overlay.named()
    labels = overlay.labels() if labelled else GUESSED
    trails = [[], []]
    writer = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, order[0])
    next_expected = order[0]

    for index in order:
        if index != next_expected:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        next_expected = index + 1
        if not ok:
            break

        src_h, src_w = frame.shape[:2]
        height = int(round(src_h * width / src_w))
        canvas = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        rx0, ry0, rx1, ry1 = _roi_box(width, height)
        sample = by_index[index]

        def to_screen(cx, cy):
            return int(round(rx0 + cx * scale)), int(round(ry0 + cy * scale))

        cv2.rectangle(canvas, (rx0, ry0), (rx1, ry1), (70, 70, 70), 1)

        # Gates first, so boxes and trails sit on top of them.
        for i, gate in enumerate(sample.gates):
            if gate is None:
                continue
            gx, gy, radius = gate
            _dashed_circle(canvas, to_screen(gx, gy), int(round(radius * scale)),
                           COLOURS[i])

        for i, det in enumerate(sample.slots):
            if det is None:
                trails[i].append(None)
            else:
                trails[i].append(to_screen(det.cx, det.cy))
            trails[i] = trails[i][-TRAIL_LENGTH:]

        for i, path in enumerate(trails):
            overlay.trail(canvas, path, COLOURS[i])

        for i, det in enumerate(sample.slots):
            if det is None:
                continue
            colour = COLOURS[i]
            x0, y0 = to_screen(det.x, det.y)
            x1, y1 = to_screen(det.x + det.w, det.y + det.h)
            if det.source == "predicted":
                # No pixels backed this: corner ticks only, never a solid box.
                arm = max(6, (x1 - x0) // 4)
                for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                                         (x0, y1, 1, -1), (x1, y1, -1, -1)):
                    cv2.line(canvas, (cx, cy), (cx + dx * arm, cy), colour, 1, cv2.LINE_AA)
                    cv2.line(canvas, (cx, cy), (cx, cy + dy * arm), colour, 1, cv2.LINE_AA)
            else:
                cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 2)
            tag = {"measured": "", "split": "  split", "predicted": "  predicted"}[det.source]
            cv2.putText(canvas, "P%d%s" % (i, tag), (x0, max(14, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
            cv2.drawMarker(canvas, to_screen(det.cx, det.cy), colour,
                           cv2.MARKER_CROSS, 11, 1, cv2.LINE_AA)

        # --- HUD -----------------------------------------------------------
        overlay.panel(canvas, 0, 0, width, 74)
        cv2.putText(canvas, "Squash player tracking", (14, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, INK, 2, cv2.LINE_AA)
        cv2.putText(canvas, "t = %6.2f s   frame %d" % (sample.time_s, sample.frame_index),
                    (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM, 1, cv2.LINE_AA)
        for i in (0, 1):
            det = sample.slots[i]
            state = "lost" if det is None else det.source
            top = 16 + 24 * i
            cv2.rectangle(canvas, (width - 360, top), (width - 344, top + 14),
                          COLOURS[i], -1)
            # The shirt colour this track is keyed on -- the evidence that keeps
            # identity from swapping when the players cross.
            if det is not None and det.shirt is not None:
                overlay.swatch(canvas, width - 340, top, det.rgb)
            # With a roster, show how far this measurement sits from the shirt
            # colour that was named for the slot -- the number the colour gate
            # acts on, so a viewer can see identity being checked, not assumed.
            gap = ""
            if labelled and det is not None and det.source != "predicted":
                distance = _anchor_distance(i, det.shirt)
                if distance == distance:                 # not NaN
                    gap = "  d=%.0f" % distance
            cv2.putText(canvas, "%s  %s%s" % (labels[i], state, gap),
                        (width - 314, top + 13), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, INK if det is not None else DIM, 1, cv2.LINE_AA)
        if sample.merged:
            cv2.putText(canvas, "players merged - no position claimed",
                        (14, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WARN, 1, cv2.LINE_AA)

        overlay.panel(canvas, 0, height - 30, width, 30)
        cv2.putText(canvas, "dashed ring = motion gate   corner ticks = coasted on prediction   "
                            "second swatch = tracked shirt colour   " +
                    ("d = distance from the shirt colour picked for that player; "
                     "over 40 is refused"
                     if labelled else "identity is unvalidated"),
                    (14, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, DIM, 1, cv2.LINE_AA)

        if writer is None:
            writer = overlay.writer(out, fps / stride, (width, height))
        writer.write(canvas)

    cap.release()
    if writer:
        writer.release()


def main(argv=None) -> None:
    p = cli.parser("Render a demo clip of the player tracker.", __doc__)
    cli.add_span_arguments(p, stride=2, start=300.0, duration=20.0)
    cli.add_tracking_arguments(p)
    p.add_argument("--width", type=int, default=DEMO_WIDTH, help="output width in px")
    p.add_argument("--out", default="demo.mp4", help="output mp4")
    args = p.parse_args(argv)
    cli.apply_tracking(args)

    samples, _ = analyse(args.video, args.start, args.duration, args.stride,
                         cli.progress("frames analysed"))
    print(summarise(samples))
    print("rendering...", flush=True)
    render_demo(args.video, samples, args.out, args.stride, args.width)
    print("demo -> " + args.out)


if __name__ == "__main__":
    main()
