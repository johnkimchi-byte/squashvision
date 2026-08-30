"""Side-by-side demo: camera view next to a bird's-eye plan of the court.

    python -m squashvision birdseye VIDEO --start 300 --duration 20 --out plan.mp4

Court positions come from each player's foot contact point, not the blob
centroid -- see `court.py` for why that distinction matters.
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import cli, overlay
from ..detect.players import PLAYER_WIDTH, _roi_box, summarise
from ..geometry import tracks as T

CAMERA_WIDTH = 900
TRAIL = 40                  # samples of court-position history to draw

COLOURS = overlay.TRACK_COLOURS
INK, DIM = overlay.INK, overlay.DIM


def render(video: str, samples, tracks, solid, court, out: str, stride: int) -> None:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    by_index = {s.frame_index: s for s in samples}
    order = sorted(by_index)
    plan_w, plan_h = court.plan_size()
    cam_h = None
    writer = None
    blank = court.draw_plan()
    names = overlay.labels()
    history = [[], []]
    cap.set(cv2.CAP_PROP_POS_FRAMES, order[0])
    next_expected = order[0]

    for k, index in enumerate(order):
        if index != next_expected:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        next_expected = index + 1
        if not ok:
            break
        src_h, src_w = frame.shape[:2]
        cam_h = int(round(src_h * CAMERA_WIDTH / src_w))
        camera = cv2.resize(frame, (CAMERA_WIDTH, cam_h), interpolation=cv2.INTER_AREA)
        scale = CAMERA_WIDTH / PLAYER_WIDTH
        rx0, ry0, _, _ = _roi_box(CAMERA_WIDTH, cam_h)
        sample = by_index[index]

        for i, det in enumerate(sample.slots):
            if det is None:
                continue
            colour = COLOURS[i]
            x0 = int(rx0 + det.x * scale)
            y0 = int(ry0 + det.y * scale)
            x1 = int(rx0 + (det.x + det.w) * scale)
            y1 = int(ry0 + (det.y + det.h) * scale)
            thin = det.source == "predicted"
            cv2.rectangle(camera, (x0, y0), (x1, y1), colour, 1 if thin else 2)
            # Mark the foot point: this, not the centroid, is what gets mapped.
            cv2.drawMarker(camera, ((x0 + x1) // 2, y1), colour,
                           cv2.MARKER_TRIANGLE_UP, 13, 2)

        plan = blank.copy()
        for i in (0, 1):
            position = tracks[i][k] if k < len(tracks[i]) else None
            history[i].append(position)
            history[i] = history[i][-TRAIL:]
            overlay.trail(plan, [None if p is None else court.plan_point(*p)
                                 for p in history[i]], COLOURS[i])
            if position is not None:
                centre = court.plan_point(*position)
                measured = solid[i][k] if k < len(solid[i]) else False
                # Hollow means the path is being carried by momentum across a
                # gap: a real place to be, but not one the pixels confirmed.
                cv2.circle(plan, centre, 9, COLOURS[i], -1 if measured else 2,
                           cv2.LINE_AA)
                cv2.circle(plan, centre, 9, (40, 40, 45), 1, cv2.LINE_AA)

        canvas = np.full((max(cam_h, plan_h), CAMERA_WIDTH + plan_w + 14, 3),
                         22, dtype=np.uint8)
        canvas[:cam_h, :CAMERA_WIDTH] = camera
        canvas[:plan_h, CAMERA_WIDTH + 14:] = plan

        # Dark panels: the court is bright enough to swallow plain text.
        for px, py, pw, ph in ((0, 0, 330, 92), (0, cam_h - 26, CAMERA_WIDTH, 26)):
            overlay.panel(canvas, px, py, pw, ph, alpha=0.65)

        cv2.putText(canvas, "t = %6.2f s" % sample.time_s, (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, INK, 1, cv2.LINE_AA)
        for i in (0, 1):
            p = tracks[i][k] if k < len(tracks[i]) else None
            text = "%s  --" % names[i] if p is None else \
                "%s  x=%.1f m  y=%.1f m" % (names[i], p[0], p[1])
            cv2.putText(canvas, text, (12, 50 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, COLOURS[i] if p is not None else DIM, 1, cv2.LINE_AA)
        cv2.putText(canvas, "triangle = foot contact point, the part mapped to the floor",
                    (12, cam_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, DIM, 1, cv2.LINE_AA)

        if writer is None:
            writer = overlay.writer(out, fps / stride,
                                    (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)

    cap.release()
    if writer:
        writer.release()


def main(argv=None) -> None:
    p = cli.parser("Bird's-eye view of tracked players.", __doc__)
    cli.add_span_arguments(p, stride=2, start=300.0, duration=20.0)
    cli.add_tracking_arguments(p)
    cli.add_court_argument(p)
    p.add_argument("--out", default="birdseye.mp4")
    p.add_argument("--smoothing", choices=sorted(T.SMOOTHING), default="normal",
                   help="how much momentum to give the plotted path")
    args = p.parse_args(argv)
    cli.apply_tracking(args)

    court = cli.load_court(args)
    print(court.report())
    depth = court.visible_depth()
    if depth < court.spec.length - 0.05:
        print("note: the camera sees %.1f m of the %.1f m court; anything beyond "
              "that is unobservable, not absent." % (depth, court.spec.length))

    run = cli.analyse_in_court(args, court, args.smoothing)
    print(summarise(run.samples))
    print("smoothing: %s (alpha=%s), sample interval %.3f s, %d points bridged"
          % (args.smoothing, T.SMOOTHING[args.smoothing], run.dt, run.coasted))
    print(T.summarise_court(run.raw, run.tracks, run.dropped, court, run.dt))
    render(args.video, run.samples, run.tracks, run.solid, court, args.out,
           args.stride)
    print("birds-eye -> " + args.out)


if __name__ == "__main__":
    main()
