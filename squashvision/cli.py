"""The argument plumbing every command shares.

Six commands used to repeat the same three blocks: declare `--start/--duration/
--stride`, declare `--profile/--roster` and apply them, then open the video to
read its size and frame rate.  Repeating them meant they drifted -- `train` and
`play` grew `--roster` at different times, and the defaults for `--stride`
disagreed for no reason anyone recorded.  They live here now, so a command
declares what it needs and the meaning is the same everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2


# Set by __main__ to the command the user actually typed, so `--help` shows
# `python -m squashvision demo` rather than the package name alone.  Left None
# when a module is run directly, where argparse's own default is right.
PROG = None


def invocation() -> str:
    """How to run this package, spelled the way it works on this machine.

    Printing a bare `python -m squashvision` is wrong on a stock Windows
    install, where `python` opens the Microsoft Store, and wrong again inside a
    virtual environment that has not been activated.  Naming the interpreter
    that is actually running gives a line that can be pasted back in.
    """
    import os
    import sys

    exe = sys.executable or "python"
    try:                                 # a path relative to here reads better
        relative = os.path.relpath(exe)
        if len(relative) < len(exe):
            exe = relative
    except ValueError:                   # different drive on Windows
        pass
    if os.name == "nt":
        # Forward slashes, deliberately.  PowerShell accepts them, and so does
        # bash -- where a backslash path is silently eaten as escapes, turning
        # `.\.venv\Scripts\python.exe` into `..venvScriptspython.exe` and
        # reporting it as missing.  One spelling that works in both is worth
        # more here than the native one.
        exe = exe.replace("\\", "/")
        # PowerShell will not run a program in the current tree without an
        # explicit `./` -- and `.venv/...` does not count, despite the dot.
        if not os.path.isabs(exe) and not exe.startswith("./"):
            exe = "./" + exe
    if " " in exe:
        exe = '"%s"' % exe
    return exe + " -m squashvision"


def parser(description: str, epilog: str | None = None):
    """An ArgumentParser that names itself correctly however it was invoked."""
    import argparse
    return argparse.ArgumentParser(
        prog=PROG, description=description, epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)


@dataclass
class VideoInfo:
    """What every command needs to know about a file before analysing it."""

    width: int
    height: int
    fps: float
    frames: int

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps else 0.0

    def dt(self, stride: int) -> float:
        """Seconds between analysed samples at this stride."""
        return stride / self.fps


def video_info(path: str) -> VideoInfo:
    """Size and frame rate of a video, or exit saying it cannot be opened."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("cannot open " + path)
    try:
        return VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            # A file with no frame-rate metadata reports 0; 30 is the least
            # surprising stand-in, and every caller divides by this.
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
            frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def add_span_arguments(parser, stride: int = 2, start: float = 0.0,
                       duration=None) -> None:
    """`video`, and which part of it to work on."""
    parser.add_argument("video")
    parser.add_argument("--start", type=float, default=start,
                        help="start time in seconds")
    parser.add_argument("--duration", type=float, default=duration,
                        help="seconds to analyse (default: to the end)")
    parser.add_argument("--stride", type=int, default=stride,
                        help="analyse every Nth frame")


def add_tracking_arguments(parser) -> None:
    """The two files that tell the tracker about this particular match."""
    parser.add_argument("--profile", metavar="FILE",
                        help="detection settings from `squashvision autotune`")
    parser.add_argument("--roster", metavar="FILE",
                        help="who the players are, from `squashvision roster`")


def apply_tracking(args) -> None:
    """Adopt whichever of --profile / --roster were given."""
    from .detect.players import apply_profile, apply_roster
    apply_profile(getattr(args, "profile", None))
    apply_roster(getattr(args, "roster", None))


def progress(label: str = "frames"):
    """A progress callback for `analyse`, printed sparsely enough to read."""
    def report(done: int, total: int) -> None:
        print("  %d/%d %s" % (done, total, label), flush=True)
    return report


@dataclass
class Analysis:
    """One pass over a video, taken all the way to court metres.

    `raw` is what the projection gave; `tracks` is that after smoothing, and
    `solid[i][k]` says whether tracks[i][k] came from a measurement rather than
    the smoother coasting.  Keeping both is what lets a caller report how much
    of a path was actually seen.
    """

    samples: list
    raw: list
    tracks: list
    solid: list
    dropped: int
    dt: float

    @property
    def coasted(self) -> int:
        """Points carried by momentum across a gap rather than measured."""
        return sum(1 for i in (0, 1) for p, s in zip(self.tracks[i], self.solid[i])
                   if p is not None and not s)


def analyse_in_court(args, court, smoothing: str = "normal",
                     show_progress: bool = False) -> Analysis:
    """Detect, track, project onto the floor, and smooth -- the usual pipeline.

    Three commands ran these four steps in the same order with the same
    arguments; the only thing they disagreed on was whether to print progress.
    """
    from .detect.players import analyse
    from .geometry import court as C
    from .geometry import tracks as T

    video = video_info(args.video)
    samples, _ = analyse(args.video, args.start, args.duration, args.stride,
                         progress() if show_progress else None)
    raw, dropped = T.project(samples, C.small_size(video.width, video.height), court)
    dt = video.dt(args.stride)
    prepared = [T.prepare(track, dt, smoothing) for track in raw]
    return Analysis(samples=samples, raw=raw,
                    tracks=[p[0] for p in prepared], solid=[p[1] for p in prepared],
                    dropped=dropped, dt=dt)


def load_court(args):
    """The calibration named on the command line, or the built-in one."""
    from .geometry.court import Court
    return Court.load(args.calibration) if getattr(args, "calibration", None) else Court()


def add_court_argument(parser) -> None:
    parser.add_argument("--calibration", metavar="FILE",
                        help="court calibration from `squashvision calibrate` "
                             "(default: the built-in Bates calibration)")
