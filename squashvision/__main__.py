"""One front door: `python -m squashvision COMMAND [options]`.

The modules are filed by what they do -- detect, geometry, score, rally, fit,
view -- which is right for reading the code and wrong for typing at a shell.
This maps a flat command name onto whichever module implements it, so the
folders can be rearranged without rewriting anyone's notes.

    python -m squashvision                 list the commands
    python -m squashvision demo --help     help for one of them

On Windows, and inside an unactivated virtual environment, `python` is not the
right word -- so the usage line it prints names the interpreter that is actually
running instead.  See cli.invocation.
"""

from __future__ import annotations

import importlib
import sys

# command -> (module, one-line description).  Order is the order they are
# normally run in, not alphabetical, because that is the useful order.
COMMANDS = {
    "calibrate": ("squashvision.geometry.calibrate",
                  "click the court landmarks once per camera angle"),
    "autotune": ("squashvision.fit.autotune",
                 "fit detection settings to this camera and encode"),
    "roster": ("squashvision.detect.roster",
               "say who the two players are, and who is not playing"),
    "track": ("squashvision.detect.players",
              "detect and follow the players; write positions to CSV"),
    "demo": ("squashvision.view.demo",
             "annotated clip showing what the tracker believes"),
    "birdseye": ("squashvision.view.birdseye",
                 "camera view beside a bird's-eye plan of the court"),
    "play": ("squashvision.detect.play",
             "split a match into play and the breaks between games"),
    "scoreboard": ("squashvision.score.scoreboard",
                   "rally ends from changes in the score box"),
    "scoredigits": ("squashvision.score.scoredigits",
                    "read the score itself, cell by cell"),
    "label": ("squashvision.rally.label",
              "hand-mark rally ends to check the detector against"),
    "rallies": ("squashvision.rally.rallies",
                "split play into rallies and score against a reference"),
    "shots": ("squashvision.rally.shots",
              "detect shot instants within each rally, per player"),
    "winners": ("squashvision.rally.winners",
                "classify rallies as winners, errors or lets"),
    "train": ("squashvision.fit.train",
              "fit the rally detector on scoreboard labels, held-out scored"),
    "volleys": ("squashvision.fit.volleys",
               "fit the volley classifier on hand labels, held-out scored"),
}


def usage() -> str:
    from . import cli

    width = max(len(name) for name in COMMANDS)
    lines = ["usage: %s COMMAND [options]" % cli.invocation(), "", "commands:"]
    lines += ["  %-*s  %s" % (width, name, description)
              for name, (_module, description) in COMMANDS.items()]
    lines += ["", "Any command takes --help.  Roughly in running order: calibrate "
                  "and autotune once", "per camera, roster once per match, then the "
                  "rest as often as you like."]
    return "\n".join(lines)


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return
    name = argv[0]
    if name not in COMMANDS:
        matches = [c for c in COMMANDS if c.startswith(name)]
        if len(matches) == 1:
            name = matches[0]
        else:
            print("unknown command %r\n" % argv[0])
            print(usage())
            raise SystemExit(2)
    from . import cli
    # So `--help` on a subcommand names the subcommand, not just the package.
    cli.PROG = cli.invocation() + " " + name
    module = importlib.import_module(COMMANDS[name][0])
    module.main(argv[1:])


if __name__ == "__main__":
    main()
