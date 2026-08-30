# squashvision — brief 1 of 2: shot segmentation

Continuing work on `squashvision`, CPU-only squash video analysis.
**S1–S4 are established. Do not re-derive, re-measure, or open files to confirm them.**
Do not explore beyond the files named in S5/S6.

## S1 · Goal

The tool finds **rally boundaries** but has **no shot events at all**. This brief
builds shot detection and the ground truth to check it against.

A later brief classifies those shots as volleys and winners. Do not attempt that
here — but do not make it harder either: shots must carry which player hit them.

## S2 · Closed — do not retry or relitigate

- **Ball tracking** — tried, resolved 0/14 labelled impacts. Not a tuning problem.
- **Audio** — arena crowd noise buries impacts on broadcast encodes.
- **Connecticut video** — 15 fps, unusable for shot-level work.
- Never add an unmeasured magic number.
- Never report a fitted score without a held-out one beside it.

## S3 · Facts — do not re-measure

**Video to use:** `~/Downloads/squashvisiontest.mp4` — 28.365 fps, 3016×1696, 596 s
- play t=300–400 and t=505–590; break t=405–505
- no roster, no detection profile → run with module defaults (`DIFF_THRESHOLD=25`),
  i.e. omit `--profile` and `--roster`
- court: `profiles/bates_court.json`, 10 landmarks, 11.3 cm mean residual (good)

**Video to avoid:** `~/Downloads/Court 4 public⧸Bates  vs. Connecticut … Trim.mp4`
- 15 fps, 1920×1080. Rally-scale only, never shot-scale.
- filename contains U+29F8 → any command touching it needs `PYTHONIOENCODING=utf-8`
  or cp1252 raises `UnicodeEncodeError`

**Stride** (squashvisiontest t=300–400, defaults, no roster; % = both players resolved):
`stride 1 → 28.4/s → 73%` · **`stride 2 → 14.2/s → 77% ← use this`** · `stride 4 → 7.1/s → 65%`
Stride 2 beats stride 1: the larger `dt` widens the Kalman gates favourably.
Use stride 1 only where temporal resolution is the actual constraint.

## S4 · Environment

- Interpreter `./.venv/Scripts/python.exe`. Bare `python` opens the Microsoft Store and fails.
- `PYTHONIOENCODING=utf-8` for anything touching the Connecticut filename.
- Not a git repo. Git Bash and PowerShell both available.

## S5 · API — complete for these tasks; do not open modules to rediscover it

```
cli.parser(desc, epilog) -> ArgumentParser
cli.add_span_arguments(p, stride=N)   # video/--start/--duration/--stride
cli.add_tracking_arguments(p)         # --profile/--roster
cli.add_court_argument(p)             # --calibration
cli.apply_tracking(args); cli.load_court(args); cli.progress(label)
cli.video_info(path) -> VideoInfo(width,height,fps,frames); .dt(stride); .duration_s
cli.analyse_in_court(args, court, smoothing="normal", show_progress=False)
    -> Analysis(samples, raw, tracks, solid, dropped, dt); .coasted
       tracks = [t0, t1]; each a list of (x_m, y_m) or None, in court metres

players.analyse(path, start, duration, stride, progress) -> (samples, roi_size)
Sample(frame_index, time_s, slots[2], merged, blobs, in_play, waiting, gates)
Detection(cx, cy, x, y, w, h, area, brightness, shirt, foot, source)
    source in {"measured","split","predicted"}   # predicted = coasted, no pixels

play.mark_play(samples, dt) -> [Segment(start_s, end_s, playing)]
play.suppress_breaks(samples) -> int

rallies.find_rallies(samples, tracks, court, dt, quiet_speed, min_break,
                     serve_bonus, min_rally) -> ([Rally(start_s,end_s)], breaks)
rallies.speeds(track, dt)                      # None where unmeasurable
rallies.in_service_box(p, court, margin) -> "L"|"R"|None
rallies.serve_ready(p0, p1, court) -> bool
rallies.evaluate(detected, reference, tolerance)
    -> {matched, false_alarms, missed, precision, recall, offsets}

scoredigits.read(video, start, duration, box, progress) -> Reading
Reading(points, final, sampled); Point(time_s, row, game, score)
    row 0 = top player on board, row 1 = bottom
SCORE_BOX lives in score/scoreboard.py

court.Court: .to_court(fx,fy), .spec, .inside(), .plan_point()
court.spec: width 6.40, length 9.75, short_line 4.26, box 1.60, .half 3.20

label.py: Mark(time_s, kind); save(path, video, marks); load(path); Labeller.run()
overlay: labels(), trail(), writer(), panel(), TRACK_COLOURS, INK/DIM/GO/WARN
__main__.COMMANDS: name -> (module_path, one_line_description)
```

## S6 · Conventions — load-bearing, follow them

1. Every threshold constant carries the measurement that set it in a comment,
   including what was tried first and why it failed. If unmeasured, say so.
2. Never pass inferred data off as observed — carry provenance like `Detection.source`.
3. Anything fitted is also scored on data it was not fitted on. Print both; label
   the optimistic one. See `fit/train.py`.
4. Register commands in `__main__.COMMANDS` by *running* order, one-line lowercase desc.
5. Module docstring = the idea + the failure it avoids; pass as `cli.parser(desc, __doc__)`.
6. Reuse `cli.add_span_arguments` / `add_tracking_arguments` / `add_court_argument`.

## S7 · Tasks

Chain: `T1 → T2`, and `T3` independently. T3 gates the *next* brief, not this one.

### T1 — shot marks in `rally/label.py` · needs: —

Add keys in `Labeller.run()`: `s` shot, `v` volley, `n` winner (rally-ending shot
not reached). Extend `Mark.kind` beyond `"rally"`/`"game"`. Add headless flags
`--shot`, `--volley`, `--winner SECONDS`, mirroring `--mark`/`--game-mark`.
Existing label files must still load.

Volley and winner marks are not used in this brief — they are the training data for
the next one. Add them now so labelling happens once, not twice.

DONE: a 60 s span labels both interactively and headlessly; `--show` prints new kinds.

### T2 — `rally/shots.py`, command `shots` · needs: T1

Detect shot instants within each rally from movement. A shot is one excursion from
the T and back → candidates are per-player speed minima, or reversals of radial
velocity about the T. Use `rallies.speeds()` on court-metre tracks from
`cli.analyse_in_court`.

Attribution without a ball, via two constraints:
- Shots **strictly alternate** between players within a rally.
- The **server** is identifiable at rally start via `serve_ready()` / `in_service_box()`.
  Server + alternation attributes every shot in the rally.
- Re-anchor each rally start, so one missed shot does not flip parity match-wide.

Emit `Shot(time_s, player, position_m, rally_index, source)`; `source` records how
the instant was derived. Write CSV. Print shots/rally, inter-shot interval
distribution, and count of rallies that failed to anchor.

Sanity bounds (not ground truth): rallies run 5–20 shots; median inter-shot interval
0.8–2.0 s. Outside that means the detector is wrong, not the match unusual.

DONE: runs on squashvisiontest t=300–400 within those bounds, and scored against T1
labels via `rallies.evaluate()` at a stated tolerance.

### T3 — verify the score box · needs: —

`SCORE_BOX` in `score/scoreboard.py` is hard-coded to one broadcast. Run `scoredigits`
on squashvisiontest.mp4 and confirm the final tally equals the score on the board.
**If it does not match, fix the box fractions now.** A wrong box mixes two games
together and never reports an error. The next brief is built entirely on this reading,
so a failure here is cheap to fix now and expensive to discover later.

DONE: final tally matches the visible board, or the box fractions are corrected until
it does. Report the reading either way.

## S8 · Report back when done

The next brief is written against these. Say explicitly:

1. The final field list of `Shot` — if it differs from `(time_s, player,
   position_m, rally_index, source)`, name the actual fields.
2. The `Mark.kind` strings you settled on for shot / volley / winner.
3. The `shots` command name and its CSV column order.
4. T2's held-out score against the T1 labels, and the tolerance used.
5. Whether T3 passed, or what the corrected `SCORE_BOX` fractions are.

## Done — 2026-08-30

1. `Shot` fields unchanged: `(time_s, player, position_m, rally_index, source)`.
2. `Mark.kind`: `rally`, `game`, `shot`, `volley`, `winner` — exactly as specified.
   Keys `s`/`v`/`n`; headless flags `--shot`/`--volley`/`--winner SECONDS`.
3. Command `shots`; CSV columns `time_s,player,position_x_m,position_y_m,rally_index,source`.
4. **Real hand labels now exist** (`labels.json`, 61 marks: 45 shot, 10 volley, 5
   winner, 1 game — collected across all 5 real rallies in both measured play
   spans; see the note below on which "rallies" are real). Scored against 60
   shot-equivalent labels (shot+volley+winner) at tolerance 1.0s:
   **precision 0.67, recall 0.57** (34 matched, 17 false, 26 missed).
5. T3 **passed** — `scoredigits` final tally (player0 6-0-0-0-0, player1
   11-4-0-0-0) verified against the actual last frame of the clip (t≈595.9s)
   by eye. `SCORE_BOX` unchanged, no fix needed.

**One correction to S3 found while hand-labelling:** of the 9 "rallies"
`find_rallies` reports across t=300-590 with module defaults, only **5 are
real points** — indices 1, 2, 3, 7, 8. Indices 0, 4, 5, and 6 are
break/warm-up artifacts (human-verified: only one player on court), not
under-segmentation like the known 57s-merged-rally issue — a second,
independent failure mode in the rally detector. `winners.py`'s "let"
classification (no score change matched nearby) already absorbs these
harmlessly. Do not treat "N rallies found" as N real points without checking
against the scoreboard or hand labels.

Also: `rally/shots.py`'s hand-label scoring originally only counted
`kind=="shot"` marks as ground truth, silently excluding `volley`/`winner`
marks even though those are shots too — fixed before labelling started, or
every labelled volley/winner would have counted as a false alarm.
