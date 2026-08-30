# squashvision

Squash match video → player positions in court metres → rallies, shots, winners,
volleys. Fixed-camera footage, classical computer vision, **no neural networks and
no GPU**. Everything is offline batch; nothing runs live.

## Running it

```
./.venv/Scripts/python.exe -m squashvision            # lists all commands
./.venv/Scripts/python.exe -m squashvision demo --help
```

**Bare `python` opens the Microsoft Store and fails.** Always use the venv
interpreter. Commands are registered in `__main__.COMMANDS`; run with no arguments
to see them rather than duplicating the list here.

Typical order: `calibrate` → `autotune` → `roster` (once per camera/encode/match),
then `track` / `demo` / `birdseye` / `rallies` / `shots` / `winners` as needed.

## Environment

- Windows. Git Bash and PowerShell both available; each needs its own syntax.
- Git repo on `master`.
- `PYTHONIOENCODING=utf-8` is **required** for any command touching the Connecticut
  video — its filename contains U+29F8 (`⧸`, not a slash) and the cp1252 console
  raises `UnicodeEncodeError` before doing any work.

## Dependencies: numpy and opencv only

This is deliberate and worth preserving. Logistic regression in `fit/volleys.py` is
hand-rolled (`_sigmoid`, `fit_logistic`) rather than pulling in sklearn. Before
adding any third-party package, check whether ~30 lines of numpy would do.

## Architecture

Dependencies point downward only. The rendering layer knows about the core; the
core never knows about rendering.

```
__main__.py            one front door, maps command names onto modules
view/  rally/  fit/    consumers: draw, segment, fit
detect/ geometry/ score/   core: detect+track, court maths, scoreboard
cli.py  overlay.py     shared: argument plumbing, palette, drawing
```

**The detection/tracking seam is the most important boundary.** In
`detect/players.py`, `Tracker.step()` touches the detector in exactly two lines
(`_foreground`, `_components`). Everything after that operates purely on
`Detection` objects. Keep it that way — it is what makes the detector replaceable.

## Conventions — load-bearing, follow them

1. **Every threshold constant carries the measurement that set it** in a comment,
   including what was tried first and why it failed. Never add a bare magic number.
   If it has not been measured, the comment says so. Read a few in
   `detect/players.py` before adding one.
2. **Never pass inferred data off as observed.** Provenance travels with the data —
   `Detection.source` is `measured` / `split` / `predicted`, the CSV records it, the
   demo draws coasted positions as corner ticks, the plan draws them hollow.
3. **Anything fitted is scored on data it was not fitted on.** Print both numbers
   and label the optimistic one as optimistic. See `fit/train.py`. Config files
   carry a `note` field saying which figure to quote.
4. New commands register in `__main__.COMMANDS` in **running order**, not
   alphabetically, with a one-line lowercase description.
5. Module docstring explains the idea *and the failure it avoids*; pass it through
   as `cli.parser(desc, __doc__)`.
6. Reuse `cli.add_span_arguments` / `add_tracking_arguments` / `add_court_argument`
   rather than redeclaring arguments.

## Videos and profiles

| File | Use |
|---|---|
| `squashvisiontest.mp4` — 28.365 fps, 3016×1696 | **Shot-level work.** Play t=300–400, t=505–590; break t=405–505. No roster or profile; module defaults are correct. Court: `profiles/bates_court.json` (11.3 cm residual). |
| `Court 4 public⧸Bates vs. Connecticut…mp4` — 15 fps, 1920×1080 | Rally-scale only. Half-rate, unusable below rally granularity. Uses `conn_profile.json` + `conn_roster.json`; `mycourt.json` is poor (45.2 cm). |

Detection profiles are **per camera *and* per encode** — a re-encode at another
resolution needs its own. Court calibrations transfer between recordings of the
same camera. Rosters are per match.

**Use `--stride 2` by default, not 1.** Measured 77% both-players-resolved at
stride 2 against 73% at stride 1 — the larger `dt` widens the Kalman gates
favourably. Stride 1 only where temporal resolution is the actual constraint.

## Closed decisions — do not reopen without new evidence

- **Ball tracking** — tried, resolved 0/14 labelled impacts. A ball is four pixels
  across; so is compression noise. Not a tuning problem — reopening it needs new
  footage (higher resolution, or a purpose-placed camera), not new code.
- **Audio** — arena crowd noise buries the impacts on broadcast encodes.
- **Silhouette height as a volley feature** — measured. Noise vs a 2 s baseline is
  ~11% at every stride, and at the 0.2–0.3 s reach timescale the p75 exceeds the
  ~15% signal. Volleys come from position and timing instead.
- **The `serve_bonus` formation term** — looked good in-sample (F1 0.92 vs 0.79),
  reversed under held-out scoring (0.85 vs 0.71). Defaults to off. The knob exists;
  do not turn it on because it looks better on the training points.
- **Inter-shot adjacency features** — anything that is a difference between two
  *detected* shots (`destination = the next shot's position`, `interval = gap to the
  next detected shot`). Rejected on the contamination below, not on how it scores.
  `fit/volleys.py`'s `interval_norm` is the existing instance, and is why that
  module is not the base for shot-type work.

## Gotchas that bite

- **Coordinate frames.** Detections are in ROI pixels at `PLAYER_WIDTH = 480`
  analysis scale, after `_roi_box()` crops. `court.foot_fraction()` undoes the crop
  and downscale. Get this wrong and the homography silently projects garbage —
  nothing raises.
- **Foot-point precision is the accuracy bottleneck.** One ROI pixel of vertical
  error costs 4–11 cm of court position, worst near the *front* wall (the camera
  sits behind the back wall). Anything touching `foot` deserves care.
- **`Detection.source` is a closed set.** `view/demo.py` does a bare dict lookup on
  it; a new value raises `KeyError` at render time.
- **Box heights are unreliable** and censored at both ends by the size filters.
  Don't build features on them.
- **`SCORE_BOX` is hard-coded to one broadcast.** Verify with `scoredigits` before
  trusting anything built on it — a wrong box mixes two games together and never
  reports an error.
- **A player who stops moving disappears.** Background subtraction only sees what
  moved. This explains `loitering()`, the `IDENTITY_MEMORY` averaging, and most
  coasted detections. It is the detector's central weakness.
- **Anything computed across two rows of `shots.csv` is contaminated.** Shot timing
  is precision 0.67 / recall 0.57 at tolerance 1.0 s, and a miss does not blank a
  row — it splices two non-adjacent shots into an apparent adjacency, giving a
  destination one shot too far downstream and a doubled interval, with nothing
  raised. The misses are also *label-correlated*: `MIN_PROMINENCE` and
  `MIN_SHOT_GAP` discard small brief excursions, which are volleys, counter-drops
  and tight front-court exchanges. And `find_shots`'s out-of-turn drop means one
  miss costs two **adjacent** shots. Estimated one row in four carries correct
  cross-row features. Build features from one instant plus the continuous tracks
  instead. See `HANDOFF_4_shot_type.md` S3.
- **Disputed number:** both-players-resolved at stride 4 appears in the record as
  both 65% and 78% — probably measured through different pipeline stages
  (`P.summarise()` counts raw per-frame slots; `cli.analyse_in_court` reports after
  smoothing and coasting). State which metric you mean.

## Documentation

- `docs/squashvision-guide.html` — the user-facing manual: setup, every command,
  troubleshooting, and how the pipeline works.
- `HANDOFF_1_shots.md`, `HANDOFF_2_winners_volleys.md` — **implemented**; kept as
  the record of what was built and why.
- `HANDOFF_3_segmentation.md` — **deferred**. Swapping background subtraction for
  instance segmentation, with a go/no-go gate. Read its S0 preconditions before
  starting any of it.
- `HANDOFF_4_shot_type.md` — **gated, not started**. Shot-type classification,
  and the feature contamination that blocks it. Its S0.4 precondition (a second
  shot-capable recording) is not currently met.
