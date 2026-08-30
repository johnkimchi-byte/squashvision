# squashvision — brief 3 of 3: instance segmentation detector

**DEFERRED. Do not start until S0's preconditions are met.**

Replace background-subtraction detection with an instance-segmentation model,
preserving the `Detection` contract so the tracker and everything downstream is
untouched. This is an **experiment with a go/no-go gate at T5**, not a migration.

## S0 · Preconditions — check before starting

Do not begin until all four hold:

1. Briefs 1 and 2 are complete: `rally/shots.py` and `fit/volleys.py` exist and
   have held-out scores recorded.
2. Those scores are written down somewhere durable. They are the regression test.
3. Nobody is mid-flight on shot or volley work. This change invalidates
   detection-layer measurements while it is in progress.
4. You have a device that can run the model size chosen in S4.

If any fails, stop and say so.

## S1 · Goal

Background subtraction has one structural weakness that costs more than any
tuning can recover: **a player who stops moving disappears** — he becomes his own
background. What survives the subtraction is whatever twitched.

That single fact is the root of a surprising amount of the codebase: `loitering()`,
the `IDENTITY_MEMORY` averaging, ~20-27% coasted detections, most of the seeding
difficulty during knock-ups, and the ~15% of rally frames lost to unsplittable
merges.

A segmentation model sees a motionless player exactly as well as a sprinting one.

## S2 · Settled — do not relitigate

- **Instance segmentation, not bounding-box detection.** A plain detector is a
  *regression* here. `foot` is the silhouette's centroid-x paired with its lowest
  row — measured as markedly better than box-based (|dx| p90 0.11 m vs 0.33 m) —
  and `shirt` is the median Lab over the *masked* torso band. Both need a mask.
- **Not a nano model by default.** squashvision is offline batch, never live:
  `track`, `demo`, `birdseye`, `rallies` and `train` all process recorded files.
  A large model at 2-5 fps is ~20-50 min for a ten-minute match, which is a fine
  batch job. Model size and input resolution are **config, not constants** — see
  S4. Reserve nano for an actual live-on-Jetson deployment.
- **Keep the tracker.** `_Kalman`, `_Track`, `_assign`, `_seed`, `_revive`,
  `_reconcile`, `_separate` are ~290 lines of measured identity logic. They
  operate purely on `Detection` objects and must survive unchanged.
- **Keep the roster.** A model knows "person", not "Smith". Identity, shirt
  anchors and `IGNORE_ZONES` all stay.
- **Spectators get worse, not better.** Background subtraction ignores a
  motionless spectator; a detector finds every person in frame. Fencing becomes
  more load-bearing. Court-geometry filtering will not save you — spectators'
  feet are hidden behind the glass, so they map to ~8.1 m, *inside* the court.
- Never add an unmeasured magic number.
- Never report a fitted score without a held-out one beside it.

## S3 · Facts

**The seam** (`detect/players.py`, line numbers as of 2026-08-30):

```
572-697   _foreground, _components, _measure,          ~126 lines   REPLACE
          _valley_split, _guided_split
698-991   Tracker + _assign/_seed/_revive/             ~290 lines   KEEP
          _reconcile/_separate
```

`Tracker.step()` touches the detector in exactly two lines:

```python
mask = _foreground(gray, background)
boxes = _components(mask)
```

**`_measure()` survives almost verbatim.** It takes `(gray, lab, mask, box, source)`
and works on `mask[py:py+ph, px:px+pw]`. Hand it one instance's mask instead of a
background-subtracted one and it computes the same `cx/cy/area/shirt/brightness/foot`.
Normalise mask values to 0/255 first — it does `area = sub.sum() // 255`.

**Untouched:** all of `geometry/`, `rally/`, `score/`, `fit/train.py`,
`fit/volleys.py`. **Dies:** the background-model chunking in `analyse()`, the
detection constants, `fit/autotune.py` entirely, both `*_profile.json` files.

### Foot-point precision is the accuracy bottleneck

`foot` feeds the homography, and its **vertical** component is the most
error-sensitive quantity in the pipeline. Measured against `bates_court.json`:

```
vertical foot error -> court error, at mid-court x
  depth      1 px      3 px      4 px
  1.00 m     10.7      31.6      41.9  cm
  4.26 m      7.0      20.9      27.7  cm
  8.00 m      3.8      11.3      15.0  cm
```

One ROI pixel costs **4-11 cm**, and it is worst near the **front** wall — the
camera sits behind the back wall, so front-court ground is furthest away and each
pixel spans more of it. Front court is where drops and kills happen, which is
exactly what brief 2's winner work depends on.

**This is the regression risk.** YOLO-seg masks come off a prototype grid at
input/4 resolution, so at 640 input the boundary quantises to ~4 px — about 3 px
in ROI space, i.e. **21-42 cm of front-court error**, worse than the entire
homography residual of 11.3 cm. The swap could halve the merge rate and double
court-position error at the same time, and none of the obvious metrics
(both-players-resolved, merge rate, coasted fraction) can see it happen.

**Baseline to beat** (squashvisiontest.mp4, t=300-400, module defaults, no roster):
both-players-resolved 73% / 77% at stride 1 / 2. **The stride-4 figure is disputed
— 65% and 78% both appear in the record. T1 resolves this.**

## S4 · Accuracy levers — choose deliberately, record what you chose

| # | Lever | Gain | Cost |
|---|-------|------|------|
| 1 | Model size: use a large variant offline, not nano | Large | ~zero |
| 2 | Full-resolution masks (`retina_masks=True`) | Large — protects `foot` directly | Low |
| 3 | Crop to the ROI *before* inference | ~45% more pixels per player, same input budget | Low |
| 4 | Input resolution 1280 rather than 640 | Back-court players ~60 px → ~120 px | Low, offline |
| 5 | Hybrid mask (T3 fallback) | Medium-large | Medium |
| 6 | Pose keypoints for the torso band | Medium — attacks identity at the root | Medium |
| 7 | Height-vs-image-row prior for spectators | Medium | Low |
| 8 | hflip test-time augmentation | Small | Low, offline |
| 9 | Fine-tune on bootstrapped masks | Potentially large | High, risky |

Levers 1-4 are close to free in a batch context and should be the defaults.

**Hybrid mask (5).** Neural mask for what it is uniquely good at — separating
merged players, finding motionless ones — and background subtraction for what
*it* is good at: a crisp, native-resolution boundary. Within each instance box,
intersect the two; fall back to the neural mask alone where background
subtraction is empty, which is exactly the stationary-player case. `_measure()`
needs no change — it takes a mask and does not care how it was built.

**Pose keypoints (6).** `TORSO_TOP, TORSO_BOTTOM = 0.15, 0.55` are fixed
fractions of the box, so a lunging player whose box is wide and short gets his
shirt sampled from somewhere that is not his torso. Shoulder and hip keypoints
define the real one. Identity is this codebase's most-fought-over problem —
`SEED_MARGIN`, `ANCHOR_MAX_DISTANCE`, `ASSOCIATION_RATIO` and `IDENTITY_MEMORY`
all compensate for shirt-colour noise — so a cleaner sample attacks the cause.

> **Do not take x from ankle keypoints.** Feet-based lateral position was
> measured at three times worse than silhouette centroid-x, because feet swing
> with every stride. Keep centroid-x from the mask; use ankles only as a
> cross-check on ground *y*.

## S5 · Environment

- Interpreter `./.venv/Scripts/python.exe`. Bare `python` opens the Microsoft Store.
- Not a git repo. Git Bash and PowerShell both available.
- One new third-party dependency (onnxruntime, ultralytics, or TensorRT). This is
  the first beyond numpy/opencv — keep it to one.

## S6 · The contract that must be reproduced exactly

```
Detection(cx, cy, x, y, w, h, area, brightness, shirt, foot, source)

cx, cy      silhouette centroid, ROI px          (xs.mean(), ys.mean())
x,y,w,h     box in ROI px
area        mask pixel count
brightness  median gray over the torso band
shirt       median Lab over the masked torso band
              band = box rows TORSO_TOP(0.15) .. TORSO_BOTTOM(0.55)
foot        (px + xs.mean(), py + ys.max() + 1.0)   <- the critical one
source      "measured" | "split" | "predicted"
```

**Coordinate frame is the likeliest silent breakage.** Detections live in ROI
pixels at `PLAYER_WIDTH = 480` analysis scale, after `_roi_box()` crops.
`court.foot_fraction()` undoes the crop and the downscale. If the model runs at
another resolution, every detection must be mapped back into that 480-wide ROI
frame or the homography silently projects garbage. Nothing raises.

**`source` values are a closed set.** `view/demo.py` does a bare dict lookup:
`{"measured": "", "split": "…", "predicted": "…"}[det.source]`. A new string
raises `KeyError` at render time.

## S7 · Conventions — load-bearing

1. Every threshold constant carries the measurement that set it in a comment,
   including what was tried first and why it failed.
2. Never pass inferred data off as observed — carry provenance.
3. Anything fitted is also scored on data it was not fitted on. Print both; label
   the optimistic one.
4. Register commands in `__main__.COMMANDS` by *running* order.
5. Module docstring = the idea + the failure it avoids.

## S8 · Tasks

`T1 → T2 → T3 → T4 → [T5 decision] → T6`. T6 runs only on a "go".

### T1 — freeze the baseline · needs: —

Record the current detection layer's numbers as a machine-comparable JSON:
both-players-resolved at strides 1/2/4, merge rate, coasted fraction, split
fraction, plus the shot and volley held-out scores from briefs 1 and 2.

**Include a foot-point precision metric — without it T5 cannot see the main
regression risk.** Two are available:

- *Purpose-built:* over near-stationary spans (the `loitering()` logic already
  identifies these via `STILL_WANDER` over 3 s windows), measure the court-metre
  spread of the player's foot point. A player standing still should not move, so
  whatever spread you measure is foot-point noise in the units that matter.
- *Ready-made:* the **unsmoothed** peak speed already printed by
  `tracks.summarise_court` ("was %.1f m/s unsmoothed"). Jitter inflates it.

**Resolve the stride-4 discrepancy while here.** Two figures are in the record
(65% and 78%), probably measured through different pipeline stages —
`P.summarise()` counts raw per-frame slots, `cli.analyse_in_court` reports after
smoothing and coasting. State which metric the baseline uses and use only that one.

DONE: a committed `baseline.json` including foot-point precision, plus one
paragraph naming the metric.

### T2 — model spike, no integration · needs: T1

Get an instance-segmentation model running on single frames. Choose size and input
resolution per S4 — default to a large variant at 1280 with `retina_masks=True`
and the ROI cropped before inference, unless a live deployment forbids it.

Measure and report:
1. inference fps on the target device;
2. **mask boundary precision at the bottom edge** — how many ROI pixels does the
   silhouette's lowest row move between consecutive frames on a stationary player?
   Multiply by the S3 table to get court centimetres;
3. masks eyeballed on front court, back corners, and a merge.

**If (2) is worse than ~1 ROI px, stop and plan for the hybrid mask in T3.** If
masks are poor at the back of the court, where players are smallest, stop and
report — that would sink the change and it is cheap to discover now.

DONE: fps, a foot-edge precision figure in ROI px and cm, and visual confirmation
on all three cases.

### T3 — the adapter · needs: T2

New function producing `Detection` objects from model output, satisfying S6
exactly. Reuse `_measure()` — pass it a per-instance 0/255 mask.

**If T2's edge precision disappointed, build the hybrid mask here** (S4 lever 5)
rather than accepting the regression: intersect the neural instance mask with the
background-subtracted mask inside each box, falling back to the neural mask alone
where background subtraction is empty.

Handle the coordinate mapping explicitly and write a test that pushes a known
image point through detection → `court.foot_fraction` → `court.to_court` and
checks it lands where it should. This is the one thing that fails silently.

DONE: adapter emits Detections whose `foot` values project to sane court metres on
a hand-checked frame.

### T4 — wire into Tracker · needs: T3

Change `Tracker.step()` to take detections (or a frame) instead of
`gray, lab, background`. Delete the background-model chunking from `analyse()`.
Keep every other Tracker method untouched — if you find yourself editing
`_assign`, `_seed` or `_reconcile`, stop: the contract is wrong, not the tracker.

DONE: `demo` and `track` run end to end on squashvisiontest.mp4.

### T5 — DECISION GATE: measure, then decide · needs: T4

Re-run T1's measurements against the new detector, same metric, same spans.
Report a side-by-side table.

**Gate on three things, not one:**

| Must improve | Must not regress | Watch |
|---|---|---|
| both-players-resolved, merge rate, coasted fraction | **foot-point precision** (T1's metric) | spectator false positives |
| | shot + volley held-out scores | |

A result that improves merges while degrading foot-point precision is a **no-go**,
not a trade — court position is what everything downstream is built on.

Then say which:
- **Go** — improvement on the left, no regression in the middle.
- **No-go** — revert. Legitimate outcome, and cheap: T1-T4 touched nothing
  downstream. Record which lever in S4 you would try next.

DONE: side-by-side table, an explicit go or no-go, and the reasoning.

### T6 — cleanup, only on a go · needs: T5 = go

Delete `fit/autotune.py` and its `__main__.COMMANDS` entry. Mark both
`*_profile.json` files obsolete. Update `--profile` handling and every doc that
quotes a detection-layer number: the manual, briefs 1 and 2, and the constants'
own comments. Record the chosen model, input resolution and mask settings as
constants carrying their measurements, per S7.1.

DONE: no command references a dead profile; no doc quotes a superseded number.

## S9 · Do not

- Do not start before S0's preconditions hold.
- Do not use a bounding-box-only detector.
- Do not default to a nano model — this is a batch pipeline (S2).
- Do not skip the foot-point metric in T1; T5 is blind without it.
- Do not take lateral position from ankle keypoints (S4).
- Do not edit the Tracker's association or identity logic.
- Do not drop the roster or the ignore zones.
- Do not treat "go" as the default outcome.
