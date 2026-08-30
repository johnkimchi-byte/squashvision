# squashvision — brief 4: shot-type classification

Label each detected shot with what kind of shot it was, so a match becomes a
searchable directory of clips. **Not deferred; T4 is a hard decision gate.**

The obstacle is not the classifier. It is that every feature anyone would
reach for is a *difference between two detected shots*, computed on top of a
detector that misses 43% of them. This brief removes that coupling first and
only then fits anything.

**This is now measured, not argued** — see S3. The same feature scores r =
−0.205 from hand-marked instants and −0.010 from detected ones, and tripling
the label set moved the volley classifier *backwards*.

## S0 · Preconditions — check before starting

1. Briefs 1 and 2 are complete. **Holds** — `rally/shots.py` and
   `fit/volleys.py` exist with held-out scores recorded.
2. `labels.json` exists with hand-marked shot instants. **Holds** — 174 marks
   (145 shot, 20 volley, 8 winner, 1 game) spanning t=2.93–587.34, in **20
   rally clusters** (shot-equivalents grouped at a >4 s gap; the count is
   stable for gaps 3–6 s). Eight clusters end in a `winner` mark.
   *Check before relying on the count:* brief 1 found 4 of 9 detected rallies
   were break artifacts, so verify these 20 against `scoredigits` rather than
   assuming 20 real points.
3. You accept that the deliverable taxonomy is the collapsed set in S4, not
   the full squash taxonomy. Drop-vs-kill and boast-vs-cross-drop are not
   recoverable here at any level of effort — see S2.
4. **A second recording usable at shot granularity.** *Wanted, no longer
   blocking.* At 61 marks across 5 rallies, hold-out-by-rally was meaningless
   and only a second match could gate T4. With 20 clusters it is a legitimate
   if weaker gate. A second match still buys the one thing rally-holdout
   cannot — generalisation across camera, calibration and encode — so treat it
   as the strongest available evidence at T4, not as a precondition for
   starting.

T1–T3 proceed regardless. T4 runs on rally-holdout when no second match
exists, and says which it used when reporting.

## S1 · Goal

A shot at time `t` gets a class. Classes are assigned from **where the
opponent got to**, which is a property of the continuous court tracks, not of
a second detection.

The current shape is:

```
tracks -> detect shot instants -> difference consecutive instants -> classify
```

Every failure in S3 is manufactured by that third step. This brief's shape is:

```
tracks -> shot instants (with explicit uncertainty) -> window features from ONE instant -> classify
```

**The core move: you do not need the next shot detected. You need where the
opponent got to, and that is already in the tracks.** The opponent's first
radial apex about the T after `t` is the moment they reached the ball. Reading
it locally requires no prominence floor, no minimum gap, no alternation parity
and no accept/reject decision.

## S2 · Settled — do not relitigate

Carried forward, still closed:

- **Ball tracking** — 0/14 labelled impacts resolved. Four pixels of ball,
  four pixels of compression noise. Reopening needs *new footage*, not new
  code: a higher-resolution or purpose-placed camera changes the premise, a
  better algorithm on this footage does not.
- **Audio** — arena crowd noise buries the impacts.
- **Silhouette height as a feature** — measured, noise exceeds signal at the
  reach timescale.
- **`serve_bonus`** — stays off. It is the standing example of a feature that
  looked good in-sample (F1 0.92 vs 0.79) and reversed held out (0.85 vs 0.71).

New, settled by this brief:

- **No inter-shot adjacency features.** Not `destination = next shot's
  position`, not `interval = gap to next detected shot`. Any feature requiring
  two detections is rejected on the S3 evidence regardless of how well it
  scores. `fit/volleys.py`'s `interval_norm` is the existing instance of this,
  and is why that module is scoped out of this brief rather than extended.
- **Not end-to-end.** A sequence model over raw tracks emitting events and
  types jointly is the right long-term shape and needs precisely the corpus
  that does not exist. Revisit when T7 has produced one.
- **Brief 3 is not a prerequisite.** It improves the tracks feeding all of
  this, but nothing here blocks on it and its own T5 can return no-go. See S3
  for the one measurement that would legitimately trigger it.
- **Keep `rally/shots.py`'s public output.** `Shot` and the CSV columns stay;
  T6 adds values to `source`, it does not change the shape.

## S3 · Facts — do not re-measure

**Shot timing, against 60 shot-equivalent hand labels at tolerance 1.0 s:**
precision **0.67**, recall **0.57** (34 matched, 17 false, 26 missed).

**Volley classification**, the existing instance of features-across-two-shots.
Two runs, same code, same video:

| labels | rallies | matched | held-out P/R | fitted P/R |
|---|---|---|---|---|
| 61 (brief 2) | 5 real | — | 0.10 / 0.22 | 1.00 / 0.14 |
| 174 (full video, t=0–596) | 18 detected | 102 | **0.03 / 0.17** | 0.00 / 0.00 |

**Tripling the labels made it worse.** Brief 2 recorded the cause as the
five-rally ceiling — "no further labelling within them can fix" it. Labelling
*outside* them, 5 → 18 rallies and 61 → 174 marks, moved held-out P/R
backwards. **Label quantity was never the blocker**, and that diagnosis is
withdrawn.

Also from that run: **only 102 of 173 hand-marked shots matched a detected
shot** within 1.0 s. 41% of the labelling effort never reaches the fit.

**Only 5 of the 9 rallies** `find_rallies` reported across t=300–590 in brief
1 were real points (indices 1, 2, 3, 7, 8); the rest were break artifacts. The
full-video run reports 18 — unverified against the board.

**Tracks:** both-players-resolved 73% / 77% at stride 1 / 2 (stride 4 disputed,
65% and 78% both in the record). ~20–27% of detections coasted. ~15% of rally
frames lost to unsplittable merges.

### The contamination, and why it is not noise

Features built on consecutive `shots.csv` rows are correct only if the row is
a true detection, its neighbour is a true detection, **and no real shot was
missed between them**. A miss does not blank a row — it splices two
non-adjacent shots into an apparent adjacency, giving a destination one shot
too far downstream and an interval roughly doubled. Nothing raises. The row is
well-formed and wrong.

**Estimated** clean-row fraction from the figures above — one row in three is a
false alarm; of the rest, ~43% have a miss breaking the adjacency; plus false
alarms landing between two true detections — **roughly one row in four**. This
is an estimate from recorded numbers, *not a measurement*. T1 measures it.

### Measured: the same feature, from two sources

`interval_norm` — time since the previous shot, normalised by that rally's
median — computed on identical labels and footage, differing only in where the
shot instants came from:

| Instants from | point-biserial r vs volley | mean | sd |
|---|---|---|---|
| Hand marks (`labels.json`) | **−0.205** | 1.052 | 0.328 |
| Detected (`shots.csv`) | **−0.010** | 1.396 | 1.900 |

The hand figure is r = −0.205, 95% CI **[−0.337, −0.114]** (cluster bootstrap
over rallies, 4000 reps, negative in 100% of them), stable at −0.246 to −0.205
across clustering gaps of 3–6 s. The detected figure is indistinguishable from
zero and lies outside that interval.

−0.205 puts `interval_norm` level with `depth_short` (−0.244) and
`depth_median` (−0.212). **It is not a weak feature. It is a feature of
ordinary strength that the detector deletes.**

The distribution shows the mechanism directly: a median-normalised interval
must centre near 1.0, and the hand series does (1.052, sd 0.328). The detected
series sits 33% high with six times the spread — missed predecessors doubling
intervals.

Note also `depth_short` and `depth_median` are collinear at r = 0.97, which
flips `depth_median`'s sign in the multivariate fit (−0.494 refit alone). So
the four features were really two, and one of the two was broken.

**Correction to an earlier claim in this brief.** This section previously
argued the misses are *label-correlated* — that `MIN_PROMINENCE = 0.4` m,
`MIN_SHOT_GAP = 0.4` s and the 0.3 s smoothing window preferentially discard
the small brief excursions that volleys and counter-drops produce. Measured,
the effect is not there at this sample size: volleys matched a detected shot
11/20 (55%), non-volleys 91/153 (59.5%). **The destruction above is fully
accounted for by uniform adjacency breakage** — ~41% of predecessors missing
is enough on its own. The mechanism remains plausible and worth measuring
per-type at T5, but do not treat it as established, and do not build on it.

### One-instant features are immune to missing shots, not to mistimed ones

Measured while rebuilding the volley feature set. `radial_v` (rate of change
of the striker's distance from the T) and `out_reach` (how far above its
recent minimum it sits) come from the continuous tracks, so no missing shot
can break them — the premise of S1. But they are *read at* the shot instant,
and a 1.0 s matching tolerance lets that instant sit well away from the strike:

| feature | hand instants | detected instants |
|---|---|---|
| `radial_v` | −0.391 | −0.239 |
| `out_reach` | −0.369 | −0.215 |

Roughly 40% of the signal, lost to mistiming alone, with no adjacency involved.

**So S1's move removes one failure mode and not the other.** Features from one
instant plus the tracks are safe from *adjacency breakage*; they stay sensitive
to *instant accuracy*. T5's timing work is therefore load-bearing for the
feature quality, not only for recall — and T2's `quality` flag should carry a
timing confidence beside its `solid` fraction.

### One miss costs two shots, and they are adjacent

`find_shots` walks pooled candidates and drops anything arriving out of turn:

```python
expected = server if server is not None else (pooled[0][1] if pooled else 0)
for t, player, _prominence in pooled:
    if player != expected:
        dropped += 1
        continue
    ...
    expected = 1 - expected
```

True sequence A, B, A, B with A's first peak missing: `expected` stays A, so
B's *genuine* shot arrives out of turn and is also dropped. Symmetrically, an
accepted false alarm flips parity and costs the next genuine shot. Either way
the two losses are **adjacent** — precisely the relation the features need.
Losses arrive in pairs, not as scattered singletons.

Derived by reading the loop, not yet measured. T1 confirms it.

This is a good design for its original purpose — the docstring says it "trades
recall for never fabricating an attribution," which is right when reporting
timing. It is the wrong shape of error when the next stage consumes
adjacencies.

### What would legitimately trigger brief 3

The window approach in T2 removes the adjacency dependence. It does not remove
*tracker*-induced bias: the ~15% unsplittable merges concentrate in tight
front-court exchanges, which may well be type-correlated — though note the
volley measurement above found no such effect for the detector, so treat this
as a hypothesis, not a known. If T5's type-conditional recall is non-uniform
**and the residual is attributable to merges**, that is evidence-driven grounds
to start brief 3. "The detector could be better" is not.

## S4 · The taxonomy — label rich, train collapsed

(origin, destination) determines direction and is degenerate for everything
differing by height, pace or wall path:

| Pair | Origin | Destination | Separable |
|---|---|---|---|
| straight drive vs cross-court drive | same | differ by up to ~3.2 m in x | **yes** |
| deep vs short | same | differ by metres in y | **yes** |
| boast vs cross-court drop | same | same | no — side-wall path |
| lob vs drive | same | same | no — height |
| drop vs kill | same | same | no — pace |

Court is 6.40 m wide, 9.75 m long, short line at 4.26 m (`CourtSpec`).

**Train on the collapsed set**: `{straight, cross} x {front, back}`, four
classes. Straight/cross from the sign of `(x - spec.half)` at origin versus
destination; front/back from destination `y` against `spec.short_line`.

**Label the full set anyway** — drive, cross, drop, boast, lob, kill, nick,
alongside the existing volley flag. Relabelling is the expensive operation and
the collapse is a property of today's sensors, not of squash. This follows the
precedent already in `Mark`'s docstring, where volley and winner were kept
distinct from shot "so the training data for brief 2 needs no relabelling."

Collapse at fit time, in one function, so the mapping stays visible and
revisable.

## S5 · Environment

- Windows. Git Bash and PowerShell both available; each needs its own syntax.
- `./.venv/Scripts/python.exe -m squashvision` — bare `python` opens the
  Microsoft Store and fails.
- numpy and opencv only. Check whether ~30 lines of numpy would do before
  adding a package; `fit/volleys.py` hand-rolls logistic regression for this
  reason.
- `PYTHONIOENCODING=utf-8` for anything touching the Connecticut video.
- `--stride 2` by default. Stride 1 only where temporal resolution is the
  actual constraint — which for `flight_time` it may be. Measure, do not
  assume.
- Working video: `squashvisiontest.mp4`, 28.365 fps, play t=300–400 and
  t=505–590, break t=405–505. Court `profiles/bates_court.json`, 11.3 cm
  residual. No roster or profile needed; module defaults are correct.

## S6 · API — complete for these tasks

```python
cli.analyse_in_court(args, court) -> Analysis
Analysis(samples, raw, tracks, solid, dropped, dt)
    tracks[i][k]   (x, y) court metres, smoothed, or None
    solid[i][k]    True if this came from a measurement, not the smoother coasting
    samples[k].time_s

court.spec.width / .length / .short_line / .half    6.40 / 9.75 / 4.26 / 3.20

rally.shots.Shot(time_s, player, position_m, rally_index, source)
    CSV: time_s,player,position_x_m,position_y_m,rally_index,source
rally.label.Mark(time_s, kind)
    kind in ("rally", "game", "shot", "volley", "winner")
    load() reads m.get("kind", "rally") -- extra JSON keys are already tolerated
```

`solid` is the provenance mechanism this brief needs; do not invent another.
Note `view/demo.py` does a bare dict lookup on `Detection.source`, so a new
value there raises `KeyError` at render time.

## S7 · Conventions — load-bearing

1. Every threshold carries the measurement that set it, including what was
   tried first and why it failed. No bare magic numbers. If unmeasured, the
   comment says so.
2. Never pass inferred data off as observed. This brief's central application
   of the rule: `Shot.source` currently has exactly one value, so the shot
   layer does not honour the convention `Detection.source` honours one layer
   down. T6 fixes that.
3. Anything fitted is scored on data it was not fitted on. Print both, label
   the optimistic one.
4. New commands register in `__main__.COMMANDS` in running order.
5. Module docstring explains the idea *and the failure it avoids*.
6. Reuse `cli.add_span_arguments` / `add_tracking_arguments` / `add_court_argument`.

## S8 · Tasks

### T0 — clip export, command `clips` · needs: —

Walk a `shots.csv` and write a clip per shot. `overlay.writer(path, fps, size)`
exists. Filename carries time, rally index and player.

Do this first, and it is not busywork: every measurement below otherwise
arrives as a summary statistic. Clips make the failures watchable, which is how
you tell a recovery-not-a-shot false alarm from tracker jitter.

DONE: a directory of clips for t=300–400, and one sentence on what the 17 known
false alarms actually are.

### T1 — measure the contamination · needs: —

Against the hand labels, measure and report:

1. Fraction of `shots.csv` rows whose adjacency to the next row is clean —
   both true detections, no hand-labelled shot between them. Compare against
   the ~1-in-4 estimate in S3.
2. Whether one miss costs two adjacent shots. Count `dropped` out-of-turn
   candidates that match a hand label, per rally.
3. Distribution of hand-labelled inter-shot intervals — this sets T2's search
   window, per convention 1.

DONE: three numbers, and the S3 estimate either confirmed or corrected.

### T2 — `rally/features.py` · needs: T1

Feature vectors from **one** shot instant plus the tracks. For a shot at `t` by
player `p`, opponent `q = 1 - p`:

- `dest` — `tracks[q][k*]` where `k*` is the argmax of `dist(tracks[q][k], tee)`
  over `[t + MIN_FLIGHT, t + MAX_FLIGHT]`. Bounded argmax, **no prominence
  floor, no gap rule, no accept/reject**. Bounds from T1.3, commented with them.
- `lateral` — sign of `(dest.x - half)` versus `(origin.x - half)`; straight or
  cross.
- `dest_depth` — `dest.y` relative to `short_line`.
- `flight_time` — `t* - t`. The pace proxy, and it needs no second detection.
- `opponent_start` — `tracks[q]` at `t`; a cross-court to a player already
  forward is not the same event as one to a player at the back.
- `striker_recovery` — `p`'s court velocity just after `t`.
- `quality` — fraction of `solid[q][k]` true over `[t, t*]`, and whether the
  apex sample itself is solid. A destination read off coasted samples is
  inferred, and the row says so. **Also carry a timing confidence**: S3 shows
  one-instant features lose ~40% of their signal to a mistimed instant even
  though no adjacency is involved, so `solid` alone does not describe the row.

**The interface takes instants, not a CSV** — hand marks from `label.py` and
rows from `shots.csv` both go in the same way. This is what makes T4 possible,
and it keeps the train/deploy gap measurable permanently.

Invariant to test: for the 34 currently-matched shots, feature vectors are
identical whether instants come from `labels.json` or from `shots.csv`.

DONE: the module, the invariant passing, and the fraction of rows flagged
inferred by `quality`.

### T3 — shot type in `rally/label.py` · needs: —

Add a type axis to `Mark`. Full taxonomy per S4, one keystroke per shot. Keep
`kind` as it is and add a separate optional field, so existing `labels.json`
loads unchanged — `load()` already tolerates extra keys.

Label the full t=300–400 and t=505–590 spans, and the second match from S0.4.

DONE: label count per type, per match; and the count in the four collapsed
classes, so T4 knows its base rate before fitting anything.

### T4 — DECISION GATE: does the concept work at all · needs: T2, T3

Fit the four collapsed classes on **hand-marked instants only**. `shots.py` is
not in this loop. Ridge multinomial logistic in numpy, held out **by match if a
second match exists, otherwise by rally** over S0.2's 20 clusters. State which.

Report the confusion matrix, not just accuracy — which classes collapse into
which is the actual finding, and it defines the taxonomy the sensors support.

- **Go** — beats base rate by a clear margin held out, on features whose
  `quality` flag is clean.
- **No-go** — the concept does not survive on clean features, and no detector
  work will rescue it. Record the confusion matrix and stop. This costs a few
  hundred labels rather than a few thousand, which is the entire point of
  ordering the brief this way.

Treating "go" as the default outcome here would repeat the `serve_bonus`
mistake with a larger budget.

DONE: confusion matrix, held-out numbers beside fitted ones, explicit go or
no-go.

### T5 — joint-evidence candidate scoring · needs: T4 = go

A shot is a two-body event: `p`'s radial apex coincides with the onset of `q`'s
movement. Score candidates on both, with a threshold far below the current
`MIN_PROMINENCE`, letting the joint term suppress the false alarms a low
threshold alone would admit.

This is aimed squarely at the S3 bias: a volley taken at the body produces
almost no excursion in the striker, but the receiver still has to respond.

**The gate is type-conditional recall, not overall recall.** Success is drops
and volleys being missed at roughly the rate drives are. Report recall per type,
before and after. A uniform 0.70 beats a lopsided 0.80.

DONE: per-type recall table before and after; a statement on whether the
residual non-uniformity is attributable to merges (see S3).

### T6 — sequence decode with explicit gaps · needs: T5

Replace the greedy alternating walk with a DP over scored candidates in a
rally: states are which player is due, transitions are accept-candidate or
hypothesise-a-miss at a cost measured from the hand labels.

One miss then inserts a gap marker instead of flipping parity for the rest of
the rally. `Shot.source` gains `measured` / `inferred` / `uncertain`, and
`features.py` refuses to compute across an inferred gap rather than computing a
wrong value silently. Update `view/demo.py`'s dict lookup in the same commit.

DONE: the clean-adjacency fraction from T1.1, re-measured. Side by side.

### T7 — `fit/shottype.py`, and the number that matters · needs: T6

Train on hand instants, evaluate on detected instants, over the same footage.
The difference between those two scores is the deploy gap, and it is the honest
headline for the whole tool. Register `shottype` in `__main__.COMMANDS` in
running order, after `shots`.

DONE: both scores, the gap, and a `note` field in the config saying which
figure to quote.

## S9 · Report back when done

1. T1's three numbers, and whether S3's estimate held.
2. T4's confusion matrix and the go/no-go.
3. T5's per-type recall table.
4. T7's deploy gap.
5. Whether S0.4's second match was obtained, and what it is.

## S10 · Do not

- Do not build any feature that requires two detections.
- Do not read T4's rally-holdout as evidence of cross-camera generalisation.
  Say which hold-out was used.
- Do not answer a poor score by labelling more. That was tried: 61 → 174 marks
  moved held-out P/R backwards (S3).
- Do not fit anything on `shots.csv` instants before T4 has passed on hand ones.
- Do not report overall recall as T5's gate; it hides the bias that motivated
  the brief.
- Do not extend `fit/volleys.py` to multi-class — it is built on the adjacency
  features this brief rejects.
- Do not start brief 3 on general grounds. T5's merge attribution is the
  trigger.
- Do not treat "go" as the default outcome at T4.
- Do not promise drop-vs-kill or boast-vs-cross-drop. They are not in the data.
