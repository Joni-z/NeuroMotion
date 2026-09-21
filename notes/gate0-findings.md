# Gate 0 findings

Status: **interim, P1–P2 only** (download of the remaining ten participants is running).
Reproduce with `PYTHONPATH=src python3 scripts/gate0.py --json runs/gate0.json`.

Headline: the FASTRAK worry we flagged in the literature review does not materialise in the
data; the clock confound we flagged does; the reach is shorter than either document assumes;
and a fifth measurement we had not planned turned up a fourth route from EEG to the target that
bypasses motor cortex entirely, which none of our planned controls would catch.

---

## 0.1 Trial inventory — confirmed, 294 per participant

`P1_AllLifts` and `P2_AllLifts` both have **294 rows**, not 328. Trials per series:

    P1  [34, 34, 28, 34, 28, 34, 34, 34, 34]
    P2  [28, 34, 34, 34, 34, 28, 34, 34, 34]

Seven series of 34 and two of 28, so one 34-lift weight series is missing from each, exactly as
expected from the withheld competition series. `HS_P{p}_ST.mat` is present (EEG only) and
`HS_P{p}_S1..S9` carry all modalities.

This confirms the number in the literature review: **3,528 usable multimodal trials**, and
2,352–2,646 training pairs after holding out 3–4 subjects. The proposal's figure of
2,600–2,900 is still wrong and should be corrected.

## 0.2 Kinematic temporal resolution — the 30 Hz worry does not hold up

We expected the 500 Hz kinematic stream to be an artefact of a FASTRAK running at
120 Hz ÷ 4 sensors ≈ 30 Hz. Three independent tests say otherwise.

| test | what upsampling from ~30 Hz would look like | observed (P1 / P2) |
|---|---|---|
| second difference zero | ~94% (piecewise linear between knots 16–17 samples apart) | **7.1% / 9.2%** |
| exact repeats of the 3D sample | ~94% with a 16-sample hold | **11.9% / 18.5%**, median spacing 1 sample |
| PSD | cliff at the 15 Hz Nyquist of a 30 Hz source | smooth roll-off, **no cliff**; noise floor only from 62–67 Hz |

So the stream is not zero-order held and not linearly interpolated. The apparent repeats are
amplitude quantisation, not temporal hold: the step sizes are 0.01 (x), 0.002 (y), 0.001 (z)
file units, and x takes only 1,372 distinct values over a 239 s series.

The binding limit is the movement itself, which is very slow:

    99%    of wrist-position power below 1.0 Hz (P1) / 0.7 Hz (P2)
    99.99% of wrist-position power below 2.6 Hz (P1) / 1.8 Hz (P2)

**Consequence.** Timing metrics are not bounded at ~33 ms by the acquisition hardware, as the
literature review cautioned they might be. They are bounded by a ~3 Hz signal bandwidth and by
amplitude quantisation. That is a better position than we assumed, and the caution in the
review should be replaced with this measurement. It also means a path/timing decomposition has
less timing detail to recover than a 500 Hz label rate suggests — worth keeping in view when we
choose the number of progress-map parameters.

## 0.3 Dur_Reach — the reach is shorter than the proposal assumes

Pooled over 588 trials:

    median 0.849 s   IQR [0.708, 1.171]   5–95% [0.619, 1.552]   range [0.400, 2.098]

The proposal and the literature review both describe a "1–2 s" reach. The measured reach is
**median 0.85 s**, and 95% of trials finish inside 1.55 s. Two things follow.

1. The generation horizon should be set from this distribution, not from the assumed 1–2 s.
   Something like 1.6 s covers 95% of reaches.
2. The argument for whole-segment generation was that over 300–500 ms the displacement is
   nearly a line segment and path/timing parameters become collinear. With a median reach of
   0.85 s the margin above that degenerate regime is smaller than we claimed. Gate 2's
   collinearity test matters more, not less.

## 0.4 Within-series drift — the clock confound is real, and uneven

Two measurements.

**Per-trial features against recording time.** Spearman ρ of each feature on `StartTime`,
averaged within series: mean ρ is near zero for `Dur_Reach`, `GF_Max`, `LF_Max` and
`GripAparture_Max`, but mean |ρ| runs 0.16–0.31. With ~30 trials per series the null
expectation for |ρ| is about 0.14, so there is some structure, strongest in
`GripAparture_Max` (0.31).

**Trajectory similarity against time gap.** The direct test of the Toeplitz concern: build the
pairwise distance matrix between reach trajectories within a series and regress distance on the
absolute difference in recording time.

| | n | ρ(gap, distance) | p | same object condition | different condition |
|---|---|---|---|---|---|
| P1 S1 | 34 | **+0.248** | 2.7e-09 | +0.289 | +0.208 |
| P1 S2 | 34 | **+0.174** | 3.6e-05 | +0.117 | +0.195 |
| P1 S3 | 28 | +0.068 | 0.19 | −0.027 | +0.105 |
| P2 S1 | 28 | +0.028 | 0.59 | +0.111 | −0.010 |
| P2 S2 | 34 | +0.033 | 0.44 | +0.114 | −0.003 |
| P2 S3 | 34 | +0.016 | 0.71 | +0.115 | −0.023 |

Positive ρ means trials recorded closer together in time have more similar wrist trajectories.

Three observations:

* **It is real where it appears.** P1's first two series show it at p < 1e-4.
* **It is not the object condition.** Weight and surface change unpredictably between trials,
  and restricting to pairs sharing both gives an equal or larger ρ. The gradient is time, not
  condition.
* **It is uneven.** P2 shows almost nothing at the series level, though the same-condition ρ
  sits at a consistent +0.11 across all three of its series. The effect looks strongest early
  in a session, which is what one would expect from settling or adaptation.

**Consequence.** A uniform permutation null, whose hypothesis is "no association at all", would
be defeated by this gradient in at least some series. The confound cannot be assumed absent, so
hypothesis (6) stands: run the timestamp-only baseline, report identification error against the
true inter-trial gap, and compute the primary cross-conditioning matrix on across-series pairs
only. On this evidence we should also report it per participant rather than pooled, since P1
and P2 differ.

## 0.5 The pre-movement window crosses the cue — a fourth route to the target

Not on the original list; it surfaced while designing the windowing and is the most consequential
thing here.

The proposal takes **~1 s of EEG ending at movement onset**. Reaction time, pooled over 588
trials, is:

    median 0.477 s   IQR [0.318, 0.649]   5–95% [0.266, 1.349]   min −0.250 (anticipatory)

So the window reaches back past the LED in most trials:

| window | trials where it stays after the cue |
|---|---|
| 0.50 s | 48% |
| 0.75 s | 16% |
| **1.00 s** | **10%** |
| 1.50 s | 4% |

At the proposed 1 s, **90% of windows contain the cue-evoked response**. Because the window is
aligned to movement onset, the cue sits at position (window length − reaction time), so the
latency of that response *is* the reaction time.

That only matters if reaction time predicts the kinematics. It does:

| | Dur_Reach | peak reach speed | GF_Max |
|---|---|---|---|
| P1 | **−0.215** (p=2e-04) | +0.086 | −0.091 |
| P2 | **+0.231** (p=6e-05) | **−0.511** (p=6e-21) | **+0.268** (p=3e-06) |

P2's reaction time explains a large share of peak reach speed, and note that the sign of the
`Dur_Reach` relationship **flips between the two participants** — which is itself a reason to
expect a cross-subject model to behave oddly.

**So there is a complete path from EEG to the target that never touches motor cortex:** read the
cue-evoked response, recover its latency, recover the reaction time, exploit the
reaction-time-to-kinematics relationship. Both links are measured above. This does not show that
any published model takes that route, only that the route is open and nobody has closed it.

**None of our five planned controls catches it.** The template ignores its input; shuffled
pairing destroys the association during training and returns chance rather than a warning; the
EMG control is unaffected; and the frontal-channel ablation is aimed at Fp1/Fp2/F7/F8 while a
visual evoked response lives over O1/O2/Oz/PO9/PO10. So we need:

* a **reaction-time-only baseline**, predicting the trajectory from reaction time alone — the
  same logic as the timestamp-only baseline, and the direct test of this route;
* a **cue-aligned** variant of the EEG window as well as the onset-aligned one, so the two can
  be compared;
* or a window short enough to stay inside the reaction time, which at 0.5 s still only holds for
  48% of trials and at 0.25 s costs most of the pre-movement signal we wanted.

This is now hypothesis (7) in the plan.

---

## Incidental notes

**Event-time convention, verified rather than assumed.** `StartTime` is the absolute LED-on time
within a series; every `t*` column is relative to a window that opens 2 s before the LED (which
is why `LEDOn` is exactly 2.000 in every row). So

    absolute_time = StartTime + t_event - 2.0

Checked on P1 S1: mean wrist speed over the 300 ms after `StartTime + tHandStart − 2` is 28×
the 300 ms before it; without the −2 the ratio is 0.78, i.e. no movement. Encoded as
`neuromotion.io.absolute_event_time`.

**`tPeakVelHandReach` is a speed, not a time — resolved.** Values run 37–75 where a time
relative to trial start would be ~2.5–3.5 s. Computing peak reach speed ourselves from the
wrist trace gives a median of 75.4 against the column's 73.4, so the column is peak velocity
magnitude in cm/s and the `t` prefix is a misnomer. Peak-velocity *timing*, which we do want,
has to be computed from the trace.

**Units are centimetres — resolved.** Checked against three known scales on P1 S1:

    reach displacement          median 26.1   (a natural reach is 25-40 cm)
    peak reach speed            median 75.4   (a natural reach peaks at 50-100 cm/s)
    object vertical excursion   median  5.8   (data paper: "a few centimetres")

Millimetre errors are therefore the file value x 10 (`neuromotion.io.CM_TO_MM`). The earlier
±130 range was the whole 36-channel `kin` matrix, which mixes positions with angles in degrees;
the wrist itself moves over ~14 x 27 x 10 cm within a series.

## What changes as a result

1. Correct `proposal.tex` (2,600–2,900 → 2,352–2,646).
2. Replace the FASTRAK caution in `litreview.tex` with the measurement.
3. Set the generation horizon from the `Dur_Reach` distribution, not the assumed 1–2 s.
4. Keep hypothesis (6); add per-participant reporting of the gap analysis.
5. Units and `tPeakVelHandReach` are resolved; both are encoded in `neuromotion.io`.
6. **Add hypothesis (7) and a reaction-time-only baseline**, and decide between onset-aligned
   and cue-aligned EEG windows on evidence rather than by default.
