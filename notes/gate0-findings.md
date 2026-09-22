# Gate 0 findings

Status: **all 12 participants**, run on torch against `/scratch/zz5070/data/way-eeg-gal`.
Reproduce with

    PYTHONPATH=src NEUROMOTION_DATA=/scratch/zz5070/data/way-eeg-gal \
      python scripts/gate0.py --json runs/gate0_all12.json

Headline: the FASTRAK worry we flagged in the literature review does not materialise in the
data; the clock confound we flagged does; the reach is shorter than either document assumes;
and a fifth measurement we had not planned turned up a fourth route from EEG to the target that
bypasses motor cortex entirely, which none of our planned controls would catch.

---

## 0.1 Trial inventory — confirmed, 294 per participant

Every one of the twelve `P*_AllLifts` files has **294 rows**, not 328, in 9 series of
[34, 34, 28, 34, 28, 34, 34, 34, 34] or a permutation of it — seven series of 34 and two of 28.
One 34-lift weight series is missing from each, exactly as expected from the withheld
competition series. `HS_P{p}_ST.mat` is present for all twelve (EEG only) and
`HS_P{p}_S1..S9` carry all modalities.

Total: **3,528 usable multimodal trials**, matching the literature review exactly, and
2,352–2,646 training pairs after holding out 3–4 subjects. The proposal's figure of
2,600–2,900 is wrong and should be corrected.

## 0.2 Kinematic temporal resolution — the 30 Hz worry does not hold up

We expected the 500 Hz kinematic stream to be an artefact of a FASTRAK running at
120 Hz ÷ 4 sensors ≈ 30 Hz. Three independent tests say otherwise.

| test | what upsampling from ~30 Hz would look like | observed, all 12 |
|---|---|---|
| second difference zero | ~94% (piecewise linear between knots 16–17 samples apart) | **6.1–9.2%** |
| exact repeats of the 3D sample | ~94% with a 16-sample hold | **10.3–18.5%**, median spacing 1 sample |
| PSD | cliff at the 15 Hz Nyquist of a 30 Hz source | smooth roll-off, **no cliff**; noise floor only from 55–78 Hz |

So the stream is not zero-order held and not linearly interpolated. The apparent repeats are
amplitude quantisation, not temporal hold: the step sizes are 0.01 (x), 0.002 (y), 0.001 (z)
file units, and x takes only 1,372 distinct values over a 239 s series.

The binding limit is the movement itself, which is very slow:

    99%    of wrist-position power below 0.6-1.0 Hz   (all 12)
    99.99% of wrist-position power below 1.8-2.6 Hz   (all 12)

The pattern is uniform: no participant comes close to the signature of an upsampled stream.

**Consequence.** Timing metrics are not bounded at ~33 ms by the acquisition hardware, as the
literature review cautioned they might be. They are bounded by a ~3 Hz signal bandwidth and by
amplitude quantisation. That is a better position than we assumed, and the caution in the
review should be replaced with this measurement. It also means a path/timing decomposition has
less timing detail to recover than a 500 Hz label rate suggests — worth keeping in view when we
choose the number of progress-map parameters.

## 0.3 Dur_Reach — the reach is shorter than the proposal assumes

Pooled over all 3,528 trials:

    median 0.998 s   IQR [0.782, 1.186]   5–95% [0.608, 1.459]   range [0.400, 2.230]

The proposal and the literature review both describe a "1–2 s" reach. The median is almost
exactly 1 s, but the spread is much narrower and sits lower than "1–2 s" implies: **95% of
reaches finish inside 1.46 s**, and the 5th percentile is 0.61 s. Two things follow.

1. The generation horizon should be set from this distribution: **1.5 s covers 95% of reaches**,
   and quoting "1–2 s" overstates the upper half.
2. The argument for whole-segment generation was that over 300–500 ms the displacement is
   nearly a line segment and path/timing parameters become collinear. A median of 1.0 s leaves
   more margin than the two-participant sample suggested, but the 5th percentile at 0.61 s means
   a tail of trials sits close to that regime. Gate 2's collinearity test should be run on the
   short tail, not just on the median trial.

## 0.4 Within-series drift — the clock confound is real, and uneven

Two measurements.

**Per-trial features against recording time.** Spearman ρ of each feature on `StartTime`,
averaged within series and then over all 12 participants:

| feature | mean ρ | mean abs ρ |
|---|---|---|
| `Dur_Reach` | **−0.181** | 0.277 |
| `GF_Max` | −0.031 | 0.249 |
| `LF_Max` | +0.010 | 0.182 |
| `GripAparture_Max` | −0.014 | 0.250 |

The two-participant sample showed mean ρ near zero for everything. With all twelve, `Dur_Reach`
has a **consistent negative drift**: reaches get faster as a series progresses. That is a
directional, subject-general effect, not noise, and it is exactly the kind of gradient that puts
band structure into a trial-similarity matrix. Mean |ρ| of 0.18–0.28 against a null expectation
of about 0.14 for ~30 trials confirms the other features carry series-specific structure too,
just without a consistent sign.

**Trajectory similarity against time gap.** The direct test of the Toeplitz concern: build the
pairwise distance matrix between reach trajectories within a series and regress distance on the
absolute difference in recording time. Run over all 12 participants on series 1, 5 and 9.

Over 36 participant-series (participants 1–12, series 1, 5 and 9):

    mean rho +0.201   median +0.158   range [+0.026, +0.601]
    36 of 36 positive,  23 of 36 significant at p < 0.01

The strongest cases, with the same-object-condition value alongside:

| | ρ(gap, distance) | p | same object condition |
|---|---|---|---|
| P3 S1 | **+0.601** | 1.9e-56 | +0.590 |
| P6 S5 | **+0.513** | 8.2e-27 | +0.438 |
| P1 S9 | **+0.443** | 2.3e-28 | +0.435 |
| P6 S9 | **+0.435** | 2.8e-27 | +0.453 |

The two-participant sample made this look uneven — strong in P1's early series, absent in P2.
With all twelve it is **universal**: every single participant-series has a positive sign, and
two thirds reach p < 0.01. The same-condition column tracks the overall value closely, so the
gradient is time, not the object.

Positive ρ means trials recorded closer together in time have more similar wrist trajectories.
The effect is not the object condition: weight and surface change unpredictably between trials,
and restricting to pairs sharing both gives an equal or larger ρ.

**Consequence.** A uniform permutation null, whose hypothesis is "no association at all", is
defeated by this gradient everywhere, not just in some series. Hypothesis (6) is therefore not a
precaution but a requirement: run the timestamp-only baseline, report identification error
against the true inter-trial gap, and compute the primary cross-conditioning matrix on
across-series pairs only. Given that `Dur_Reach` also drifts consistently downward within a
series (mean ρ −0.181), part of this is a real, directional adaptation effect rather than
measurement drift, which makes it harder to filter out and easier to mistake for signal.

## 0.5 The pre-movement window crosses the cue — a fourth route to the target

Not on the original list; it surfaced while designing the windowing and is the most consequential
thing here.

The proposal takes **~1 s of EEG ending at movement onset**. Reaction time, pooled over all
3,528 trials, is:

    median 0.414 s   IQR [0.334, 0.577]   5–95% [0.276, 1.116]

So the window reaches back past the LED in nearly every trial:

| window | trials where it stays after the cue |
|---|---|
| 0.50 s | 35% |
| 0.75 s | 10% |
| **1.00 s** | **6%** |
| 1.50 s | 3% |

At the proposed 1 s, **94% of windows contain the cue-evoked response**. Because the window is
aligned to movement onset, the cue sits at position (window length − reaction time), so the
latency of that response *is* the reaction time.

That only matters if reaction time predicts the kinematics. In **11 of 12 participants** it does
(p < 0.01; only P5 shows nothing):

| relationship | participants significant | range of ρ | sign |
|---|---|---|---|
| reaction time → **peak reach speed** | **9 of 12** | −0.175 to **−0.624** | **always negative** |
| reaction time → `Dur_Reach` | 8 of 12 | −0.215 to +0.393 | **flips** |
| reaction time → `GF_Max` | 5 of 12 | −0.237 to +0.343 | mostly positive |

**The peak-speed relationship is the one that matters.** It appears in three quarters of
participants and its sign is the same in every one of them: a slower reaction is followed by a
slower reach. That is a subject-general regularity, so a model that learns it **transfers across
subjects**.

That last point connects directly to the anomaly we built the literature review around. Jain and
Kumar report leave-one-subject-out performance within 0.03–0.15 PCC of within-subject
performance, which we called unexpected for a signal as subject-idiosyncratic as EEG. Here is a
mechanism that would produce exactly that: the cue-evoked response is large, stereotyped and
essentially subject-general; its latency gives the reaction time; and reaction time predicts peak
reach speed with a consistent sign across participants. A decoder exploiting that chain would
show almost no drop under LOSO, because nothing in the chain is subject-specific. We are not
claiming this is what those models do. We are saying the route is open, it is measured, and it
predicts the specific anomaly we flagged.

Note also that the `Dur_Reach` relationship **flips sign between participants**, so a
cross-subject model would be pushed toward the transferable peak-speed component and away from
the duration component — a testable prediction.

**None of our five planned controls catches it.** The template ignores its input; shuffled
pairing destroys the association during training and returns chance rather than a warning; the
EMG control is unaffected; and the frontal-channel ablation is aimed at Fp1/Fp2/F7/F8 while a
visual evoked response lives over O1/O2/Oz/PO9/PO10. So we need:

* a **reaction-time-only baseline**, predicting the trajectory from reaction time alone — the
  same logic as the timestamp-only baseline, and the direct test of this route;
* a **cue-aligned** variant of the EEG window as well as the onset-aligned one, so the two can
  be compared;
* an **occipital ablation** alongside the frontal one, since that is where the cue response sits;
* or a window short enough to stay inside the reaction time, which at 0.5 s holds for only 35%
  of trials and at 0.25 s costs most of the pre-movement signal we wanted.

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
