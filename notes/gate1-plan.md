# Gate 1 — plan of record

Gate 1 asks one question: **does EEG on WAY-EEG-GAL carry trajectory information above an
empirically estimated null, under the exact split we intend to use?**

Everything here is ordered so that the cheap things that could invalidate later work happen
first. Nothing in steps 0–1 requires a model.

## Step 0 — measurements that precede modelling

These are preconditions. Each one can change the design, so none of the modelling starts
until all four have an answer.

| # | Question | Why it matters | Method |
|---|---|---|---|
| 0.1 | Is `P*_AllLifts` 328 rows or 294? How many trials actually carry kinematics? | The literature review claims 3,528 usable multimodal trials, not the advertised 3,936. If that is wrong, the training-set size in the proposal is wrong. | Count rows per participant; cross-check against the series files present. |
| 0.2 | What is the **effective** sampling rate of the FASTRAK kinematics? | Stored at a nominal 500 Hz, but the hardware updates at 120 Hz divided by active sensors (≈30 Hz with four). If the stream is zero-order held, every timing metric we plan to report is bounded by ~33 ms, and the timing branch of the path/timing decomposition is far less informative than assumed. | Fraction of exactly-repeated consecutive samples; run-length distribution; power spectrum of wrist position and its numerical derivative. |
| 0.3 | What is the distribution of `Dur_Reach`? | The proposal assumes a 1–2 s reach. Untested. Sets the generation horizon and the spline knot count. | Histogram, median, IQR, per-participant spread. |
| 0.4 | How much do wrist trajectories drift **within** a series? | Li et al. (2021): if kinematic similarity is a smooth function of recording time and EEG is autocorrelated on the same scale, a trial-identification measure gets diagonal structure from the clock alone. This is the failure mode none of our five planned controls catches. | Regress trajectory features (endpoint, peak speed, duration) on trial index within series; report the slope and the autocorrelation length. |

Deliverable: `notes/gate0-findings.md` with numbers, plus the figures behind 0.2 and 0.4.

## Step 1 — splits and windowing

Built before any baseline so every model sees identical data.

- Split **by subject**, then by series, then by trial, **before** any windowing.
- Hold out 3–4 subjects. LOSO for the cross-subject numbers.
- Normalisation statistics from training data only.
- Causal filtering only; any zero-phase filter is used solely to reproduce a prior protocol and
  is labelled as such.
- Record, for every trial, its absolute recording time. Needed for 0.4 and for the
  gap-matched null in step 3.

## Step 2 — baselines, in the order they answer a hypothesis

1. **Phase-conditioned template** (ignores its input). Tests H1: does a model with no EEG reach
   published PCC?
2. **Timestamp-only** (predicts from trial index). New, from Li et al. If this scores well, any
   identification measure we compute is suspect.
3. **Shuffled pairing**, retrained 50–100 times. Gives the empirical null for both metrics.
   Tests H2.
4. **EMG-conditioned** positive control. If EMG cannot beat the template, the pipeline is broken
   and no EEG result means anything.
5. **Lightweight EEG regressor.** The actual question.
6. **M3T-Attention reproduction.** Tests H4 (LOSO vs within-subject).

## Step 3 — scoring

Every model scored under **both** metrics on the same predictions:

- mean 3D displacement error (mm) — primary
- per-axis PCC — for comparison with the literature only

H3 is confirmed if the rank order differs between them.

Frontal-channel ablation (Fp1, Fp2, F7, F8) on the EEG regressor, reported against the template
level rather than against zero.

## Pass criterion, fixed in advance

The EEG model's mean displacement error on held-out subjects falls below the 5th percentile of
the shuffle distribution.

## Where things live

- Code: this repo, `src/neuromotion/` and `scripts/`.
- Data: `~/NeuroMotion-data/way-eeg-gal/P1..P12/` on the Mac mini (outside the repo).
  ~10.3 GB compressed, ~15 GB expanded. Mirror to `/scratch/zz5070/` on torch when we need GPUs.
- Gate 1 is CPU-light and runs locally. Torch is not needed until the generative models.
