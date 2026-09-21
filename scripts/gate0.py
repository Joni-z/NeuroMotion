#!/usr/bin/env python3
"""Gate 0 -- the four measurements that precede any modelling.

0.1  trial inventory: is AllLifts 294 rows, and does it match the series files?
0.2  effective temporal resolution of the FASTRAK kinematics
0.3  distribution of Dur_Reach (the proposal assumes a 1--2 s reach)
0.4  within-series drift, the confound Li et al. (2021) warn about

Run:  PYTHONPATH=src python3 scripts/gate0.py [--root DIR] [--out notes/gate0-findings.md]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import signal, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuromotion.io import (  # noqa: E402
    WINDOW_PRE_LED_S,
    absolute_event_time,
    alllifts_column,
    load_alllifts,
    load_series,
    participants,
    series_paths,
)

def _default_root() -> Path:
    """Where the extracted archives live.

    ``NEUROMOTION_DATA`` first, so the same command works on torch
    (/scratch/$USER/data/way-eeg-gal) and on a local sample.
    """
    env = os.environ.get("NEUROMOTION_DATA")
    if env:
        return Path(env)
    scratch = Path(f"/scratch/{os.environ.get('USER', '')}/data/way-eeg-gal")
    if scratch.exists():
        return scratch
    return Path.home() / "NeuroMotion-data" / "way-eeg-gal"


# ----------------------------------------------------------------- 0.1

def trial_inventory(root: Path, ps: list[int]) -> dict:
    rows = []
    for p in ps:
        table, cols = load_alllifts(root / f"P{p}_AllLifts.mat")
        run = alllifts_column(table, cols, "Run").astype(int)
        per_run = [int((run == r).sum()) for r in sorted(set(run))]
        released = series_paths(root, p)
        st = (root / f"HS_P{p}_ST.mat").exists()
        rows.append(dict(participant=p, n_trials=int(table.shape[0]),
                         n_series_in_table=len(per_run), trials_per_series=per_run,
                         n_hs_released=len(released), has_ST=st))
    return dict(rows=rows,
                total_trials=sum(r["n_trials"] for r in rows),
                n_participants=len(rows))


# ----------------------------------------------------------------- 0.2

def kinematics_resolution(root: Path, ps: list[int], per_participant_series: int = 1) -> dict:
    """Is the 500 Hz kinematic stream real, or upsampled from the ~30 Hz FASTRAK?

    Three independent tests:
      * cumulative power -- where does the movement actually live?
      * piecewise-linear knots -- linear interpolation leaves a zero second
        difference between knots spaced fs/source_rate apart
      * zero-order hold -- sample-and-hold leaves exact repeats at the same spacing
    """
    out = []
    for p in ps:
        path = root / f"HS_P{p}_S{per_participant_series}.mat"
        if not path.exists():
            continue
        s = load_series(path)
        w = s.position("wrist")
        fs = s.fs_kin

        f, P = signal.welch(w - w.mean(0), fs=fs, nperseg=4096, axis=0)
        Pm = P.sum(1)
        cum = np.cumsum(Pm) / Pm.sum()
        band = {f"p{q}": float(f[np.searchsorted(cum, q / 100)])
                for q in (95, 99, 99.9, 99.99)}

        # noise floor: first frequency where the PSD is within 3 dB of its median above 50 Hz
        hi = Pm[f > 50]
        floor = float(np.median(hi))
        above = np.where(Pm <= floor * 2)[0]
        floor_onset = float(f[above[0]]) if len(above) else float("nan")

        d2 = np.abs(np.diff(w, n=2, axis=0)).sum(1)
        pw_linear_frac = float((d2 <= 1e-12).mean())

        same = np.all(np.diff(w, axis=0) == 0, axis=1)
        zoh_frac = float(same.mean())
        gaps = np.diff(np.where(~same)[0])
        zoh_spacing = float(np.median(gaps)) if len(gaps) else float("nan")

        steps = {}
        for ax, nm in zip(range(3), "xyz"):
            dv = np.abs(np.diff(w[:, ax]))
            dv = dv[dv > 0]
            steps[nm] = dict(min_step=float(dv.min()) if len(dv) else float("nan"),
                             n_unique=int(len(np.unique(w[:, ax]))))

        out.append(dict(participant=p, series=per_participant_series,
                        fs_nominal=fs, power_below_hz=band,
                        noise_floor_onset_hz=floor_onset,
                        frac_second_diff_zero=pw_linear_frac,
                        frac_exact_repeat=zoh_frac,
                        median_update_spacing_samples=zoh_spacing,
                        quantisation=steps))
    return dict(per_participant=out)


# ----------------------------------------------------------------- 0.3

def reach_duration(root: Path, ps: list[int]) -> dict:
    allv, per = [], []
    for p in ps:
        table, cols = load_alllifts(root / f"P{p}_AllLifts.mat")
        v = alllifts_column(table, cols, "Dur_Reach")
        v = v[np.isfinite(v) & (v > 0)]
        allv.append(v)
        per.append(dict(participant=p, n=int(v.size), median=float(np.median(v)),
                        p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95))))
    v = np.concatenate(allv)
    return dict(per_participant=per,
                pooled=dict(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)),
                            p1=float(np.percentile(v, 1)), p5=float(np.percentile(v, 5)),
                            p25=float(np.percentile(v, 25)), p75=float(np.percentile(v, 75)),
                            p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)),
                            min=float(v.min()), max=float(v.max())))


# ----------------------------------------------------------------- 0.4

def within_series_drift(root: Path, ps: list[int]) -> dict:
    """Do trajectory features track recording time within a series?

    If they do, a trial-identification measure can be diagonally dominant from
    the clock alone, which is the false positive Li et al. (2021) describe.
    """
    feats = ("Dur_Reach", "GF_Max", "LF_Max", "GripAparture_Max")
    rows = []
    for p in ps:
        table, cols = load_alllifts(root / f"P{p}_AllLifts.mat")
        run = alllifts_column(table, cols, "Run").astype(int)
        start = alllifts_column(table, cols, "StartTime")
        for feat in feats:
            y = alllifts_column(table, cols, feat)
            rs = []
            for r in sorted(set(run)):
                m = (run == r) & np.isfinite(y) & np.isfinite(start)
                if m.sum() < 8:
                    continue
                rho, _ = stats.spearmanr(start[m], y[m])
                if np.isfinite(rho):
                    rs.append(float(rho))
            if rs:
                rows.append(dict(participant=p, feature=feat, n_series=len(rs),
                                 mean_rho=float(np.mean(rs)),
                                 mean_abs_rho=float(np.mean(np.abs(rs)))))
    by_feat = {}
    for feat in feats:
        vals = [r["mean_rho"] for r in rows if r["feature"] == feat]
        avals = [r["mean_abs_rho"] for r in rows if r["feature"] == feat]
        if vals:
            by_feat[feat] = dict(mean_rho=float(np.mean(vals)),
                                 mean_abs_rho=float(np.mean(avals)),
                                 n_participants=len(vals))
    return dict(per_participant=rows, by_feature=by_feat)


def trajectory_similarity_vs_gap(root: Path, p: int, series: int = 1) -> dict:
    """Does wrist-trajectory similarity decay with inter-trial time gap?

    This is the direct test of the Toeplitz concern: build the pairwise distance
    matrix between reach trajectories in one series and regress distance on the
    absolute difference in recording time.
    """
    path = root / f"HS_P{p}_S{series}.mat"
    if not path.exists():
        return {}
    s = load_series(path)
    table, cols = load_alllifts(root / f"P{p}_AllLifts.mat")
    run = alllifts_column(table, cols, "Run").astype(int)
    m = run == series
    onset_t = absolute_event_time(table, cols, "tHandStart")[m]
    weight = alllifts_column(table, cols, "CurW")[m]
    surface = alllifts_column(table, cols, "CurS")[m]

    fs = s.fs_kin
    w = s.position("wrist")
    horizon = int(round(1.2 * fs))
    trajs, times, wts, sfs = [], [], [], []
    for t0, wt, sf in zip(onset_t, weight, surface):
        if not np.isfinite(t0):
            continue
        onset = int(round(t0 * fs))
        if onset < 0 or onset + horizon >= len(w):
            continue
        seg = w[onset:onset + horizon]
        trajs.append(seg - seg[0])          # displacement relative to onset position
        times.append(t0)
        wts.append(wt)
        sfs.append(sf)
    if len(trajs) < 10:
        return {}
    X, T = np.stack(trajs), np.asarray(times)
    W, S = np.asarray(wts), np.asarray(sfs)
    k = len(X)
    D = np.stack([np.sqrt(((X - X[i]) ** 2).sum(-1)).mean(-1) for i in range(k)])
    iu = np.triu_indices(k, 1)
    gap, dist = np.abs(T[:, None] - T[None, :])[iu], D[iu]
    rho, pval = stats.spearmanr(gap, dist)

    # Is the association just the object condition? Weight and surface change
    # unpredictably between trials, so they should not track the clock -- but check.
    same_cond = ((W[:, None] == W[None, :]) & (S[:, None] == S[None, :]))[iu]
    rho_same, _ = stats.spearmanr(gap[same_cond], dist[same_cond]) if same_cond.sum() > 20 else (np.nan, np.nan)
    rho_diff, _ = stats.spearmanr(gap[~same_cond], dist[~same_cond]) if (~same_cond).sum() > 20 else (np.nan, np.nan)
    rho_w_time, _ = stats.spearmanr(T, W)

    # Neighbouring vs distant pairs, in the units the metric actually uses.
    near = dist[gap <= np.percentile(gap, 20)]
    far = dist[gap >= np.percentile(gap, 80)]

    return dict(participant=p, series=series, n_trials=k,
                spearman_gap_vs_distance=float(rho), p_value=float(pval),
                rho_within_same_condition=float(rho_same),
                rho_across_conditions=float(rho_diff),
                spearman_time_vs_weight=float(rho_w_time),
                median_distance_nearest20pct=float(np.median(near)),
                median_distance_farthest20pct=float(np.median(far)),
                note="positive rho means trials recorded closer in time have more similar "
                     "trajectories, which would give a trial-identification measure "
                     "diagonal structure from the clock alone")


# ----------------------------------------------------------------- report

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    if args.root is None:
        args.root = _default_root()

    ps = participants(args.root)
    if not ps:
        sys.exit(f"no participants found under {args.root}")
    print(f"# Gate 0 over {len(ps)} participant(s): {ps}\n", flush=True)

    res = {"participants": ps}
    res["inventory"] = trial_inventory(args.root, ps)
    print("## 0.1 trial inventory")
    inv = res["inventory"]
    for r in inv["rows"]:
        print(f"  P{r['participant']:<3d} {r['n_trials']} trials, "
              f"{r['n_series_in_table']} series {r['trials_per_series']}, "
              f"HS released={r['n_hs_released']}, ST present={r['has_ST']}")
    print(f"  total {inv['total_trials']} trials over {inv['n_participants']} participants\n")

    res["kinematics"] = kinematics_resolution(args.root, ps)
    print("## 0.2 kinematic temporal resolution")
    for r in res["kinematics"]["per_participant"]:
        b = r["power_below_hz"]
        print(f"  P{r['participant']:<3d} nominal {r['fs_nominal']} Hz | "
              f"99% power < {b['p99']:.1f} Hz, 99.99% < {b['p99.99']:.1f} Hz | "
              f"noise floor from {r['noise_floor_onset_hz']:.0f} Hz | "
              f"2nd-diff zero {r['frac_second_diff_zero']*100:.1f}% | "
              f"exact repeats {r['frac_exact_repeat']*100:.1f}%")
    print()

    res["reach_duration"] = reach_duration(args.root, ps)
    d = res["reach_duration"]["pooled"]
    print("## 0.3 Dur_Reach")
    print(f"  n={d['n']}  median={d['median']:.3f}s  IQR=[{d['p25']:.3f}, {d['p75']:.3f}]  "
          f"5-95%=[{d['p5']:.3f}, {d['p95']:.3f}]  range=[{d['min']:.3f}, {d['max']:.3f}]\n")

    res["drift"] = within_series_drift(args.root, ps)
    print("## 0.4 within-series drift (Spearman rho of feature vs recording time)")
    for feat, v in res["drift"]["by_feature"].items():
        print(f"  {feat:<20s} mean rho={v['mean_rho']:+.3f}  mean |rho|={v['mean_abs_rho']:.3f}")
    sim = trajectory_similarity_vs_gap(args.root, ps[0])
    res["similarity_vs_gap"] = sim
    if sim:
        print(f"  trajectory distance vs time gap (P{sim['participant']} S{sim['series']}, "
              f"{sim['n_trials']} trials): rho={sim['spearman_gap_vs_distance']:+.3f} "
              f"(p={sim['p_value']:.2g})")
    print()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
