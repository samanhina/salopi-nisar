"""
Real-data test of the SALOPI phase hypothesis on NISAR quad-pol scenes of the Rann of Kutch.

Natural experiment, zones defined by GEOGRAPHY (not by the radar):
  SAND  - Thar desert dunes north of the Rann (non-saline, dry)        -> phase reference (ε'' ≈ 0)
  SALT  - Great Rann of Kutch salt flat (solonchak, salt crust)         -> test zone
  BET   - Khadir/Bela 'bets': rocky, vegetated islands inside the Rann -> control (not salt flat)
Only BARE pixels are used (low HV/HH, above noise), so vegetation does not disturb the phase.
Questions:
  1. Is the salt flat phase shifted relative to sand by the 4-9° the physics model predicts?
  2. Does the shift keep its size across June (dry), July (monsoon), October although moisture changes?
  3. How well does phase alone separate salt-flat pixels from sand pixels (accuracy, AUC)?
"""
import json, glob, os
import numpy as np, h5py
from pyproj import Transformer
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/claude/hackathon/theory"; os.makedirs(OUT, exist_ok=True)
ZONES = {  # lon_min, lat_min, lon_max, lat_max (WGS84)
    "SAND": [(69.80, 24.42, 70.40, 24.56), (69.35, 24.45, 69.70, 24.68)],
    "SALT": [(69.95, 23.98, 70.25, 24.18), (70.55, 23.98, 71.00, 24.14)],
    "BET":  [(70.55, 23.76, 70.90, 23.88)],
}
dates = ["20260618", "20260712", "20261016"]
tr = Transformer.from_crs(32642, 4326, always_xy=True)


def sm_at(date, lon, lat):
    p = glob.glob(f"data/indus/*SME2*{date}*.h5")[0]
    with h5py.File(p) as f:
        g = f["science/LSAR/SME2/grids"]
        sm = g["soilMoisture"][:]; xs = g["xCoordinates"][:]; ys = g["yCoordinates"][:]
    X, Y = Transformer.from_crs(4326, 6933, always_xy=True).transform(lon, lat)
    ci = np.rint((X - xs[0]) / (xs[1] - xs[0])).astype(int); ri = np.rint((Y - ys[0]) / (ys[1] - ys[0])).astype(int)
    ok = (ci >= 0) & (ci < len(xs)) & (ri >= 0) & (ri < len(ys))
    out = np.full(lon.shape, np.nan); out[ok] = sm[ri[ok], ci[ok]]
    out[(out < 0) | (out > 1)] = np.nan
    return out


res = {"zones": ZONES, "dates": {}}
samples = {}
for d in dates:
    D = np.load(f"overviews/indus_{d}.npz")
    XX, YY = np.meshgrid(D["x"], D["y"])
    lon, lat = tr.transform(XX, YY)
    hh, hv, vv, c = D["HHHH"], D["HVHV"], D["VVVV"], D["HHVV"]
    hhdb, hvdb = 10 * np.log10(hh), 10 * np.log10(hv)
    bare = (hvdb - hhdb < -5) & (hhdb > -25) & (10 * np.log10(vv) > -25)   # surface scattering, above noise   # surface scattering, above noise
    sm = sm_at(d, lon, lat)
    masks = {}
    for z, boxes in ZONES.items():
        m = np.zeros(hh.shape, bool)
        for b in boxes:
            m |= (lon >= b[0]) & (lon <= b[2]) & (lat >= b[1]) & (lat <= b[3])
        masks[z] = m & np.isfinite(c.real)
    # reference phase = circular mean over bare sand
    u = c / np.abs(c)
    ref = np.nanmean(u[masks["SAND"] & bare]); ref /= abs(ref)
    ph = np.degrees(np.angle(c * np.conj(ref)))
    dd = {"sand_ref_absolute_phase_deg": round(float(np.degrees(np.angle(ref))), 1)}
    for z, m in masks.items():
        mb = m & bare
        z_u = np.nanmean(u[mb]) * np.conj(ref)
        dd[z] = {"n_pixels": int(mb.sum()), "bare_fraction": round(float(mb.sum() / max(m.sum(), 1)), 2),
                 "phase_vs_sand_deg": round(float(np.degrees(np.angle(z_u))), 2),
                 "phase_spread_deg_IQR": [round(float(np.nanpercentile(ph[mb], 25)), 1), round(float(np.nanpercentile(ph[mb], 75)), 1)],
                 "HH_dB": round(float(np.nanmedian(hhdb[mb])), 2), "VV_dB": round(float(np.nanmedian(10*np.log10(vv[mb]))), 2),
                 "HV_dB": round(float(np.nanmedian(hvdb[mb])), 2),
                 "soil_moisture": None if np.isnan(np.nanmedian(sm[mb])) else round(float(np.nanmedian(sm[mb])), 3)}
        samples.setdefault(d, {})[z] = ph[mb]
    res["dates"][d] = dd

# separability: phase alone, SALT vs SAND (per date), best threshold accuracy + AUC
def auc(a, b):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    x = np.concatenate([a, b]); r = x.argsort().argsort() + 1
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))
res["separability_salt_vs_sand"] = {}
for d in dates:
    s, n = samples[d]["SALT"], samples[d]["SAND"]
    A = auc(s, n); A = max(A, 1 - A)
    thr = np.linspace(-30, 30, 601)
    accs = [max(((s < t).mean() + (n >= t).mean()) / 2, ((s >= t).mean() + (n < t).mean()) / 2) for t in thr]
    res["separability_salt_vs_sand"][d] = {"AUC": round(A, 3), "balanced_accuracy": round(float(max(accs)), 3)}

json.dump(res, open(f"{OUT}/rann_test.json", "w"), indent=1)
print(json.dumps(res, indent=1))

# figure: phase distributions per zone per date
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
col = {"SAND": "#d9a441", "SALT": "#3b6fb6", "BET": "#4a8a5a"}
for i, d in enumerate(dates):
    for z in ZONES:
        v = samples[d][z]; v = v[np.isfinite(v)]
        if len(v) > 20:
            ax[i].hist(v, bins=np.linspace(-30, 30, 61), density=True, alpha=.55, color=col[z],
                       label=f"{z}: {res['dates'][d][z]['phase_vs_sand_deg']:+.1f}°")
    ax[i].axvline(0, color="k", lw=.7)
    smv = res["dates"][d]["SALT"]["soil_moisture"]
    ax[i].set_title(f"{d[:4]}-{d[4:6]}-{d[6:]}   salt-flat moisture {smv}")
    ax[i].set_xlabel("HH−VV phase relative to Thar sand, °"); ax[i].legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/rann_phase_zones.png", dpi=120)
