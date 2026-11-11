"""
Build web-map layers for the SALOPI x NISAR app.

Input : overviews/<region>_<date>.npz  (200 m multilooked GCOV, UTM)  +  data/<folder>/*SME2*.h5
Output: web/data/<region>/<layer>_<date>.png   transparent PNG overlays on a lat/lon grid
        web/data/<region>/cube.bin              uint8 data cube at coarser grid, for click-to-inspect time series
        web/data/meta.json                      bounds, dates, layers, statistics, quantisation rules
"""
import glob, json, os
import numpy as np, h5py
from pyproj import Transformer
import matplotlib; matplotlib.use("Agg")
from matplotlib import cm, colors
from PIL import Image

REGIONS = {
    # key: (data folder, lon_min, lat_min, lon_max, lat_max, mode, title)
    "rann": ("indus", 69.00, 23.45, 71.62, 24.82, "precision", "Rann of Kutch / Thar (Pakistan–India)"),
    "mirzachul": ("mirzachul", 68.05, 39.18, 71.82, 42.06, "global", "Mirzachul & Syrdarya (Uzbekistan)"),
}
STEP = 0.002        # ~200 m picture grid
CUBE_STEP = 0.006   # ~600 m grid for click-to-inspect numbers
OUT = "docs/data"

QUANT = {  # value = lo + q*(hi-lo)/250 ; 255 = no data
    "HH": (-35, 5), "HV": (-40, 0), "VV": (-35, 5), "PHASE": (-45, 45), "SM": (0, 0.6), "RISK": (0, 1),
}


def grid(b, step):
    lons = np.arange(b[0] + step / 2, b[2], step)
    lats = np.arange(b[3] - step / 2, b[1], -step)
    return np.meshgrid(lons, lats)


def sample_regular(arr, xs, ys, X, Y):
    """Nearest-neighbour lookup of arr (regular x/y axes) at map coordinates X, Y."""
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
    ci = np.rint((X - xs[0]) / dx).astype(int); ri = np.rint((Y - ys[0]) / dy).astype(int)
    ok = (ci >= 0) & (ci < len(xs)) & (ri >= 0) & (ri < len(ys))
    out = np.full(X.shape, np.nan, dtype=arr.dtype if np.iscomplexobj(arr) else "float32")
    out[ok] = arr[ri[ok], ci[ok]]
    return out


def to_png(rgba, path):
    # WebP with alpha: ~8x smaller than PNG, supported by all modern browsers
    Image.fromarray((np.clip(rgba, 0, 1) * 255).astype("uint8"), "RGBA").save(path.replace(".png", ".webp"), quality=82, method=6)


def colorize(a, cmap, lo, hi):
    rgba = matplotlib.colormaps[cmap](colors.Normalize(lo, hi, clip=True)(np.nan_to_num(a, nan=lo)))
    rgba[..., 3] = np.isfinite(a)
    return rgba


def quant(a, key):
    lo, hi = QUANT[key]
    q = np.rint((np.clip(a, lo, hi) - lo) / (hi - lo) * 250)
    q[~np.isfinite(a)] = 255
    return q.astype("uint8")


def rel_phase(c):
    u = c / np.abs(c)
    ref = np.nansum(u); ref /= abs(ref)
    return np.degrees(np.angle(c * np.conj(ref))), float(np.degrees(np.angle(ref)))


def sme2_files(folder):
    out = {}
    for p in glob.glob(f"data/{folder}/*SME2*.h5"):
        out[os.path.basename(p).split("_")[-6][:8]] = p
    return out


def load_sm(path, LON, LAT):
    with h5py.File(path, "r") as f:
        g = f["science/LSAR/SME2/grids"]
        sm = g["soilMoisture"][:].astype("float32"); sm[(sm < 0) | (sm > 1)] = np.nan
        xs = g["xCoordinates"][:]; ys = g["yCoordinates"][:]; epsg = int(g["projection"][()])
    X, Y = Transformer.from_crs(4326, epsg, always_xy=True).transform(LON, LAT)
    return sample_regular(sm, xs, ys, X, Y)


def pct_rank(a, ref):
    """Percentile rank of a within the valid values of ref (0..1)."""
    v = np.sort(ref[np.isfinite(ref)])
    r = np.searchsorted(v, a) / max(len(v), 1)
    r[~np.isfinite(a)] = np.nan
    return r


def build(key):
    folder, *b, mode, title = REGIONS[key]
    od = f"{OUT}/{key}"; os.makedirs(od, exist_ok=True)
    files = sorted(glob.glob(f"overviews/{folder}_*.npz"))
    dates = [os.path.basename(f)[len(folder) + 1:-4] for f in files]
    smf = sme2_files(folder)
    grids = {}
    for name, step in (("pic", STEP), ("cube", CUBE_STEP)):
        LON, LAT = grid(b, step)
        grids[name] = (LON, LAT)
    meta = {"title": title, "mode": mode, "bounds": [[b[1], b[0]], [b[3], b[2]]], "dates": dates,
            "cube_shape": list(grids["cube"][0].shape), "layers": [], "stats": {}}
    per = {"pic": {}, "cube": {}}
    for f, d in zip(files, dates):
        D = np.load(f)
        for gname, (LON, LAT) in grids.items():
            X, Y = Transformer.from_crs(4326, int(D["epsg"]), always_xy=True).transform(LON, LAT)
            L = {"HH": 10 * np.log10(sample_regular(D["HHHH"], D["x"], D["y"], X, Y)),
                 "HV": 10 * np.log10(sample_regular(D["HVHV"], D["x"], D["y"], X, Y))}
            if "HHVV" in D:
                L["VV"] = 10 * np.log10(sample_regular(D["VVVV"], D["x"], D["y"], X, Y))
                c = sample_regular(D["HHVV"], D["x"], D["y"], X, Y)
                L["PHASE"], ref = rel_phase(c)
                if gname == "pic":
                    meta["stats"].setdefault(d, {})["phase_offset_deg"] = round(ref, 1)
            L["SM"] = load_sm(smf[d], LON, LAT) if d in smf else np.full(LON.shape, np.nan, "float32")
            L["SM"][~np.isfinite(L["HH"])] = np.nan
            per[gname][d] = L

    # --- salinization risk screening (global mode, works with HH+HV only) ---------------------
    # Idea: salt-affected irrigated land is typically WET (shallow saline groundwater, waterlogging)
    # but stays POORLY VEGETATED through the season (crops fail). So:  risk = wetness x (1 - vegetation)
    # wetness   = percentile rank of mean SME2 soil moisture
    # vegetation= percentile rank of the season MAXIMUM HV backscatter (volume scattering of plants)
    # Open water (HH < -20 dB on every date) is excluded.
    for gname in ("pic", "cube"):
        P = per[gname]
        hv_max = np.nanmax(np.stack([P[d]["HV"] for d in dates]), 0)
        hh_max = np.nanmax(np.stack([P[d]["HH"] for d in dates]), 0)
        sm_mean = np.nanmean(np.stack([P[d]["SM"] for d in dates]), 0)
        ref_pic = per["pic"]
        hv_ref = np.nanmax(np.stack([ref_pic[d]["HV"] for d in dates]), 0)
        sm_ref = np.nanmean(np.stack([ref_pic[d]["SM"] for d in dates]), 0)
        risk = pct_rank(sm_mean, sm_ref) * (1 - pct_rank(hv_max, hv_ref))
        risk[hh_max < -20] = np.nan
        # express as percentile within the region: 0.9 = riskier than 90% of the region's land
        if gname == "pic":
            risk_pic_raw = risk.copy()
        risk = pct_rank(risk, risk_pic_raw)
        for d in dates:
            P[d]["RISK"] = risk

    # --- pictures -------------------------------------------------------------------------------
    first = dates[0]
    for d in dates:
        L = per["pic"][d]
        hh, hv = L["HH"], L["HV"]
        R = np.clip((hh + 25) / 25, 0, 1); G = np.clip((hv + 32) / 22, 0, 1); B = np.clip((hh - hv - 3) / 12, 0, 1)
        rgb = np.dstack([R, G, B, np.isfinite(hh)])
        to_png(rgb, f"{od}/RGB_{d}.png")
        to_png(colorize(hv, "gray", -32, -10), f"{od}/HV_{d}.png")
        to_png(colorize(L["SM"], "YlGnBu", 0.02, 0.40), f"{od}/SM_{d}.png")
        if d != first:
            to_png(colorize(hv - per["pic"][first]["HV"], "RdBu_r", -6, 6), f"{od}/dHV_{d}.png")
            to_png(colorize(hh - per["pic"][first]["HH"], "RdBu_r", -6, 6), f"{od}/dHH_{d}.png")
        if "PHASE" in L:
            to_png(colorize(L["PHASE"], "PuOr_r", -25, 25), f"{od}/PHASE_{d}.png")
        st = meta["stats"].setdefault(d, {})
        for k in ("HH", "HV", "VV", "SM", "PHASE"):
            if k in L:
                st[f"{k}_median"] = round(float(np.nanmedian(L[k])), 3)
    to_png(colorize(np.where(per["pic"][first]["RISK"] >= 0.75, per["pic"][first]["RISK"], np.nan), "YlOrRd", 0.75, 1.0), f"{od}/RISK.png")
    meta["layers"] = ["RGB", "HV", "dHV", "dHH", "SM", "RISK"] + (["PHASE"] if "PHASE" in per["pic"][first] else [])

    # --- data cube for click-to-inspect -------------------------------------------------------
    keys = [k for k in ("HH", "HV", "VV", "PHASE", "SM") if k in per["cube"][first]]
    blocks, order = [], []
    for d in dates:
        for k in keys:
            blocks.append(quant(per["cube"][d][k], k)); order.append([d, k])
    blocks.append(quant(per["cube"][first]["RISK"], "RISK")); order.append(["all", "RISK"])
    # base64 text so it can be served as a plain .txt file by any static host
    import base64
    open(f"{od}/cube.txt", "w").write(base64.b64encode(np.stack(blocks).tobytes()).decode())
    meta["cube_order"] = order
    print(key, "done:", dates, meta["layers"], "cube", meta["cube_shape"], len(order), "bands")
    return meta


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    meta = {"quant": QUANT, "regions": {k: build(k) for k in REGIONS}}
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
