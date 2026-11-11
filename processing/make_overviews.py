"""
Make 200 m multilooked overviews from NISAR GCOV (20 m) files.
Block-averages power terms (HHHH, HVHV, VVVV) and the complex HHVV term over FACTOR x FACTOR pixels.
Averaging the complex term before taking the angle is the standard way to get a clean HH-VV phase.
Output: overviews/<region>_<date>.npz  (linear power + complex HHVV + x/y coordinates, EPSG)
"""
import glob, os, sys
import numpy as np, h5py

FACTOR = 10          # 20 m * 10 = 200 m  (matches SME2 soil moisture resolution)
G = "science/LSAR/GCOV/grids/frequencyA"
OUT = "overviews"
os.makedirs(OUT, exist_ok=True)


def block_mean(a, k):
    r, c = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    a = a[:r, :c]
    valid = np.isfinite(a) if not np.iscomplexobj(a) else np.isfinite(a.real)
    a = np.where(valid, a, 0)
    s = a.reshape(r // k, k, c // k, k).sum(axis=(1, 3))
    n = valid.reshape(r // k, k, c // k, k).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / n
    m[n < k * k * 0.5] = np.nan
    return m


def process(path):
    region = path.split(os.sep)[-2]
    date = os.path.basename(path).split("_")[-6][:8]
    out = f"{OUT}/{region}_{date}.npz"
    if os.path.exists(out):
        print("skip", out); return
    f = h5py.File(path, "r"); g = f[G]
    ny, nx = g["HHHH"].shape
    ny_k = (ny // FACTOR) * FACTOR
    terms = [t for t in ("HHHH", "HVHV", "VVVV", "HHVV") if t in g]
    res = {t: [] for t in terms}
    strip = 512 if 512 % FACTOR == 0 else FACTOR * 50   # rows per read, multiple of FACTOR
    strip = FACTOR * 50
    for r0 in range(0, ny_k, strip):
        r1 = min(r0 + strip, ny_k)
        m = g["mask"][r0:r1, :]
        bad = (m == 0) | (m == 255)
        for t in terms:
            a = g[t][r0:r1, :]
            a = a.astype("complex64") if np.iscomplexobj(a) else a.astype("float32")
            if np.iscomplexobj(a):
                a[bad] = np.nan + 1j * np.nan
            else:
                a[bad] = np.nan; a[a <= 0] = np.nan
            res[t].append(block_mean(a, FACTOR))
        sys.stdout.write(f"\r{region} {date}: {r1}/{ny_k} rows"); sys.stdout.flush()
    x = g["xCoordinates"][:]; y = g["yCoordinates"][:]
    xs = x[: (nx // FACTOR) * FACTOR].reshape(-1, FACTOR).mean(1)
    ys = y[:ny_k].reshape(-1, FACTOR).mean(1)
    data = {t: np.vstack(v) for t, v in res.items()}
    np.savez_compressed(out, x=xs, y=ys, epsg=int(g["projection"][()]), **data)
    print(f"\n  saved {out}  shape {data['HHHH'].shape}")


if __name__ == "__main__":
    for p in sorted(glob.glob("data/*/*GCOV*.h5")):
        process(p)
