"""
SALOPI x NISAR - practice script (preparation, not the hackathon project itself).

What it does with one NISAR L2 GCOV file (.h5):
  1. Lists everything inside the file (all datasets, their shapes and types).
  2. Reads the backscatter layers HHHH, HVHV, VVVV (whatever exists) and shows them in dB.
  3. If the file is quad-pol (has HHVV), computes the HH-VV phase difference - our salt signature.
  4. Saves quick-look pictures next to the file and prints simple statistics.

Install once:   pip install h5py numpy matplotlib pyproj
Run:            python nisar_explore.py  PATH_TO_FILE.h5
Optional crop:  python nisar_explore.py  FILE.h5  --bbox 68.0 40.2 68.8 40.8     (lon_min lat_min lon_max lat_max)
Only list:      python nisar_explore.py  FILE.h5  --list
"""
import argparse, os, sys
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/science/LSAR/GCOV/grids"


def list_contents(f):
    print("\n=== Everything inside the file ===")
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name:70s} {str(obj.shape):18s} {obj.dtype}")
    f.visititems(show)


def find_grid(f):
    """Return the frequencyA group (or frequencyB if A is missing)."""
    for fr in ("frequencyA", "frequencyB"):
        p = f"{BASE}/{fr}"
        if p in f:
            return f[p], fr
    sys.exit("No GCOV grids found. Is this a GCOV file? Run with --list to see what is inside.")


def epsg_of(grid):
    if "projection" in grid:
        ds = grid["projection"]
        for key in ("epsg_code", "epsg"):
            if key in ds.attrs:
                return int(ds.attrs[key])
        try:
            return int(ds[()])
        except Exception:
            pass
    return None


def crop_window(grid, bbox):
    """Turn a lon/lat box into row/column limits. Returns slices for the whole image if not possible."""
    x = grid["xCoordinates"][:]
    y = grid["yCoordinates"][:]
    full = (slice(0, len(y)), slice(0, len(x)))
    if bbox is None:
        return full, x, y
    epsg = epsg_of(grid)
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        xs, ys = tr.transform([bbox[0], bbox[2], bbox[0], bbox[2]], [bbox[1], bbox[1], bbox[3], bbox[3]])
    except Exception as e:
        print(f"! Could not convert lon/lat to map coordinates ({e}); using the whole image.")
        return full, x, y
    cols = np.where((x >= min(xs)) & (x <= max(xs)))[0]
    rows = np.where((y >= min(ys)) & (y <= max(ys)))[0]
    if len(cols) == 0 or len(rows) == 0:
        print("! The box does not overlap this image; using the whole image.")
        return full, x, y
    return (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)), x, y


def to_db(a):
    a = np.where(a > 0, a, np.nan)
    return 10 * np.log10(a)


def stride_for(shape, target=1500):
    return max(1, int(max(shape) / target))


def main():
    ap = argparse.ArgumentParser(description="Look inside a NISAR GCOV file")
    ap.add_argument("file")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    ap.add_argument("--list", action="store_true", help="only list the contents")
    a = ap.parse_args()

    out_dir = os.path.splitext(a.file)[0] + "_quicklook"
    with h5py.File(a.file, "r") as f:
        list_contents(f)
        if a.list:
            return
        grid, fr = find_grid(f)
        print(f"\nUsing {BASE}/{fr}   EPSG: {epsg_of(grid)}")
        (rs, cs), x, y = crop_window(grid, a.bbox)
        layers = [n for n in ("HHHH", "HVHV", "VVVV", "VHVH") if n in grid]
        has_hhvv = "HHVV" in grid
        mode = "QUAD-POL (precision mode possible)" if has_hhvv else "dual/single-pol (global mode only)"
        print(f"Backscatter layers: {layers}   ->   {mode}")

        os.makedirs(out_dir, exist_ok=True)
        shape = (rs.stop - rs.start, cs.stop - cs.start)
        st = stride_for(shape)
        print(f"Area: {shape[0]} x {shape[1]} pixels (showing every {st}th pixel in pictures)\n")

        stats = {}
        for n in layers:
            db = to_db(grid[n][rs, cs][::st, ::st].astype("float32"))
            stats[n] = db
            v = db[np.isfinite(db)]
            print(f"{n}: mean {v.mean():6.2f} dB   median {np.median(v):6.2f} dB   5-95% {np.percentile(v,5):6.2f} .. {np.percentile(v,95):6.2f} dB")
            plt.figure(figsize=(7, 6))
            plt.imshow(db, cmap="gray", vmin=np.nanpercentile(db, 2), vmax=np.nanpercentile(db, 98))
            plt.colorbar(label=f"{n}, dB"); plt.title(f"{n} backscatter, dB"); plt.axis("off")
            plt.savefig(os.path.join(out_dir, f"{n}_dB.png"), dpi=130, bbox_inches="tight"); plt.close()

        if "HHHH" in stats and "HVHV" in stats:
            ratio = stats["HVHV"] - stats["HHHH"]
            print(f"HV/HH ratio: median {np.nanmedian(ratio):6.2f} dB  (higher = more vegetation / volume scattering)")

        if has_hhvv:
            c = grid["HHVV"][rs, cs][::st, ::st]
            phase = np.degrees(np.angle(c))
            v = phase[np.isfinite(phase)]
            print(f"HH-VV phase difference: median {np.median(v):6.2f} deg   5-95% {np.percentile(v,5):6.2f} .. {np.percentile(v,95):6.2f} deg")
            print("  (our SALOPI model: salt shifts this by about -2.5 to -8 deg, moisture only ~ +0.2 deg)")
            plt.figure(figsize=(7, 6))
            plt.imshow(phase, cmap="twilight", vmin=-30, vmax=30)
            plt.colorbar(label="HH-VV phase difference, degrees"); plt.title("HH-VV phase difference"); plt.axis("off")
            plt.savefig(os.path.join(out_dir, "HHVV_phase.png"), dpi=130, bbox_inches="tight"); plt.close()

        if "mask" in grid:
            print("Note: the file has a 'mask' layer (invalid / shadow / layover pixels) - use it to hide bad pixels.")
    print(f"\nPictures saved in: {out_dir}")


if __name__ == "__main__":
    main()
