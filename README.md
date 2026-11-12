# ε″ · Salt from Space

**Detecting soil salinity from NASA-ISRO NISAR L-band radar polarimetry.**
Team **ε″ (Epsilon Double Prime)** · NASA Space Apps Challenge 2026 · challenge *"Dancing with the SARs"*.

Live app: https://samanhina.github.io/salopi-nisar/  (EN / RU / UZ)

Dissolved salt raises the imaginary part of soil permittivity, ε″. That rotates the phase between HH and VV radar echoes far more than water does — so radar phase can separate salt from moisture where amplitude cannot.

## Two modes
| Mode | Data | What it shows |
|---|---|---|
| **Global** (Mirzachul & Syrdarya, Uzbekistan) | GCOV dual-pol HH+HV + SME2 soil moisture | Season change (harvest, flooding) and a **salinization risk index**: land that is *wet* but *stays bare* all season |
| **Precision** (Rann of Kutch / Thar, Pakistan–India) | GCOV quad-pol (HH, HV, VH, VV, complex HH·VV*) | **HH–VV phase difference** and a persistent **salt signature** |

## Evidence
- **Forward model** (`processing/phase_theory.py`, `processing/soil_model.py`): Stogryn saline water + Dobson mixing at 1.257 GHz with Bragg (SPM) backscatter. Salt crust shifts HH–VV phase by **+4…+9°**, moisture alone by **−0.2°**; salinity explains 74% of phase variance.
- **Natural experiment** (`processing/rann_test.py`): Great Rann salt flat vs non-saline Thar dunes, zones by geography, bare pixels only. Salt flat sits **+5…+14°** above sand on all three dates; NISAR SME2 says it is *drier*; phase alone separates the two with **77–90%** accuracy (AUC 0.75–0.95). With roughness matched (equal depolarisation) the shift stays **+8°, +12°, +4°**.
- **Global-mode check:** the risk index puts 35% of the Mirzachul (Hungry Steppe) zone in the top-risk class vs 13% scene-wide (2.6×); in Syrdarya region 99.1% of irrigated land is slightly-to-moderately saline (JIRCAS Working Report 88).
Figures and numbers: `results/`.

## Impact
1.4 billion ha of land (10.7%) is salt-affected and another 1 billion ha is at risk (FAO 2024); salt-induced degradation of irrigated land costs ~US$27 bn per year (Qadir et al. 2014). In Uzbekistan more than half of irrigated land is saline.

## Data
NISAR L2 GCOV and L3 SME2, **provisional** products (June–October 2026), NASA Earthdata / ASF DAAC.
- Rann: track 063, frame 076, 2026-06-18, 07-12, 10-16 (QPDH)
- Mirzachul: track 020, frame 068, 2026-06-27, 08-26, 10-13 (DHDH)

## Pipeline
```
pip install h5py numpy matplotlib pyproj pillow scipy
python processing/nisar_explore.py FILE.h5 --list     # look inside a granule
python processing/make_overviews.py                    # 20 m -> 200 m multilook (power + complex HH·VV*)
python processing/build_layers.py                      # lat/lon WebP layers + data cube for the app
python processing/phase_theory.py                      # forward model
python processing/rann_test.py                         # natural experiment
cd docs && python -m http.server                       # open http://localhost:8000
```
Put granules in `data/indus/` and `data/mirzachul/`.

## Limitations
Provisional products (per-scene HH–VV phase offset removed by referencing to dry dunes); one quad-pol site; the risk index is a screening tool that needs field ECe samples to calibrate.

## License
MIT. Leaflet (BSD-2). Basemap © Esri. Fonts: IBM Plex (OFL).
