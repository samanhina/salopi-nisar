# SALOPI × NISAR

**Listening for salt in radar echoes.** An open-source web app that uses NASA-ISRO **NISAR** L-band radar to track soil change and flag land at risk of salinization.
Team **ε″ (Epsilon Double Prime)** · NASA Space Apps Challenge 2026 · challenge *"Dancing with the SARs"*.

## Two modes
| Mode | Data | What it shows |
|---|---|---|
| **Global** (Mirzachul & Syrdarya, Uzbekistan) | GCOV dual-pol HH+HV + SME2 soil moisture | Season change (harvest, flooding), and a **salinization risk screening**: land that is *wet* but *stays bare* all season |
| **Precision** (Rann of Kutch / Thar, Pakistan–India) | GCOV quad-pol (HH, HV, VH, VV, complex HH·VV*) | **HH–VV phase difference** — the dielectric fingerprint of dissolved salt (ε″) |

## Data
NISAR L2 GCOV and L3 SME2, **provisional** products (June–October 2026), from NASA Earthdata / ASF DAAC.
- Rann: track 063, frame 076, 2026-06-18, 07-12, 10-16 (QPDH)
- Mirzachul: track 020, frame 068, 2026-06-27, 08-26, 10-13 (DHDH)

## Pipeline
```
pip install h5py numpy matplotlib pyproj pillow
python processing/nisar_explore.py FILE.h5 --list     # look inside a granule
python processing/make_overviews.py                    # 20 m -> 200 m multilook (power + complex HH·VV*)
python processing/build_layers.py                      # lat/lon WebP layers + data cube for the app
cd docs && python -m http.server                        # open http://localhost:8000
```
Put granules in `data/indus/` and `data/mirzachul/`.

## Is it really salt? (evidence)
`processing/phase_theory.py` – SALOPI soil physics at 1.257 GHz with a Bragg (SPM) surface model: salt crust shifts HH–VV phase by **+4…+9°**, moisture alone by **−0.2°**; salt explains 74% of phase variance.
`processing/rann_test.py` – natural experiment on real NISAR quad-pol scenes: Great Rann salt flat vs Thar sand dunes (zones by geography, bare pixels only). Salt flat sits **+9°, +14°, +5°** above sand on all three dates, while NISAR SME2 says it is *drier*; phase alone separates the two with **77–90% accuracy** (AUC 0.75–0.95). Figures and numbers in `results/`.

**Global-mode check:** the risk index puts 35% of the Mirzachul (Hungry Steppe) zone in the top-risk class vs 13% scene-wide (2.6×); in Syrdarya region 99.1% of irrigated land was rated slightly-to-moderately saline (MAWR 2003, in JIRCAS Working Report 88).

## Method notes and honest limits
- Complex HH·VV* is averaged over 10×10 pixels **before** taking the angle (standard multilook phase).
- Provisional scenes carry a constant HH–VV phase offset (+68° Jun/Jul, −91° Oct); phase is measured **relative to dry Thar sand dunes** (natural target, ε″ ≈ 0).
- Risk = percentile(soil moisture) × (1 − percentile(season-max HV)), open water removed. A screening indicator, not a salinity measurement; it needs field ECe samples to calibrate.
- Physics background (Stogryn + Dobson + Fresnel model, 240-case virtual experiment) comes from our SALOPI P-band CubeSat project.

## License
MIT. Leaflet (BSD-2). Basemap © Esri.
