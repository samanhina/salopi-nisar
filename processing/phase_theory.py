"""
SALOPI physics -> NISAR: what HH-VV phase should salt and moisture produce at L-band (1.257 GHz)?

Same soil physics as the SALOPI CubeSat model (Stogryn saline water + Dobson mixing),
but a BACKSCATTER surface model instead of specular reflection:
Small Perturbation Model (first-order Bragg), the standard model for co-pol phase of bare soil.
    a_hh = (e-1) / (cos t + sqrt(e - sin^2 t))^2
    a_vv = (e-1) * ((e-1) sin^2 t + e) / (e cos t + sqrt(e - sin^2 t))^2
    HH-VV phase = arg(a_hh * conj(a_vv))
Roughness cancels in the SPM phase (it multiplies both channels equally), so the phase depends
only on the complex permittivity -> on the imaginary part e'' (salt) much more than on e' (water).
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = open("/home/claude/salopi/salopi_model.py", encoding="utf-8").read().split("# ---------------- ")[0]
src_all = open("/home/claude/salopi/salopi_model.py", encoding="utf-8").read()
start = src_all.index("E0 = 8.854e-12"); end = src_all.index("SALT_LEVELS")
ns = {"np": np}; exec(src_all[start:end], ns)
soil_eps, saline_water_eps, ece_estimate = ns["soil_eps"], ns["saline_water_eps"], ns["ece_estimate"]

F_NISAR = 1.257e9
OUT = "/home/claude/hackathon/theory"; os.makedirs(OUT, exist_ok=True)


def spm_phase(eps, theta_deg):
    t = np.deg2rad(theta_deg); s2 = np.sin(t) ** 2; c = np.cos(t)
    r = np.sqrt(eps - s2)
    ahh = (eps - 1) / (c + r) ** 2
    avv = (eps - 1) * ((eps - 1) * s2 + eps) / (eps * c + r) ** 2
    return np.degrees(np.angle(ahh * np.conj(avv))), 10 * np.log10(np.abs(ahh) ** 2), 10 * np.log10(np.abs(avv) ** 2)


def phase(mv, salt, theta=40, f=F_NISAR):
    e = soil_eps(f, np.asarray(mv, float), np.asarray(salt, float))
    return spm_phase(e, theta)


res = {}
# 1) grid: salt x moisture at 40 deg
mv = np.linspace(0.03, 0.40, 60)
salts = [0, 0.5, 1.5, 4, 10, 30]              # g/kg (0 ... 3 % -> salt crust / solonchak)
plt.figure(figsize=(7.5, 5))
for s in salts:
    ph, _, _ = phase(mv, s)
    plt.plot(mv, ph, lw=2.4, label=f"salt {s} g/kg (ECe≈{ece_estimate(s):.0f} dS/m)")
plt.xlabel("Soil moisture, m³/m³"); plt.ylabel("HH−VV phase, degrees (model convention)")
plt.title("NISAR L-band (1.26 GHz), 40°: phase vs moisture for each salt level")
plt.axhline(0, color="k", lw=.6); plt.legend(fontsize=8.5); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(f"{OUT}/phase_vs_moisture.png", dpi=130); plt.close()

# 2) sensitivities around typical states
def dphi(mv1, s1, mv2, s2, th=40):
    return float(phase(mv2, s2, th)[0] - phase(mv1, s1, th)[0])

res["moisture_effect_nonsaline_0.05_to_0.35"] = dphi(0.05, 0, 0.35, 0)
res["salt_effect_at_mv0.25_0_to_4gkg"] = dphi(0.25, 0, 0.25, 4)
res["salt_effect_at_mv0.25_0_to_30gkg"] = dphi(0.25, 0, 0.25, 30)
res["salt_effect_at_mv0.10_0_to_30gkg"] = dphi(0.10, 0, 0.10, 30)
# Rann-specific contrast: dry non-saline sand (Thar dunes) vs damp salt crust (Great Rann)
sand = dict(mv=0.04, s=0.0); crust_dry = dict(mv=0.10, s=30.0); crust_wet = dict(mv=0.30, s=30.0)
wet_fresh = dict(mv=0.30, s=0.0)
for name, a in (("damp_salt_crust_mv0.10_30gkg", crust_dry), ("wet_salt_crust_mv0.30_30gkg", crust_wet),
                ("wet_nonsaline_mv0.30", wet_fresh)):
    res[f"vs_dry_sand: {name}"] = dphi(sand["mv"], sand["s"], a["mv"], a["s"])
# incidence-angle dependence (NISAR swath ~33-47 deg)
res["by_angle_salt_crust_vs_sand"] = {th: dphi(0.04, 0, 0.15, 30, th) for th in (33, 37, 40, 44, 47)}
res["by_angle_wet_fresh_vs_sand"] = {th: dphi(0.04, 0, 0.30, 0, th) for th in (33, 37, 40, 44, 47)}

# 3) map: phase over (moisture, salt) plane
MV, LS = np.meshgrid(np.linspace(0.03, 0.40, 120), np.linspace(-1, np.log10(50), 120))
PH = phase(MV, 10 ** LS)[0]
plt.figure(figsize=(7, 5))
cs = plt.contourf(MV, 10 ** LS, PH, levels=21, cmap=__import__("matplotlib.colors",fromlist=["x"]).LinearSegmentedColormap.from_list("d",["#105666","#7FA9B0","#F7F4D5","#E2B5AC","#9C4F45"])); plt.colorbar(cs, label="HH−VV phase, °")
plt.contour(MV, 10 ** LS, PH, levels=[-10, -5, -2, 0], colors="#0A3323", linewidths=.7)
plt.yscale("log"); plt.xlabel("Soil moisture, m³/m³"); plt.ylabel("Salt, g/kg soil")
plt.title("Model: phase follows SALT (vertical), barely MOISTURE (horizontal)")
plt.tight_layout(); plt.savefig(f"{OUT}/phase_map.png", dpi=130); plt.close()

# variance split like the SALOPI ANOVA: random soils, how much of the phase variance is salt vs moisture
rng = np.random.default_rng(1)
n = 20000
mv_r = rng.uniform(0.03, 0.40, n); s_r = 10 ** rng.uniform(-1, np.log10(30), n); th_r = rng.uniform(33, 47, n)
ph_r = phase(mv_r, s_r, th_r)[0]
def r2(x, y):
    # binned-mean explained variance (eta^2)
    bins = np.quantile(x, np.linspace(0, 1, 21)); idx = np.clip(np.digitize(x, bins) - 1, 0, 19)
    m = np.array([y[idx == i].mean() for i in range(20)])
    return float(1 - ((y - m[idx]) ** 2).sum() / ((y - y.mean()) ** 2).sum())
res["eta2_phase_explained_by_log_salt"] = r2(np.log10(s_r), ph_r)
res["eta2_phase_explained_by_moisture"] = r2(mv_r, ph_r)
res["eta2_phase_explained_by_angle"] = r2(th_r, ph_r)
json.dump(res, open(f"{OUT}/theory_results.json", "w"), indent=1)
print(json.dumps(res, indent=1))
