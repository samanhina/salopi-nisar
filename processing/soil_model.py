"""
soil — численное моделирование (виртуальный эксперимент) отражения P-диапазонного
сигнала от засолённой почвы при разной влажности.

Цепочка расчёта:
 1) соль + вода -> солёность поровой воды (г/кг) -> нормальность NaCl
 2) модель Стогрина (Stogryn, 1971): ε и проводимость σ солёной воды (Дебай + ионная проводимость)
 3) полуэмпирическая модель смеси Добсона (Dobson et al., 1985): ε почвы
 4) формулы Френеля: коэффициенты отражения Γh, Γv, круговая поляризация (как у SNoOPI)
 5) поправка на шероховатость, глубина проникновения
 6) виртуальный эксперимент 4×4 условия × 3 образца × 5 повторов = 240 измерений с шумами
 7) анализ признаков + машинное обучение (классификация / регрессия)
"""
import numpy as np, pandas as pd, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

OUT = "soil_model_out"
import os; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)
E0 = 8.854e-12
C = 2.998e8

# ---------------- почва (суглинок) ----------------
SAND, CLAY = 0.40, 0.20
RHO_S = 2.66                     # плотность твёрдой фазы, г/см3
RHO_B = 1.40                     # плотность сложения, г/см3
ALPHA = 0.65
BETA = 1.2748 - 0.519*SAND - 0.152*CLAY
EPS_SOLID = (1.01 + 0.44*RHO_S)**2 - 0.062
T = 25.0                         # °C
THETA = np.deg2rad(40.0)         # угол падения
BG_SALT = 0.1                    # фоновая соль в "незасолённой" почве, г/кг

def saline_water_eps(f, S_ppt):
    """Стогрин (1971): комплексная ε солёной воды, σ (См/м). Конвенция ε = ε' - jε''."""
    S = np.asarray(S_ppt, float)
    N = S*(1.707e-2 + 1.205e-5*S + 4.058e-9*S**2)
    eps0_T = 87.74 - 0.40008*T + 9.398e-4*T**2 + 1.410e-6*T**3
    a = 1 - 0.2551*N + 5.151e-2*N**2 - 6.889e-3*N**3
    tau2pi_T = 1.1109e-10 - 3.824e-12*T + 6.938e-14*T**2 - 5.096e-16*T**3
    b = 0.1463e-2*N*T + 1 - 0.04896*N - 0.02967*N**2 + 5.644e-3*N**3
    eps_s, tau2pi = eps0_T*a, tau2pi_T*b
    sigma = N*(10.394 - 2.3776*N + 0.68258*N**2 - 0.13538*N**3 + 1.0086e-2*N**4)  # 25 °C
    eps_inf = 4.9
    eps = eps_inf + (eps_s - eps_inf)/(1 + 1j*tau2pi*f) - 1j*sigma/(2*np.pi*f*E0)
    return eps, sigma

def pore_salinity(salt_gkg, mv, rho_b=RHO_B):
    """г соли на кг поровой воды. Масса воды на кг сухой почвы = mv/rho_b (кг)."""
    return np.minimum((salt_gkg + BG_SALT)*rho_b/mv, 300.0)

def soil_eps(f, mv, salt_gkg, rho_b=RHO_B):
    S = pore_salinity(salt_gkg, mv, rho_b)
    efw, _ = saline_water_eps(f, S)
    em_a = 1 + (rho_b/RHO_S)*(EPS_SOLID**ALPHA - 1) + mv**BETA*efw**ALPHA - mv
    return em_a**(1/ALPHA)

def fresnel(eps, theta=THETA):
    c, s2 = np.cos(theta), np.sin(theta)**2
    r = np.sqrt(eps - s2)
    gh = (c - r)/(c + r)
    gv = (eps*c - r)/(eps*c + r)
    g_cross = (gv - gh)/2          # RHCP -> LHCP при зеркальном отражении (схема SoOp)
    return gh, gv, g_cross

def roughness(f, s_m, theta=THETA):
    k = 2*np.pi*f/C
    return np.exp(-2*(k*s_m*np.cos(theta))**2)   # множитель для мощности

def pen_depth(f, eps):
    lam = C/f
    e1, e2 = eps.real, -eps.imag
    return lam*np.sqrt(e1)/(2*np.pi*e2)

def ece_estimate(salt_gkg, theta_sat=0.45):
    """Оценка ECe (дСм/м) — проводимость вытяжки насыщенной пасты."""
    S = pore_salinity(salt_gkg, theta_sat)
    _, s = saline_water_eps(1e9, S)
    return s*10

# ---------------- план эксперимента ----------------
SALT_LEVELS = np.array([0.0, 0.5, 1.5, 4.0])            # г/кг (0; 0.05; 0.15; 0.4 %)
SALT_NAMES = ["без соли", "низкая", "средняя", "высокая"]
MV_LEVELS = np.array([0.05, 0.15, 0.25, 0.35])          # м3/м3
FREQS = np.array([240, 255, 270, 300, 330, 360, 370, 380])*1e6   # P-диапазон (SNoOPI 240–380 МГц)
F_L = 1575.42e6

summary = {}
summary["soil"] = dict(sand=SAND, clay=CLAY, rho_b=RHO_B, beta=round(BETA,3), eps_solid=round(EPS_SOLID,2))
summary["levels"] = [dict(name=n, salt_gkg=float(s), salt_pct=float(s/10), ECe_dSm=round(float(ece_estimate(s)),1))
                     for n, s in zip(SALT_NAMES, SALT_LEVELS)]

# ---------------- 1. детерминированная таблица 16 условий ----------------
rows = []
for s, sn in zip(SALT_LEVELS, SALT_NAMES):
    for mv in MV_LEVELS:
        r = dict(Засолённость=sn, Соль_г_кг=s, Влажность_м3м3=mv,
                 Солёность_поровой_воды_г_кг=round(float(pore_salinity(s, mv)),1))
        for f in [255e6, 370e6, F_L]:
            e = soil_eps(f, mv, s); _,_,gc = fresnel(e)
            tag = f"{int(round(f/1e6))}МГц"
            r[f"ε'_{tag}"] = round(e.real,2); r[f"ε''_{tag}"] = round(-e.imag,2)
            r[f"Отражение_дБ_{tag}"] = round(10*np.log10(abs(gc)**2),2)
            r[f"Фаза_град_{tag}"] = round(np.degrees(np.angle(gc)),2)
            r[f"Глубина_проникн_см_{tag}"] = round(pen_depth(f, e)*100,1)
        rows.append(r)
det = pd.DataFrame(rows)

# ---------------- 2. виртуальный эксперимент (240 измерений) ----------------
N_SAMPLES, N_REP = 3, 5
AMP_NOISE_DB, PHASE_NOISE_DEG = 0.25, 2.0
meas = []
sid = 0
for si, s in enumerate(SALT_LEVELS):
    for mi, mv in enumerate(MV_LEVELS):
        for k in range(N_SAMPLES):
            sid += 1
            mv_a = max(0.02, mv + rng.normal(0, 0.01))            # фактическая влажность
            s_a = max(0.0, s*(1 + rng.normal(0, 0.05)))            # фактическая соль
            rb_a = rng.normal(RHO_B, 0.03)                         # уплотнение образца
            rough = rng.uniform(0.003, 0.012)                       # шероховатость поверхности, м
            ece = float(ece_estimate(s_a))*(1 + rng.normal(0, 0.03))  # контрольный кондуктометр
            for rep in range(N_REP):
                gain = rng.normal(0, 0.15)                          # дрейф калибровки, дБ (общий для всех частот)
                row = dict(ID_образца=sid, Повтор=rep+1, Класс_засолённости=si, Засолённость=SALT_NAMES[si],
                           Соль_номинал_г_кг=s, Влажность_номинал=mv, Соль_факт_г_кг=round(s_a,3),
                           Влажность_факт=round(mv_a,4), Плотность_г_см3=round(rb_a,3),
                           Шероховатость_мм=round(rough*1000,1), ECe_контроль_дСм_м=round(ece,2))
                for f in FREQS:
                    e = soil_eps(f, mv_a, s_a, rb_a); _,_,gc = fresnel(e)
                    p_db = 10*np.log10(abs(gc)**2*roughness(f, rough)) + gain + rng.normal(0, AMP_NOISE_DB)
                    ph = np.degrees(np.angle(gc)) + rng.normal(0, PHASE_NOISE_DEG)
                    tag = int(f/1e6)
                    row[f"R_{tag}_дБ"] = round(p_db, 3)
                    row[f"Ф_{tag}_град"] = round(ph, 2)
                meas.append(row)
df = pd.DataFrame(meas)
R_COLS = [f"R_{int(f/1e6)}_дБ" for f in FREQS]
P_COLS = [f"Ф_{int(f/1e6)}_град" for f in FREQS]
fghz = FREQS/1e9
# производные признаки: спектральный наклон отражения и фазы (устойчивы к общему дрейфу усиления)
df["Наклон_R_дБ_на_100МГц"] = np.polyfit(fghz*10, df[R_COLS].values.T, 1)[0]
df["Наклон_Ф_град_на_100МГц"] = np.polyfit(fghz*10, df[P_COLS].values.T, 1)[0]
df["R255_минус_R370_дБ"] = df["R_255_дБ"] - df["R_370_дБ"]

# ---------------- 3. анализ признаков: доля дисперсии (2-факторный ANOVA, η²) ----------------
def eta2(col):
    g = df.groupby(["Соль_номинал_г_кг", "Влажность_номинал"])[col].mean().unstack()
    y = df[col]; tot = ((y - y.mean())**2).sum()
    gm = y.mean()
    ss_s = N_SAMPLES*N_REP*len(MV_LEVELS)*((g.mean(axis=1) - gm)**2).sum()
    ss_m = N_SAMPLES*N_REP*len(SALT_LEVELS)*((g.mean(axis=0) - gm)**2).sum()
    ss_cells = N_SAMPLES*N_REP*((g - gm)**2).values.sum()
    return dict(признак=col, соль=ss_s/tot, влажность=ss_m/tot, взаимодействие=(ss_cells-ss_s-ss_m)/tot,
                шум=1-ss_cells/tot)
FEATS_ANOVA = ["R_370_дБ", "R_255_дБ", "Ф_370_град", "Ф_255_град", "R255_минус_R370_дБ",
               "Наклон_R_дБ_на_100МГц", "Наклон_Ф_град_на_100МГц"]
anova = pd.DataFrame([eta2(c) for c in FEATS_ANOVA])
corr = pd.DataFrame([dict(признак=c,
                          r_с_солью=np.corrcoef(df[c], df["ECe_контроль_дСм_м"])[0,1],
                          r_с_влажностью=np.corrcoef(df[c], df["Влажность_факт"])[0,1]) for c in FEATS_ANOVA])

# ---------------- 4. машинное обучение ----------------
groups = df["ID_образца"].values           # повторы одного образца никогда не попадают и в обучение, и в тест
y_cls = df["Класс_засолённости"].values
SETS = {
    "A. Одна частота, только амплитуда (370 МГц)": ["R_370_дБ"],
    "B. Две частоты SNoOPI, амплитуда (255+370)": ["R_255_дБ", "R_370_дБ"],
    "C. Амплитуда по спектру 240–380 МГц": R_COLS,
    "D. Амплитуда + фаза по спектру": R_COLS + P_COLS,
}
def rf(): return RandomForestClassifier(n_estimators=400, random_state=0, min_samples_leaf=2)
ml = []
cv = GroupKFold(n_splits=3)
# схема групп: внутри каждого условия образцы 1,2,3 -> разные фолды
fold_id = ((df["ID_образца"]-1) % 3).values
preds = {}
for name, cols in SETS.items():
    X = df[cols].values
    p = cross_val_predict(rf(), X, y_cls, groups=fold_id, cv=cv)
    preds[name] = p
    # тест на переносимость: обучение на 3 уровнях влажности, тест на 4-м (по очереди)
    accs = []
    for mv in MV_LEVELS:
        tr = df["Влажность_номинал"] != mv
        m = rf().fit(X[tr], y_cls[tr]); accs.append(accuracy_score(y_cls[~tr], m.predict(X[~tr])))
    ml.append(dict(набор=name, точность_CV=accuracy_score(y_cls, p),
                   точность_на_новой_влажности=np.mean(accs), мин_по_влажности=np.min(accs)))
ml = pd.DataFrame(ml)

# регрессия ECe + одновременная оценка влажности на большом Монте-Карло наборе (непрерывные условия)
NMC = 3000
mc = []
for i in range(NMC):
    mv = rng.uniform(0.04, 0.38); s = rng.uniform(0, 5)
    rb = rng.normal(RHO_B, 0.03); rough = rng.uniform(0.003, 0.012); gain = rng.normal(0, 0.15)
    r = dict(mv=mv, salt=s, ECe=float(ece_estimate(s)))
    for f in FREQS:
        e = soil_eps(f, mv, s, rb); _,_,gc = fresnel(e)
        r[f"R_{int(f/1e6)}_дБ"] = 10*np.log10(abs(gc)**2*roughness(f, rough)) + gain + rng.normal(0, AMP_NOISE_DB)
        r[f"Ф_{int(f/1e6)}_град"] = np.degrees(np.angle(gc)) + rng.normal(0, PHASE_NOISE_DEG)
    mc.append(r)
mc = pd.DataFrame(mc)
tr = np.arange(NMC) < 2000
reg = []
reg_pred = {}
for name, cols in SETS.items():
    m = RandomForestRegressor(n_estimators=400, random_state=0, min_samples_leaf=3).fit(mc.loc[tr, cols], mc.loc[tr, "ECe"])
    p = m.predict(mc.loc[~tr, cols]); reg_pred[name] = p
    mm = RandomForestRegressor(n_estimators=300, random_state=0, min_samples_leaf=3).fit(mc.loc[tr, cols], mc.loc[tr, "mv"])
    pm = mm.predict(mc.loc[~tr, cols])
    reg.append(dict(набор=name, R2_ECe=r2_score(mc.loc[~tr,"ECe"], p), MAE_ECe_дСм_м=mean_absolute_error(mc.loc[~tr,"ECe"], p),
                    R2_влажность=r2_score(mc.loc[~tr,"mv"], pm)))
reg = pd.DataFrame(reg)

# важность признаков (набор D, классификация)
mD = rf().fit(df[R_COLS+P_COLS].values, y_cls)
imp = pd.DataFrame(dict(признак=R_COLS+P_COLS, важность=mD.feature_importances_)).sort_values("важность", ascending=False)
print("Доля важности: амплитуда", round(imp[imp.признак.str.startswith("R_")].важность.sum(),3), " фаза", round(imp[imp.признак.str.startswith("Ф_")].важность.sum(),3))

# ---------------- 5. якобиан: математическая идентифицируемость ----------------
def feat_vec(mv, s):
    out = []
    for f in [255e6, 370e6]:
        _,_,gc = fresnel(soil_eps(f, mv, s))
        out += [10*np.log10(abs(gc)**2), np.degrees(np.angle(gc))]
    return np.array(out)    # [R255, Ф255, R370, Ф370]
jac = []
for s in SALT_LEVELS:
    for mv in MV_LEVELS:
        h1, h2 = 1e-4, 1e-3
        J = np.column_stack([(feat_vec(mv+h1, s) - feat_vec(mv-h1, s))/(2*h1),
                             (feat_vec(mv, s+h2+BG_SALT*0) - feat_vec(mv, max(s-h2,0)))/((s+h2)-max(s-h2,0))])
        # нормировка на шум измерений -> «отношение сигнал/шум» чувствительности
        W = np.diag([1/AMP_NOISE_DB, 1/PHASE_NOISE_DEG, 1/AMP_NOISE_DB, 1/PHASE_NOISE_DEG])
        Jw = W @ J
        # только амплитуды двух частот
        Ja = Jw[[0,2], :]
        Fi = Jw.T @ Jw; cov = np.linalg.inv(Fi)
        Fa = Ja.T @ Ja; cova = np.linalg.inv(Fa) if abs(np.linalg.det(Fa)) > 1e-12 else np.full((2,2), np.inf)
        jac.append(dict(соль_г_кг=s, влажность=mv,
                        dR370_dсоль_дБ_на_г_кг=J[2,1], dФ370_dсоль_град_на_г_кг=J[3,1],
                        dR370_dвлажн_дБ_на_0_01=J[2,0]*0.01,
                        число_обусловленности=np.linalg.cond(Jw),
                        σ_соль_амплит_фаза_г_кг=np.sqrt(cov[1,1]), σ_соль_только_амплит_г_кг=np.sqrt(cova[1,1])))
jac = pd.DataFrame(jac)

# ---------------- сохранение ----------------
with pd.ExcelWriter(f"{OUT}/soil_расчётные_данные.xlsx") as w:
    pd.DataFrame(summary["levels"]).to_excel(w, sheet_name="Уровни засолённости", index=False)
    det.to_excel(w, sheet_name="16 условий (модель)", index=False)
    df.to_excel(w, sheet_name="240 измерений", index=False)
    anova.round(3).to_excel(w, sheet_name="ANOVA η²", index=False)
    corr.round(3).to_excel(w, sheet_name="Корреляции", index=False)
    ml.round(3).to_excel(w, sheet_name="ML классификация", index=False)
    reg.round(3).to_excel(w, sheet_name="ML регрессия", index=False)
    imp.round(4).to_excel(w, sheet_name="Важность признаков", index=False)
    jac.round(3).to_excel(w, sheet_name="Якобиан", index=False)
df.to_csv(f"{OUT}/soil_240_измерений.csv", index=False, encoding="utf-8-sig")

res = dict(summary=summary, anova=anova.round(3).to_dict("records"), corr=corr.round(3).to_dict("records"),
           ml=ml.round(3).to_dict("records"), reg=reg.round(3).to_dict("records"),
           imp=imp.head(6).round(4).to_dict("records"), jac=jac.round(3).to_dict("records"),
           det=det.to_dict("records"))
json.dump(res, open(f"{OUT}/results.json", "w"), ensure_ascii=False, indent=1, default=float)

# ---------------- графики ----------------
plt.rcParams.update({"contour.negative_linestyle": "solid", "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#9a9893", "figure.dpi": 150, "savefig.bbox": "tight"})
SEQ = ["#a9cbf2", "#5f9ee8", "#2a78d6", "#123f78"]        # засолённость: светлый -> тёмный
MVC = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]        # влажность: категориальные

# Рис.1 ε' и ε'' от солёности
sg = np.linspace(0, 5, 100)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for mv, c in zip(MV_LEVELS, MVC):
    e = soil_eps(370e6, mv, sg)
    ax[0].plot(sg, e.real, color=c, lw=2, label=f"влажность {mv:.2f}")
    ax[1].plot(sg, -e.imag, color=c, lw=2, label=f"влажность {mv:.2f}")
ax[0].set_title("Действительная часть ε′ (370 МГц)", loc="left"); ax[1].set_title("Мнимая часть ε″ — потери (370 МГц)", loc="left")
for a in ax: a.set_xlabel("Добавленная соль, г/кг сухой почвы")
ax[1].legend(frameon=False, fontsize=8)
fig.savefig(f"{OUT}/fig1_dielectric.png"); plt.close(fig)

# Рис.2 спектр отражения
fr = np.linspace(100e6, 2000e6, 300)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for a, mv in zip(ax, [0.10, 0.30]):
    for s, c, n in zip(SALT_LEVELS, SEQ, SALT_NAMES):
        _,_,gc = fresnel(soil_eps(fr, mv, s))
        a.plot(fr/1e6, 10*np.log10(abs(gc)**2), color=c, lw=2, label=f"{n} ({s} г/кг)")
    a.axvspan(240, 380, color="#eb6834", alpha=0.10, lw=0)
    a.text(400, -7.6, "← P-диапазон SNoOPI", fontsize=8, color="#52514e")
    a.set_title(f"Влажность {mv:.2f} м³/м³", loc="left"); a.set_xlabel("Частота, МГц")
ax[0].set_ylabel("Отражательная способность, дБ"); ax[1].legend(frameon=False, fontsize=8)
fig.savefig(f"{OUT}/fig2_spectrum.png"); plt.close(fig)

# Рис.3 неоднозначность и её разрешение
mvg = np.linspace(0.04, 0.38, 160); sgr = np.linspace(0, 5, 160)
MM, SS = np.meshgrid(mvg, sgr)
_,_,g370 = fresnel(soil_eps(370e6, MM, SS)); _,_,g255 = fresnel(soil_eps(255e6, MM, SS))
R370 = 10*np.log10(abs(g370)**2); P370 = np.degrees(np.angle(g370)); DR = 10*np.log10(abs(g255)**2) - R370
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
cs = ax[0].contour(MM, SS, R370, levels=10, colors="#2a78d6", linewidths=1.2)
ax[0].clabel(cs, fmt="%.1f дБ", fontsize=7)
ax[0].set_title("Только амплитуда 370 МГц:\nодно значение — много комбинаций", loc="left")
cs1 = ax[1].contour(MM, SS, R370, levels=10, colors="#2a78d6", linewidths=1.0)
cs2 = ax[1].contour(MM, SS, P370, levels=10, colors="#eb6834", linewidths=1.0)
ax[1].clabel(cs2, fmt="%.0f°", fontsize=7)
ax[1].set_title("Амплитуда (синие) + фаза (оранжевые):\nлинии пересекаются -> решение единственное", loc="left")
for a in ax: a.set_xlabel("Объёмная влажность, м³/м³"); a.set_ylabel("Соль, г/кг")
fig.savefig(f"{OUT}/fig3_ambiguity.png"); plt.close(fig)

# Рис.4 глубина проникновения
fig, ax = plt.subplots(figsize=(6, 3.8))
for s, c, n in zip(SALT_LEVELS, SEQ, SALT_NAMES):
    ax.plot(mvg, pen_depth(370e6, soil_eps(370e6, mvg, s))*100, color=c, lw=2, label=f"{n}")
ax.plot(mvg, pen_depth(F_L, soil_eps(F_L, mvg, 0))*100, color="#52514e", lw=1.5, ls="--", label="L-диапазон (GNSS), без соли")
ax.set_yscale("log"); ax.set_xlabel("Объёмная влажность, м³/м³"); ax.set_ylabel("Глубина проникновения, см")
ax.set_title("Глубина проникновения сигнала 370 МГц", loc="left"); ax.legend(frameon=False, fontsize=8)
fig.savefig(f"{OUT}/fig4_penetration.png"); plt.close(fig)

# Рис.5 виртуальный эксперимент: признаки по условиям
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
for a, col, ttl in zip(ax, ["R_370_дБ", "Ф_370_град", "Наклон_R_дБ_на_100МГц"],
                       ["Амплитуда 370 МГц, дБ", "Фаза 370 МГц, °", "Спектральный наклон, дБ/100 МГц"]):
    for si, (c, n) in enumerate(zip(SEQ, SALT_NAMES)):
        sub = df[df["Класс_засолённости"] == si]
        x = sub["Влажность_номинал"] + (si-1.5)*0.012
        a.scatter(x, sub[col], s=10, color=c, alpha=0.75, label=n, edgecolor="none")
    a.set_title(ttl, loc="left"); a.set_xlabel("Влажность, м³/м³"); a.set_xticks(MV_LEVELS)
ax[2].legend(frameon=False, fontsize=8, markerscale=1.5)
fig.savefig(f"{OUT}/fig5_virtual_experiment.png"); plt.close(fig)

# Рис.6 ML
fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw=dict(width_ratios=[1.3, 1]))
lbl = ["A. 1 частота\nамплитуда", "B. 255+370\nамплитуда", "C. спектр\nамплитуда", "D. спектр\nамплитуда+фаза"]
x = np.arange(4)
ax[0].bar(x-0.2, ml["точность_CV"], 0.38, color="#2a78d6", label="новые образцы (CV)")
ax[0].bar(x+0.2, ml["точность_на_новой_влажности"], 0.38, color="#eb6834", label="новая влажность")
ax[0].axhline(0.25, color="#52514e", ls=":", lw=1, label="случайное угадывание (25%)")
for i in range(4):
    ax[0].text(i-0.2, ml["точность_CV"][i]+0.02, f"{ml['точность_CV'][i]:.0%}", ha="center", fontsize=8)
    ax[0].text(i+0.2, ml["точность_на_новой_влажности"][i]+0.02, f"{ml['точность_на_новой_влажности'][i]:.0%}", ha="center", fontsize=8)
ax[0].set_xticks(x); ax[0].set_xticklabels(lbl, fontsize=8); ax[0].set_ylim(0, 1.12); ax[0].set_ylabel("Точность классификации")
ax[0].set_title("Классификация 4 уровней засолённости", loc="left"); ax[0].legend(frameon=False, fontsize=8, loc="upper left")
cm = confusion_matrix(y_cls, preds[list(SETS)[3]], normalize="true")
ax[1].imshow(cm, cmap="Blues", vmin=0, vmax=1); ax[1].grid(False)
for i in range(4):
    for j in range(4):
        ax[1].text(j, i, f"{cm[i,j]:.0%}", ha="center", va="center", fontsize=9, color="white" if cm[i,j] > 0.5 else "#0b0b0b")
ax[1].set_xticks(range(4)); ax[1].set_yticks(range(4)); ax[1].set_xticklabels(SALT_NAMES, fontsize=8); ax[1].set_yticklabels(SALT_NAMES, fontsize=8)
ax[1].set_xlabel("Предсказано"); ax[1].set_ylabel("Истинно"); ax[1].set_title("Матрица ошибок, набор D", loc="left")
fig.savefig(f"{OUT}/fig6_ml.png"); plt.close(fig)

# Рис.7 регрессия ECe
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for a, key in zip(ax, [list(SETS)[1], list(SETS)[3]]):
    a.scatter(mc.loc[~tr, "ECe"], reg_pred[key], s=6, color="#2a78d6", alpha=0.5, edgecolor="none")
    mx = mc["ECe"].max(); a.plot([0, mx], [0, mx], color="#52514e", lw=1, ls="--")
    r2 = reg.set_index("набор").loc[key, "R2_ECe"]
    a.set_title(f"{key.split('.')[0]}: R² = {r2:.2f}", loc="left"); a.set_xlabel("Истинная ECe, дСм/м")
ax[0].set_ylabel("Оценка модели, дСм/м")
fig.savefig(f"{OUT}/fig7_regression.png"); plt.close(fig)

print(pd.DataFrame(summary["levels"]))
print(det[["Засолённость","Влажность_м3м3","ε'_370МГц","ε''_370МГц","Отражение_дБ_370МГц","Фаза_град_370МГц","Глубина_проникн_см_370МГц"]].to_string())
print(anova.round(3).to_string()); print(corr.round(3).to_string())
print(ml.round(3).to_string()); print(reg.round(3).to_string()); print(imp.head(8).to_string())
print(jac.round(3).to_string()); print(confusion_matrix(y_cls, preds[list(SETS)[3]]))
