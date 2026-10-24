"""Builds Figures 1-2, Tables 1-2 and the graphical abstract from results/*.csv (run kbace.py first)."""
from pathlib import Path
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).parent; R = ROOT / "results"; F = ROOT / "figures"; F.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titlesize": 10,
                     "axes.titleweight": "bold", "savefig.dpi": 600, "pdf.fonttype": 42})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# categorical slots in fixed order (validated default palette), plus a neutral for references
STYLE = {  # policy: (colour, linestyle, marker, short label)
    "KB-ACE": ("#2a78d6", "-", "o", "KB-ACE (proposed)"),
    "Cost-aware cascade": ("#eb6834", "--", "s", "Cost-aware cascade"),
    "Gradient boosting (all evidence)": ("#1baf7a", "-.", "^", "Gradient boosting, all evidence"),
    "Random forest": ("#eda100", ":", "D", "Random forest, all evidence"),
    "KB-ACE w/o trust anchoring": ("#6250d6", (0, (5, 1, 1, 1)), "v", "KB-ACE without trust anchoring"),
}
ORDER = ["Expert scorecard (rules only)", "Logistic regression", "Random forest", "Gradient boosting (URL only)",
         "Gradient boosting (all evidence)", "Cost-aware cascade", "KB-ACE"]
ABL = ["KB-ACE", "KB-ACE + monotone constraints", "KB-ACE w/o trust anchoring", "KB-ACE w/o threat augmentation",
       "Cost-aware cascade"]


def save(fig, name):
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight"); fig.savefig(F / f"{name}.png", bbox_inches="tight")
    fig.savefig(F / f"{name}.tiff", bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


per = pd.read_csv(R / "per_split.csv"); cur = pd.read_csv(R / "attack_curve.csv")
sen = pd.read_csv(R / "sensitivity.csv"); aud = pd.read_csv(R / "knowledge_audit.csv")
tests = pd.read_csv(R / "paired_tests.csv"); run = json.load(open(R / "run_summary.json"))
K = run["forgeable"]
full_att = cur[cur.k == K].groupby(["policy"]).recall.agg(["mean", "std"])

# ---------------------------------------------------------------- Figure 1
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw=dict(wspace=0.30))
a = ax[0]
for p, (c, ls, mk, lab) in STYLE.items():
    g = cur[cur.policy == p].groupby("k").recall.agg(["mean", "std"])
    a.fill_between(g.index, (g["mean"] - g["std"]).clip(0), (g["mean"] + g["std"]).clip(None, 1), color=c, alpha=.12, lw=0)
    a.plot(g.index, g["mean"], color=c, ls=ls, lw=1.8, marker=mk, ms=4, markevery=4, label=lab)
a.set(xlabel="Forged cues per phishing site (k)", ylabel="Phishing recall",
      xlim=(0, K), ylim=(-0.02, 1.02)); a.grid(axis="y", color=GRID, lw=.6)
a.set_title("(a) Evasion by cue forgery", loc="left")
fig.legend(*a.get_legend_handles_labels(), frameon=False, fontsize=7.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13))

b = ax[1]
pis = np.linspace(0, 0.1, 201)
cm = per.groupby("policy")[["cost", "cost_att"]].mean()
for p, (c, ls, mk, lab) in STYLE.items():
    if p == "KB-ACE w/o trust anchoring":
        continue
    y = (1 - pis) * cm.loc[p, "cost"] + pis * cm.loc[p, "cost_att"]
    b.plot(pis * 100, y, color=c, ls=ls, lw=1.8, label=lab)
rivals = ["Cost-aware cascade", "Gradient boosting (all evidence)", "Random forest"]
def breakeven(kb, rv):
    dc, da = kb["cost"] - rv["cost"], kb["cost_att"] - rv["cost_att"]
    return 0.0 if dc <= 0 else dc / (dc - da)
pstar = max(breakeven(cm.loc["KB-ACE"], cm.loc[r]) for r in rivals)
yb = (1 - pstar) * cm.loc["KB-ACE", "cost"] + pstar * cm.loc["KB-ACE", "cost_att"]
b.axvline(pstar * 100, color=MUTED, lw=.8, ls=":")
b.annotate(f"break-even: {pstar * 100:.1f}%", xy=(pstar * 100, yb), xytext=(pstar * 100 + 1.2, 0.08),
           fontsize=7.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=.7))
b.set(xlabel="Phishing sites forging all cues (%)", ylabel="Expected cost per URL (false hold = 1)",
      xlim=(0, 10), ylim=(0, 0.75)); b.grid(axis="y", color=GRID, lw=.6)
b.set_title("(b) Expected decision cost under attack", loc="left")
save(fig, "Figure_1")

# ---------------------------------------------------------------- Figure 2
fig, ax = plt.subplots(1, 2, figsize=(7.2, 4.4), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.55))
a = ax[0]
col = {"agree": "#2a78d6", "conflict": "#e34948", "weak": "#b8b6ae", "no prior": "#b8b6ae"}
d = aud.sort_values(["tier", "rho"]).reset_index(drop=True)
ypos = np.arange(len(d))[::-1] + d.tier.map({1: 0, 2: 1, 3: 2}).to_numpy()[::-1] * 0  # placeholder
gap = d.tier.map({1: 2.0, 2: 1.0, 3: 0.0}).to_numpy()
ypos = np.arange(len(d))[::-1] + gap
for i, r in d.iterrows():
    a.barh(ypos[i], r.rho, color=col[r.status], height=.72, hatch="////" if r.forgeable else None,
           edgecolor="white", lw=0)
a.set_yticks(ypos); a.set_yticklabels(d.feature.str.replace("_", " "), fontsize=6.3)
a.axvline(0, color=MUTED, lw=.7)
for t, lab in ((1, "Tier 1: URL string"), (2, "Tier 2: page fetch"), (3, "Tier 3: registry/reputation")):
    yy = ypos[d.tier.to_numpy() == t]
    a.text(0.98, yy.max() + .55, lab, fontsize=7, color=MUTED, ha="right", transform=a.get_yaxis_transform())
a.set(xlabel="Spearman ρ with legitimacy (documented direction: +)", ylim=(-1, ypos.max() + 1.6))
a.grid(axis="x", color=GRID, lw=.6)
from matplotlib.patches import Patch
a.legend(handles=[Patch(color=col["agree"], label="agrees with documentation"),
                  Patch(color=col["conflict"], label="contradicts documentation"),
                  Patch(color=col["weak"], label="weak / no documented prior"),
                  Patch(facecolor="white", edgecolor=MUTED, hatch="////", label="forgeable by attacker")],
         frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.42, -0.1), ncol=2)
a.set_title("(a) Knowledge–data audit of 30 cues", loc="left")

b = ax[1]
rs, ms = sorted(sen.r.unique()), sorted(sen.m.unique())
mean = sen.groupby(["policy", "r", "m"])[["cost", "cost_att"]].mean()
Z = np.zeros((len(rs), len(ms)))
for i, rr in enumerate(rs):
    for j, mm in enumerate(ms):
        kb = mean.loc[("KB-ACE", rr, mm)]
        Z[i, j] = max(breakeven(kb, mean.loc[(p, rr, mm)]) for p in rivals) * 100
blues = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
im = b.imshow(Z, cmap=blues, aspect="auto", origin="lower", vmin=0, vmax=max(5, np.ceil(Z.max())))
for i in range(len(rs)):
    for j in range(len(ms)):
        b.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center", fontsize=7.5,
               color="white" if Z[i, j] > 0.55 * im.get_clim()[1] else INK)
b.set_xticks(range(len(ms))); b.set_xticklabels([f"×{m:g}" for m in ms], fontsize=7.5)
b.set_yticks(range(len(rs))); b.set_yticklabels([f"{r:g}:1" for r in rs])
b.set(xlabel="Evidence-acquisition cost multiplier", ylabel="Miss : false-hold cost ratio")
for s in b.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=b, fraction=.05, pad=.03); cb.set_label("Break-even forging share (%)", fontsize=7.5)
cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)
b.set_title("(b) Break-even attack share", loc="left")
save(fig, "Figure_2")

# ---------------------------------------------------------------- Tables
def ms_(m, s, f="{:.3f}"):
    return f"{f.format(m)} ± {f.format(s)}"
g = per.groupby("policy"); mu, sd = g.mean(numeric_only=True), g.std(numeric_only=True)
t1 = []
for p in ORDER:
    t1.append({"Policy": p, "ROC-AUC": "—" if np.isnan(mu.loc[p, "auc"]) else ms_(mu.loc[p, "auc"], sd.loc[p, "auc"]),
               "FPR": ms_(mu.loc[p, "fpr"], sd.loc[p, "fpr"]), "FNR": ms_(mu.loc[p, "fnr"], sd.loc[p, "fnr"]),
               "Acquisition cost": f"{mu.loc[p, 'acq']:.3f}", "Expected cost (clean)": ms_(mu.loc[p, "cost"], sd.loc[p, "cost"]),
               "Recall, full forgery": ms_(full_att.loc[p, "mean"], full_att.loc[p, "std"]),
               "Expected cost (full forgery)": f"{mu.loc[p, 'cost_att']:.3f}",
               "Knowledge violations": f"{mu.loc[p, 'viol']:.3f}"})
pd.DataFrame(t1).to_csv(F / "Table_1.csv", index=False)
t2 = []
for p in ABL:
    t2.append({"Configuration": p, "Trust anchoring": "✓" if p in ("KB-ACE", "KB-ACE + monotone constraints", "KB-ACE w/o threat augmentation") else "–",
               "Threat augmentation": "✓" if p in ("KB-ACE", "KB-ACE + monotone constraints", "KB-ACE w/o trust anchoring") else "–",
               "Monotone constraints": "✓" if p == "KB-ACE + monotone constraints" else "–",
               "Reach tier 3": f"{mu.loc[p, 'reach3']:.3f}", "FPR": f"{mu.loc[p, 'fpr']:.3f}",
               "Expected cost (clean)": ms_(mu.loc[p, "cost"], sd.loc[p, "cost"]),
               "Recall, full forgery": ms_(full_att.loc[p, "mean"], full_att.loc[p, "std"]),
               "Expected cost (full forgery)": f"{mu.loc[p, 'cost_att']:.3f}",
               "Knowledge violations": f"{mu.loc[p, 'viol']:.3f}"})
pd.DataFrame(t2).to_csv(F / "Table_2.csv", index=False)
json.dump({"breakeven_pct": pstar * 100, "breakeven_grid": Z.round(2).tolist(), "r": [float(x) for x in rs], "m": [float(x) for x in ms]},
          open(R / "breakeven.json", "w"), indent=2)

# ---------------------------------------------------------------- Graphical abstract (schematic, drawn from results)
fig = plt.figure(figsize=(13.3, 5.3)); a = fig.add_axes([0, 0, 1, 1]); a.set_xlim(0, 100); a.set_ylim(0, 40); a.axis("off")
def box(x, y, w, h, title, body, fc, ec):
    a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=1.4))
    a.text(x + w / 2, y + h - 1.8, title, ha="center", va="top", fontsize=12, weight="bold", color=INK)
    a.text(x + w / 2, y + h - 5.0, body, ha="center", va="top", fontsize=9.5, color=MUTED, linespacing=1.4)
a.text(50, 38.2, "KB-ACE: knowledge-based, cost-aware evidence acquisition for phishing triage", ha="center",
       fontsize=15, weight="bold", color=INK)
box(1.5, 20, 17, 14.5, "Knowledge base", "evidence tier & cost\nforgeability tag\ndocumented polarity\n(audited against data)", "#f3f2ef", MUTED)
tiers = [("Tier 1 · URL string", "cost 0 · 8 cues"), ("Tier 2 · page fetch", "cost 0.02 · 13 cues"),
         ("Tier 3 · reputation", "cost 0.10 · 9 cues")]
for i, (t, bdy) in enumerate(tiers):
    x = 24 + i * 19
    box(x, 22, 15.5, 10.5, t, bdy + ("\nmay HOLD, never RELEASE" if i < 2 else "\nmay HOLD or RELEASE"),
        "#e8f1fc", "#2a78d6")
    if i < 2:
        a.annotate("", xy=(x + 18.4, 27.2), xytext=(x + 15.9, 27.2), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
        a.text(x + 17.15, 28.2, "unsure", ha="center", fontsize=8, color=MUTED)
a.annotate("", xy=(23.7, 27.2), xytext=(19.1, 27.2), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
box(82.5, 20, 16, 14.5, "Learners", "gradient boosting\nper evidence tier\n+ threat-informed\naugmentation", "#f3f2ef", MUTED)
a.annotate("", xy=(78.8, 27.2), xytext=(82.3, 27.2), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
kb, cc, gb = (per[per.policy == p] for p in ("KB-ACE", "Cost-aware cascade", "Gradient boosting (all evidence)"))
facts = [(f"{full_att.loc['Gradient boosting (all evidence)', 'mean'] * 100:.0f}% → {full_att.loc['KB-ACE', 'mean'] * 100:.0f}%",
          "phishing recall when every\nforgeable cue is faked\n(gradient boosting → KB-ACE)"),
         (f"{(1 - kb.acq.mean() / gb.acq.mean()) * 100:.0f}% less",
          "evidence-acquisition spend than\nalways querying every source"),
         (f"{pstar * 100:.1f}%", "share of forging phishing at which\nKB-ACE becomes the cheapest policy"),
         (f"{run['data']['duplicate_rows']:,} / {run['data']['rows']:,}", "rows were exact duplicates:\ngroup-aware splits avoid leakage")]
for i, (big, small) in enumerate(facts):
    x = 2 + i * 24.5
    a.add_patch(FancyBboxPatch((x, 2), 22.5, 13.5, boxstyle="round,pad=0.25,rounding_size=1.2", fc="white", ec=GRID, lw=1.2))
    a.text(x + 11.25, 12.6, big, ha="center", fontsize=17, weight="bold", color="#2a78d6")
    a.text(x + 11.25, 10.3, small, ha="center", va="top", fontsize=9.3, color=MUTED, linespacing=1.35)
a.text(50, 17.2, f"UCI Phishing Websites · {run['data']['rows']:,} sites · {run['splits']} group-aware splits · "
       "cost ratio miss : false hold = 10 : 1", ha="center", fontsize=9.5, color=MUTED)
fig.savefig(F / "Graphical_Abstract.pdf"); fig.savefig(F / "Graphical_Abstract.png", dpi=300)
fig.savefig(F / "Graphical_Abstract.tiff", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
print("figures and tables written to", F)
