"""KB-ACE: knowledge-based, cost-aware evidence acquisition for phishing triage.

One script reproduces every number in the paper:
    python kbace.py                # full run (20 group-aware splits)
    python kbace.py --splits 3     # quick check

Data: UCI Phishing Websites (Mohammad, Thabtah & McCluskey), CC BY 4.0,
doi:10.24432/C51W2X, stored as data/phishing.csv (values -1/0/1, Result -1 = phishing).
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

ROOT = Path(__file__).parent
OUT = ROOT / "results"

# --------------------------------------------------------------------------
# 1. Knowledge base: the analyst-owned evidence ontology
#    tier      : 1 = URL string (no network), 2 = sandboxed page fetch + TLS,
#                3 = third-party registry / reputation look-ups
#    forgeable : attacker can set the value cheaply within a campaign
#    doc_pol   : documented polarity (+1: higher value = more legitimate,
#                0: no usable prior), from the dataset's feature documentation
# --------------------------------------------------------------------------
KB = {
    "having_IP_Address": (1, 1, 1), "URL_Length": (1, 1, 1), "Shortining_Service": (1, 1, 1),
    "having_At_Symbol": (1, 1, 1), "double_slash_redirecting": (1, 1, 1), "Prefix_Suffix": (1, 1, 1),
    "having_Sub_Domain": (1, 1, 1), "HTTPS_token": (1, 1, 1),
    "SSLfinal_State": (2, 0, 1), "Favicon": (2, 1, 1), "port": (2, 1, 1), "Request_URL": (2, 1, 1),
    "URL_of_Anchor": (2, 1, 1), "Links_in_tags": (2, 1, 1), "SFH": (2, 1, 1),
    "Submitting_to_email": (2, 1, 1), "Redirect": (2, 1, 0), "on_mouseover": (2, 1, 1),
    "RightClick": (2, 1, 1), "popUpWidnow": (2, 1, 1), "Iframe": (2, 1, 1),
    "Domain_registeration_length": (3, 1, 1), "Abnormal_URL": (3, 1, 1), "age_of_domain": (3, 0, 1),
    "DNSRecord": (3, 0, 1), "web_traffic": (3, 0, 1), "Page_Rank": (3, 0, 1), "Google_Index": (3, 0, 1),
    "Links_pointing_to_page": (3, 0, 1), "Statistical_report": (3, 0, 1),
}
BASE = dict(c_fp=1.0, c_fn=10.0, acq=(0.0, 0.02, 0.10))   # cost units: one false block = 1
L_GRID = [-1, .005, .01, .02, .05, .1, .2, .3]               # -1 = never accept early
U_GRID = [.5, .7, .8, .9, .95, .98, .99, 2]                  # 2 = never block early
T_GRID = np.round(np.linspace(.02, .98, 49), 2)


UCI_ZIP = "https://archive.ics.uci.edu/static/public/327/phishing+websites.zip"


def fetch_data(path):
    """Download the UCI archive once and convert 'Training Dataset.arff' to CSV."""
    import io, urllib.request, zipfile
    raw = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(UCI_ZIP).read())).read("Training Dataset.arff").decode()
    lines = raw.splitlines()
    cols = [l.split()[1] for l in lines if l.lower().startswith("@attribute")]
    start = next(i for i, l in enumerate(lines) if l.lower().startswith("@data")) + 1
    rows = [list(map(int, l.split(","))) for l in lines[start:] if l.strip() and not l.startswith("%")]
    path.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def load():
    path = ROOT / "data" / "phishing.csv"
    if not path.exists():
        fetch_data(path)
    d = pd.read_csv(path)
    feats = list(KB)
    X = d[feats].to_numpy(float)
    y = (d["Result"].to_numpy() == -1).astype(int)          # 1 = phishing
    groups = pd.factorize(d[feats].astype(str).agg("|".join, axis=1))[0]
    return X, y, groups, feats


def audit(X, y, feats, alpha=0.01):
    """Knowledge-data consistency audit: does the data agree with the documented polarity?"""
    rows = []
    for j, f in enumerate(feats):
        rho, p = stats.spearmanr(X[:, j], 1 - y)            # association with legitimacy
        pol = KB[f][2]
        if pol == 0 or p >= alpha:
            status = "no prior" if pol == 0 else "weak"
        else:
            status = "agree" if np.sign(rho) == pol else "conflict"
        rows.append(dict(feature=f, tier=KB[f][0], forgeable=KB[f][1], rho=rho, p=p, status=status))
    return pd.DataFrame(rows)


def hgb(cst, seed):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
                                          l2_regularization=1.0, early_stopping=False,
                                          monotonic_cst=cst, random_state=seed)


def augment(X, y, forg, rng):
    """Threat-informed augmentation: each phishing site gets one copy in which a random subset of its
    forgeable cues is spliced from a randomly drawn legitimate training site (label stays phishing)."""
    P, L = X[y == 1].copy(), X[y == 0]
    for row in P:
        k = rng.integers(1, len(forg) + 1)
        idx = rng.choice(forg, k, replace=False)
        row[idx] = L[rng.integers(len(L)), idx]
    return np.vstack([X, P]), np.concatenate([y, np.ones(len(P), int)])


class Stages:
    """Three nested evidence stages (T1, T1+T2, T1+T2+T3), each an HGB risk model."""

    def __init__(self, mono, aug, cols, cst, forg, legit_val, seed):
        self.mono, self.aug, self.cols, self.cst = mono, aug, cols, cst
        self.forg, self.legit_val, self.seed = forg, legit_val, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        if self.aug:
            X, y = augment(X, y, self.forg, rng)
        self.m = []
        for c in self.cols:
            cst = self.cst[c] if self.mono else None
            self.m.append(hgb(cst, self.seed).fit(X[:, c], y))
        return self

    def proba(self, X):
        return np.column_stack([m.predict_proba(X[:, c])[:, 1] for m, c in zip(self.m, self.cols)])


def oof(make, X, y, g, k=5):
    """Out-of-fold probabilities on the training split (thresholds never see test data)."""
    out = None
    for tr, va in GroupKFold(k).split(X, y, g):
        p = make().fit(X[tr], y[tr]).proba(X[va])
        out = np.zeros((len(y), p.shape[1])) if out is None else out
        out[va] = p
    return out


# --------------------------------------------------------------------------
# 2. Decision layer: cost-optimal cascade thresholds
# --------------------------------------------------------------------------
def cascade_table(P, y, anchored):
    """Pre-compute, for every threshold combination, the cost-independent counts."""
    Ls = [-1] if anchored else L_GRID
    combos, early_fp, early_fn, n2, R3 = [], [], [], [], []
    for l1 in Ls:
        for u1 in U_GRID:
            b1, a1 = P[:, 0] >= u1, (P[:, 0] <= l1) & (P[:, 0] < u1)
            r2 = ~(b1 | a1)
            for l2 in Ls:
                for u2 in U_GRID:
                    b2 = r2 & (P[:, 1] >= u2)
                    a2 = r2 & ~b2 & (P[:, 1] <= l2)
                    r3 = r2 & ~b2 & ~a2
                    combos.append((l1, u1, l2, u2))
                    early_fp.append(((b1 | b2) & (y == 0)).sum())
                    early_fn.append(((a1 | a2) & (y == 1)).sum())
                    n2.append(r2.sum())
                    R3.append(r3)
    R3 = np.array(R3, float)
    A = (P[:, [2]] >= T_GRID) & (y[:, None] == 0)            # stage-3 false blocks per t
    B = (P[:, [2]] < T_GRID) & (y[:, None] == 1)             # stage-3 misses per t
    return dict(combos=combos, fp=np.array(early_fp), fn=np.array(early_fn), n2=np.array(n2),
                n3=R3.sum(1), FP3=R3 @ A, FN3=R3 @ B, n=len(y))


def best_cascade(tab, cost):
    c1, c2, c3 = cost["acq"]
    tot = (cost["c_fp"] * (tab["fp"][:, None] + tab["FP3"]) + cost["c_fn"] * (tab["fn"][:, None] + tab["FN3"])
           + (tab["n"] * c1 + tab["n2"][:, None] * c2 + tab["n3"][:, None] * c3))
    i, t = np.unravel_index(np.argmin(tot), tot.shape)
    return tab["combos"][i] + (T_GRID[t],)


def run_cascade(P, th):
    l1, u1, l2, u2, t = th
    d = np.zeros(len(P), int); stage = np.ones(len(P), int)
    b1, a1 = P[:, 0] >= u1, (P[:, 0] <= l1) & (P[:, 0] < u1)
    r2 = ~(b1 | a1)
    b2 = r2 & (P[:, 1] >= u2); a2 = r2 & ~b2 & (P[:, 1] <= l2); r3 = r2 & ~b2 & ~a2
    d[b1 | b2] = 1; d[r3] = (P[r3, 2] >= t)
    stage[r2] = 2; stage[r3] = 3
    return d, stage


def best_t(p, y, cost):
    """Cost-minimising threshold over every distinct score (block if p >= t)."""
    o = np.argsort(-p); ps, ys = p[o], y[o]
    fp = np.concatenate([[0], np.cumsum(ys == 0)])          # blocking the top-i scores
    fn = (y == 1).sum() - np.concatenate([[0], np.cumsum(ys == 1)])
    tot = cost["c_fp"] * fp + cost["c_fn"] * fn
    cut = np.r_[ps[0] + 1, ps]                              # threshold for blocking top-i
    ok = np.r_[True, ps[1:] != ps[:-1], True]               # only cut between distinct scores
    i = np.flatnonzero(ok)[np.argmin(tot[ok])]
    return ps[i - 1] if i > 0 else ps[0] + 1


def score(d, y, stage, cost):
    acq = np.cumsum(cost["acq"])[stage - 1]
    fp, fn = ((d == 1) & (y == 0)).sum(), ((d == 0) & (y == 1)).sum()
    return dict(fpr=fp / (y == 0).sum(), fnr=fn / (y == 1).sum(), acc=(d == y).mean(),
                acq=acq.mean(), cost=(cost["c_fp"] * fp + cost["c_fn"] * fn + acq.sum()) / len(y),
                reach2=(stage >= 2).mean(), reach3=(stage == 3).mean())


def mono_violations(predict, X, agree_idx, rng, n=400):
    """Share of single-step moves towards documented-and-data-confirmed phishing evidence that lower the
    predicted risk by more than one percentage point."""
    S = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    base, viol, tot = predict(S), 0, 0
    for j in agree_idx:
        dom = np.unique(X[:, j]); Z = S.copy()
        can = Z[:, j] > dom.min()
        Z[can, j] = [dom[dom < v].max() for v in Z[can, j]]
        p = predict(Z)
        viol += (p[can] < base[can] - 0.01).sum(); tot += can.sum()
    return viol / max(tot, 1)


def mimic(X, y, order, legit_val, k):
    Z = X.copy()
    if k:
        Z[np.ix_(y == 1, order[:k])] = legit_val[order[:k]]
    return Z


# --------------------------------------------------------------------------
# 3. Experiment
# --------------------------------------------------------------------------
def main(n_splits, seed):
    t0 = time.time(); OUT.mkdir(exist_ok=True)
    X, y, g, feats = load()
    tier = np.array([KB[f][0] for f in feats]); forg = np.where([KB[f][1] for f in feats])[0]
    cols = [np.where(tier <= s)[0] for s in (1, 2, 3)]
    full_audit = audit(X, y, feats); full_audit.to_csv(OUT / "knowledge_audit.csv", index=False, float_format="%.6g")
    dup = dict(rows=len(y), unique_patterns=int(g.max() + 1),
               duplicate_rows=int(len(y) - (g.max() + 1)),
               conflicting_patterns=int(pd.Series(y).groupby(g).nunique().gt(1).sum()),
               phishing=int(y.sum()), legitimate=int((1 - y).sum()))
    variants = {"plain": (0, 0), "aug": (0, 1), "kb": (1, 1)}   # (monotone constraints, threat augmentation)
    policies = {  # name: (variant, trust-anchored)
        "KB-ACE": ("aug", 1), "KB-ACE + monotone constraints": ("kb", 1),
        "KB-ACE w/o trust anchoring": ("aug", 0), "KB-ACE w/o threat augmentation": ("plain", 1),
        "Cost-aware cascade": ("plain", 0)}
    r_grid, m_grid = [2, 5, 10, 20], [0.25, 0.5, 1, 2, 4]
    rows, curve, sens = [], [], []

    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=seed)
    for s, (tr, te) in enumerate(splitter.split(X, y, g)):
        Xtr, ytr, gtr, Xte, yte = X[tr], y[tr], g[tr], X[te], y[te]
        aud = audit(Xtr, ytr, feats)
        agree = np.where(aud.status == "agree")[0]
        cst = np.where(aud.status == "agree", -1, 0)        # higher value -> lower phishing risk
        # attacker's value map: for every cue, the value with the highest legitimacy rate in training data
        legit_val = np.array([pd.Series(1 - ytr).groupby(Xtr[:, j]).mean().idxmax() for j in range(X.shape[1])])
        mi = mutual_info_classif(Xtr[:, forg], ytr, discrete_features=True, random_state=seed)
        order = forg[np.argsort(-mi)]                       # attacker forges strongest cues first
        rng = np.random.default_rng(seed + s)
        mk = lambda v: (lambda: Stages(*variants[v], cols, cst, forg, legit_val, seed + s))
        fitted, P_oof, P_te = {}, {}, {}
        for v in variants:
            P_oof[v] = oof(mk(v), Xtr, ytr, gtr)
            fitted[v] = mk(v)().fit(Xtr, ytr)
            P_te[v] = fitted[v].proba(Xte)
        tabs = {(v, a): cascade_table(P_oof[v], ytr, a) for v, a in set(policies.values())}

        # single-model baselines (full evidence unless stated)
        lr = LogisticRegression(max_iter=2000, C=1.0)
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed + s)
        singles = {}
        for name, mdl in (("Logistic regression", lr), ("Random forest", rf)):
            po = np.zeros(len(ytr))
            for a, b in GroupKFold(5).split(Xtr, ytr, gtr):
                po[b] = mdl.fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]
            m = mdl.fit(Xtr, ytr)
            singles[name] = (po, lambda Z, m=m: m.predict_proba(Z)[:, 1], 3)
        singles["Gradient boosting (all evidence)"] = (P_oof["plain"][:, 2], lambda Z: fitted["plain"].proba(Z)[:, 2], 3)
        singles["Gradient boosting (URL only)"] = (P_oof["plain"][:, 0], lambda Z: fitted["plain"].proba(Z)[:, 0], 1)
        sc = lambda Z: 1 / (1 + np.exp(np.clip(Z, -1, 1) @ np.array([KB[f][2] for f in feats], float)))
        singles["Expert scorecard (rules only)"] = (sc(Xtr), sc, 3)

        def evaluate(cost, record):
            res = {}
            Za = mimic(Xte, yte, order, legit_val, len(forg))    # every phishing site forges every cue
            for name, (po, f, st) in singles.items():
                t = best_t(po, ytr, cost)
                p = f(Xte); d = (p >= t).astype(int)
                r = score(d, yte, np.full(len(yte), st), cost)
                r["auc"] = roc_auc_score(yte, p)
                r["cost_att"] = score((f(Za) >= t).astype(int), yte, np.full(len(yte), st), cost)["cost"]
                if record:
                    for k in range(len(forg) + 1):
                        Z = mimic(Xte, yte, order, legit_val, k)
                        curve.append(dict(split=s, policy=name, k=k, recall=((f(Z) >= t)[yte == 1]).mean()))
                    r["viol"] = mono_violations(f, Xte, agree, rng)
                res[name] = r
            for name, (v, a) in policies.items():
                th = best_cascade(tabs[(v, a)], cost)
                d, stg = run_cascade(P_te[v], th)
                r = score(d, yte, stg, cost); r["auc"] = np.nan
                da, sa = run_cascade(fitted[v].proba(Za), th)
                r["cost_att"] = score(da, yte, sa, cost)["cost"]
                if record:
                    for k in range(len(forg) + 1):
                        Z = mimic(Xte, yte, order, legit_val, k)
                        dz, _ = run_cascade(fitted[v].proba(Z), th)
                        curve.append(dict(split=s, policy=name, k=k, recall=dz[yte == 1].mean()))
                    r["viol"] = mono_violations(lambda Z: fitted[v].proba(Z)[:, 2], Xte, agree, rng)
                    r.update(dict(zip(["l1", "u1", "l2", "u2", "t"], th)))
                res[name] = r
            return res

        for name, r in evaluate(BASE, True).items():
            rows.append(dict(split=s, policy=name, **r))
        for rr in r_grid:
            for mm in m_grid:
                cost = dict(c_fp=1.0, c_fn=float(rr), acq=tuple(mm * c for c in BASE["acq"]))
                for name, r in evaluate(cost, False).items():
                    if name in ("KB-ACE", "Cost-aware cascade", "Gradient boosting (all evidence)",
                                "Random forest"):
                        sens.append(dict(split=s, policy=name, r=rr, m=mm, cost=r["cost"], cost_att=r["cost_att"],
                                         fpr=r["fpr"], fnr=r["fnr"], acq=r["acq"]))
        print(f"split {s + 1}/{n_splits} done  ({time.time() - t0:.0f}s)", flush=True)

    per = pd.DataFrame(rows); per.to_csv(OUT / "per_split.csv", index=False, float_format="%.6g")
    cur = pd.DataFrame(curve); cur.to_csv(OUT / "attack_curve.csv", index=False, float_format="%.6g")
    sen = pd.DataFrame(sens); sen.to_csv(OUT / "sensitivity.csv", index=False, float_format="%.6g")
    full_k = len(forg)
    att = cur[cur.k == full_k].groupby(["split", "policy"]).recall.first().unstack()
    att10 = cur[cur.k == 10].groupby(["split", "policy"]).recall.first().unstack()

    # paired tests: KB-ACE vs every other policy (Wilcoxon + Holm, Nadeau-Bengio corrected t)
    def nb_t(diff, n_tr, n_te):
        J = len(diff); var = diff.var(ddof=1)
        tstat = diff.mean() / np.sqrt((1 / J + n_te / n_tr) * var) if var > 0 else np.inf
        return 2 * stats.t.sf(abs(tstat), J - 1)
    cost_tab = per.pivot(index="split", columns="policy", values="cost")
    tests = []
    n_te = len(y) * 0.3; n_tr = len(y) - n_te
    for metric, tab in (("expected cost", cost_tab), ("recall, full mimicry", att), ("recall, 10 forged cues", att10)):
        for other in tab.columns.drop("KB-ACE"):
            diff = (tab["KB-ACE"] - tab[other]).to_numpy()
            pw = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            tests.append(dict(metric=metric, comparison=f"KB-ACE vs {other}", kb=tab["KB-ACE"].mean(),
                              other=tab[other].mean(), diff=diff.mean(), p_wilcoxon=pw, p_nb=nb_t(diff, n_tr, n_te)))
    tests = pd.DataFrame(tests)
    for m in tests.metric.unique():                        # Holm within each metric family
        idx = tests.index[tests.metric == m]; p = tests.loc[idx, "p_wilcoxon"].to_numpy()
        o = np.argsort(p); adj = np.maximum.accumulate(np.minimum(1, (len(p) - np.arange(len(p))) * p[o]))
        tests.loc[idx[o], "p_holm"] = adj
    tests.to_csv(OUT / "paired_tests.csv", index=False, float_format="%.6g")

    summ = per.groupby("policy").agg(["mean", "std"]).drop(columns="split")
    summ.columns = ["_".join(c) for c in summ.columns]
    summ["recall_mimicry_mean"] = att.mean(); summ["recall_mimicry_std"] = att.std()
    summ["recall_k10_mean"] = att10.mean()
    summ.to_csv(OUT / "summary.csv", float_format="%.6g")
    json.dump(dict(attack_order=[feats[j] for j in order], attack_values={feats[j]: float(legit_val[j]) for j in forg},
                   data=dup, splits=n_splits, seed=seed, costs=BASE, forgeable=len(forg),
                   audit=full_audit.status.value_counts().to_dict(),
                   conflicts=full_audit.query("status=='conflict'").feature.tolist(),
                   runtime_s=round(time.time() - t0)), open(OUT / "run_summary.json", "w"), indent=2)
    print(summ[["cost_mean", "fpr_mean", "fnr_mean", "acq_mean", "reach3_mean", "recall_mimicry_mean", "viol_mean"]].round(4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    main(a.splits, a.seed)
