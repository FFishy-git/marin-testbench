"""Every number the Muon re-baselining needs, recomputed from archived runs."""
import gzip, json
from pathlib import Path
import numpy as np

IDEA = Path("/root/dl/runtime/script/marin-testbench/optimizer_workspace/ideas/output-side-normalization")
REC, RES = IDEA / "experiment_records", IDEA / "paper" / "results"
CAMP = "campaigns/confirmation-20261121/runs"


def series(run, field="selection_loss"):
    with gzip.open(REC / f"{CAMP}/{run}/metrics_from_000000.jsonl.gz", "rt") as f:
        rows = [json.loads(l) for l in f]
    pts = [(int(r["step"]), float(r[field])) for r in rows if r.get(field) is not None]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def pal(name):
    d = json.load(open(REC / f"paloma_eval/{name}.json"))
    return d["paloma_macro_mean"], {k: v for k, v in d["scores"].items() if k.startswith("paloma/")}


def phase(title, runs, pal_names, late_from):
    print("=" * 94)
    print(title)
    base = runs["Muon"]
    st_b, v_b = series(base)
    late_b = v_b[st_b >= late_from]
    macro, dom = {}, {}
    for k, n in pal_names.items():
        macro[k], dom[k] = pal(n)
    print(f"  {'arm':12s} {'final sel':>10s} {'d vs Muon':>10s} {'d vs MuonT':>11s} "
          f"{'late mean d vs Muon (wins)':>28s} {'paloma d vs Muon (dom)':>24s}")
    for k, run in runs.items():
        st, v = series(run)
        late = v[st >= late_from]
        n = min(len(late), len(late_b))
        dlate = late_b[-n:] - late[-n:]
        wins = int((dlate > 0).sum())
        dsel_mu = (v_b[-1] - v[-1]) * 1e3
        dsel_mt = (series(runs["MuonT"])[1][-1] - v[-1]) * 1e3
        if k in macro:
            dwin = sum(1 for kk in dom["Muon"] if dom[k][kk] < dom["Muon"][kk])
            ptxt = f"{(macro['Muon'] - macro[k]) * 1e3:+7.2f} ({dwin}/{len(dom['Muon'])})"
        else:
            ptxt = "---"
        print(f"  {k:12s} {v[-1]:10.4f} {dsel_mu:+10.2f} {dsel_mt:+11.2f} "
              f"{np.mean(dlate) * 1e3:+16.2f} ({wins}/{n}) {ptxt:>24s}")
    print("  (deltas x1e-3 nats; positive = better than the reference)")
    print()


phase("Phase 3: 130M 4x Chinchilla (80 TPP, seed 0, eta = 0.01056), late window from step 10241",
      {"Muon": "wave3a/muon_130m_4x_seed0",
       "MuonT": "wave3a/muont_130m_4x_seed0",
       "OD-All": "wave3b/od_all_130m_4x_seed0_lr0p01056",
       "WT-MLP-Lag": "wave3b/wt_mlp_lag_130m_4x_seed0_lr0p01056"},
      {"Muon": "muon_130m_4x_seed0", "MuonT": "muont_130m_4x_seed0",
       "OD-All": "od_all_130m_4x_seed0_lr0p01056", "WT-MLP-Lag": "wt_mlp_lag_130m_4x_seed0_lr0p01056"},
      10241)

phase("Phase 4: 520M 1x Chinchilla (d=1024, seed 0), late window from step 5001",
      {"Muon": "wave4a/muon_520m_1x_seed0",
       "MuonT": "wave4a/muont_520m_1x_seed0",
       "OD-All": "wave4a/od_all_520m_1x_seed0",
       "WT-MLP-Lag": "wave4a/wt_mlp_lag_520m_1x_seed0"},
      {"Muon": "muon_520m_1x_seed0", "MuonT": "muont_520m_1x_seed0",
       "OD-All": "od_all_520m_1x_seed0", "WT-MLP-Lag": "wt_mlp_lag_520m_1x_seed0"},
      5001)

print("=" * 94)
print("Phase 2: 130M 1x, fresh seeds 1-3 (paired, x1e-3 nats, + = better than Muon)")
p2 = json.load(open(RES / "confirmation_campaign.json"))["phase2_3seed"]
mu = {m: np.array(p2["muon"][f"{m}_seeds"]) for m in ("sel", "c4", "paloma")}
for arm in ("muont", "od_all", "wt_mlp_lag"):
    out = []
    for m in ("sel", "c4", "paloma"):
        d = (mu[m] - np.array(p2[arm][f"{m}_seeds"])) * 1e3
        out.append(f"{m} {d.mean():+7.2f} +- {d.std(ddof=1):4.2f} ({int((d > 0).sum())}/3)")
    print(f"  {arm:11s} " + "   ".join(out))

print()
print("=" * 94)
print("Phase 1: 130M 1x seed 0, matched LR (absolute finals from the campaign record)")
runs = json.load(open(RES / "confirmation_campaign.json"))["runs"]
for lr, names in (("0.016", {"Muon": "ref_muon_0p016", "MuonT": "wave1a/muont_130m_1x_seed0_lr0p016",
                             "WT-MLP-Lag": "wave1a/wt_mlp_lag_130m_1x_seed0_lr0p016"}),
                  ("0.02263", {"Muon": "wave1b/muon_130m_1x_seed0_lr0p0226", "MuonT": "ref_muont_0p0226",
                               "Null-WT": "wave1b/null_wt_130m_1x_seed0_lr0p0226",
                               "OD-All": "wave1b/od_all_130m_1x_seed0_lr0p0226", "WT-MLP-Lag": "ref_wt_0p0226"})):
    print(f"  matched eta = {lr}")
    b = runs[names["Muon"]]
    for k, n in names.items():
        r = runs[n]
        print(f"    {k:11s} final_sel {r['final_sel']:.4f} ({(b['final_sel'] - r['final_sel']) * 1e3:+6.2f} vs Muon)"
              f"   final_c4 {r['final_c4']:.4f} ({(b['final_c4'] - r['final_c4']) * 1e3:+6.2f} vs Muon)")
