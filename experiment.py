"""
Production v3: All reviewer bugs fixed.
Bug 1: Missing feature now hides from model only, not from DGP
Bug 2: Two-step now re-estimates theta using feature-predicted difficulty
Bug 3: Stress tests crossed with sparsity
Bug 4: Portable paths
"""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from scipy.stats import spearmanr
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.family':'serif','font.size':10,
    'pdf.fonttype':42,'ps.fonttype':42})
import json
np.random.seed(42)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

TW = np.array([1.5, 0.8, 0.6, 1.2, 0.3])
FNAMES = ['reasoning_depth','domain_spec','prompt_complex','answer_ambig','context_len']
NM, NI, NS = 25, 100, 5

# ═══ Core functions ═══
def gen_data(nm, ni, tw, sp=1., noise_feats=0, nonlinear=False, seed=42):
    """Generate benchmark data. Difficulty ALWAYS uses all true weights."""
    np.random.seed(seed)
    ta = np.linspace(-2.5, 2.5, nm); np.random.shuffle(ta)
    nf = len(tw)
    F = np.random.rand(ni, nf)
    td = F @ tw
    if nonlinear:
        td += 0.8 * F[:, 0] * F[:, 3]
    td -= td.mean()
    td += np.random.normal(0, 0.3, ni)
    if noise_feats > 0:
        F = np.concatenate([F, np.random.rand(ni, noise_feats)], axis=1)
    disc = np.random.uniform(0.5, 2.0, ni)
    pr = expit(disc[None, :] * (ta[:, None] - td[None, :]))
    R = (np.random.rand(nm, ni) < pr).astype(float)
    if sp < 1.:
        m = np.random.rand(nm, ni) < sp
        Rs = R.copy(); Rs[~m] = np.nan
    else:
        Rs = R
    return Rs, R, ta, td, F, disc

def simple_avg(R):
    with np.errstate(all='ignore'):
        return np.nan_to_num(np.nanmean(R, axis=1), nan=0.5)

def fit_irt(R, iters=10):
    n_p, n_i = R.shape; has_nan = np.any(np.isnan(R))
    theta = np.clip(np.log((np.nanmean(R,1)+.01)/(1-np.nanmean(R,1)+.01)), -4, 4)
    b = np.clip(-np.log((np.nanmean(R,0)+.01)/(1-np.nanmean(R,0)+.01)), -4, 4)
    a = np.ones(n_i) * 0.7
    for _ in range(iters):
        for j in range(n_i):
            v = ~np.isnan(R[:,j]) if has_nan else np.ones(n_p, bool)
            if v.sum() < 3: continue
            Rj, tj = R[v,j], theta[v]
            def oi(p):
                if p[0] < .01: return 1e10
                z = p[0]*(tj-p[1]); pr = np.clip(expit(z), 1e-8, 1-1e-8)
                return -np.sum(Rj*np.log(pr)+(1-Rj)*np.log(1-pr))
            r = minimize(oi, [a[j],b[j]], method='Nelder-Mead', options={'maxiter':30})
            a[j] = max(.01, min(5, r.x[0])); b[j] = np.clip(r.x[1], -5, 5)
        for i in range(n_p):
            v = ~np.isnan(R[i,:]) if has_nan else np.ones(n_i, bool)
            if v.sum() < 3: continue
            Ri, ai, bi = R[i,v], a[v], b[v]
            def ot(t):
                z = ai*(t[0]-bi); pr = np.clip(expit(z), 1e-8, 1-1e-8)
                return -np.sum(Ri*np.log(pr)+(1-Ri)*np.log(1-pr))
            r = minimize(ot, [theta[i]], method='Nelder-Mead', options={'maxiter':30})
            theta[i] = np.clip(r.x[0], -5, 5)
    return theta, a, b

def fit_lltm(R, F, iters=10, reg=0.01):
    n_p, n_i = R.shape; nf = F.shape[1]; has_nan = np.any(np.isnan(R))
    Fm = F.mean(0); Fs = F.std(0)+1e-8; Fn = (F-Fm)/Fs
    theta = np.clip(np.log((np.nanmean(R,1)+.01)/(1-np.nanmean(R,1)+.01)), -4, 4)
    w = np.zeros(nf)
    for _ in range(iters):
        for i in range(n_p):
            v = ~np.isnan(R[i,:]) if has_nan else np.ones(n_i, bool)
            if v.sum() < 3: continue
            Ri, Fi = R[i,v], Fn[v]
            def ot(t):
                z = t[0]-Fi@w; p = np.clip(expit(z), 1e-8, 1-1e-8)
                return -np.sum(Ri*np.log(p)+(1-Ri)*np.log(1-p))
            r = minimize(ot, [theta[i]], method='Nelder-Mead', options={'maxiter':30})
            theta[i] = np.clip(r.x[0], -5, 5)
        if has_nan:
            def ow(ww):
                nll = 0
                for i in range(n_p):
                    v = ~np.isnan(R[i,:])
                    if v.sum() < 1: continue
                    z = theta[i]-Fn[v]@ww; p = np.clip(expit(z), 1e-8, 1-1e-8)
                    nll -= np.sum(R[i,v]*np.log(p)+(1-R[i,v])*np.log(1-p))
                nll += reg*np.sum(ww**2)
                return nll
        else:
            pidx = np.repeat(np.arange(n_p), n_i)
            Rf = R.flatten()
            def ow(ww):
                z = theta[pidx]-np.tile(Fn@ww, n_p)
                p = np.clip(expit(z), 1e-8, 1-1e-8)
                return -np.sum(Rf*np.log(p)+(1-Rf)*np.log(1-p))+reg*np.sum(ww**2)
        r = minimize(ow, w, method='L-BFGS-B', options={'maxiter':80})
        w = r.x
    return theta, w, Fn

def two_step_rank(R, F, iters=10, alpha=1.0):
    """BUG 2 FIX: Two-step that actually uses feature-predicted difficulty
    for ability re-estimation."""
    # Step 1: Fit standard IRT
    _, a_irt, b_irt = fit_irt(R, iters)
    # Step 2: Ridge regression on features
    Fm = F.mean(0); Fs = F.std(0)+1e-8; Fn = (F-Fm)/Fs
    from numpy.linalg import solve
    n = Fn.shape[1]
    w = solve(Fn.T@Fn + alpha*np.eye(n), Fn.T@b_irt)
    # Step 3: Re-estimate theta using feature-predicted difficulty
    b_pred = Fn @ w
    n_p, n_i = R.shape; has_nan = np.any(np.isnan(R))
    theta = np.clip(np.log((np.nanmean(R,1)+.01)/(1-np.nanmean(R,1)+.01)), -4, 4)
    for _ in range(5):
        for i in range(n_p):
            v = ~np.isnan(R[i,:]) if has_nan else np.ones(n_i, bool)
            if v.sum() < 3: continue
            Ri, bi = R[i,v], b_pred[v]
            def ot(t):
                z = t[0]-bi; p = np.clip(expit(z), 1e-8, 1-1e-8)
                return -np.sum(Ri*np.log(p)+(1-Ri)*np.log(1-p))
            r = minimize(ot, [theta[i]], method='Nelder-Mead', options={'maxiter':30})
            theta[i] = np.clip(r.x[0], -5, 5)
    return theta, w, b_pred

print("="*70)
print("PRODUCTION v3 — ALL BUGS FIXED")
print("="*70)

# ═══ EXP 1: Sparsity sweep ═══
print("\n[EXP 1] Sparsity sweep...")
sps = [1., .7, .5, .3, .1]
e1 = {s: {'avg':[], 'irt':[], 'lltm':[], '2step':[]} for s in sps}
for sp in sps:
    for sd in range(NS):
        Rs,_,ta,_,F,_ = gen_data(NM, NI, TW, sp, seed=sd*100+7)
        e1[sp]['avg'].append(spearmanr(ta, simple_avg(Rs))[0])
        ti,_,_ = fit_irt(Rs, 6); e1[sp]['irt'].append(spearmanr(ta, ti)[0])
        tl,_,_ = fit_lltm(Rs, F, 6); e1[sp]['lltm'].append(spearmanr(ta, tl)[0])
        t2,_,_ = two_step_rank(Rs, F, 6); e1[sp]['2step'].append(spearmanr(ta, t2)[0])
    m = {k: np.mean(v) for k,v in e1[sp].items()}
    print(f"  sp={sp:.0%}: avg={m['avg']:.4f} irt={m['irt']:.4f} lltm={m['lltm']:.4f} 2s={m['2step']:.4f}")

# ═══ EXP 2: Noisy features × sparsity ═══
print("\n[EXP 2] Noisy features × sparsity...")
noise_counts = [0, 5, 20]
test_sps = [1., .5]
e2 = {}
for nn in noise_counts:
    for sp in test_sps:
        key = f"n{nn}_s{sp}"
        e2[key] = {'lltm':[], '2step':[]}
        for sd in range(NS):
            Rs,_,ta,_,F,_ = gen_data(NM, NI, TW, sp, noise_feats=nn, seed=sd*100+11)
            tl,_,_ = fit_lltm(Rs, F, 6); e2[key]['lltm'].append(spearmanr(ta, tl)[0])
            t2,_,_ = two_step_rank(Rs, F, 6); e2[key]['2step'].append(spearmanr(ta, t2)[0])
        print(f"  noise={nn:>2}, sp={sp:.0%}: lltm={np.mean(e2[key]['lltm']):.4f} 2s={np.mean(e2[key]['2step']):.4f}")

# ═══ EXP 3: Missing true feature (BUG 1 FIXED) ═══
print("\n[EXP 3] Missing true feature (FIXED: feature hidden from model only)...")
e3 = {}
for sp in [1., .3]:
    for cond in ['complete', 'missing']:
        key = f"{cond}_s{sp}"
        e3[key] = {'lltm':[], '2step':[], 'irt':[]}
        for sd in range(NS):
            # Generate data with ALL features affecting difficulty
            Rs,Rf,ta,td,F,_ = gen_data(NM, NI, TW, sp, seed=sd*100+22)
            if cond == 'complete':
                F_model = F  # model sees all features
            else:
                F_model = F[:, 1:]  # model CANNOT see reasoning_depth
                # but true difficulty STILL depends on it
            ti,_,_ = fit_irt(Rs, 6); e3[key]['irt'].append(spearmanr(ta, ti)[0])
            tl,_,_ = fit_lltm(Rs, F_model, 8); e3[key]['lltm'].append(spearmanr(ta, tl)[0])
            t2,_,_ = two_step_rank(Rs, F_model, 8); e3[key]['2step'].append(spearmanr(ta, t2)[0])
        m = {k: np.mean(v) for k,v in e3[key].items()}
        print(f"  {cond:>8}, sp={sp:.0%}: irt={m['irt']:.4f} lltm={m['lltm']:.4f} 2s={m['2step']:.4f}")

# ═══ EXP 4: Nonlinear interactions × sparsity ═══
print("\n[EXP 4] Nonlinear interactions × sparsity...")
e4 = {}
for sp in [1., .3]:
    for nl in [False, True]:
        key = f"{'nl' if nl else 'lin'}_s{sp}"
        e4[key] = {'lltm':[], '2step':[]}
        for sd in range(NS):
            Rs,_,ta,_,F,_ = gen_data(NM, NI, TW, sp, nonlinear=nl, seed=sd*100+33)
            tl,_,_ = fit_lltm(Rs, F, 6); e4[key]['lltm'].append(spearmanr(ta, tl)[0])
            t2,_,_ = two_step_rank(Rs, F, 6); e4[key]['2step'].append(spearmanr(ta, t2)[0])
        print(f"  {'nonlin' if nl else 'linear':>7}, sp={sp:.0%}: lltm={np.mean(e4[key]['lltm']):.4f} 2s={np.mean(e4[key]['2step']):.4f}")

# ═══ EXP 5: Unseen item difficulty prediction ═══
print("\n[EXP 5] Difficulty prediction for unseen items...")
tfs = [.9, .7, .5, .3]
e5 = {t: {'lltm':[], '2step':[]} for t in tfs}
for tf in tfs:
    for sd in range(NS):
        _,Rf,ta,td,F,_ = gen_data(NM, NI, TW, 1., seed=sd*100+44)
        nt = int(NI*tf); idx = np.random.permutation(NI)
        tri, tei = idx[:nt], idx[nt:]
        if len(tei) < 5: continue
        _,wl,_ = fit_lltm(Rf[:,tri], F[tri], 6)
        Ftn = (F[tei]-F[tri].mean(0))/(F[tri].std(0)+1e-8)
        rl,_ = spearmanr(td[tei], Ftn@wl); e5[tf]['lltm'].append(rl)
        _,w2,_ = two_step_rank(Rf[:,tri], F[tri], 6)
        Ftn2 = (F[tei]-F[tri].mean(0))/(F[tri].std(0)+1e-8)
        r2,_ = spearmanr(td[tei], Ftn2@w2); e5[tf]['2step'].append(r2)
    print(f"  train={tf:.0%}: lltm={np.mean(e5[tf]['lltm']):.4f} 2s={np.mean(e5[tf]['2step']):.4f}")

# ═══ EXP 6: Feature weight recovery ═══
print("\n[EXP 6] Feature weight recovery...")
_,Rf,_,_,F,_ = gen_data(NM, NI, TW, 1., seed=42)
_,wl,_ = fit_lltm(Rf, F, 8)
_,w2,_ = two_step_rank(Rf, F, 8)
wln = wl/np.abs(wl).sum()*np.abs(TW).sum()
w2n = w2/np.abs(w2).sum()*np.abs(TW).sum()
print(f"  {'Feature':<18} {'True':>6} {'LLTM':>6} {'2Step':>6}")
for i,n in enumerate(FNAMES):
    print(f"  {n:<18} {TW[i]:>6.3f} {wln[i]:>6.3f} {w2n[i]:>6.3f}")

# ═══ FIGURES ═══
print("\nGenerating figures...")

# Fig 1: Sparsity sweep
fig, ax = plt.subplots(figsize=(5.5, 4))
for m,lab,c,mk in [('avg','Simple Averaging','#c0392b','o'),('irt','Standard IRT','#2980b9','s'),
                     ('lltm','LLTM (joint)','#27ae60','D'),('2step','Two-Step (ridge)','#8e44ad','^')]:
    mn = [np.mean(e1[s][m]) for s in sps]
    se = [np.std(e1[s][m])/np.sqrt(NS) for s in sps]
    ax.errorbar([1-s for s in sps], mn, yerr=se, label=lab, color=c, marker=mk, ms=6, lw=2, capsize=3)
ax.set_xlabel('Missing Data Rate'); ax.set_ylabel(r'Spearman $\rho$')
ax.legend(fontsize=8, loc='lower right'); ax.set_ylim(.55, 1.02); ax.grid(True, alpha=.3)
plt.tight_layout(); plt.savefig(OUT/'fig1_sparsity.pdf', bbox_inches='tight'); print("  fig1 ✓")

# Fig 2: 2×2 stress tests
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

# 2a: Noisy features at different sparsity
ax = axes[0,0]
for sp, ls, lbl in [(1., '-', 'full'), (.5, '--', '50% miss')]:
    vals = [np.mean(e2[f"n{nn}_s{sp}"]['lltm']) for nn in noise_counts]
    ax.plot(noise_counts, vals, f'D{ls}', color='#27ae60', lw=2, ms=5, label=f'LLTM ({lbl})')
    vals2 = [np.mean(e2[f"n{nn}_s{sp}"]['2step']) for nn in noise_counts]
    ax.plot(noise_counts, vals2, f'^{ls}', color='#8e44ad', lw=1.5, ms=4, label=f'2-Step ({lbl})')
ax.set_xlabel('Irrelevant Features Added'); ax.set_ylabel(r'Spearman $\rho$')
ax.set_title('(a) Noisy Features $\\times$ Sparsity', fontweight='bold')
ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=.3)

# 2b: Missing feature (FIXED)
ax = axes[0,1]
sp_labels = ['0%', '70%']
sp_vals = [1., .3]
x = np.arange(len(sp_vals)); wid = .2
for idx, (cond, col, label) in enumerate([
    ('complete', '#27ae60', 'LLTM (all feat)'),
    ('missing', '#e67e22', 'LLTM (miss strongest)'),
]):
    vals = [np.mean(e3[f"{cond}_s{sp}"]['lltm']) for sp in sp_vals]
    ax.bar(x + idx*wid - wid/2, vals, wid, label=label, color=col, edgecolor='black', lw=.5)
irt_vals = [np.mean(e3[f"complete_s{sp}"]['irt']) for sp in sp_vals]
ax.plot(x, irt_vals, 'ks--', ms=5, lw=1.5, label='Standard IRT')
ax.set_xticks(x); ax.set_xticklabels(sp_labels)
ax.set_xlabel('Missing Data Rate'); ax.set_ylabel(r'Spearman $\rho$')
ax.set_title('(b) Missing True Feature', fontweight='bold')
ax.legend(fontsize=7); ax.grid(True, alpha=.3, axis='y')

# 2c: Nonlinear × sparsity
ax = axes[1,0]
x = np.arange(len(sp_vals)); wid = .2
for idx, (cond, col, label) in enumerate([
    ('lin', '#27ae60', 'LLTM (linear)'),
    ('nl', '#e67e22', 'LLTM (nonlinear)'),
]):
    vals = [np.mean(e4[f"{cond}_s{sp}"]['lltm']) for sp in sp_vals]
    ax.bar(x + idx*wid - wid/2, vals, wid, label=label, color=col, edgecolor='black', lw=.5)
ax.set_xticks(x); ax.set_xticklabels(sp_labels)
ax.set_xlabel('Missing Data Rate'); ax.set_ylabel(r'Spearman $\rho$')
ax.set_title('(c) Nonlinear Interactions × Sparsity', fontweight='bold')
ax.legend(fontsize=7); ax.grid(True, alpha=.3, axis='y')

# 2d: Difficulty prediction (with error bars)
ax = axes[1,1]
x = np.arange(len(tfs)); wid = .3
lltm_means = [np.mean(e5[t]['lltm']) for t in tfs]
lltm_se = [np.std(e5[t]['lltm'])/np.sqrt(NS) for t in tfs]
ts_means = [np.mean(e5[t]['2step']) for t in tfs]
ts_se = [np.std(e5[t]['2step'])/np.sqrt(NS) for t in tfs]
ax.bar(x-wid/2, lltm_means, wid, yerr=lltm_se, capsize=3,
       label='LLTM', color='#27ae60', edgecolor='black', lw=.5)
ax.bar(x+wid/2, ts_means, wid, yerr=ts_se, capsize=3,
       label='Two-Step', color='#8e44ad', edgecolor='black', lw=.5)
ax.set_xticks(x); ax.set_xticklabels([f'{int(t*100)}%' for t in tfs])
ax.set_xlabel('Training Items (% of total)'); ax.set_ylabel(r'Prediction $\rho$')
ax.set_title('(d) Unseen Item Difficulty', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=.3, axis='y'); ax.set_ylim(0, 1)

plt.tight_layout(); plt.savefig(OUT/'fig2_stress.pdf', bbox_inches='tight'); print("  fig2 ✓")

# Fig 3: Feature weights
fig, ax = plt.subplots(figsize=(6, 3.8))
x = np.arange(5); wd = .25
ax.bar(x-wd, TW, wd, label='True', color='#2c3e50', edgecolor='black', lw=.5)
ax.bar(x, wln, wd, label='LLTM', color='#27ae60', edgecolor='black', lw=.5)
ax.bar(x+wd, w2n, wd, label='Two-Step', color='#8e44ad', edgecolor='black', lw=.5)
ax.set_xticks(x)
ax.set_xticklabels(['reasoning\ndepth','domain\nspec','prompt\ncomplex','answer\nambig','context\nlen'], fontsize=8)
ax.set_ylabel('Feature Weight'); ax.legend(fontsize=9); ax.grid(True, alpha=.3, axis='y')
plt.tight_layout(); plt.savefig(OUT/'fig3_weights.pdf', bbox_inches='tight'); print("  fig3 ✓")

# Save results
def _ms(v): return {"mean": float(np.mean(v)), "se": float(np.std(v)/np.sqrt(NS))}
all_res = {
    "exp1_sparsity": {str(s): {m: _ms(v) for m,v in d.items()} for s,d in e1.items()},
    "exp2_noisy_features": {k: {m: _ms(v) for m,v in d.items()} for k,d in e2.items()},
    "exp3_missing_feature": {k: {m: _ms(v) for m,v in d.items()} for k,d in e3.items()},
    "exp4_nonlinear": {k: {m: _ms(v) for m,v in d.items()} for k,d in e4.items()},
    "exp5_unseen_items": {str(t): {m: _ms(v) for m,v in d.items()} for t,d in e5.items()},
    "exp6_weight_recovery": {
        "feature_names": list(FNAMES),
        "true_weights": [float(x) for x in TW],
        "lltm_normalized": [float(x) for x in wln],
        "twostep_normalized": [float(x) for x in w2n],
    },
    "config": {"n_models": NM, "n_items": NI, "n_features": len(TW),
               "n_seeds": NS, "random_seed_base": 42},
}
with open(ROOT/'results.json', 'w') as f:
    json.dump(all_res, f, indent=2)

print("\n" + "="*70)
print("ALL EXPERIMENTS COMPLETE")
print("="*70)
