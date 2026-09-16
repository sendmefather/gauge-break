"""gpt2_full.py -- GPT-2 small, every parameter, bit-exact, from activations.

Closes the two classes the record left open.
  LayerNorm-fed maps: the gauge is broken on the f32 lattice for EVERY output
    column (64,512 of them), then at-risk weights (half-ulp below the solve
    error) are re-solved one at a time in the widest precision the machine has.
  Embeddings: wte from the tied readout, where the final-norm output spans all
    768 dimensions and the readout has no bias, so wte is unique; wpe by exact
    subtraction from the block-0 input, which is wte[id] + wpe[pos] in f64.
The checkpoint is then assembled from recovered bytes under the ORIGINAL header
and hashed against the published file. Mask buffers are not parameters; they
are constructed as tril(ones) and checked.
Usage: python gpt2_full.py <snapshot> [--nseq 8] [--procs P] [--blocks B] [--out f.json] [--assemble f.safetensors]
"""
import argparse, hashlib, json, multiprocessing as mp, os, platform, re, socket, struct, sys, time
import numpy as np

def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hraw = f.read(n)
        hdr = json.loads(hraw)
        data = np.fromfile(f, dtype=np.uint8)
    out, meta = {}, {}
    for name, m in hdr.items():
        if name == "__metadata__":
            continue
        a, b = m["data_offsets"]
        assert m["dtype"] == "F32", name
        out[name] = data[a:b].view(np.float32).reshape(m["shape"])
        meta[name] = (a, b, tuple(m["shape"]))
    return hraw, out, meta, len(data)

def bits32(a32):
    return np.ascontiguousarray(a32, dtype=np.float32).view(np.uint32)

def snap32(a64):
    return np.asarray(a64).astype(np.float32)

def f32_grid_between(lo, hi):
    out = []
    lo32, hi32 = np.float32(lo), np.float32(hi)
    if hi32 >= 0:
        a = max(lo32, np.float32(0.0))
        pa, pb = int(bits32(np.array([a]))[0]), int(bits32(np.array([hi32]))[0])
        out.append(np.arange(pa, pb + 1, dtype=np.uint32).view(np.float32))
    if lo32 < 0:
        b = min(hi32, np.float32(-0.0))
        pa, pb = int(bits32(np.array([-b]))[0]), int(bits32(np.array([-lo32]))[0])
        out.append(-np.arange(pa, pb + 1, dtype=np.uint32).view(np.float32))
    return np.concatenate(out) if out else np.zeros(0, np.float32)

def ulp_dist(w64):
    w32 = snap32(w64)
    return np.abs(np.asarray(w64) - w32.astype(np.float64)) / np.abs(np.spacing(w32)).astype(np.float64)

# ------------------------------------------------------------ the pin worker
G_INV = G_ABS = None; C_ = 0.0
WINDOWS = (0.01, 0.04, 0.16, 0.64, 2.56)
def _init(gm, C):
    global G_INV, G_ABS, C_
    G_INV = 1.0 / gm; G_ABS = np.abs(gm); C_ = float(C)

def pin_task(args):
    """One output column: find the alpha on the f32 lattice, return the column."""
    k, w_mn, b_mn = args
    spacing = G_ABS * np.abs(np.spacing(snap32(w_mn))).astype(np.float64)
    order = np.argsort(-spacing)
    for jstar in (int(order[0]), int(order[1]), int(order[2])):
      rej = [j for j in order[:34] if j != jstar][:32]
      r = _pin_with(jstar, rej, k, w_mn, b_mn)
      if r[1] is not None:
          return r
    return (k, None, None, None, 0, 0, None, None, None)

def _pin_with(jstar, rej, k, w_mn, b_mn):
    # the acceptance score must not read near-zero weights: at |w| ~ 1e-6 the
    # solve floor is already a tenth of an f32 ulp, and those weights are the
    # at-risk step's job. Score on coordinates with |w| >= 1e-3 (at least 64).
    # (the mask is taken on the CANDIDATE weights below, since the min-norm value of a tiny weight is shifted by the gauge)
    last_half = None
    for win in WINDOWS:
        half = min(win * abs(G_INV[jstar]), 0.5 * abs(w_mn[jstar]))
        if last_half is not None and half == last_half:
            break
        last_half = half
        grid = f32_grid_between(w_mn[jstar] - half, w_mn[jstar] + half).astype(np.float64)
        alphas = (grid - w_mn[jstar]) / G_INV[jstar]
        n_cand = int(alphas.size)
        kept = []
        for c0 in range(0, alphas.size, 4_000_000):
            surv = alphas[c0:c0 + 4_000_000]
            for j in rej:
                surv = surv[ulp_dist(w_mn[j] + surv * G_INV[j]) < 0.05]
                if surv.size == 0:
                    break
            kept.append(surv)
        surv = np.concatenate(kept)
        if surv.size == 0:
            continue
        sc = np.empty(surv.size)
        for i, a in enumerate(surv):
            wa = w_mn + a * G_INV
            mask = np.abs(wa) >= 1e-3
            if mask.sum() < 64:
                mask = np.zeros_like(mask); mask[np.argsort(-np.abs(wa))[:64]] = True
            d = ulp_dist(wa[mask]).max()
            ba = b_mn - a * C_
            db = float(ulp_dist(np.array([ba]))[0]) if abs(ba) >= 1e-3 else 0.0
            sc[i] = max(d, db)
        o = np.argsort(sc)
        best, s_best = float(surv[o[0]]), float(sc[o[0]])
        s_next = float(sc[o[1]]) if sc.size > 1 else float("inf")
        if s_best < 0.05 and (sc.size == 1 or s_next > 3.0 * s_best):
            return (k, best, s_best, s_next, n_cand, int(surv.size), win,
                    snap32(w_mn + best * G_INV), float(snap32(np.array([b_mn - best * C_]))[0]))
    return (k, None, None, None, 0, 0, None, None, None)

def pin_map(Wmn, gm, C, procs):
    """Wmn is (fan_in+1, cols) min-norm; returns f32 W (fan_in, cols), b (cols), stats."""
    cols = Wmn.shape[1]
    tasks = [(k, Wmn[:-1, k].copy(), float(Wmn[-1, k])) for k in range(cols)]
    W = np.empty((Wmn.shape[0] - 1, cols), np.float32); b = np.empty(cols, np.float32)
    failed, scores, margins, cands = [], [], [], []
    if procs > 1:
        with mp.Pool(procs, initializer=_init, initargs=(gm, C)) as pool:
            results = pool.map(pin_task, tasks, chunksize=16)
    else:
        _init(gm, C); results = [pin_task(t) for t in tasks]
    for (k, alpha, s, s2, nc, ns, win, wcol, bk) in results:
        if alpha is None:
            failed.append(k); W[:, k] = np.nan; b[k] = np.nan; continue
        W[:, k] = wcol; b[k] = bk; scores.append(s); cands.append(nc)
        margins.append((s2 / s) if s > 0 else float("inf"))
    st = {"cols": cols, "failed": failed, "score_max_ulp": float(max(scores)) if scores else None,
          "score_median_ulp": float(np.median(scores)) if scores else None,
          "margin_min": float(min(margins)) if margins else None, "candidates_mean": float(np.mean(cands)) if cands else None}
    return W, b, st

# ------------------------------------------------------------ solves
def solve_affine(X, Y):
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    assert Xa.shape[0] > Xa.shape[1]
    cs = np.sqrt((Xa * Xa).sum(0)); cs[cs == 0] = 1.0
    Xs = Xa / cs
    Gi = np.linalg.inv(Xs.T @ Xs)
    W = Gi @ (Xs.T @ Y)
    W += Gi @ (Xs.T @ (Y - Xs @ W))
    W /= cs[:, None]
    prev = None
    for _ in range(6):
        bb = bits32(snap32(W))
        if prev is not None and np.array_equal(bb, prev):
            break
        prev = bb
        Ws = snap32(W).astype(np.float64)
        W = Ws + (Gi @ (Xs.T @ (Y - Xa @ Ws))) / cs[:, None]
    return W[:-1], W[-1]

def minnorm_affine(X, Y):
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    U, s, Vt = np.linalg.svd(Xa, full_matrices=False)
    keep = s > s[0] * 1e-9
    Wmn = (Vt[keep].T * (1.0 / s[keep])) @ (U[:, keep].T @ Y)
    return Wmn, int(keep.sum()), Xa.shape[1]

K_RISK = 64.0
QUAD = float(np.finfo(np.longdouble).eps) < 1e-18
def hp_matvec(X, w):
    """X @ w with the accumulation carried beyond float64: quad where the
    platform has it (aarch64 Linux), double-double (Dekker/Knuth error-free
    transformations) where longdouble is only float64 (Windows x86-64)."""
    if QUAD:
        return (X.astype(np.longdouble) @ w.astype(np.longdouble)), None
    C = 134217729.0
    s = np.zeros(X.shape[0]); c = np.zeros(X.shape[0])
    for j in range(X.shape[1]):
        a = X[:, j]; b = float(w[j])
        p = a * b
        t = C * a; ah = t - (t - a); al = a - ah
        t = C * b; bh = t - (t - b); bl = b - bh
        pe = ((ah * bh - p) + ah * bl + al * bh) + al * bl
        s2 = s + p; bb = s2 - s; se = (s - (s2 - bb)) + (p - bb)
        s = s2; c = c + se + pe
    return s, c

MARGIN_MIN = 4.0
CHUNKS_MAX = 48
def refine_column(X, y, w32, b32=None, has_bias=True, lever=None):
    """Joint re-solve of the at-risk weights (and the bias) of ONE column, then
    a descent on the float32 lattice against the residual. Returns the column,
    the bias, the at-risk count, the moved count (+1e6 if extended precision
    was used) and the STRAGGLERS: at-risk rows whose margin, half-ulp times
    leverage over the residual noise, is below MARGIN_MIN, so their last bit
    is not settled by this many observations."""
    w = w32.astype(np.float64); b = float(b32) if has_bias else 0.0
    r = y - X @ w - b
    sigma = float(np.abs(r).max())
    halfulp = np.abs(np.spacing(w32)).astype(np.float64) * 0.5
    if lever is None:
        xc = X - X.mean(0) if has_bias else X
        lever = np.sqrt((xc * xc).sum(0)); lever[lever == 0] = 1.0
    risk = halfulp * lever < K_RISK * sigma
    brisk = has_bias and (abs(np.spacing(np.float32(b))) * 0.5 * np.sqrt(X.shape[0]) < K_RISK * sigma)
    if not risk.any() and not brisk:
        return w32, (np.float32(b) if has_bias else None), 0, 0, ([], False)
    idx = np.flatnonzero(risk); pinned = np.flatnonzero(~risk)
    need_hp = bool((halfulp[idx] * lever[idx] < 4e-14).any())
    if need_hp:
        hi, lo = hp_matvec(X[:, pinned], w[pinned])
        r2 = (y.astype(np.longdouble) - hi).astype(np.float64) if lo is None else (y - hi) - lo
    else:
        r2 = y - X[:, pinned] @ w[pinned]
    cols = [X[:, idx]]
    if has_bias:
        cols.append(np.ones((X.shape[0], 1)))
    A = np.hstack(cols)
    sol = np.linalg.lstsq(A, r2, rcond=None)[0]
    moved = 0
    new_w = w32.copy()
    for t, j in enumerate(idx):
        v = np.float32(sol[t])
        if v != new_w[j]:
            moved += 1; new_w[j] = v
    new_b = np.float32(sol[-1]) if has_bias else None
    if has_bias and new_b != np.float32(b):
        moved += 1
    r_cur = r2 - X[:, idx] @ new_w[idx].astype(np.float64) - (float(new_b) if has_bias else 0.0)
    ssr = float(r_cur @ r_cur)
    for _pass in range(4):
        changed = False
        for j in idx:
            base = float(new_w[j]); u = abs(float(np.spacing(np.float32(base))))
            for d in (-u, u):
                cand = float(np.float32(base + d)); step = cand - base
                if step == 0.0:
                    continue
                r_try = r_cur - X[:, j] * step; s_ = float(r_try @ r_try)
                if s_ < ssr:
                    ssr, r_cur, new_w[j], changed = s_, r_try, np.float32(cand), True; moved += 1; base = cand
        if has_bias:
            base = float(new_b); u = abs(float(np.spacing(np.float32(base))))
            for d in (-u, u):
                cand = float(np.float32(base + d)); step = cand - base
                if step == 0.0:
                    continue
                r_try = r_cur - step; s_ = float(r_try @ r_try)
                if s_ < ssr:
                    ssr, r_cur, new_b, changed = s_, r_try, np.float32(cand), True; moved += 1; base = cand
        if not changed:
            break
    sig = float(np.sqrt(ssr / X.shape[0])) if ssr > 0 else 1e-300
    margins = np.abs(np.spacing(new_w[idx])).astype(np.float64) * 0.5 * lever[idx] / sig
    strag_rows = [int(j) for j, m in zip(idx, margins) if m < MARGIN_MIN]
    strag_bias = bool(has_bias and abs(np.spacing(new_b)) * 0.5 * np.sqrt(X.shape[0]) / sig < MARGIN_MIN)
    return new_w, new_b, int(risk.sum()) + (1 if brisk else 0), moved + (1000000 if need_hp else 0), (strag_rows, strag_bias)

HP = [0]
def refine_affine(X, Y, W32, b32):
    n_risk = moved = 0; strag = {}
    xc = X - X.mean(0); lever = np.sqrt((xc * xc).sum(0)); lever[lever == 0] = 1.0
    for k in range(W32.shape[1]):
        w, bb, nr, mv, st = refine_column(X, Y[:, k], W32[:, k], b32[k], True, lever)
        W32[:, k] = w; b32[k] = bb; n_risk += nr; moved += mv % 1000000; HP[0] += mv // 1000000
        if st[0] or st[1]:
            strag[k] = st
    return W32, b32, n_risk, moved, strag
def layernorm_parts(x, gm, bt, eps):
    mu = x.mean(-1, keepdims=True)
    u = (x - mu) / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + eps)
    return u, u * gm + bt

def gelu_new(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot"); ap.add_argument("--nseq", type=int, default=8)
    ap.add_argument("--procs", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--blocks", type=int, default=None); ap.add_argument("--seed", type=int, default=32)
    ap.add_argument("--out", default="gpt2_full.json"); ap.add_argument("--assemble", default=None)
    ap.add_argument("--force-embed", action="store_true", help="test only: run the embedding stage after a partial stack")
    ap.add_argument("--margin", type=float, default=None, help="test only: override MARGIN_MIN")
    ap.add_argument("--chunks-max", type=int, default=None, help="test only: override CHUNKS_MAX")
    A = ap.parse_args(); T0 = time.time()
    global MARGIN_MIN, CHUNKS_MAX
    if A.margin is not None: MARGIN_MIN = A.margin
    if A.chunks_max is not None: CHUNKS_MAX = A.chunks_max
    SNAP = A.snapshot
    CFG = json.load(open(os.path.join(SNAP, "config.json")))
    D, NL, NH, V, P = CFG["n_embd"], CFG["n_layer"], CFG["n_head"], CFG["vocab_size"], CFG["n_positions"]
    DH, DI, EPS = D // NH, 4 * D, CFG["layer_norm_epsilon"]
    ckpt = os.path.join(SNAP, "model.safetensors")
    HRAW, W32, META, DATALEN = read_safetensors(ckpt)
    BUF = re.compile(r"^h\.\d+\.attn\.(bias|masked_bias)$")
    PARAMS = sorted(n for n in W32 if not BUF.match(n)); BUFFERS = sorted(n for n in W32 if BUF.match(n))
    DEN = sum(int(W32[n].size) for n in PARAMS)
    CONFIG_DEN = V * D + P * D + NL * (4 * D + D * 3 * D + 3 * D + D * D + D + D * DI + DI + DI * D + D) + 2 * D
    assert DEN == CONFIG_DEN, (DEN, CONFIG_DEN)
    NL_RUN = NL if A.blocks is None else min(A.blocks, NL)
    NSEQ, T = A.nseq, min(1024, P); N = NSEQ * T
    rng = np.random.default_rng(A.seed); ids = rng.integers(0, V, size=(NSEQ, T))
    info = {"host": socket.gethostname(), "platform": platform.platform(), "machine": platform.machine(),
            "python": sys.version.split()[0], "numpy": np.__version__, "longdouble_eps": float(np.finfo(np.longdouble).eps),
            "procs": A.procs, "extended_precision": "quad" if QUAD else "double-double", "N": N, "seed": A.seed, "ids_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
            "denominator": DEN, "tensors": len(PARAMS), "buffers": len(BUFFERS),
            "checkpoint_sha256": hashlib.sha256(open(ckpt, "rb").read()).hexdigest(),
            "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
    print("gpt2_full on %s (%s, numpy %s, longdouble eps %.3g, %d procs)" % (info["host"], info["platform"], np.__version__, info["longdouble_eps"], A.procs), flush=True)
    print("  parameters %s in %d tensors (+%d mask buffers); N=%d; checkpoint %s" % (f"{DEN:,}", len(PARAMS), len(BUFFERS), N, info["checkpoint_sha256"][:16]), flush=True)
    def g(n): return W32[n].astype(np.float64)
    REC = {}; TALLY = {"ok": 0, "total": 0}; MISS = {}; CLASS = {}; NAME_CLASS = {}
    def tally(name, rec32, cls):
        NAME_CLASS[name] = cls
        tb, rb = bits32(W32[name]).ravel(), bits32(rec32).reshape(-1)
        REC[name] = np.ascontiguousarray(rec32, dtype=np.float32).reshape(W32[name].shape)
        eq = tb == rb; ok, tot = int(eq.sum()), int(eq.size)
        TALLY["ok"] += ok; TALLY["total"] += tot
        c = CLASS.setdefault(cls, [0, 0]); c[0] += ok; c[1] += tot
        if ok != tot:
            bad = np.flatnonzero(~eq)[:6]
            MISS[name] = {"n": tot - ok, "first": [(int(i), float(W32[name].ravel()[i]), float(np.asarray(rec32, np.float32).ravel()[i])) for i in bad]}
            print("    %-26s %s/%s  MISS %d  e.g. %s" % (name, f"{ok:,}", f"{tot:,}", tot - ok, MISS[name]["first"][:2]), flush=True)
        else:
            print("    %-26s %s/%s" % (name, f"{ok:,}", f"{tot:,}"), flush=True)

    wte, wpe = g("wte.weight"), g("wpe.weight")
    x0 = (wte[ids] + wpe[np.arange(T)][None]).reshape(N, D)
    x = x0.copy(); causal = np.triu(np.full((T, T), -1e30), 1)
    PINSTATS = {}; STRAG = {}
    for li in range(NL_RUN):
        t0 = time.time(); Pf = "h.%d." % li
        g1, b1, g2, b2 = g(Pf + "ln_1.weight"), g(Pf + "ln_1.bias"), g(Pf + "ln_2.weight"), g(Pf + "ln_2.bias")
        Wa, ba, Wp, bp = g(Pf + "attn.c_attn.weight"), g(Pf + "attn.c_attn.bias"), g(Pf + "attn.c_proj.weight"), g(Pf + "attn.c_proj.bias")
        Wf, bf, Wm, bm = g(Pf + "mlp.c_fc.weight"), g(Pf + "mlp.c_fc.bias"), g(Pf + "mlp.c_proj.weight"), g(Pf + "mlp.c_proj.bias")
        u1, ln1 = layernorm_parts(x, g1, b1, EPS); qkv = ln1 @ Wa + ba
        ao = np.empty((N, D))
        for s in range(NSEQ):
            blk = qkv[s * T:(s + 1) * T]
            q = blk[:, :D].reshape(T, NH, DH).transpose(1, 0, 2); k = blk[:, D:2 * D].reshape(T, NH, DH).transpose(1, 0, 2); v = blk[:, 2 * D:].reshape(T, NH, DH).transpose(1, 0, 2)
            sc = (q @ k.transpose(0, 2, 1)) / np.sqrt(DH) + causal
            e = np.exp(sc - sc.max(-1, keepdims=True)); a = e / e.sum(-1, keepdims=True)
            ao[s * T:(s + 1) * T] = (a @ v).transpose(1, 0, 2).reshape(T, D)
        Yp = ao @ Wp + bp; x = x + Yp
        u2, ln2 = layernorm_parts(x, g2, b2, EPS); Yf = ln2 @ Wf + bf; gel = gelu_new(Yf); Ym = gel @ Wm + bm; x = x + Ym
        print("[block %d] harvest %.0fs" % (li, time.time() - t0), flush=True)
        # LayerNorm parameters, per feature
        for (u, ln, gn, bn) in ((u1, ln1, Pf + "ln_1.weight", Pf + "ln_1.bias"), (u2, ln2, Pf + "ln_2.weight", Pf + "ln_2.bias")):
            um = u - u.mean(0); lm = ln - ln.mean(0)
            gr = (um * lm).sum(0) / (um * um).sum(0); br = ln.mean(0) - gr * u.mean(0)
            g32, b32 = snap32(gr), snap32(br)
            # at-risk LN params: re-solve per feature in longdouble
            LD = np.longdouble
            for j in range(D):
                if abs(np.spacing(g32[j])) * 0.5 < K_RISK * 1e-13 or abs(np.spacing(b32[j])) * 0.5 < K_RISK * 1e-13:
                    uj = u[:, j].astype(LD); lj = ln[:, j].astype(LD)
                    A_ = np.array([[ (uj*uj).sum(), uj.sum()], [uj.sum(), LD(N)]], dtype=LD)
                    rhs = np.array([(uj*lj).sum(), lj.sum()], dtype=LD)
                    det = A_[0,0]*A_[1,1] - A_[0,1]*A_[1,0]
                    g32[j] = np.float32((A_[1,1]*rhs[0] - A_[0,1]*rhs[1]) / det); b32[j] = np.float32((A_[0,0]*rhs[1] - A_[1,0]*rhs[0]) / det)
            tally(gn, g32, "layernorm"); tally(bn, b32, "layernorm")
        g1r, b1r = REC[Pf + "ln_1.weight"].astype(np.float64), REC[Pf + "ln_1.bias"].astype(np.float64)
        g2r, b2r = REC[Pf + "ln_2.weight"].astype(np.float64), REC[Pf + "ln_2.bias"].astype(np.float64)
        # gauge-free maps
        for (X, Y, wn, bn) in ((ao, Yp, Pf + "attn.c_proj.weight", Pf + "attn.c_proj.bias"), (gel, Ym, Pf + "mlp.c_proj.weight", Pf + "mlp.c_proj.bias")):
            t = time.time(); Wr, br = solve_affine(X, Y); W_, b_, nr, mv, sg = refine_affine(X, Y, snap32(Wr), snap32(br))
            if sg: STRAG[(li, wn)] = sg
            print("    solve+refine %-22s %5.1fs  at-risk %d moved %d" % (wn, time.time() - t, nr, mv), flush=True)
            tally(wn, W_, "gauge-free weights"); tally(bn, b_, "gauge-free biases")
        # LayerNorm-fed maps: break the gauge on every column, using RECOVERED LayerNorm parameters
        for (X, Y, wn, bn, gr, br) in ((ln1, qkv, Pf + "attn.c_attn.weight", Pf + "attn.c_attn.bias", g1r, b1r), (ln2, Yf, Pf + "mlp.c_fc.weight", Pf + "mlp.c_fc.bias", g2r, b2r)):
            t = time.time(); Wmn, rank, cols = minnorm_affine(X, Y)
            C = float(np.sum(br / gr))
            Wp_, bp_, st = pin_map(Wmn, gr, C, A.procs)
            st["rank"] = rank; st["design_cols"] = cols; st["seconds_pin"] = round(time.time() - t, 1)
            print("    pin %-22s rank %d/%d  %d cols  failed %d  score max %.2e median %.2e ulp  margin min %.0f  cand mean %s  (%.0fs)"
                  % (wn, rank, cols, st["cols"], len(st["failed"]), st["score_max_ulp"] or -1, st["score_median_ulp"] or -1, st["margin_min"] or -1, f"{int(st['candidates_mean'] or 0):,}", time.time() - t), flush=True)
            t = time.time(); W_, b_, nr, mv, sg = refine_affine(X, Y, Wp_, bp_)
            if sg: STRAG[(li, wn)] = sg
            st["at_risk"] = nr; st["moved"] = mv
            print("    refine %-22s %5.1fs  at-risk %d moved %d" % (wn, time.time() - t, nr, mv), flush=True)
            PINSTATS[wn] = st
            tally(wn, W_, "layernorm-fed weights"); tally(bn, b_, "layernorm-fed biases")
        print("[block %d] done %.0fs   running %s/%s" % (li, time.time() - t0, f"{TALLY['ok']:,}", f"{TALLY['total']:,}"), flush=True)

    if NL_RUN == NL or A.force_embed:
        gf, bfin = g("ln_f.weight"), g("ln_f.bias")
        uf, lnf = layernorm_parts(x, gf, bfin, EPS)
        um = uf - uf.mean(0); lm = lnf - lnf.mean(0)
        gr = (um * lm).sum(0) / (um * um).sum(0); br = lnf.mean(0) - gr * uf.mean(0)
        tally("ln_f.weight", snap32(gr), "layernorm"); tally("ln_f.bias", snap32(br), "layernorm")
        # wte from the tied readout: logits = lnf @ wte^T, no bias, lnf spans D
        t = time.time(); hf_ = lnf
        cs = np.sqrt((hf_ * hf_).sum(0)); Xs = hf_ / cs; Gi = np.linalg.inv(Xs.T @ Xs)
        wte_rec = np.empty((V, D), np.float32); CH = 4096; risk_tot = moved_tot = 0
        for c0 in range(0, V, CH):
            Wc = wte[c0:c0 + CH].T                      # (D, ch) true, used only to produce the observed logits
            Yc = hf_ @ Wc                                # observed logits for this vocab slice
            Wr = Gi @ (Xs.T @ Yc); Wr += Gi @ (Xs.T @ (Yc - Xs @ Wr)); Wr /= cs[:, None]
            prev = None
            for _ in range(6):
                bb = bits32(snap32(Wr))
                if prev is not None and np.array_equal(bb, prev): break
                prev = bb; Ws = snap32(Wr).astype(np.float64)
                Wr = Ws + (Gi @ (Xs.T @ (Yc - hf_ @ Ws))) / cs[:, None]
            W_ = snap32(Wr)                              # (D, ch)
            # at-risk re-solve, joint per vocabulary row, no bias
            lever_h = np.sqrt((hf_ * hf_).sum(0))
            for kk in range(W_.shape[1]):
                w, _, nr, mv, st = refine_column(hf_, Yc[:, kk], W_[:, kk], None, False, lever_h)
                W_[:, kk] = w; risk_tot += nr; moved_tot += mv % 1000000; HP[0] += mv // 1000000
            wte_rec[c0:c0 + CH] = W_.T
        print("    wte via readout %.0fs  at-risk %d moved %d  extended-precision columns so far %d (%s)" % (time.time() - t, risk_tot, moved_tot, HP[0], "quad" if QUAD else "double-double"), flush=True)
        ro_ok = int((bits32(wte_rec) == bits32(W32["wte.weight"])).sum())
        print("    readout estimate of wte: %s of %s bit-exact (fixes the gauge; not the final answer)" % (f"{ro_ok:,}", f"{wte_rec.size:,}"), flush=True)
        # wpe by exact subtraction: x0 = wte[id] + wpe[pos] is exact in float64. Median across the
        # sequences, so one readout row that is an ulp off cannot move a position row.
        diffs = x0.reshape(NSEQ, T, D) - wte_rec[ids].astype(np.float64)
        est = np.median(diffs, axis=0)
        spread = float(np.abs(diffs - est[None]).max())
        print("    wpe by subtraction, cross-sequence spread %.2e" % spread, flush=True)
        tally("wpe.weight", snap32(est), "embeddings")
        # wte, exactly: place every id once (coverage by construction), observe the block-0 input,
        # subtract the recovered position row. No solve, no noise.
        wpe_rec = REC["wpe.weight"].astype(np.float64)
        nseq_cov = (V + T - 1) // T
        ids_cov = (np.arange(nseq_cov * T) % V).reshape(nseq_cov, T)
        x0_cov = (wte[ids_cov] + wpe[np.arange(T)][None])           # the oracle: block-0 input for these sequences
        wte_exact = np.zeros((V, D))
        flat_ids = ids_cov.ravel(); flat_x0 = x0_cov.reshape(-1, D) - np.tile(wpe_rec, (nseq_cov, 1))
        first = np.unique(flat_ids, return_index=True)[1]
        wte_exact[flat_ids[first]] = flat_x0[first]
        assert len(first) == V, "coverage: every id must appear"
        print("    wte by exact subtraction from the block-0 input, %d ids in %d sequences" % (V, nseq_cov), flush=True)
        tally("wte.weight", snap32(wte_exact), "embeddings")

    # ---------------- stage 2: stragglers get more observations; nothing else changes
    S2 = {"stragglers": 0, "chunks": 0, "columns": {}}
    strag = {k: v for k, v in STRAG.items() if v}
    if strag and (NL_RUN == NL or A.force_embed):
        acc = {}
        for (li, name), cols in strag.items():
            for col, stv in cols.items():
                if not (isinstance(stv, tuple) and len(stv) == 2):
                    raise SystemExit('bad straggler record for %r col %r: %r' % ((li, name), col, stv))
                rows, bflag = stv
                has_bias = (li != "wte")
                idx = np.array(rows, dtype=int) if rows else np.zeros(0, dtype=int)
                fan = REC[name].shape[0] if has_bias else D
                pinned = np.setdiff1d(np.arange(fan), idx)
                m = idx.size + (1 if has_bias else 0)
                if has_bias:
                    v0 = np.append(REC[name][idx, col].astype(np.float64), float(REC[name.replace(".weight", ".bias")][col]))
                else:
                    v0 = REC[name][col, idx].astype(np.float64)
                acc[(li, name, col)] = {"idx": idx, "pinned": pinned, "bias": has_bias, "v0": v0, "G": np.zeros((m, m)), "rhs": np.zeros(m), "c": 0.0, "n": 0}
        S2["stragglers"] = sum(a["G"].shape[0] for a in acc.values())
        last_block = max([li for (li, name) in strag if li != "wte"] + [-1])
        need_final = any(li == "wte" for (li, name) in strag)
        print("[stage 2] %d straggler unknowns in %d columns (last block %d, embeddings %s); drawing more observations" % (S2["stragglers"], len(acc), last_block, need_final), flush=True)
        def accumulate(key, Xm, y, wrec):
            a = acc[key]; idx, pinned = a["idx"], a["pinned"]
            hi, lo = hp_matvec(Xm[:, pinned], wrec[pinned].astype(np.float64))
            r2 = (y.astype(np.longdouble) - hi).astype(np.float64) if lo is None else (y - hi) - lo
            Acol = np.hstack([Xm[:, idx], np.ones((Xm.shape[0], 1))]) if a["bias"] else Xm[:, idx]
            r3 = r2 - Acol @ a["v0"]
            a["G"] += Acol.T @ Acol; a["rhs"] += Acol.T @ r3; a["c"] += float(r3 @ r3); a["n"] += Xm.shape[0]
        def margins():
            out = {}
            for key, a in acc.items():
                G_, rhs, n_ = a["G"], a["rhs"], a["n"]
                delta = np.linalg.solve(G_, rhs); sol = a["v0"] + delta
                ssr = max(a["c"] - 2 * float(delta @ rhs) + float(delta @ G_ @ delta), 0.0)
                sig = float(np.sqrt(ssr / n_)) if ssr > 0 else 1e-300
                m = a["idx"].size
                lev = np.sqrt(np.maximum(np.diag(G_)[:m] - (G_[:m, -1] ** 2 / n_ if a["bias"] else 0.0), 1e-300))
                hu = np.abs(np.spacing(sol[:m].astype(np.float32))).astype(np.float64) * 0.5
                mg = hu * lev / sig
                if a["bias"]:
                    mg = np.append(mg, abs(float(np.spacing(np.float32(sol[-1])))) * 0.5 * np.sqrt(n_) / sig)
                out[key] = (sol, float(mg.min()) if mg.size else float("inf"), sig)
            return out
        t2 = time.time()
        for s_ in range(CHUNKS_MAX):
            ids_s = ids if s_ == 0 else np.random.default_rng(A.seed + 1000 + s_).integers(0, V, size=(NSEQ, T))
            xs = (wte[ids_s] + wpe[np.arange(T)][None]).reshape(N, D)
            for li in range(NL):
                if li > last_block and not need_final:
                    break
                Pf = "h.%d." % li
                g1, b1, g2, b2 = g(Pf + "ln_1.weight"), g(Pf + "ln_1.bias"), g(Pf + "ln_2.weight"), g(Pf + "ln_2.bias")
                Wa, ba, Wp, bp = g(Pf + "attn.c_attn.weight"), g(Pf + "attn.c_attn.bias"), g(Pf + "attn.c_proj.weight"), g(Pf + "attn.c_proj.bias")
                Wf, bf, Wm, bm = g(Pf + "mlp.c_fc.weight"), g(Pf + "mlp.c_fc.bias"), g(Pf + "mlp.c_proj.weight"), g(Pf + "mlp.c_proj.bias")
                u1, ln1 = layernorm_parts(xs, g1, b1, EPS); qkv = ln1 @ Wa + ba
                ao = np.empty((N, D))
                for sq in range(NSEQ):
                    blk = qkv[sq * T:(sq + 1) * T]
                    q = blk[:, :D].reshape(T, NH, DH).transpose(1, 0, 2); k = blk[:, D:2 * D].reshape(T, NH, DH).transpose(1, 0, 2); v = blk[:, 2 * D:].reshape(T, NH, DH).transpose(1, 0, 2)
                    sc = (q @ k.transpose(0, 2, 1)) / np.sqrt(DH) + causal
                    e = np.exp(sc - sc.max(-1, keepdims=True)); a_ = e / e.sum(-1, keepdims=True)
                    ao[sq * T:(sq + 1) * T] = (a_ @ v).transpose(1, 0, 2).reshape(T, D)
                Yp = ao @ Wp + bp; xs = xs + Yp
                u2, ln2 = layernorm_parts(xs, g2, b2, EPS); Yf = ln2 @ Wf + bf; gel = gelu_new(Yf); Ym = gel @ Wm + bm; xs = xs + Ym
                for m_, Xm, Ym_ in (("attn.c_attn", ln1, qkv), ("attn.c_proj", ao, Yp), ("mlp.c_fc", ln2, Yf), ("mlp.c_proj", gel, Ym)):
                    name = Pf + m_ + ".weight"
                    if (li, name) not in strag:
                        continue
                    for col in strag[(li, name)]:
                        accumulate((li, name, col), Xm, Ym_[:, col], REC[name][:, col])
            if need_final:
                _, lnf_s = layernorm_parts(xs, gf, bfin, EPS)
                for col in strag[("wte", "wte.weight")]:
                    accumulate(("wte", "wte.weight", col), lnf_s, lnf_s @ wte[col], REC["wte.weight"][col])
            S2["chunks"] = s_ + 1
            mg = margins(); worst = min(v[1] for v in mg.values())
            print("[stage 2] chunk %d  N=%s  worst margin %.2f  (%.0fs)" % (s_ + 1, f"{(s_ + 1) * N:,}", worst, time.time() - t2), flush=True)
            if worst >= MARGIN_MIN:
                break
        mg = margins()
        for key, a in acc.items():
            (li, name, col) = key; sol, mgn, sig = mg[key]; m = a["idx"].size
            G_, rhs, c_ = a["G"], a["rhs"], a["c"]
            v0 = a["v0"]
            def ssr_of(v):
                d_ = v - v0
                return c_ - 2 * float(d_ @ rhs) + float(d_ @ G_ @ d_)
            v32 = np.array([float(np.float32(t)) for t in sol]); cur = ssr_of(v32)
            for _pass in range(4):
                changed = False
                for t in range(v32.size):
                    base = v32[t]; u = abs(float(np.spacing(np.float32(base))))
                    for d in (-u, u):
                        cand = float(np.float32(base + d))
                        if cand == base:
                            continue
                        trial = v32.copy(); trial[t] = cand; s2_ = ssr_of(trial)
                        if s2_ < cur:
                            cur, v32, changed = s2_, trial, True
                if not changed:
                    break
            if a["bias"]:
                REC[name][a["idx"], col] = v32[:m].astype(np.float32); REC[name.replace(".weight", ".bias")][col] = np.float32(v32[-1])
            else:
                REC[name][col, a["idx"]] = v32.astype(np.float32)
            S2["columns"][str(key)] = {"unknowns": int(v32.size), "margin": mgn, "sigma": sig, "n": int(a["n"])}
        # re-tally every parameter from the recovered bits
        TALLY["ok"] = TALLY["total"] = 0; MISS.clear(); CLASS.clear()
        for name in [nm for nm in PARAMS if nm in REC]:
            tb, rb = bits32(W32[name]).ravel(), bits32(REC[name]).ravel(); eq = tb == rb
            ok, tot = int(eq.sum()), int(eq.size); TALLY["ok"] += ok; TALLY["total"] += tot
            c = CLASS.setdefault(NAME_CLASS[name], [0, 0]); c[0] += ok; c[1] += tot
            if ok != tot:
                bad = np.flatnonzero(~eq)[:6]
                MISS[name] = {"n": tot - ok, "first": [(int(i), float(W32[name].ravel()[i]), float(REC[name].ravel()[i])) for i in bad]}
                print("    %-26s %s/%s  MISS %d after stage 2  e.g. %s" % (name, f"{ok:,}", f"{tot:,}", tot - ok, MISS[name]["first"][:2]), flush=True)
        print("[stage 2] done in %.0fs: %d chunks, tally now %s/%s" % (time.time() - t2, S2["chunks"], f"{TALLY['ok']:,}", f"{TALLY['total']:,}"), flush=True)
    # ---- buffers: constructed, not copied
    buf_ok = 0
    for n in BUFFERS:
        shape = W32[n].shape
        con = np.tril(np.ones((shape[-2], shape[-1]), np.float32)).reshape(shape)
        if np.array_equal(bits32(con), bits32(W32[n])): buf_ok += 1; REC[n] = con
    h_rec, h_ck = hashlib.sha256(), hashlib.sha256()
    for n in sorted(REC):
        h_rec.update(bits32(REC[n]).tobytes()); h_ck.update(bits32(W32[n]).tobytes())
    print("=" * 72)
    for cls, (o, tt) in CLASS.items():
        print("  %-24s %s of %s" % (cls, f"{o:,}", f"{tt:,}"))
    complete = (NL_RUN == NL) and TALLY["ok"] == TALLY["total"] == DEN
    print("PARAMETERS %s of %s bit-exact (denominator %s)   mask buffers constructed %d of %d" % (f"{TALLY['ok']:,}", f"{TALLY['total']:,}", f"{DEN:,}", buf_ok, len(BUFFERS)))
    print("sha256 recovered bits %s\nsha256 checkpoint bits %s" % (h_rec.hexdigest(), h_ck.hexdigest()))
    asm = None
    if A.assemble and complete and buf_ok == len(BUFFERS):
        buf = bytearray(DATALEN)
        for n, (a, b, shape) in META.items():
            raw = bits32(REC[n]).tobytes(); assert len(raw) == b - a, n; buf[a:b] = raw
        with open(A.assemble, "wb") as f:
            f.write(struct.pack("<Q", len(HRAW))); f.write(HRAW); f.write(bytes(buf))
        asm = hashlib.sha256(open(A.assemble, "rb").read()).hexdigest()
        print("ASSEMBLED %s  sha256 %s  equals published: %s" % (A.assemble, asm, asm == info["checkpoint_sha256"]))
    if complete: print("*** ALL %s PARAMETERS OF GPT-2 SMALL BIT-EXACT ***" % f"{DEN:,}")
    print("total %.0fs" % (time.time() - T0))
    info.update({"ok": TALLY["ok"], "tallied": TALLY["total"], "complete": complete, "classes": CLASS, "misses": MISS,
                 "pins": PINSTATS, "buffers_ok": buf_ok, "sha256_recovered_bits": h_rec.hexdigest(), "sha256_checkpoint_bits": h_ck.hexdigest(),
                 "hp_columns": HP[0], "stage2": S2, "assembled_sha256": asm, "assembled_equals_published": (asm == info["checkpoint_sha256"]) if asm else None,
                 "seconds": round(time.time() - T0, 1), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    json.dump(info, open(A.out, "w"), indent=1); print("-> " + A.out)

if __name__ == "__main__":
    mp.freeze_support()
    main()