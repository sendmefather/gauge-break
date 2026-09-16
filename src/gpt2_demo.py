"""gpt2_demo.py -- the GPT-2 small recovery record, replayed on two devices.

GPT-2 is stored in float32, so a recovered value must land within half an
f32 ulp (2^-24 relative) of the stored one, 2^15 tighter than bf16. Three
findings of the record are re-measured here, numpy only, f64 arithmetic:
  A  gauge-free maps (attn.c_proj, fed by the attention concat; mlp.c_proj,
     fed by GELU) solve bit-exactly by least squares with lattice rounds.
  B  LayerNorm-fed maps (attn.c_attn, mlp.c_fc) cannot: the design [ln, 1]
     is rank-deficient by exactly one, its null direction is (1/g, -C) with
     C = sum_j b_j / g_j, so one free scalar per output column leaves the
     output unchanged. Measured: rank, null cosine, the min-norm solution
     bit-exactness (about zero) and its output error (about 1e-14).
  C  quantisation is information: the gauge orbit is a line, and the true
     column is where that line meets the f32 grid in all 768 coordinates
     at once. Candidates are enumerated on the coarsest coordinate and
     survivors scored by the worst grid distance; the best pins the column.
Usage: python gpt2_demo.py <snapshot_dir> [--blocks B] [--nseq S] [--pin-cols K] [--out f.json]
"""
import argparse, hashlib, json, os, platform, re, socket, struct, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("snapshot")
ap.add_argument("--blocks", type=int, default=None)
ap.add_argument("--nseq", type=int, default=8)
ap.add_argument("--seed", type=int, default=32)
ap.add_argument("--pin-cols", type=int, default=8)
ap.add_argument("--window", type=float, default=0.25, help="alpha window for the lattice search")
ap.add_argument("--no-control", action="store_true",
                help="skip the shuffled-X falsifier (it is on by default)")
ap.add_argument("--out", default="gpt2_demo.json")
A = ap.parse_args()
T0 = time.time()

def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        data = np.fromfile(f, dtype=np.uint8)
    out = {}
    for name, m in hdr.items():
        if name == "__metadata__":
            continue
        a, b = m["data_offsets"]
        if m["dtype"] != "F32":
            raise SystemExit("expected an f32 checkpoint, got %s for %s" % (m["dtype"], name))
        out[name] = data[a:b].view(np.float32).reshape(m["shape"])
    return out

SNAP = A.snapshot
CFG = json.load(open(os.path.join(SNAP, "config.json")))
D, NL, NH, V, P = CFG["n_embd"], CFG["n_layer"], CFG["n_head"], CFG["vocab_size"], CFG["n_positions"]
DH, DI, EPS = D // NH, 4 * D, CFG["layer_norm_epsilon"]
ckpt_path = os.path.join(SNAP, "model.safetensors")
W32 = read_safetensors(ckpt_path)
BUFFER = re.compile(r"^h\.\d+\.attn\.(bias|masked_bias)$")
PARAMS = sorted(n for n in W32 if not BUFFER.match(n))
DEN = sum(int(W32[n].size) for n in PARAMS)
CONFIG_DEN = V * D + P * D + NL * (4 * D + D * 3 * D + 3 * D + D * D + D + D * DI + DI + DI * D + D) + 2 * D
assert DEN == CONFIG_DEN, "denominator mismatch %d vs %d" % (DEN, CONFIG_DEN)
NL_RUN = NL if A.blocks is None else min(A.blocks, NL)
NSEQ, T = A.nseq, min(1024, P)
N = NSEQ * T
rng = np.random.default_rng(A.seed)
ids = rng.integers(0, V, size=(NSEQ, T))
ids_sha = hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()

def g(name):
    return W32[name].astype(np.float64)

def bits32(a32):
    return np.ascontiguousarray(a32).view(np.uint32)

def snap32(a64):
    return a64.astype(np.float32)             # round to nearest even

info = {"host": socket.gethostname(), "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "longdouble_eps": float(np.finfo(np.longdouble).eps), "snapshot": os.path.abspath(SNAP),
        "D": D, "L": NL, "NH": NH, "V": V, "N": N, "NSEQ": NSEQ, "seed": A.seed, "ids_sha256": ids_sha,
        "blocks_run": NL_RUN, "denominator": DEN, "tensors": len(PARAMS),
        "checkpoint_sha256": hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest(),
        "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
print("gpt2_demo on %s (%s, numpy %s)" % (info["host"], info["platform"], np.__version__), flush=True)
print("  GPT-2 D=%d L=%d NH=%d V=%d  parameters %s (%d tensors, %d mask buffers skipped)"
      % (D, NL, NH, V, f"{DEN:,}", len(PARAMS), len(W32) - len(PARAMS)))
print("  N=%d (%d x %d) seed %d  ids sha256 %s" % (N, NSEQ, T, A.seed, ids_sha[:16]))
print("  checkpoint sha256 %s" % info["checkpoint_sha256"], flush=True)

REC = {}
CK = {}
REC_LN = REC
TALLY = {"ok": 0, "total": 0, "miss": {}}
CONTROL = {}
# Separate stream on purpose: drawing the permutation from `rng` would shift
# every later draw and the controlled run would stop being the same run.
crng = np.random.default_rng(A.seed + 991)
def tally(name, rec32, part=None):
    tb = bits32(W32[name]).ravel() if part is None else bits32(W32[name])[part].ravel()
    rb = bits32(np.ascontiguousarray(rec32)).ravel()
    key = name if part is None else "%s[%s]" % (name, part)
    REC[key] = rb; CK[key] = tb
    eq = tb == rb
    ok, tot = int(eq.sum()), int(eq.size)
    TALLY["ok"] += ok; TALLY["total"] += tot
    if ok != tot:
        TALLY["miss"][key] = tot - ok
        print("    %-30s %s/%s  MISS %d" % (key, f"{ok:,}", f"{tot:,}", tot - ok), flush=True)
    else:
        print("    %-30s %s/%s" % (key, f"{ok:,}", f"{tot:,}"), flush=True)
    return ok, tot

def layernorm_parts(x, gm, bt):
    mu = x.mean(-1, keepdims=True)
    u = (x - mu) / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + EPS)
    return u, u * gm + bt

def gelu_new(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

def solve_affine(X, Y):
    """Least squares for Y = X W + b, full-rank design, f32 lattice rounds."""
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
        b = bits32(snap32(W))
        if prev is not None and np.array_equal(b, prev):
            break
        prev = b
        Ws = snap32(W).astype(np.float64)
        W = Ws + (Gi @ (Xs.T @ (Y - Xa @ Ws))) / cs[:, None]
    res = float(np.abs(Y - Xa @ W).max())
    return W[:-1], W[-1], res

def minnorm_affine(X, Y, gm, bt):
    """LN-fed map: measure the rank deficiency, the null direction, and the min-norm fit."""
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    U, s, Vt = np.linalg.svd(Xa, full_matrices=False)
    rank = int((s > s[0] * 1e-9).sum())
    null = Vt[-1]
    pred = np.append(1.0 / gm, -np.sum(bt / gm))
    cos = float(abs(null @ pred) / (np.linalg.norm(null) * np.linalg.norm(pred)))
    keep = s > s[0] * 1e-9
    Wmn = (Vt[keep].T * (1.0 / s[keep])) @ (U[:, keep].T @ Y)
    res = float(np.abs(Y - Xa @ Wmn).max())
    return Wmn, rank, Xa.shape[1], cos, s[-1] / s[0], res

def f32_grid_between(lo, hi):
    """Every float32 in [lo, hi], enumerated as consecutive bit patterns."""
    out = []
    lo32, hi32 = np.float32(lo), np.float32(hi)
    if hi32 >= 0:
        a = max(lo32, np.float32(0.0))
        pa, pb = int(bits32(np.array([a], np.float32))[0]), int(bits32(np.array([hi32], np.float32))[0])
        out.append(np.arange(pa, pb + 1, dtype=np.uint32).view(np.float32))
    if lo32 < 0:
        b = min(hi32, np.float32(-0.0))
        pa, pb = int(bits32(np.array([-b], np.float32))[0]), int(bits32(np.array([-lo32], np.float32))[0])
        out.append(-np.arange(pa, pb + 1, dtype=np.uint32).view(np.float32))
    return np.concatenate(out) if out else np.zeros(0, np.float32)

def ulp_dist(w64):
    """Distance of f64 values to the nearest f32 grid point, in ulps of that point."""
    w32 = snap32(w64)
    return np.abs(w64 - w32.astype(np.float64)) / np.abs(np.spacing(w32)).astype(np.float64)

def pin_column(w_mn, b_mn, gm, C, window, n_reject=24):
    """Break the gauge on one column: true w_j = w_mn_j + alpha/g_j, b = b_mn - alpha*C."""
    inv_g = 1.0 / gm
    spacing = np.abs(gm) * np.abs(np.spacing(snap32(w_mn))).astype(np.float64)
    jstar = int(np.argmax(spacing))
    # the window never crosses zero, where the f32 grid is dense: half-width
    # is the alpha window mapped to this coordinate, capped at half its value
    half = min(window * abs(inv_g[jstar]), 0.5 * abs(w_mn[jstar]))
    lo, hi = w_mn[jstar] - half, w_mn[jstar] + half
    grid = f32_grid_between(lo, hi).astype(np.float64)
    alphas = (grid - w_mn[jstar]) * gm[jstar]
    n_cand = int(alphas.size)
    order = np.argsort(-spacing); rej = [j for j in order if j != jstar][:n_reject]
    kept = []
    for c0 in range(0, alphas.size, 4_000_000):
        surv = alphas[c0:c0 + 4_000_000]
        for j in rej:
            d = ulp_dist(w_mn[j] + surv * inv_g[j])
            surv = surv[d < 0.05]
            if surv.size == 0:
                break
        kept.append(surv)
    surv = np.concatenate(kept)
    scores = []
    for a in surv:
        d = ulp_dist(w_mn + a * inv_g).max()
        db = float(ulp_dist(np.array([b_mn - a * C]))[0])
        scores.append(max(d, db))
    scores = np.array(scores)
    if scores.size == 0:
        return None
    o = np.argsort(scores)
    best = float(surv[o[0]]); s_best = float(scores[o[0]])
    s_next = float(scores[o[1]]) if scores.size > 1 else float("inf")
    w_rec = snap32(w_mn + best * inv_g); b_rec = snap32(np.array([b_mn - best * C]))[0]
    return {"jstar": jstar, "candidates": n_cand, "survivors": int(surv.size), "alpha_window": half * abs(gm[jstar]),
            "alpha": best, "score_ulp": s_best, "runner_up_ulp": s_next,
            "margin": (s_next / s_best) if s_best > 0 else float("inf"),
            "w": w_rec, "b": b_rec}

# -------------------------------------------------------------------- forward
wte, wpe = g("wte.weight"), g("wpe.weight")
x = wte[ids] + wpe[np.arange(T)][None]
x = x.reshape(N, D)
seqpos = np.arange(T)
causal = np.triu(np.full((T, T), -1e30), 1)
A_class, B_class, LN_class, PINS = [], [], [], []
for li in range(NL_RUN):
    t0 = time.time(); Pf = "h.%d." % li
    g1, b1 = g(Pf + "ln_1.weight"), g(Pf + "ln_1.bias")
    g2, b2 = g(Pf + "ln_2.weight"), g(Pf + "ln_2.bias")
    Wa, ba = g(Pf + "attn.c_attn.weight"), g(Pf + "attn.c_attn.bias")
    Wp, bp = g(Pf + "attn.c_proj.weight"), g(Pf + "attn.c_proj.bias")
    Wf, bf = g(Pf + "mlp.c_fc.weight"), g(Pf + "mlp.c_fc.bias")
    Wm, bm = g(Pf + "mlp.c_proj.weight"), g(Pf + "mlp.c_proj.bias")
    u1, ln1 = layernorm_parts(x, g1, b1)
    qkv = ln1 @ Wa + ba
    ao = np.empty((N, D))
    for s in range(NSEQ):
        blk = qkv[s * T:(s + 1) * T]
        q = blk[:, :D].reshape(T, NH, DH).transpose(1, 0, 2)
        k = blk[:, D:2 * D].reshape(T, NH, DH).transpose(1, 0, 2)
        v = blk[:, 2 * D:].reshape(T, NH, DH).transpose(1, 0, 2)
        sc = (q @ k.transpose(0, 2, 1)) / np.sqrt(DH) + causal
        e = np.exp(sc - sc.max(-1, keepdims=True)); a = e / e.sum(-1, keepdims=True)
        ao[s * T:(s + 1) * T] = (a @ v).transpose(1, 0, 2).reshape(T, D)
    Yp = ao @ Wp + bp
    x = x + Yp
    u2, ln2 = layernorm_parts(x, g2, b2)
    Yf = ln2 @ Wf + bf
    gel = gelu_new(Yf)
    Ym = gel @ Wm + bm
    x = x + Ym
    print("[block %d] harvest %.0fs" % (li, time.time() - t0), flush=True)
    # LayerNorm parameters: two unknowns per feature, no gauge
    for (u, ln, gn, bn) in ((u1, ln1, Pf + "ln_1.weight", Pf + "ln_1.bias"), (u2, ln2, Pf + "ln_2.weight", Pf + "ln_2.bias")):
        Wl, bl, _ = solve_affine(u, ln)          # per-feature: ln_j = u_j*g_j + b_j, but solved jointly is diagonal
        gm_rec, bt_rec = np.diag(Wl).copy(), bl.copy()
        # the joint solve returns a diagonal map; recover each feature by its own 1-D regression, as the record does
        um = u - u.mean(0); lm = ln - ln.mean(0)
        gm_rec = (um * lm).sum(0) / (um * um).sum(0)
        bt_rec = ln.mean(0) - gm_rec * u.mean(0)
        ok1, t1 = tally(gn, snap32(gm_rec)); ok2, t2 = tally(bn, snap32(bt_rec))
        LN_class.append((ok1 + ok2, t1 + t2))
    # A: gauge-free maps
    for (X, Y, wn, bn) in ((ao, Yp, Pf + "attn.c_proj.weight", Pf + "attn.c_proj.bias"), (gel, Ym, Pf + "mlp.c_proj.weight", Pf + "mlp.c_proj.bias")):
        t = time.time(); Wr, br, res = solve_affine(X, Y)
        print("    SOLVE %-22s %5.1fs residual %.3e" % (wn, time.time() - t, res), flush=True)
        ok1, t1 = tally(wn, snap32(Wr)); ok2, t2 = tally(bn, snap32(br))
        A_class.append({"map": wn, "ok": ok1 + ok2, "total": t1 + t2, "residual": res})
        # FALSIFIER. Permute the rows of X, breaking the pairing with Y, and run the
        # identical solve. If the recovered bits were arithmetic on the stored weights
        # rather than a fit to the observations, this would change nothing. It must
        # collapse. Deliberately does NOT call tally(): the control is not a result
        # and must never enter the denominator.
        if not A.no_control and li == 0:
            tc = time.time()
            Wc, bc, resc = solve_affine(X[crng.permutation(X.shape[0])], Y)
            tb = bits32(W32[wn]).ravel()
            rb = bits32(np.ascontiguousarray(snap32(Wc))).ravel()
            frac = float((tb == rb).mean())
            CONTROL[wn] = {"shuffled_bitmatch": frac, "residual": resc,
                           "threshold": 0.001, "passed": bool(frac < 0.001)}
            print("    CONTROL shuffled-X %-22s bit-match %.6f  residual %.3e  %5.1fs  (must be < 0.001)"
                  % (wn, frac, resc, time.time() - tc), flush=True)
    # B: LayerNorm-fed maps, measured as the record measured them
    for (X, Y, wn, bn, gm, bt) in ((ln1, qkv, Pf + "attn.c_attn.weight", Pf + "attn.c_attn.bias", g1, b1), (ln2, Yf, Pf + "mlp.c_fc.weight", Pf + "mlp.c_fc.bias", g2, b2)):
        t = time.time(); Wmn, rank, cols, cos, smin, res = minnorm_affine(X, Y, gm, bt)
        frac = float((bits32(snap32(Wmn[:-1])) == bits32(W32[wn])).mean())
        print("    %-22s design rank %d of %d, null cos %.6f, smallest sv ratio %.1e, output error %.2e, min-norm bit-exact %.6f  (%.1fs)"
              % (wn, rank, cols, cos, smin, res, frac, time.time() - t), flush=True)
        B_class.append({"map": wn, "rank": rank, "cols": cols, "null_cos": cos, "sv_ratio": float(smin),
                        "output_error": res, "minnorm_bitexact": frac, "weights": int(W32[wn].size)})
        if li == 0 and wn.endswith("c_attn.weight") and A.pin_cols > 0:
            # C: break the gauge on the f32 lattice, using the RECOVERED LayerNorm parameters
            gm_rec = REC_LN[Pf + "ln_1.weight"].view(np.float32).astype(np.float64); bt_rec = REC_LN[Pf + "ln_1.bias"].view(np.float32).astype(np.float64)
            C = float(np.sum(bt_rec / gm_rec))
            Wtrue = W32[wn]; btrue = W32[bn]
            recw = np.empty((D, A.pin_cols), np.float32); recb = np.empty(A.pin_cols, np.float32)
            for kcol in range(A.pin_cols):
                t = time.time()
                r = pin_column(Wmn[:-1, kcol], Wmn[-1, kcol], gm_rec, C, A.window)
                if r is None:
                    print("    PIN col %d: no survivor inside the window" % kcol); recw[:, kcol] = np.nan; recb[kcol] = np.nan
                    PINS.append({"col": kcol, "survivors": 0}); continue
                okb = int((bits32(r["w"]) == bits32(Wtrue[:, kcol])).sum()) + int(bits32(np.array([r["b"]]))[0] == bits32(btrue)[kcol])
                alpha_true = float(((Wtrue[:, kcol].astype(np.float64) - Wmn[:-1, kcol]) * gm_rec).mean())
                print("    PIN col %d: j*=%d  %s candidates -> %d survivors  window %.3f alpha %.6e (true %.6e)  score %.2e ulp, runner-up %.2e, margin %.0fx  bits %d/%d  (%.1fs)"
                      % (kcol, r["jstar"], f"{r['candidates']:,}", r["survivors"], r["alpha_window"], r["alpha"], alpha_true, r["score_ulp"], r["runner_up_ulp"], r["margin"], okb, D + 1, time.time() - t), flush=True)
                recw[:, kcol] = r["w"]; recb[kcol] = r["b"]
                PINS.append({"col": kcol, "jstar": r["jstar"], "candidates": r["candidates"], "survivors": r["survivors"],
                             "alpha": r["alpha"], "alpha_true": alpha_true, "alpha_window": r["alpha_window"], "score_ulp": r["score_ulp"],
                             "runner_up_ulp": r["runner_up_ulp"], "margin": r["margin"], "bits_ok": okb, "bits_total": D + 1})
            tally(wn, recw, part=(slice(None), slice(0, A.pin_cols)))
            tally(bn, recb, part=slice(0, A.pin_cols))
    print("[block %d] done %.0fs   running %s/%s" % (li, time.time() - t0, f"{TALLY['ok']:,}", f"{TALLY['total']:,}"), flush=True)

# LayerNorm parameters live in REC under their names; expose for the pin step (defined lazily above)
if NL_RUN == NL:
    gf, bfin = g("ln_f.weight"), g("ln_f.bias")
    uf, lnf = layernorm_parts(x, gf, bfin)
    um = uf - uf.mean(0); lm = lnf - lnf.mean(0)
    gm_rec = (um * lm).sum(0) / (um * um).sum(0); bt_rec = lnf.mean(0) - gm_rec * uf.mean(0)
    ok1, t1 = tally("ln_f.weight", snap32(gm_rec)); ok2, t2 = tally("ln_f.bias", snap32(bt_rec))
    LN_class.append((ok1 + ok2, t1 + t2))

h_rec, h_ck = hashlib.sha256(), hashlib.sha256()
for k in sorted(REC):
    h_rec.update(np.ascontiguousarray(REC[k]).tobytes())
    h_ck.update(np.ascontiguousarray(CK[k]).tobytes())
a_ok, a_tot = sum(r["ok"] for r in A_class), sum(r["total"] for r in A_class)
ln_ok, ln_tot = sum(r[0] for r in LN_class), sum(r[1] for r in LN_class)
pin_ok, pin_tot = sum(p.get("bits_ok", 0) for p in PINS), sum(p.get("bits_total", 0) for p in PINS)
print("=" * 70)
print("A  gauge-free maps (attn.c_proj, mlp.c_proj, weights+biases): %s of %s" % (f"{a_ok:,}", f"{a_tot:,}"))
print("LN LayerNorm gains and biases:                                 %s of %s" % (f"{ln_ok:,}", f"{ln_tot:,}"))
print("B  LayerNorm-fed maps, min-norm bit-exact fraction: %s" % ", ".join("%.5f" % r["minnorm_bitexact"] for r in B_class[:4]) + (" ..." if len(B_class) > 4 else ""))
print("B  design rank deficiency: %s" % ", ".join("%d/%d" % (r["rank"], r["cols"]) for r in B_class[:4]) + (" ..." if len(B_class) > 4 else ""))
print("C  lattice-pinned columns of h.0.attn.c_attn: %s of %s bits" % (f"{pin_ok:,}", f"{pin_tot:,}"))
print("TALLIED %s of %s bit-exact over %d recovered tensors" % (f"{TALLY['ok']:,}", f"{TALLY['total']:,}", len(REC)))
print("sha256 recovered bits: %s" % h_rec.hexdigest())
print("sha256 checkpoint bits: %s" % h_ck.hexdigest())
print("bit-identical: %s" % (h_rec.hexdigest() == h_ck.hexdigest()))
print("total %.0fs" % (time.time() - T0))
info.update({"A": A_class, "B": B_class, "LN": {"ok": ln_ok, "total": ln_tot}, "pins": PINS,
             "A_ok": a_ok, "A_total": a_tot, "pin_ok": pin_ok, "pin_total": pin_tot,
             "ok": TALLY["ok"], "tallied": TALLY["total"], "misses": TALLY["miss"],
             "sha256_recovered_bits": h_rec.hexdigest(), "sha256_checkpoint_bits": h_ck.hexdigest(),
             "bit_identical": h_rec.hexdigest() == h_ck.hexdigest(),
             "control": CONTROL, "seconds": round(time.time() - T0, 1), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
json.dump(info, open(A.out, "w"), indent=1)
print("-> " + A.out)