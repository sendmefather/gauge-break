"""Check the LayerNorm gauge claim against the PUBLIC GPT-2 checkpoint.

This settles one question and nothing else: is alpha a fitted knob, or is it
determined by the stored weights? It needs no recovery code, no harvested
activations, and nothing from us. It reads openai-community/gpt2, whose
sha256 is 248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707,
and recomputes alpha from scratch.

  python check_gauge.py path/to/model.safetensors

The algebra it uses is the one already agreed to be standard. After LayerNorm,
ln_j = g_j * xhat_j + beta_j with sum_j xhat_j = 0, so for EVERY input

    sum_j ln_j / g_j = sum_j beta_j / g_j =: C

which means the design [ln, 1] annihilates v = (1/g, -C). That v is the gauge
direction, and shifting W[:,k] += a/g, b_k -= a*C leaves the output unchanged.

The minimum-norm least-squares solution is by definition orthogonal to the
nullspace. So the offset between min-norm and stored is a PROJECTION:

    alpha_k = <(W[:,k], b_k), v> / <v, v>

Nothing is fitted. If the published alpha values are real, this reproduces
them from the public checkpoint alone.

Then it asks the grid question: how far can alpha move before any float32 bit
of the column changes? That number explains why two machines whose alpha
disagreed in the 16th decimal still recovered identical bits, and the control
at the end shows the test is capable of failing.
"""
import json, struct, sys
import numpy as np

# Reported by the two runs, machine-a and machine-b, h.0.attn.c_attn.
ALPHA_DESKTOP = [-2.25356813227178517e-03, 2.70811699038357809e-03, 2.26760958823000102e-03,
                 8.78706153379211824e-04, 2.02332665096445962e-04, 7.94487659798388205e-04,
                 2.34088600615995094e-03, 4.08483942033393090e-04]
ALPHA_SPARK = [-2.25356813227131593e-03, 2.70811699038388601e-03, 2.26760958823029593e-03,
               8.78706153378738569e-04, 2.02332665096622335e-04, 7.94487659798155535e-04,
               2.34088600615982127e-03, 4.08483942034695759e-04]


def tensors(path):
    f = open(path, "rb")
    hl = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(hl))
    base = 8 + hl

    def get(name):
        e = hdr[name]
        s, t = e["data_offsets"]
        f.seek(base + s)
        return np.frombuffer(f.read(t - s), dtype=np.float32).reshape(e["shape"])
    return get


def grid_tol(col_f32, g):
    """Largest move in alpha, each way, that flips no float32 bit of the column.

    W_j(a) = wmin_j + a/g_j, so d W_j = da / g_j. Two things follow that an
    up-only, sign-blind formula gets wrong, and a reviewer caught both:

      1. the distance to the next float32 up is not the distance down, at a
         power-of-two boundary they differ by a factor of two;
      2. a positive move in alpha RAISES W_j when g_j > 0 and LOWERS it when
         g_j < 0, so the sign of the gain decides which of those two distances
         binds.

    GPT-2 does carry a negative gain: h.3.ln_2.weight[266] = -2.5572e-04, the
    single non-positive LayerNorm gain in the checkpoint. Against columns fed by
    that tensor the old one-line form returned NEGATIVE half-widths, which is
    not a distance. It happened not to bite on h.0.attn.c_attn, where every gain
    is positive, but it was wrong as written.

    Returns (tol_plus, tol_minus), both positive. Checked against a bisection
    search for the first actual bit flip: they agree to every printed digit.
    """
    f64 = col_f32.astype(np.float64)
    up = (np.nextafter(col_f32, np.float32(np.inf)).astype(np.float64) - f64) / 2.0
    dn = (f64 - np.nextafter(col_f32, np.float32(-np.inf)).astype(np.float64)) / 2.0
    pos = g > 0
    return (float(np.min(np.where(pos, up * g, dn * -g))),
            float(np.min(np.where(pos, dn * g, up * -g))))


def main(path):
    T = tensors(path)
    g = T("h.0.ln_1.weight").astype(np.float64)
    beta = T("h.0.ln_1.bias").astype(np.float64)
    W = T("h.0.attn.c_attn.weight")
    b = T("h.0.attn.c_attn.bias").astype(np.float64)

    C = float(np.sum(beta / g))
    inv = 1.0 / g
    vv = float(np.sum(inv * inv) + C * C)
    print("C     = sum_j beta_j/g_j      = %+.17e" % C)
    print("<v,v> = sum_j 1/g_j^2 + C^2   = %+.17e" % vv)
    print()
    print(" col | alpha recomputed here        | alpha published (desktop)    |  rel. difference")
    print("-" * 94)
    worst = 0.0
    for k in range(8):
        a = (float(np.sum(W[:, k].astype(np.float64) * inv)) - b[k] * C) / vv
        rel = abs(a - ALPHA_DESKTOP[k]) / abs(ALPHA_DESKTOP[k])
        worst = max(worst, rel)
        print(" %3d | %+.17e | %+.17e | %.2e" % (k, a, ALPHA_DESKTOP[k], rel))
    print()
    print("worst relative difference: %.2e  (float64 round-off on a 769-term dot product)" % worst)

    print()
    print("=== grid question: how far can alpha move before a float32 bit flips? ===")
    print(" col | machines' alpha differ by |  grid half-width  |  ratio | bits, desktop a | bits, spark a | CONTROL 2x tol")
    print("-" * 118)
    assert np.all(g != 0), "a zero LayerNorm gain leaves 1/g undefined"
    ok = ctrl = True
    for k in range(8):
        col = W[:, k].copy()
        f64 = col.astype(np.float64)
        tol = min(grid_tol(col, g))
        wmin = f64 - ALPHA_DESKTOP[k] * inv

        def nbits(a):
            rec = (wmin + a * inv).astype(np.float32)
            return int(np.sum(rec.view(np.uint32) == col.view(np.uint32)))
        move = abs(ALPHA_DESKTOP[k] - ALPHA_SPARK[k])
        d, s, c = nbits(ALPHA_DESKTOP[k]), nbits(ALPHA_SPARK[k]), nbits(ALPHA_DESKTOP[k] + 2 * tol)
        ok = ok and d == 768 and s == 768
        ctrl = ctrl and c < 768
        print(" %3d | %25.3e | %17.3e | %6.4f | %15s | %13s | %s"
              % (k, move, tol, move / tol, "%d/768" % d, "%d/768" % s, "%d/768" % c))
    print()
    print("  every stored bit reproduced from both machines' alpha : %s" % ok)
    print("  a wrong alpha (2x the tolerance) breaks bits           : %s" % ctrl)
    print()
    print("  The second line is the point. If it said False the first line would mean nothing.")
    return 0 if (ok and ctrl) else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
