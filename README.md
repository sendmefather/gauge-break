# gauge-break

**Every weight of GPT-2 small, recovered bit-exactly from its own activations.**

124,439,808 of 124,439,808 parameters. The assembled file's sha256 equals the
published checkpoint's. Reproduced on two machines sharing no instruction set,
operating system, or extended-precision implementation.

The method is here. Run it yourself.

---

## The claim, in one paragraph

Watch a transformer's internal activations and you can solve for its weights.
For most layers this is ordinary least squares. For layers fed by a LayerNorm it
is **provably impossible** &mdash; the normaliser's centring creates an exact
continuous symmetry, so an entire line of weight matrices produces byte-identical
outputs, and no estimator can tell them apart. That impossibility is real in
&#8477;. It is not real in float32. The symmetry is continuous; the stored weights
sit on a discrete grid; and a line through a 769-dimensional lattice touches a
grid point in every coordinate at exactly one place.

Full derivation: **[the paper](https://www.d0re.com/paper)**.

## The mathematics, compressed

LayerNorm emits `ln_j = g_j * xhat_j + beta_j` with `sum_j xhat_j = 0`. So for
**every** input:

```
sum_j ln_j / g_j  =  sum_j beta_j / g_j  =:  C
```

The left side is observable; the right side is a constant. The augmented design
`[ln, 1]` therefore annihilates `v = (1/g, -C)`, and for any scalar alpha:

```
W' = W + alpha / g        b' = b - alpha * C        =>        X W' + b' = X W + b
```

exactly. Move every weight in the block by an unbounded amount; the output does
not change. Measured: at `alpha = 1e6` individual weights shift by `2.39e+07`
and the output shifts by `1.89e-07`, which is float round-off.

Least squares returns the minimum-norm point of that family. It reproduces the
outputs to `1e-14` and gets **0.00000** of the stored bits.

The grid breaks it. Move alpha by more than `min_j(half-ulp(W_j) * |g_j|)` and
some coordinate falls off the lattice. For one column that tolerance is
`3.2e-12` against a search window of `0.25`. Enumerate: **13,645,404 candidates,
exactly one survivor, 769 of 769 bits.**

A wrong candidate scores `~0.5 ulp`; the true one scores `3.7e-09`. The
probability a wrong alpha passes all 769 coordinates is **10^-6250**.

## Run it

Needs Python 3.10+, numpy, and about 8 GB of RAM. No GPU, no ML framework.

```bash
pip install numpy
huggingface-cli download openai-community/gpt2 --local-dir gpt2
```

**Verify the gauge algebra against the public checkpoint** (seconds, reads only
stored tensors, no recovery):

```bash
python src/check_gauge.py gpt2/model.safetensors
```

**Recover block 0 and run the falsifier** (about a minute):

```bash
python src/gpt2_demo.py gpt2 --blocks 1 --pin-cols 0 --out demo.json
```

Expected, and the second line is the point:

```
SOLVE h.0.attn.c_proj.weight   0.8s residual 2.132e-14
    h.0.attn.c_proj.weight         589,824/589,824
    CONTROL shuffled-X h.0.attn.c_proj.weight bit-match 0.000000  (must be < 0.001)
```

**The lattice break on eight columns** (a few minutes):

```bash
python src/gpt2_demo.py gpt2 --blocks 1 --pin-cols 8 --out pins.json
```

**Every parameter, assembled and hashed** (about an hour, 8 GB):

```bash
python src/gpt2_full.py gpt2 --out full.json
```

It prints `assembled_equals_published True` or it fails. There is no third
outcome.

## The falsifier

A method that returns the right answer whatever you feed it has shown nothing.
`gpt2_demo.py` permutes the rows of `X`, breaking their pairing with `Y`, and
runs the identical solve:

| map | real solve | rows of X permuted |
|---|---|---|
| `h.0.attn.c_proj.weight` | 590,592 / 590,592 bits | bit-match **0.000000000** |
| `h.0.mlp.c_proj.weight` | 2,360,064 / 2,360,064 bits | bit-match **0.000000000** |
| residual | 2.13e-14 / 2.84e-14 | 2.11e+01 / 6.47e+01 |

Exactly zero bits, residuals fifteen orders of magnitude larger.

## What is in here

```
src/check_gauge.py   gauge algebra against the public checkpoint; no recovery code
src/gpt2_demo.py     one block, the lattice break, and the shuffled control
src/gpt2_full.py     all 124,439,808 parameters, assembled and hashed
records/             the JSON run records behind every number quoted above
data/column-0.txt    one column in full: g, stored, min-norm, alpha, checksums
paper/               the paper, also at d0re.com/paper
```

## Why this is public

Byte-equality with a public checkpoint is consistent with having copied the
checkpoint. That objection cannot be answered with more numbers about GPT-2,
because the target is public. There are two ways out: recruit an auditor who
commits to a private model's hash first, or publish the method so anyone can run
it against a target of their own choosing.

This is the second. Point it at a model we have never seen and stop taking our
word for anything.

## Reproduced

| | machine-a | machine-b |
|---|---|---|
| architecture | AMD64 / Windows 11 | aarch64 / Linux |
| extended precision | double-double | quad |
| longdouble eps | 2.22e-16 | 1.93e-34 |
| high-precision columns | 88 | 89 |
| sha256 of recovered bits | `e7ddb1fe...70eec2` | `e7ddb1fe...70eec2` |

Different routes through the work. Identical bits.

The run records in `records/` are the originals with two substitutions: the
`host` field carries `machine-a`/`machine-b` in place of the real hostnames, and
`snapshot` has its home directory replaced by `<local>`. Nothing else was
touched; every measured quantity is as the run wrote it.

## Status

Not peer reviewed. Before publication the argument was put to two language
models in separate sessions and prompted to attack it; that is adversarial
review, not peer review. They found two defects, both corrected: a mislabelled
comparison column, and a one-sided grid tolerance that returned negative
half-widths against the model's single negative LayerNorm gain
(`h.3.ln_2.weight[266] = -2.5572e-04`). The mathematics came through unamended.

Target: `openai-community/gpt2`, sha256
`248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`.

MIT licensed. Dimension Zero Reverse Engineering &middot;
[d0re.com](https://www.d0re.com)