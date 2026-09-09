"""
Run this FIRST on your machine, before generating data or training:

    python tests/test_torch.py

It checks the parts that are easy to get silently wrong: the parallel scan
against a naive recurrence, chunk-boundary handling, parameter matching between
the complex model and its real twin, gradient flow, the stability of the state
parameterisation, exact phase equivariance of the complex path, and that the
model can overfit a single batch.
"""

import sys

import torch

sys.path.insert(0, ".")

from cvssd.cvssm import CVSelectiveSSM, SSMBlock, selective_scan  # noqa: E402
from cvssd.models import (MODELS, SSDNet, count_params,  # noqa: E402
                          matched_real_width)

ok = lambda n: print(f"  PASS  {n}")
torch.manual_seed(0)


def naive(a, b):
    h = torch.zeros_like(b)
    acc = torch.zeros_like(b[:, 0])
    for t in range(b.shape[1]):
        acc = a[:, t] * acc + b[:, t]
        h[:, t] = acc
    return h


# 1. scan == naive recurrence, complex and real, with awkward lengths
for dtype in (torch.cfloat, torch.float32):
    for L, chunk in [(64, 64), (100, 16), (129, 32), (1024, 64)]:
        a = torch.randn(2, L, 3, 4, dtype=dtype) * 0.3
        b = torch.randn(2, L, 3, 4, dtype=dtype)
        assert torch.allclose(selective_scan(a, b, chunk), naive(a, b),
                              atol=1e-4), (dtype, L, chunk)
ok("selective_scan == naive recurrence (complex & real, ragged lengths)")

# 2. chunking is invariant
a = torch.randn(2, 256, 3, 4, dtype=torch.cfloat) * 0.4
b = torch.randn(2, 256, 3, 4, dtype=torch.cfloat)
r = [selective_scan(a, b, c) for c in (16, 64, 256)]
assert all(torch.allclose(r[0], x, atol=1e-4) for x in r[1:])
ok("scan invariant to chunk size")

# 3. state parameterisation is contractive
ssm = CVSelectiveSSM(8, 4)
A = ssm.A()
assert torch.all(A.real < 0), "Re(A) must be negative"
assert torch.all(torch.abs(torch.exp(torch.rand(8, 4).abs() * 5 * A)) < 1.0)
ok("A parameterisation gives |exp(dt*A)| < 1 (stable)")

# 4. shapes
z = torch.randn(2, 128, 8, dtype=torch.cfloat)
assert ssm(z).shape == z.shape and ssm(z).is_complex()
blk = SSMBlock(8, 4, complex_mode=True)
assert blk(z, torch.zeros(2, 1)).shape == z.shape
ok("CVSelectiveSSM / SSMBlock preserve shape and dtype")

# 5. full models forward
x = torch.randn(4, 2, 256)
for name in MODELS:
    kw = dict(d_model=12, d_state=4, n_stages=2, blocks_per_stage=1)
    m = MODELS[name](**(kw if name in ("cvssd", "rvssd") else {}))
    y, s = m(x)
    assert y.shape == x.shape, (name, y.shape)
    assert s.shape == (4, 1), (name, s.shape)
    assert torch.isfinite(y).all(), name
ok("all models forward with correct shapes and finite output")

# 6. gradients reach every parameter
m = SSDNet(d_model=12, d_state=4, n_stages=2, blocks_per_stage=1)
y, s = m(x)
(y.square().mean() + s.square().mean()).backward()
dead = [n for n, p in m.named_parameters() if p.grad is None]
assert not dead, f"no gradient for: {dead[:5]}"
ok("gradients reach every parameter of CV-SSD")

# 7. parameter matching between complex and real twin
kw = dict(d_state=4, n_stages=2, blocks_per_stage=1)
d_real = matched_real_width(24, **kw)
n_c = count_params(SSDNet(d_model=24, complex_mode=True, **kw))
n_r = count_params(SSDNet(d_model=d_real, complex_mode=False, **kw))
rel = abs(n_c - n_r) / n_c
assert rel < 0.10, (n_c, n_r, rel)
ok(f"param-matched twin: complex d=24 ({n_c/1e3:.1f}k) vs real d={d_real} "
   f"({n_r/1e3:.1f}k), {100*rel:.1f}% apart")

# 8. exact phase equivariance of the complex path (and its absence in the real one)
def rotate(x, th):
    c, s = torch.cos(th), torch.sin(th)
    return torch.stack([c * x[:, 0] - s * x[:, 1],
                        s * x[:, 0] + c * x[:, 1]], dim=1)


th = torch.tensor(0.7)
mc = SSDNet(d_model=12, d_state=4, n_stages=2, blocks_per_stage=1,
            complex_mode=True).eval()
with torch.no_grad():
    y1 = mc(rotate(x, th))[0]
    y2 = rotate(mc(x)[0], th)
err = (y1 - y2).abs().max() / y2.abs().max()
assert err < 1e-4, f"complex model is not phase-equivariant: {err:.2e}"
ok(f"CV-SSD is exactly phase-equivariant (rel. err {err:.1e})")

mr = SSDNet(d_model=12, d_state=4, n_stages=2, blocks_per_stage=1,
            complex_mode=False).eval()
with torch.no_grad():
    e_r = ((mr(rotate(x, th))[0] - rotate(mr(x)[0], th)).abs().max()
           / mr(x)[0].abs().max())
assert e_r > 1e-3, "real twin unexpectedly equivariant; ablation is void"
ok(f"real twin is not phase-equivariant (rel. err {e_r:.1e}) - ablation is valid")

# 9. selection='riparts' must break equivariance (the ablation switch works)
ma = SSDNet(d_model=12, d_state=4, n_stages=2, blocks_per_stage=1,
            complex_mode=True, selection="riparts").eval()
with torch.no_grad():
    e_a = ((ma(rotate(x, th))[0] - rotate(ma(x)[0], th)).abs().max()
           / ma(x)[0].abs().max())
assert e_a > 1e-3, "selection ablation had no effect"
ok("selection='riparts' breaks equivariance as expected")

# 10. can overfit one batch
torch.manual_seed(0)
m = SSDNet(d_model=16, d_state=4, n_stages=2, blocks_per_stage=1)
xb = torch.randn(4, 2, 128)
tb = torch.randn(4, 2, 128) * 0.1
opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
l0 = None
for i in range(150):
    opt.zero_grad()
    loss = torch.nn.functional.mse_loss(m(xb)[0], tb)
    loss.backward()
    opt.step()
    if i == 0:
        l0 = float(loss)
assert float(loss) < 0.25 * l0, (l0, float(loss))
ok(f"overfits a single batch ({l0:.4f} -> {float(loss):.4f})")

print("\nAll torch tests passed.")
