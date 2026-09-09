import numpy as np
import sys
sys.path.insert(0, ".")

# --------------------------------------------------------------------------
# Helper functions and test setup
# --------------------------------------------------------------------------
ok = lambda n: print(f"  PASS  {n}")
rng = np.random.default_rng(0)
THETA = 0.7
ROT = np.exp(1j * THETA)

def err(f, z):
    """Relative equivariance error of a map f."""
    a, b = f(ROT * z), ROT * f(z)
    return np.abs(a - b).max() / (np.abs(b).max() + 1e-12)

# --------------------------------------------------------------------------
# Primitives (mirroring the torch code)
# --------------------------------------------------------------------------
D, N, L, B = 6, 4, 32, 2

def complex_linear(z, W):
    """Bias-free ComplexLinear"""
    return z @ W.T

def complex_layernorm(z, g):
    """ComplexLayerNorm"""
    p = (z.real ** 2 + z.imag ** 2).mean(-1, keepdims=True)
    return z * (g / np.sqrt(p + 1e-5))

def film_gain(z, g):
    """FiLM, complex mode: real gain"""
    return z * (1 + g)[None, None, :]

def cv_ssm(z, P):
    """CVSelectiveSSM, selection='mag'"""
    m = np.abs(z)
    f = np.concatenate([m, np.log1p(m)], axis=-1)
    dt = np.log1p(np.exp(f @ P["dt"].T + P["dtb"]))
    bc = (f @ P["bc"].T).reshape(z.shape[0], z.shape[1], 2, 2, N)
    Bt = bc[:, :, 0, 0] + 1j * bc[:, :, 0, 1]
    Ct = bc[:, :, 1, 0] + 1j * bc[:, :, 1, 1]
    A = (-np.exp(P["alr"]) + 1j * P["aim"])[None, None]
    dtc = dt[..., None].astype(np.complex128)
    Abar = np.exp(dtc * A)
    Bbar = dtc * Bt[:, :, None, :] * z[..., None]
    h = np.zeros_like(Bbar)
    acc = np.zeros(Bbar.shape[0::2][:1] + (D, N), dtype=complex)
    acc = np.zeros((z.shape[0], D, N), dtype=complex)
    for t in range(z.shape[1]):
        acc = Abar[:, t] * acc + Bbar[:, t]
        h[:, t] = acc
    return (h * Ct[:, :, None, :]).sum(-1) + P["D"] * z

def block(z, P, cond_gain):
    h = complex_layernorm(z, P["ln"])
    h = film_gain(h, cond_gain)
    g = np.abs(h) @ P["gate"].T
    g = g / (1 + np.exp(-g))                                  # SiLU, real
    h = cv_ssm(h, P) * g
    return z + complex_linear(h, P["out"])

def bad_block(z, P, cond_gain):
    """The version that failed: a complex gate squares the phase."""
    h = complex_layernorm(z, P["ln"])
    h = film_gain(h, cond_gain)
    gz = complex_linear(h, P["gateC"])
    gz = gz * (1 / (1 + np.exp(-np.abs(gz))))
    h = cv_ssm(h, P) * gz
    return z + complex_linear(h, P["out"])

# --------------------------------------------------------------------------
# Parameter and input initialization
# --------------------------------------------------------------------------
P = dict(
    ln=rng.normal(size=D),
    D=rng.normal(size=D),
    dt=rng.normal(size=(D, 2 * D)) * 0.2,
    dtb=rng.normal(size=D) * 0.1,
    bc=rng.normal(size=(4 * N, 2 * D)) * 0.2,
    alr=rng.normal(size=(D, N)) * 0.1,
    aim=rng.normal(size=(D, N)),
    gate=rng.normal(size=(D, D)) * 0.3,
    gateC=(rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))) * 0.3,
    out=(rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))) * 0.3,
)
cond = rng.normal(size=D) * 0.2
z = rng.normal(size=(B, L, D)) + 1j * rng.normal(size=(B, L, D))

# --------------------------------------------------------------------------
# Equivariance tests for each primitive and the full block
# --------------------------------------------------------------------------
for name, f in [
    ("bias-free ComplexLinear", lambda t: complex_linear(t, P["out"])),
    ("ComplexLayerNorm (RMS on |z|)", lambda t: complex_layernorm(t, P["ln"])),
    ("FiLM real-gain conditioning", lambda t: film_gain(t, cond)),
    ("CVSelectiveSSM, selection='mag'", lambda t: cv_ssm(t, P)),
    ("full SSMBlock (real magnitude gate)", lambda t: block(t, P, cond)),
]:
    e = err(f, z)
    assert e < 1e-10, (name, e)
    ok(f"{name}: equivariance error {e:.1e}")

# --------------------------------------------------------------------------
# Regression tests: these should break equivariance
# --------------------------------------------------------------------------

# The regression this test was written for
e_bad = err(lambda t: bad_block(t, P, cond), z)
assert e_bad > 1e-2, "complex gate should break equivariance"
ok(f"complex gate DOES break it ({e_bad:.1e}) - the bug the torch test caught")

# A complex bias also breaks it
e_bias = err(lambda t: complex_linear(t, P["out"]) + (0.3 + 0.2j), z)
assert e_bias > 1e-2
ok(f"a complex bias also breaks it ({e_bias:.1e}) - hence bias=False")

# Selection from real/imag parts breaks it (the intended ablation switch)
def cv_ssm_ri(t):
    P2 = dict(P)
    f = np.concatenate([t.real, t.imag], axis=-1)
    dt = np.log1p(np.exp(f @ P2["dt"].T + P2["dtb"]))
    bc = (f @ P2["bc"].T).reshape(t.shape[0], t.shape[1], 2, 2, N)
    Bt = bc[:, :, 0, 0] + 1j * bc[:, :, 0, 1]
    Ct = bc[:, :, 1, 0] + 1j * bc[:, :, 1, 1]
    A = (-np.exp(P2["alr"]) + 1j * P2["aim"])[None, None]
    dtc = dt[..., None].astype(np.complex128)
    Abar, Bbar = np.exp(dtc * A), dtc * Bt[:, :, None, :] * t[..., None]
    h = np.zeros_like(Bbar)
    acc = np.zeros((t.shape[0], D, N), dtype=complex)
    for s in range(t.shape[1]):
        acc = Abar[:, s] * acc + Bbar[:, s]
        h[:, s] = acc
    return (h * Ct[:, :, None, :]).sum(-1) + P2["D"] * t

e_ri = err(cv_ssm_ri, z)
assert e_ri > 1e-2
ok(f"selection='riparts' breaks it ({e_ri:.1e}) - valid ablation switch")

# --------------------------------------------------------------------------
# Final summary
# --------------------------------------------------------------------------
print("\nEquivariance verified. Composition of equivariant maps is equivariant,")
print("so the full CV-SSD (conv -> blocks -> conv, all bias-free) inherits it.")