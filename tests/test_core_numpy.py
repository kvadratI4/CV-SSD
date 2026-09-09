import numpy as np, sys
sys.path.insert(0, '.')
from cvssd.datagen import (constellation, bits_per_symbol, symbols_to_bits,
                           rrc_taps, make_sample, MODULATIONS)
from cvssd.metrics import (nmse_db, snr_gain_db, recover_symbols, ber,
                           evm_percent, evaluate_sample)
from cvssd.reference_scan import sequential_scan, parallel_scan, stable_A, discretize
from cvssd.splits import group_kfold, leave_one_snr_out, check_no_leakage

ok = lambda n: print(f"  PASS  {n}")

# 1. constellations unit power, Gray mapping = 1 bit between neighbours
for m in MODULATIONS:
    c = constellation(m)
    assert abs(np.mean(np.abs(c)**2) - 1) < 1e-9, m
    assert len(c) == 2**bits_per_symbol(m)
for m in MODULATIONS:
    c = constellation(m); k = bits_per_symbol(m)
    bits = symbols_to_bits(np.arange(len(c)), m)
    d = np.abs(c[:, None] - c[None, :]) + np.eye(len(c)) * 1e9
    dmin = d.min()
    nb = np.argwhere(d < dmin * 1.05)          # nearest-neighbour pairs
    ham = np.sum(bits[nb[:, 0]] != bits[nb[:, 1]], axis=1)
    assert np.all(ham == 1), (m, ham.max())
ok("constellations unit-power + valid Gray labelling")

# 2. RRC unit energy, Nyquist ISI-free at symbol instants
g = rrc_taps(0.35, 8, 8)
assert abs(np.sum(g**2) - 1) < 1e-9
rc = np.convolve(g, g)
c0 = (len(rc)-1)//2
taps = rc[c0::8]
assert abs(taps[0] - 1) < 1e-6 and np.max(np.abs(taps[1:])) < 2e-2, taps[:4]
ok("RRC unit energy + Nyquist criterion")

# 3. complex scan: parallel == sequential
rng = np.random.default_rng(0)
for shape in [(3,5,64),(2,1,129)]:
    a = (rng.normal(size=shape)+1j*rng.normal(size=shape))*0.3
    bb = rng.normal(size=shape)+1j*rng.normal(size=shape)
    assert np.allclose(sequential_scan(a,bb), parallel_scan(a,bb), atol=1e-9)
ok("complex parallel scan == sequential reference")

# 4. stability of the parameterisation
A = stable_A(rng.normal(size=(4,8)), rng.normal(size=(4,8))*5)
Ab = discretize(np.abs(rng.normal(size=(4,8)))+0.1, A)
assert np.all(np.abs(Ab) < 1.0), np.abs(Ab).max()
ok("stable_A gives contractive |exp(dA)| < 1")

# 5. receiver sanity: clean waveform -> ~0 BER, tiny EVM
for m in MODULATIONS:
    c, x, sym, idx, h, meta = make_sample(np.random.default_rng(1), m, 40.0)
    s = recover_symbols(c, mod=m, sps=8, n_sym=128, ref_symbols=sym, n_pilot=8)[8:-8]
    e, n = ber(s, idx[8:-8], m)
    assert e == 0, (m, e, n)
    assert evm_percent(s, sym[8:-8]) < 5.0, (m, evm_percent(s, sym[8:-8]))
ok("genie receiver: 0 BER, EVM at the ISI floor, on clean waveform at 40 dB")

# 6. BER degrades monotonically as SNR falls (QPSK)
prev = -1
for snr in [20, 10, 4, 0, -6]:
    tot = err = 0
    for k in range(30):
        c, x, sym, idx, h, meta = make_sample(np.random.default_rng([2,k]), 'QPSK', snr)
        s = recover_symbols(x, mod='QPSK', sps=8, n_sym=128, ref_symbols=sym, n_pilot=8)[8:-8]
        e, n = ber(s, idx[8:-8], 'QPSK'); err += e; tot += n
    r = err/tot
    assert r >= prev - 1e-9, (snr, r, prev)
    prev = r
assert prev > 0.05, prev
ok("BER increases monotonically as SNR drops (QPSK)")

# 7. impairments run and an oracle 'denoiser' scores a positive SNR gain
c, x, sym, idx, h, meta = make_sample(np.random.default_rng(3), '16QAM', 0.0,
    multipath=True, cfo_max=1e-4, phase_linewidth=1e-3, pa_backoff_db=8.0)
r = evaluate_sample(c, x, 0.5*(c+x), mod='16QAM', sps=8, n_sym=128,
                    symbols=sym, sym_idx=idx, cfo=meta['cfo'], channel=h)
assert r['snr_gain_db'] > 0 and r['evm_pct'] < r['evm_pct_noisy'], r
assert r['bit_errors'] <= r['bit_errors_noisy'], r
ok("multipath+CFO+phase-noise+PA chain; oracle shrink improves SNR/EVM/BER")

# 8. splits leak nothing
groups = rng.integers(0, 40, size=600); snrs = rng.choice([-10,0,10], size=600)
for tr, te in group_kfold(groups, 5):
    assert check_no_leakage(tr, te, groups) and len(te) > 0
for tr, te, s in leave_one_snr_out(snrs):
    assert set(snrs[te]) == {s} and s not in set(snrs[tr])
ok("GroupKFold + Leave-One-SNR-Out leak no groups/SNRs")

# 9. line-by-line numpy mirror of the torch selective_scan (chunking + carry)
def _hs(a, b):                      # mirrors cvssd.cvssm._hillis_steele
    C = a.shape[-3]; step = 1
    while step < C:
        a_sh = np.concatenate([np.ones_like(a[..., :step, :, :]),
                               a[..., :C-step, :, :]], axis=-3)
        b_sh = np.concatenate([np.zeros_like(b[..., :step, :, :]),
                               b[..., :C-step, :, :]], axis=-3)
        b = b + a * b_sh
        a = a * a_sh
        step *= 2
    return a, b

def torch_scan_mirror(a, b, chunk=64):   # mirrors cvssd.cvssm.selective_scan
    B, L, D, N = a.shape
    pad = (-L) % chunk
    if pad:
        sh = (B, pad, D, N)
        a = np.concatenate([a, np.ones(sh, a.dtype)], 1)
        b = np.concatenate([b, np.zeros(sh, b.dtype)], 1)
    Lp = a.shape[1]; n_chunk = Lp // chunk
    a = a.reshape(B, n_chunk, chunk, D, N); b = b.reshape(B, n_chunk, chunk, D, N)
    carry = np.zeros((B, D, N), b.dtype); outs = []
    for c in range(n_chunk):
        cp, h = _hs(a[:, c], b[:, c])
        h = h + cp * carry[:, None]
        carry = h[:, -1]; outs.append(h)
    return np.concatenate(outs, axis=1).reshape(B, Lp, D, N)[:, :L]

def naive_bld(a, b):
    B, L, D, N = a.shape
    h = np.zeros_like(b); acc = np.zeros((B, D, N), b.dtype)
    for t in range(L):
        acc = a[:, t] * acc + b[:, t]
        h[:, t] = acc
    return h

for L, ch in [(64, 64), (100, 16), (129, 32), (256, 64), (1024, 64)]:
    a = (rng.normal(size=(2, L, 3, 4)) + 1j*rng.normal(size=(2, L, 3, 4))) * 0.3
    bb = rng.normal(size=(2, L, 3, 4)) + 1j*rng.normal(size=(2, L, 3, 4))
    assert np.allclose(torch_scan_mirror(a, bb, ch), naive_bld(a, bb), atol=1e-9), (L, ch)
# chunk size must not change the result
a = (rng.normal(size=(2,256,3,4))+1j*rng.normal(size=(2,256,3,4)))*0.4
bb = rng.normal(size=(2,256,3,4))+1j*rng.normal(size=(2,256,3,4))
r = [torch_scan_mirror(a, bb, c) for c in (16, 64, 256)]
assert all(np.allclose(r[0], x, atol=1e-9) for x in r[1:])
ok("chunked scan algorithm (exact mirror of the torch code) == naive recurrence")
print("\nAll numpy-core tests passed.")
