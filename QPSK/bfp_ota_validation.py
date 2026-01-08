#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np

# ================= USER CONFIG =================
ADD_RAYLEIGH = True
RAYLEIGH_MODE = "block"   # "block" (one h for all samples) or "fast" (h per sample)
SNR_DB = 20.0             # AWGN SNR after fading (in dB). Increase to make it closer to ideal
CSI_ASSUMED = True        # True: perfect equalization y/h; False: no equalization (just to see impact)
RNG_SEED_CH = 12345
TX_DIR = "txA"                 # produced by tx_make_modulated_from_image.py
OUT_DIR = "bfpA"

WRITE_COMPRESSED = True

WIQIN = 16
BLOCK_LEN = 12
WIQOUT_LIST = [4, 6, 8, 10, 12, 14, 16]

WEXP = 4
CLIP_INPUT = True

AGC_MODE = "maxabs"
AGC_Q = 0.999
AGC_EPS = 1e-12
# ===============================================


# ---------------- I/O helpers ----------------
def read_c64(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if os.path.getsize(path) % 8 != 0:
        raise RuntimeError(f"{path} not complex64?")
    x = np.fromfile(path, dtype=np.complex64)
    if x.size == 0:
        raise RuntimeError("Empty input")
    return x

def write_c64(path: str, x: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    x.astype(np.complex64, copy=False).tofile(path)

def write_u8(path: str, x: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    x.astype(np.uint8, copy=False).tofile(path)


# ---------------- math helpers ----------------
def _sat_signed(x: np.ndarray, bits: int) -> np.ndarray:
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    return np.clip(x, lo, hi)

def _to_fixed_iq(x: np.ndarray, wiqin: int, clip_input: bool):
    S = float((1 << (wiqin - 1)) - 1)
    xr = x.real.astype(np.float32, copy=False)
    xi = x.imag.astype(np.float32, copy=False)
    if clip_input:
        xr = np.clip(xr, -1.0, 1.0 - 1.0 / S)
        xi = np.clip(xi, -1.0, 1.0 - 1.0 / S)
    I = np.round(xr * S).astype(np.int32)
    Q = np.round(xi * S).astype(np.int32)
    I = _sat_signed(I, wiqin).astype(np.int32)
    Q = _sat_signed(Q, wiqin).astype(np.int32)
    return I, Q, S

def _normalize(x: np.ndarray):
    if AGC_MODE.lower() == "quantile":
        scale = float(np.quantile(np.abs(x), AGC_Q))
    else:
        scale = float(np.max(np.abs(x)))
    if not np.isfinite(scale) or scale < AGC_EPS:
        scale = 1.0
    return x / scale, scale

def apply_rayleigh_channel(x: np.ndarray, snr_db: float, mode: str, rng: np.random.Generator):
    """
    Apply complex Rayleigh fading + AWGN:
        y = h * x + n
    where h ~ CN(0,1). For 'block': one h for whole vector.
    Noise is set to achieve target SNR (per complex sample) relative to faded signal power.

    Returns: (y, h, noise_var)
    """
    mode = mode.lower()
    n = x.size

    if mode == "block":
        h = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2.0)
        h = np.complex64(h)  # scalar
        hx = h * x
    elif mode == "fast":
        h = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) / np.sqrt(2.0)
        hx = h * x
    else:
        raise ValueError("RAYLEIGH_MODE must be 'block' or 'fast'")

    # Signal power after fading
    sigp = float(np.mean(np.abs(hx) ** 2))
    if sigp <= 0:
        noise_var = 0.0
        y = hx
        return y.astype(np.complex64), h, noise_var

    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_var = sigp / snr_lin  # per complex sample: E|n|^2 = noise_var

    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    noise *= np.sqrt(noise_var / 2.0).astype(np.float32)

    y = hx + noise
    return y.astype(np.complex64), h, float(noise_var)


# ---------------- Mod/Demod (must match TX) ----------------
def _bits_per_sym(mod: str) -> int:
    m = mod.upper()
    if m == "QPSK":
        return 2
    if m == "QAM16":
        return 4
    if m == "QAM64":
        return 6
    raise ValueError(mod)

def _qam_levels(M: int) -> np.ndarray:
    s = int(np.sqrt(M))
    if s * s != M:
        raise ValueError("M must be square QAM")
    return np.arange(-(s - 1), s, 2, dtype=np.int32)

def _inv_gray(g: int) -> int:
    x = 0
    while g:
        x ^= g
        g >>= 1
    return x

def qpsk_demod(sym: np.ndarray) -> np.ndarray:
    re = sym.real
    im = sym.imag
    sI = (re >= 0).astype(np.uint8)
    sQ = (im >= 0).astype(np.uint8)
    out = np.zeros((sym.size, 2), dtype=np.uint8)
    out[(sI == 1) & (sQ == 1)] = [0, 0]
    out[(sI == 0) & (sQ == 1)] = [0, 1]
    out[(sI == 0) & (sQ == 0)] = [1, 1]
    out[(sI == 1) & (sQ == 0)] = [1, 0]
    return out.reshape(-1)

def square_qam_demod(sym: np.ndarray, M: int) -> np.ndarray:
    k = int(np.log2(M))
    bpa = k // 2
    levels = _qam_levels(M).astype(np.float32)
    avgp = float(np.mean(levels**2)) * 2.0
    levels_n = levels / np.sqrt(avgp)

    re = sym.real.astype(np.float32)
    im = sym.imag.astype(np.float32)

    di = np.abs(re[:, None] - levels_n[None, :])
    dq = np.abs(im[:, None] - levels_n[None, :])
    idx_i_gray = np.argmin(di, axis=1).astype(np.int32)
    idx_q_gray = np.argmin(dq, axis=1).astype(np.int32)

    idx_i = np.vectorize(_inv_gray)(idx_i_gray).astype(np.int32)
    idx_q = np.vectorize(_inv_gray)(idx_q_gray).astype(np.int32)

    out = np.zeros((sym.size, k), dtype=np.uint8)
    for t in range(bpa):
        out[:, t] = ((idx_i >> (bpa - 1 - t)) & 1).astype(np.uint8)
        out[:, bpa + t] = ((idx_q >> (bpa - 1 - t)) & 1).astype(np.uint8)
    return out.reshape(-1)

def demodulate(sym: np.ndarray, scheme: str) -> np.ndarray:
    s = scheme.upper()
    if s == "QPSK":
        return qpsk_demod(sym)
    if s == "QAM16":
        return square_qam_demod(sym, 16)
    if s == "QAM64":
        return square_qam_demod(sym, 64)
    raise ValueError(scheme)


# ---------------- BFP (your doc-exact) ----------------
def bfp_compress_algorithm1(I, Q, block_len, wiqin, wiqout, wexp):
    mantissa_size = wiqout - 1
    e_max = (1 << wexp) - 1

    n_blocks = len(I) // block_len
    n_use = n_blocks * block_len
    if n_use == 0:
        raise RuntimeError("Not enough samples for one block")

    I = I[:n_use]
    Q = Q[:n_use]

    I_m = np.empty(n_use, dtype=np.int16)
    Q_m = np.empty(n_use, dtype=np.int16)
    ud = np.empty(n_blocks, dtype=np.uint8)

    lo_out = -(1 << (wiqout - 1))
    hi_out = (1 << (wiqout - 1)) - 1

    for b in range(n_blocks):
        s = b * block_len
        Ie = I[s:s + block_len]
        Qe = Q[s:s + block_len]

        maxV = int(max(Ie.max(), Qe.max()))
        minV = int(min(Ie.min(), Qe.min()))
        maxValue = max(maxV, abs(minV) - 1)

        if maxValue <= 0:
            exponent = 0
        else:
            msb = int(np.floor(np.log2(maxValue) + 1.0))
            exponent = max(msb - mantissa_size, 0)
            exponent = min(exponent, e_max)

        Ic = Ie if exponent == 0 else (Ie >> exponent)
        Qc = Qe if exponent == 0 else (Qe >> exponent)

        Ic = np.clip(Ic, lo_out, hi_out)
        Qc = np.clip(Qc, lo_out, hi_out)

        I_m[s:s + block_len] = Ic.astype(np.int16)
        Q_m[s:s + block_len] = Qc.astype(np.int16)
        ud[b] = np.uint8(exponent & 0x0F)

    return I_m, Q_m, ud, n_use

def bfp_decompress_eq10(I_m, Q_m, ud, block_len, wiqin, S):
    n_use = len(I_m)
    n_blocks = len(ud)
    assert n_use == n_blocks * block_len

    x_hat = np.empty(n_use, dtype=np.complex64)
    for b in range(n_blocks):
        s = b * block_len
        e = int(ud[b] & 0x0F)

        Ie = I_m[s:s + block_len].astype(np.int32)
        Qe = Q_m[s:s + block_len].astype(np.int32)
        if e != 0:
            Ie = Ie << e
            Qe = Qe << e

        Ie = _sat_signed(Ie, wiqin).astype(np.int32)
        Qe = _sat_signed(Qe, wiqin).astype(np.int32)

        x_hat[s:s + block_len] = (Ie.astype(np.float32) / S + 1j * Qe.astype(np.float32) / S).astype(np.complex64)

    return x_hat


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    meta_path = os.path.join(TX_DIR, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    mod_scheme = meta["mod_scheme"]
    bps = meta["bps"]

    tx_syms_path = os.path.join(TX_DIR, "tx_syms.c64")
    x = read_c64(tx_syms_path)
    print("[OK] Read tx symbols:", tx_syms_path, "N=", x.size, "MOD=", mod_scheme)

    # normalize + fixed-point
    x_n, scale = _normalize(x)
    I, Q, S = _to_fixed_iq(x_n, WIQIN, CLIP_INPUT)

    # save a copy of meta into OUT_DIR for notebook convenience
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    for wiqout in WIQOUT_LIST:
        subdir = os.path.join(OUT_DIR, mod_scheme.lower(), f"wiqout{wiqout}")
        os.makedirs(subdir, exist_ok=True)

        I_m, Q_m, ud, n_use = bfp_compress_algorithm1(I, Q, BLOCK_LEN, WIQIN, wiqout, WEXP)

        if WRITE_COMPRESSED:
            inter = np.empty(2 * n_use, dtype=np.int16)
            inter[0::2] = I_m
            inter[1::2] = Q_m
            inter.tofile(os.path.join(subdir, "mantissa_iq.i16"))
            ud.tofile(os.path.join(subdir, "udCompParam.u8"))

        x_hat_n = bfp_decompress_eq10(I_m, Q_m, ud, BLOCK_LEN, WIQIN, S)
        x_hat = (x_hat_n * scale).astype(np.complex64, copy=False)

        # ---------- Added: Rayleigh channel + (optional) equalization ----------
        if ADD_RAYLEIGH:
            rng_ch = np.random.default_rng(RNG_SEED_CH + int(wiqout))  # different per WIQOUT, reproducible
            y, h, noise_var = apply_rayleigh_channel(x_hat, SNR_DB, RAYLEIGH_MODE, rng_ch)

            if CSI_ASSUMED:
                # perfect 1-tap equalization
                x_eq = (y / h).astype(np.complex64) if np.isscalar(h) else (y / h).astype(np.complex64)
            else:
                x_eq = y.astype(np.complex64)

            # save extra diagnostics (optional but useful)
            write_c64(os.path.join(subdir, "rx_syms_ch.c64"), y)
            # store h in a tiny file
            if np.isscalar(h):
                np.array([h], dtype=np.complex64).tofile(os.path.join(subdir, "h.c64"))
            else:
                h.astype(np.complex64).tofile(os.path.join(subdir, "h.c64"))

            x_out = x_eq
        else:
            x_out = x_hat
        # ----------------------------------------------------------------------

        write_c64(os.path.join(subdir, "rx_syms.c64"), x_out)

        # demod & save rx bits
        rx_bits = demodulate(x_out, mod_scheme)
        write_u8(os.path.join(subdir, "rx_bits.u8"), rx_bits)

        print(f"[WIQOUT={wiqout:2d}] wrote rx_syms/rx_bits in {subdir}")

    print("[DONE] BFP sweep finished.")

if __name__ == "__main__":
    main()
