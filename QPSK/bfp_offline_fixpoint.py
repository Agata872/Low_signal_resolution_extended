#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

# ================= USER CONFIG =================

INFILE = "pre-com.bin"          # input complex64 IQ
OUTFILE = "post-com.bin"        # output complex64 IQ

BLOCK_LEN = 12                  # BFP block length
MANTISSA_BITS = 5               # <<< sweep this
EXPONENT_BITS = 4               # usually fixed
WIQIN = 16                      # equivalent input bitwidth
CLIP_INPUT = True               # clip float IQ to [-1,1)

# ===============================================


def read_c64(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    if os.path.getsize(path) % 8 != 0:
        raise RuntimeError(f"{path} size is not multiple of 8 bytes (not complex64?)")
    x = np.fromfile(path, dtype=np.complex64)
    if x.size == 0:
        raise RuntimeError("Empty input file")
    return x


def write_c64(path: str, x: np.ndarray) -> None:
    x.astype(np.complex64, copy=False).tofile(path)


def minv_protect_int16(a: np.ndarray, m_min: int) -> None:
    mask = (a == m_min)
    if np.any(mask):
        a[mask] = np.int16(m_min + 1)


def bfp_offline_fixpoint(x: np.ndarray) -> np.ndarray:
    m_min = -(2 ** (MANTISSA_BITS - 1))
    m_max = (2 ** (MANTISSA_BITS - 1)) - 1
    e_min = 0
    e_max = (2 ** EXPONENT_BITS) - 1

    S = float((2 ** (WIQIN - 1)) - 1)  # full-scale, e.g., 32767

    # split real/imag
    xr = x.real.astype(np.float32, copy=False)
    xi = x.imag.astype(np.float32, copy=False)

    if CLIP_INPUT:
        xr = np.clip(xr, -1.0, 1.0 - 1.0 / S)
        xi = np.clip(xi, -1.0, 1.0 - 1.0 / S)

    I_int = np.round(xr * S).astype(np.int32)
    Q_int = np.round(xi * S).astype(np.int32)

    n_blocks = len(x) // BLOCK_LEN
    n_use = n_blocks * BLOCK_LEN
    if n_use == 0:
        raise RuntimeError("Not enough samples for one block")

    if n_use != len(x):
        print(f"[WARN] Truncating {len(x) - n_use} tail samples")

    I_int = I_int[:n_use]
    Q_int = Q_int[:n_use]

    x_hat = np.empty(n_use, dtype=np.complex64)

    for b in range(n_blocks):
        s = b * BLOCK_LEN
        Ie = I_int[s:s + BLOCK_LEN]
        Qe = Q_int[s:s + BLOCK_LEN]

        max_val = int(np.max(np.maximum(np.abs(Ie), np.abs(Qe))))
        if max_val == 0:
            e = 0
        else:
            e = int(np.ceil(np.log2(max_val / float(m_max))))
            e = max(e_min, min(e_max, e))

        if e == 0:
            I_m = np.clip(Ie, m_min, m_max).astype(np.int16)
            Q_m = np.clip(Qe, m_min, m_max).astype(np.int16)
        else:
            scale = float(2 ** e)
            I_m = np.clip(np.round(Ie / scale), m_min, m_max).astype(np.int16)
            Q_m = np.clip(np.round(Qe / scale), m_min, m_max).astype(np.int16)

        minv_protect_int16(I_m, m_min)
        minv_protect_int16(Q_m, m_min)

        scale = float(2 ** e)
        I_rec = (I_m.astype(np.float32) * scale) / S
        Q_rec = (Q_m.astype(np.float32) * scale) / S

        x_hat[s:s + BLOCK_LEN] = (I_rec + 1j * Q_rec).astype(np.complex64)

    return x_hat


def main():
    print("[INFO] Offline BFP processing")
    print(f"  Input           : {INFILE}")
    print(f"  Output          : {OUTFILE}")
    print(f"  Block length    : {BLOCK_LEN}")
    print(f"  Mantissa bits   : {MANTISSA_BITS}")
    print(f"  Exponent bits   : {EXPONENT_BITS}")
    print(f"  WIQIN           : {WIQIN}")

    x = read_c64(INFILE)
    print(f"[OK] Read {x.size} samples")

    x_hat = bfp_offline_fixpoint(x)
    write_c64(OUTFILE, x_hat)

    print(f"[OK] Wrote {x_hat.size} samples")

    p_in = np.mean(np.abs(x[:x_hat.size]) ** 2)
    p_out = np.mean(np.abs(x_hat) ** 2)
    print(f"[INFO] Power in/out: {p_in:.6g} -> {p_out:.6g} (ratio {p_out/p_in:.6g})")


if __name__ == "__main__":
    main()
