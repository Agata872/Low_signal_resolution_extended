#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

# ================= USER CONFIG =================
INFILE = "bin/pre-comp.bin"   # input complex64 IQ (float32 I/Q)

OUT_DIR = "bfp"
WRITE_COMPRESSED = True

BLOCK_LEN = 12
WIQIN = 16
WIQOUT_LIST = [4, 6, 8, 10, 12, 14, 16]

WEXP = 4                 # exponent bits (low 4 bits in udCompParam)
CLIP_INPUT = True

# Normalization:
#   "maxabs"   -> scale = max(|x|)
#   "quantile" -> scale = quantile(|x|, AGC_Q)
AGC_MODE = "maxabs"
AGC_Q = 0.999
AGC_EPS = 1e-12
# ===============================================


# ---------------- I/O helpers ----------------
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
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    x.astype(np.complex64, copy=False).tofile(path)


# ---------------- math helpers ----------------
def _sat_signed(x: np.ndarray, bits: int) -> np.ndarray:
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    return np.clip(x, lo, hi)


def _to_fixed_iq(x: np.ndarray, wiqin: int, clip_input: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Map float IQ in ~[-1,1) to signed WIQIN-bit integers using full-scale S=2^(WIQIN-1)-1.
    Return (I_int32, Q_int32, S_float).
    """
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


def _normalize(x: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Return (x_norm, scale) where x = x_norm * scale.
    """
    if AGC_MODE.lower() == "quantile":
        a = np.abs(x)
        scale = float(np.quantile(a, AGC_Q))
    else:
        scale = float(np.max(np.abs(x)))

    if not np.isfinite(scale) or scale < AGC_EPS:
        scale = 1.0
    return x / scale, scale


# ---------------- BFP (doc-exact Algorithm 1 + Eq.10) ----------------
def bfp_compress_algorithm1(
    I: np.ndarray,
    Q: np.ndarray,
    block_len: int,
    wiqin: int,
    wiqout: int,
    wexp: int,
):
    """
    Doc logic (3.2.1 Algorithm 1):
      maxV/minV over I,Q in block
      maxValue = max(maxV, abs(minV)-1)
      msb = floor(log2(maxValue)+1)
      exponent = max(msb - mantissa_size, 0)
      compressed = value * 2^{-exponent}  (arithmetic right shift)
    mantissa_size = wiqout - 1 (wiqout includes sign bit)
    udCompParam: 1 byte, low 4 bits exponent.
    """
    assert wiqout >= 2, "WIQOUT must include sign + at least 1 mantissa bit"
    mantissa_size = wiqout - 1
    e_max = (1 << wexp) - 1

    n_blocks = len(I) // block_len
    n_use = n_blocks * block_len
    if n_use == 0:
        raise RuntimeError("Not enough samples for one block")
    if n_use != len(I):
        print(f"[WARN] Truncating {len(I) - n_use} tail samples")

    I = I[:n_use]
    Q = Q[:n_use]

    I_m = np.empty(n_use, dtype=np.int16)
    Q_m = np.empty(n_use, dtype=np.int16)
    ud = np.empty(n_blocks, dtype=np.uint8)

    lo_out = -(1 << (wiqout - 1))
    hi_out = (1 << (wiqout - 1)) - 1

    clip_cnt_I = 0
    clip_cnt_Q = 0
    total_cnt = 0

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

        if exponent == 0:
            Ic = Ie
            Qc = Qe
        else:
            Ic = Ie >> exponent
            Qc = Qe >> exponent

        # track clipping (should be rare if exponent is correct, but keep diagnostics)
        preI = Ic
        preQ = Qc
        Ic = np.clip(Ic, lo_out, hi_out)
        Qc = np.clip(Qc, lo_out, hi_out)

        clip_cnt_I += np.count_nonzero(preI != Ic)
        clip_cnt_Q += np.count_nonzero(preQ != Qc)
        total_cnt += Ic.size

        I_m[s:s + block_len] = Ic.astype(np.int16)
        Q_m[s:s + block_len] = Qc.astype(np.int16)
        ud[b] = np.uint8(exponent & 0x0F)

    clip_rate_I = float(clip_cnt_I / total_cnt) if total_cnt else 0.0
    clip_rate_Q = float(clip_cnt_Q / total_cnt) if total_cnt else 0.0
    return I_m, Q_m, ud, n_use, clip_rate_I, clip_rate_Q


def bfp_decompress_eq10(
    I_m: np.ndarray,
    Q_m: np.ndarray,
    ud: np.ndarray,
    block_len: int,
    wiqin: int,
    S: float,
) -> np.ndarray:
    """
    Eq.10 inverse: IQ = mantissa * 2^e (left shift), then /S to float.
    """
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

        I_f = Ie.astype(np.float32) / S
        Q_f = Qe.astype(np.float32) / S
        x_hat[s:s + block_len] = (I_f + 1j * Q_f).astype(np.complex64)

    return x_hat


# ---------------- progress bar ----------------
def _progress(i: int, n: int, prefix: str = "") -> None:
    width = 28
    frac = (i + 1) / n
    filled = int(round(width * frac))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{prefix}[{bar}] {i+1}/{n} ({frac*100:5.1f}%)", end="", flush=True)
    if i + 1 == n:
        print()


# ---------------- main sweep ----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[INFO] Offline BFP sweep (doc-exact Algorithm 1 + Eq.10)")
    print(f"  Input     : {INFILE}")
    print(f"  Out dir   : {OUT_DIR}")
    print(f"  BLOCK_LEN : {BLOCK_LEN}")
    print(f"  WIQIN     : {WIQIN}")
    print(f"  WIQOUTs   : {WIQOUT_LIST}")
    print(f"  WEXP      : {WEXP}")
    print(f"  Clip input: {CLIP_INPUT}")
    print(f"  AGC_MODE  : {AGC_MODE} (Q={AGC_Q} if quantile)")

    x = read_c64(INFILE)
    print(f"[OK] Read {x.size} complex64 samples")

    x_n, scale = _normalize(x)
    print(f"[INFO] Normalization scale = {scale:g}")

    I, Q, S = _to_fixed_iq(x_n, WIQIN, CLIP_INPUT)

    n_total = len(WIQOUT_LIST)
    for idx, wiqout in enumerate(WIQOUT_LIST):
        _progress(idx, n_total, prefix="Running ")

        I_m, Q_m, ud, n_use, clipI, clipQ = bfp_compress_algorithm1(
            I, Q, BLOCK_LEN, WIQIN, wiqout, WEXP
        )

        if WRITE_COMPRESSED:
            mant_path = os.path.join(OUT_DIR, f"mantissa_iq_wiqout{wiqout}.i16")
            ud_path = os.path.join(OUT_DIR, f"udCompParam_wiqout{wiqout}.u8")

            inter = np.empty(2 * n_use, dtype=np.int16)
            inter[0::2] = I_m
            inter[1::2] = Q_m
            inter.tofile(mant_path)
            ud.tofile(ud_path)

        x_hat_n = bfp_decompress_eq10(I_m, Q_m, ud, BLOCK_LEN, WIQIN, S)
        x_hat = x_hat_n * scale

        out_path = os.path.join(OUT_DIR, f"rx_wiqout{wiqout}.bin")
        write_c64(out_path, x_hat)

        p_in = float(np.mean(np.abs(x[:x_hat.size]) ** 2))
        p_out = float(np.mean(np.abs(x_hat) ** 2))
        ratio = (p_out / p_in) if p_in > 0 else float("nan")

        e_min = int((ud.min() & 0x0F)) if ud.size else 0
        e_max = int((ud.max() & 0x0F)) if ud.size else 0

        print(
            f"[WIQOUT={wiqout:2d}] wrote {out_path} | "
            f"clip(I,Q)=({clipI*100:.2f}%,{clipQ*100:.2f}%) | "
            f"power ratio={ratio:.6g} | exp[min,max]=[{e_min},{e_max}]"
        )

    print("[DONE] All WIQOUT sweeps finished.")


if __name__ == "__main__":
    main()
