#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

# ================= USER CONFIG =================
INFILE = "tx/pre-comp.bin"  # input complex64 IQ (float32 I/Q)

OUT_DIR = "bsc"              # output directory
WRITE_COMPRESSED = True      # write compressed payload for inspection

BLOCK_LEN = 12               # PRB size: 12 IQ samples
WIQIN = 16                   # input IQ bitwidth (signed)
WIQOUT_LIST = [4, 6, 8, 10, 12, 14, 16]

CLIP_INPUT = True            # clip float IQ to [-1, 1) before fixed-point quantization
ROUNDING = True              # symmetric rounding on shifts; rounded inverseScaler

AGC_EPS = 1e-12              # max-abs normalization epsilon
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
    """Saturate to signed two's-complement range for given bitwidth."""
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    return np.clip(x, lo, hi)


def _round_div_pow2_signed(x: np.ndarray, sh: int) -> np.ndarray:
    """
    Signed rounding division by 2^sh (sh>=0):
      round(x / 2^sh) with symmetric rounding.
    """
    if sh <= 0:
        return x.astype(np.int32, copy=False)
    add = 1 << (sh - 1)
    x64 = x.astype(np.int64, copy=False)
    pos = x64 >= 0
    x64[pos] = x64[pos] + add
    x64[~pos] = x64[~pos] - add
    return (x64 >> sh).astype(np.int32)


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


# ---------------- BS compress/decompress ----------------
def bs_compress_algorithm2(
    I: np.ndarray,
    Q: np.ndarray,
    block_len: int,
    wiqin: int,
    wiqout: int,
    rounding: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, float]:
    """
    Block Scaling compression based on thesis Algorithm 2 (with practical fixed-point handling).

    Returns:
      I_c(int16), Q_c(int16), ud(uint8 scaler per block), n_use,
      clip_rate_I, clip_rate_Q
    """
    assert wiqout >= 2, "WIQOUT must include sign + at least 1 bit"

    n_blocks = len(I) // block_len
    n_use = n_blocks * block_len
    if n_use == 0:
        raise RuntimeError("Not enough samples for one block")
    if n_use != len(I):
        print(f"[WARN] Truncating {len(I) - n_use} tail samples")

    I = I[:n_use]
    Q = Q[:n_use]

    I_c = np.empty(n_use, dtype=np.int16)
    Q_c = np.empty(n_use, dtype=np.int16)
    ud = np.empty(n_blocks, dtype=np.uint8)

    lo_out = -(1 << (wiqout - 1))
    hi_out = (1 << (wiqout - 1)) - 1

    div_for_scaler = 1 << (wiqin - 8)   # 2^(WIQIN-8)
    inv_q17_one = 1 << 7               # 2^7 for Q1.7

    # IMPORTANT: consume Q1.7 fractional (7 bits) here to avoid mantissa overflow
    sh_norm = (wiqin - wiqout + 7)

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
            scaler = 1
        else:
            scaler = int(np.ceil(maxValue / float(div_for_scaler)))
            if scaler == 0:
                scaler = 1
            if scaler > 255:
                scaler = 255

        if rounding:
            inverseScaler = (inv_q17_one + scaler // 2) // scaler
        else:
            inverseScaler = inv_q17_one // scaler

        tmpI64 = Ie.astype(np.int64) * int(inverseScaler)
        tmpQ64 = Qe.astype(np.int64) * int(inverseScaler)

        if rounding:
            tmpI32 = _round_div_pow2_signed(tmpI64, sh_norm)
            tmpQ32 = _round_div_pow2_signed(tmpQ64, sh_norm)
        else:
            tmpI32 = (tmpI64 >> sh_norm).astype(np.int32)
            tmpQ32 = (tmpQ64 >> sh_norm).astype(np.int32)

        preI = tmpI32
        preQ = tmpQ32

        tmpI32 = np.clip(tmpI32, lo_out, hi_out).astype(np.int32)
        tmpQ32 = np.clip(tmpQ32, lo_out, hi_out).astype(np.int32)

        clip_cnt_I += np.count_nonzero(preI != tmpI32)
        clip_cnt_Q += np.count_nonzero(preQ != tmpQ32)
        total_cnt += tmpI32.size

        I_c[s:s + block_len] = tmpI32.astype(np.int16)
        Q_c[s:s + block_len] = tmpQ32.astype(np.int16)
        ud[b] = np.uint8(scaler)

    clip_rate_I = float(clip_cnt_I / total_cnt) if total_cnt else 0.0
    clip_rate_Q = float(clip_cnt_Q / total_cnt) if total_cnt else 0.0
    return I_c, Q_c, ud, n_use, clip_rate_I, clip_rate_Q


def bs_decompress_algorithm3(
    I_c: np.ndarray,
    Q_c: np.ndarray,
    ud: np.ndarray,
    block_len: int,
    wiqin: int,
    wiqout: int,
    S: float,
    rounding: bool = True,
) -> np.ndarray:
    """
    Decompression (paired with the above compressor variant).

    With compressor using sh_norm = (WIQIN - WIQOUT + 7),
    the matching decompressor uses exponent:
      d = (WIQOUT - WIQIN)
    so that overall scaling closes.
    """
    n_use = len(I_c)
    n_blocks = len(ud)
    assert n_use == n_blocks * block_len

    x_hat = np.empty(n_use, dtype=np.complex64)
    d = wiqout - wiqin  # matched to compressor variant

    for b in range(n_blocks):
        s = b * block_len
        scaler = int(ud[b])

        Ie = I_c[s:s + block_len].astype(np.int64)
        Qe = Q_c[s:s + block_len].astype(np.int64)

        prodI = Ie * scaler
        prodQ = Qe * scaler

        if d >= 0:
            if rounding:
                outI = _round_div_pow2_signed(prodI, d)
                outQ = _round_div_pow2_signed(prodQ, d)
            else:
                outI = (prodI >> d).astype(np.int32)
                outQ = (prodQ >> d).astype(np.int32)
        else:
            sh = -d
            outI = (prodI << sh).astype(np.int32)
            outQ = (prodQ << sh).astype(np.int32)

        outI = _sat_signed(outI, wiqin).astype(np.int32)
        outQ = _sat_signed(outQ, wiqin).astype(np.int32)

        I_f = outI.astype(np.float32) / S
        Q_f = outQ.astype(np.float32) / S
        x_hat[s:s + block_len] = (I_f + 1j * Q_f).astype(np.complex64)

    return x_hat


# ---------------- progress bar (no external deps) ----------------
def _progress(i: int, n: int, prefix: str = "") -> None:
    width = 28
    frac = (i + 1) / n
    filled = int(round(width * frac))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{prefix}[{bar}] {i+1}/{n} ({frac*100:5.1f}%)", end="", flush=True)
    if i + 1 == n:
        print()  # newline


# ---------------- main sweep ----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[INFO] Offline BSC (Block Scaling) sweep")
    print(f"  Input     : {INFILE}")
    print(f"  Out dir   : {OUT_DIR}")
    print(f"  BLOCK_LEN : {BLOCK_LEN}")
    print(f"  WIQIN     : {WIQIN}")
    print(f"  WIQOUTs   : {WIQOUT_LIST}")
    print(f"  Rounding  : {ROUNDING}")
    print(f"  Clip input: {CLIP_INPUT}")

    x = read_c64(INFILE)
    print(f"[OK] Read {x.size} complex64 samples")

    # global max-abs normalization (recommended for BS on low-amplitude captures)
    scale = float(np.max(np.abs(x)))
    if scale < AGC_EPS:
        scale = 1.0
    x_n = x / scale
    print(f"[INFO] Max-abs scale = {scale:g}")

    # float -> fixed WIQIN
    I, Q, S = _to_fixed_iq(x_n, WIQIN, CLIP_INPUT)

    n_total = len(WIQOUT_LIST)
    for idx, wiqout in enumerate(WIQOUT_LIST):
        _progress(idx, n_total, prefix="Running ")

        # compress
        I_c, Q_c, ud, n_use, clipI, clipQ = bs_compress_algorithm2(
            I, Q, BLOCK_LEN, WIQIN, wiqout, rounding=ROUNDING
        )

        # write compressed payload
        if WRITE_COMPRESSED:
            mant_path = os.path.join(OUT_DIR, f"mantissa_iq_wiqout{wiqout}.i16")
            ud_path = os.path.join(OUT_DIR, f"udCompParam_wiqout{wiqout}.u8")

            inter = np.empty(2 * n_use, dtype=np.int16)
            inter[0::2] = I_c
            inter[1::2] = Q_c
            inter.tofile(mant_path)
            ud.tofile(ud_path)

        # decompress
        x_hat_n = bs_decompress_algorithm3(
            I_c, Q_c, ud, BLOCK_LEN, WIQIN, wiqout, S, rounding=ROUNDING
        )

        # undo normalization
        x_hat = x_hat_n * scale

        # output reconstructed IQ
        out_path = os.path.join(OUT_DIR, f"rx_wiqout{wiqout}.bin")
        write_c64(out_path, x_hat)

        # quick sanity stats
        p_in = float(np.mean(np.abs(x[:x_hat.size]) ** 2))
        p_out = float(np.mean(np.abs(x_hat) ** 2))
        ratio = (p_out / p_in) if p_in > 0 else float("nan")
        ud_min = int(ud.min()) if ud.size else 0
        ud_max = int(ud.max()) if ud.size else 0

        print(
            f"[WIQOUT={wiqout:2d}] wrote {out_path} | "
            f"clip(I,Q)=({clipI*100:.2f}%,{clipQ*100:.2f}%) | "
            f"power ratio={ratio:.6g} | scaler[min,max]=[{ud_min},{ud_max}]"
        )

    print("[DONE] All WIQOUT sweeps finished.")


if __name__ == "__main__":
    main()
