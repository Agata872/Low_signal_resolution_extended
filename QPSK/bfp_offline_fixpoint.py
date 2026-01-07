#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

# ================= USER CONFIG =================
INFILE = "bin/pre-comp.bin"           # input complex64 IQ (float32 I/Q)
OUTFILE = "bin/post-comp.bin"         # output complex64 IQ (float32 I/Q)

# Optional: write "compressed payload" for debugging / inspection
WRITE_COMPRESSED = True
COMP_MANTISSA_FILE = "bin/bfp_mantissa_iq_i16.bin"   # interleaved I,Q int16 per sample
COMP_UDCOMPPARAM_FILE = "bin/bfp_udCompParam_u8.bin" # 1 byte per block (low 4 bits exponent)

BLOCK_LEN = 12        # PRB size in the doc (12 samples) :contentReference[oaicite:4]{index=4}
WIQIN = 16            # input IQ bitwidth (e.g., 16-bit signed)
WIQOUT = 6            # output compressed IQ width = (sign + mantissa) in bits :contentReference[oaicite:5]{index=5}
WEXP = 4              # exponent bits (usually 4, range 0..15) :contentReference[oaicite:6]{index=6}

CLIP_INPUT = True     # clip float IQ to [-1, 1) before fixed-point quantization
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


def _sat_signed(x: np.ndarray, bits: int) -> np.ndarray:
    """Saturate to signed two's-complement range for given bitwidth."""
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    return np.clip(x, lo, hi)


def _to_fixed_iq(x: np.ndarray, wiqin: int, clip_input: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Map float IQ in ~[-1,1) to signed WIQIN-bit integers using full-scale S=2^(WIQIN-1)-1.
    """
    S = float((1 << (wiqin - 1)) - 1)

    xr = x.real.astype(np.float32, copy=False)
    xi = x.imag.astype(np.float32, copy=False)

    if clip_input:
        # keep within representable range
        xr = np.clip(xr, -1.0, 1.0 - 1.0 / S)
        xi = np.clip(xi, -1.0, 1.0 - 1.0 / S)

    I = np.round(xr * S).astype(np.int32)
    Q = np.round(xi * S).astype(np.int32)

    # saturate to WIQIN range (safety)
    I = _sat_signed(I, wiqin).astype(np.int32)
    Q = _sat_signed(Q, wiqin).astype(np.int32)
    return I, Q, S


def bfp_compress_algorithm1(I: np.ndarray, Q: np.ndarray,
                            block_len: int, wiqin: int, wiqout: int, wexp: int):
    """
    Exact doc logic (3.2.1 Algorithm 1):
      maxV/minV over I and Q in block
      maxValue = max(maxV, abs(minV)-1)   (two's complement adjust)
      msb = floor(log2(maxValue)+1)
      exponent = max(msb - mantissa_size, 0)
      compressed = value * 2^{-exponent}  (arithmetic right shift)
    Here mantissa_size = wiqout - 1 (since wiqout includes sign bit). :contentReference[oaicite:7]{index=7}
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

    # store mantissas as int16 (still ok even if wiqout<16)
    I_m = np.empty(n_use, dtype=np.int16)
    Q_m = np.empty(n_use, dtype=np.int16)
    ud = np.empty(n_blocks, dtype=np.uint8)

    # representable range for wiqout-bit signed
    lo_out = -(1 << (wiqout - 1))
    hi_out = (1 << (wiqout - 1)) - 1

    for b in range(n_blocks):
        s = b * block_len
        Ie = I[s:s + block_len]
        Qe = Q[s:s + block_len]

        maxV = int(max(Ie.max(), Qe.max()))
        minV = int(min(Ie.min(), Qe.min()))

        # maxValue = max(maxV, |minV| - 1)  (doc's two's complement adjustment) :contentReference[oaicite:8]{index=8}
        maxValue = max(maxV, abs(minV) - 1)

        if maxValue <= 0:
            exponent = 0
        else:
            msb = int(np.floor(np.log2(maxValue) + 1.0))  # msb = floor(log2(maxValue)+1) :contentReference[oaicite:9]{index=9}
            exponent = max(msb - mantissa_size, 0)        # exponent = max(msb - mantissa_size, 0) :contentReference[oaicite:10]{index=10}
            exponent = min(exponent, e_max)               # 4-bit range 0..15 via udCompParam :contentReference[oaicite:11]{index=11}

        # arithmetic right shift by exponent (equiv *2^{-exponent}) :contentReference[oaicite:12]{index=12}
        if exponent == 0:
            Ic = Ie
            Qc = Qe
        else:
            Ic = Ie >> exponent
            Qc = Qe >> exponent

        # Safety saturate to WIQOUT signed range (should rarely trigger if exponent computed as doc)
        Ic = np.clip(Ic, lo_out, hi_out).astype(np.int16)
        Qc = np.clip(Qc, lo_out, hi_out).astype(np.int16)

        I_m[s:s + block_len] = Ic
        Q_m[s:s + block_len] = Qc

        # udCompParam: 1 byte, lower 4 bits exponent, upper 4 bits reserved :contentReference[oaicite:13]{index=13}
        ud[b] = np.uint8(exponent & 0x0F)

    return I_m, Q_m, ud, n_use


def bfp_decompress_eq10(I_m: np.ndarray, Q_m: np.ndarray, ud: np.ndarray,
                        block_len: int, wiqin: int, wiqout: int, wexp: int,
                        S: float) -> np.ndarray:
    """
    Exact doc inverse (Eq. 10): IQ_i = m_i * 2^e (left shift). :contentReference[oaicite:14]{index=14}
    Then map back to float by dividing full-scale S.
    """
    n_use = len(I_m)
    n_blocks = len(ud)
    assert n_use == n_blocks * block_len

    x_hat = np.empty(n_use, dtype=np.complex64)

    # reconstruct to WIQIN-bit domain (int32)
    for b in range(n_blocks):
        s = b * block_len
        e = int(ud[b] & 0x0F)

        Ie = I_m[s:s + block_len].astype(np.int32)
        Qe = Q_m[s:s + block_len].astype(np.int32)

        if e != 0:
            Ie = Ie << e
            Qe = Qe << e

        # saturate to WIQIN signed range (safety)
        Ie = _sat_signed(Ie, wiqin).astype(np.int32)
        Qe = _sat_signed(Qe, wiqin).astype(np.int32)

        I_f = Ie.astype(np.float32) / S
        Q_f = Qe.astype(np.float32) / S
        x_hat[s:s + block_len] = (I_f + 1j * Q_f).astype(np.complex64)

    return x_hat


def main():
    print("[INFO] Offline BFP (doc-exact Algorithm 1 + Eq.10)")
    print(f"  Input    : {INFILE}")
    print(f"  Output   : {OUTFILE}")
    print(f"  BLOCK_LEN: {BLOCK_LEN}")
    print(f"  WIQIN    : {WIQIN}")
    print(f"  WIQOUT   : {WIQOUT}  (includes sign)")
    print(f"  WEXP     : {WEXP}")

    x = read_c64(INFILE)
    print(f"[OK] Read {x.size} samples")
    # ======== AGC / NORMALIZATION 加在这里 ========
    a = np.abs(x)
    scale = np.quantile(a, 0.999)  # 推荐 0.999 或 0.9999
    x = x / scale
    print("[INFO] AGC scale =", scale)

    I, Q, S = _to_fixed_iq(x, WIQIN, CLIP_INPUT)

    I_m, Q_m, ud, n_use = bfp_compress_algorithm1(
        I, Q, BLOCK_LEN, WIQIN, WIQOUT, WEXP
    )

    # (Optional) dump compressed payload: mantissas + udCompParam
    if WRITE_COMPRESSED:
        os.makedirs(os.path.dirname(COMP_MANTISSA_FILE), exist_ok=True)
        # interleave I,Q mantissas: [I0,Q0,I1,Q1,...]
        inter = np.empty(2 * n_use, dtype=np.int16)
        inter[0::2] = I_m
        inter[1::2] = Q_m
        inter.tofile(COMP_MANTISSA_FILE)
        ud.tofile(COMP_UDCOMPPARAM_FILE)
        print(f"[OK] Wrote compressed mantissas : {COMP_MANTISSA_FILE}")
        print(f"[OK] Wrote udCompParam (u8)    : {COMP_UDCOMPPARAM_FILE}")

    x_hat = bfp_decompress_eq10(
        I_m, Q_m, ud, BLOCK_LEN, WIQIN, WIQOUT, WEXP, S
    )
    x_hat = x_hat * scale
    write_c64(OUTFILE, x_hat)
    print(f"[OK] Wrote {x_hat.size} samples")

    p_in = np.mean(np.abs(x[:x_hat.size]) ** 2)
    p_out = np.mean(np.abs(x_hat) ** 2)
    print(f"[INFO] Power in/out: {p_in:.6g} -> {p_out:.6g} (ratio {p_out/p_in:.6g})")

    # quick sanity: exponent stats
    print(f"[INFO] Exponent stats: min={int(ud.min() & 0x0F)} max={int(ud.max() & 0x0F)}")


if __name__ == "__main__":
    main()
