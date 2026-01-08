#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from PIL import Image

# ================= USER CONFIG =================
IMG_PATH = "shannon.png"
OUT_DIR = "txA"               # output folder
MOD_SCHEME = "QAM16"          # "QPSK" / "QAM16" / "QAM64"
H = W = 1024                  # resize target
BLOCK_LEN = 12                # for your fronthaul block processing alignment
# ===============================================


# ---------- Modulation helpers (Gray, square QAM) ----------
def _bits_per_sym(mod: str) -> int:
    m = mod.upper()
    if m == "QPSK":
        return 2
    if m == "QAM16":
        return 4
    if m == "QAM64":
        return 6
    raise ValueError(f"Unknown MOD_SCHEME: {mod}")

def _qam_levels(M: int) -> np.ndarray:
    s = int(np.sqrt(M))
    if s * s != M:
        raise ValueError("M must be square QAM")
    return np.arange(-(s - 1), s, 2, dtype=np.int32)

def _gray(n: int) -> int:
    return n ^ (n >> 1)

def qpsk_mod(bits_u8: np.ndarray) -> np.ndarray:
    b = bits_u8.reshape(-1, 2)
    # Gray: 00:+ +, 01:- +, 11:- -, 10:+ -
    I = np.where((b[:, 0] == 0) & (b[:, 1] == 0),  1,
        np.where((b[:, 0] == 0) & (b[:, 1] == 1), -1,
        np.where((b[:, 0] == 1) & (b[:, 1] == 1), -1,  1)))
    Q = np.where((b[:, 0] == 0) & (b[:, 1] == 0),  1,
        np.where((b[:, 0] == 0) & (b[:, 1] == 1),  1,
        np.where((b[:, 0] == 1) & (b[:, 1] == 1), -1, -1)))
    return (I + 1j * Q).astype(np.complex64) / np.sqrt(2.0)

def square_qam_mod(bits_u8: np.ndarray, M: int) -> np.ndarray:
    k = int(np.log2(M))
    bpa = k // 2
    levels = _qam_levels(M)

    b = bits_u8.reshape(-1, k)
    bi = b[:, :bpa]
    bq = b[:, bpa:]

    ii = np.zeros(b.shape[0], dtype=np.int32)
    qq = np.zeros(b.shape[0], dtype=np.int32)
    for t in range(bpa):
        ii = (ii << 1) | bi[:, t].astype(np.int32)
        qq = (qq << 1) | bq[:, t].astype(np.int32)

    gi = np.vectorize(_gray)(ii)
    gq = np.vectorize(_gray)(qq)

    I = levels[gi]
    Q = levels[gq]
    sym = (I + 1j * Q).astype(np.complex64)

    # normalize to unit avg power
    levels_f = levels.astype(np.float32)
    avgp = float(np.mean(levels_f**2)) * 2.0
    return sym / np.sqrt(avgp)

def modulate(bits: np.ndarray, scheme: str) -> np.ndarray:
    s = scheme.upper()
    if s == "QPSK":
        return qpsk_mod(bits)
    if s == "QAM16":
        return square_qam_mod(bits, 16)
    if s == "QAM64":
        return square_qam_mod(bits, 64)
    raise ValueError(scheme)

# ---------- Bit helpers ----------
def bytes_to_bits_u8(data_u8: np.ndarray) -> np.ndarray:
    # MSB-first, length = 8*N
    return np.unpackbits(data_u8, bitorder="big")

def write_u8(path: str, x: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    x.astype(np.uint8, copy=False).tofile(path)

def write_c64(path: str, x: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    x.astype(np.complex64, copy=False).tofile(path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load & resize image -> uint8 grayscale
    img = Image.open(IMG_PATH).convert("L").resize((W, H))
    pix = np.array(img, dtype=np.uint8)   # shape (H,W)

    # 2) bytes -> bits
    data_bytes = pix.reshape(-1)  # uint8, length H*W
    bits = bytes_to_bits_u8(data_bytes).astype(np.uint8)
    bps = _bits_per_sym(MOD_SCHEME)

    # 3) Pad bits to multiple of bps so modulation is aligned
    pad_bits = (-bits.size) % bps
    if pad_bits:
        bits_pad = np.pad(bits, (0, pad_bits), mode="constant", constant_values=0)
    else:
        bits_pad = bits

    # 4) Modulate -> complex symbols
    syms = modulate(bits_pad, MOD_SCHEME).astype(np.complex64)

    # 5) Pad symbols to multiple of BLOCK_LEN (for your compressor block alignment)
    pad_syms = (-syms.size) % BLOCK_LEN
    if pad_syms:
        syms_pad = np.pad(syms, (0, pad_syms), mode="constant", constant_values=0.0 + 0.0j)
    else:
        syms_pad = syms

    # 6) Save outputs
    tx_bits_path = os.path.join(OUT_DIR, "tx_bits.u8")
    tx_syms_path = os.path.join(OUT_DIR, "tx_syms.c64")

    write_u8(tx_bits_path, bits_pad)       # padded-to-bps bits (for reference BER)
    write_c64(tx_syms_path, syms_pad)      # padded-to-BLOCK_LEN symbols (for compression input)

    meta = {
        "img_path": IMG_PATH,
        "H": H, "W": W,
        "mod_scheme": MOD_SCHEME,
        "bps": bps,
        "block_len": BLOCK_LEN,
        "n_bytes": int(data_bytes.size),
        "n_bits_raw": int(bits.size),
        "n_bits_padded": int(bits_pad.size),
        "pad_bits_to_bps": int(pad_bits),
        "n_syms_raw": int(syms.size),
        "n_syms_padded": int(syms_pad.size),
        "pad_syms_to_block": int(pad_syms),
        "bitorder": "big"
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[OK] TX generated")
    print("  tx_bits :", tx_bits_path, "len(bits)=", bits_pad.size)
    print("  tx_syms :", tx_syms_path, "len(syms)=", syms_pad.size)
    print("  meta    :", os.path.join(OUT_DIR, "meta.json"))

if __name__ == "__main__":
    main()
