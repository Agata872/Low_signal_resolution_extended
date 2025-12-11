#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORAN Fronthaul Compression BLER Simulation

Methods:
- Uncompressed (baseline, no compression)
- BFP (Block Floating Point)
- BlockScaling (Block Scaling)
- MuLaw (μ-law based compression)
- Modulation (lossless placeholder)

Channel modes:
- "mimo": centralized MIMO, i.i.d. Rayleigh
- "cellfree": cell-free with geometry-based large-scale fading

Author: (adapted for Tianzheng)
"""

import numpy as np
import matplotlib.pyplot as plt
from tool import *  # 里边应包含 rng, qam_mod, qam_demod, compress_* 等


# ============================================================
#  BLER Simulation
# ============================================================

def simulate_bler_all(
    mode="mimo",
    M_AP=8,
    K_UE=2,
    mod_order_tags=(2, 16, 64),
    bitwidths=(8, 10, 12),
    snr_db_vec=np.arange(0, 21, 4),
    Ntrials=500,
    N_sym_per_tb=50,
):
    """
    Simulate BLER for:
    - Uncompressed baseline
    - BFP, BlockScaling, MuLaw, Modulation (lossless placeholder).
    """

    # ---- 允许标量传入 ----
    if isinstance(mod_order_tags, (int, np.integer)):
        mod_order_tags = (mod_order_tags,)
    else:
        mod_order_tags = tuple(mod_order_tags)

    if isinstance(bitwidths, (int, np.integer)):
        bitwidths = (bitwidths,)
    else:
        bitwidths = tuple(bitwidths)

    snr_db_vec = np.asarray(snr_db_vec, dtype=float)

    methods = ["Uncompressed", "BFP", "BlockScaling", "MuLaw"]
    n_methods = len(methods)
    n_mod = len(mod_order_tags)
    n_bw = len(bitwidths)
    n_snr = len(snr_db_vec)

    BLER = np.zeros((n_methods, n_mod, n_bw, n_snr), dtype=float)

    layout = None
    if mode == "cellfree":
        layout = generate_layout_cellfree(M_AP, K_UE)

    for imod, mod_tag in enumerate(mod_order_tags):
        # Interpret tag: 2 -> QPSK -> M=4; 16 -> 16QAM; 64 -> 64QAM
        if mod_tag == 2:
            M_qam = 4
        else:
            M_qam = mod_tag

        bits_per_sym = int(np.log2(M_qam))

        for ibw, bw in enumerate(bitwidths):
            print(f"=== Mode={mode}, ModTag={mod_tag}, bitWidth={bw} ===")

            for isnr, snr_db in enumerate(snr_db_vec):
                snr_lin = 10 ** (snr_db / 10.0)
                tx_power = 1.0
                noise_var = tx_power / snr_lin

                blk_err = np.zeros(n_methods, dtype=int)

                for _ in range(Ntrials):
                    # 1) Generate bits and symbols for all users
                    bits_tx = rng.integers(
                        0, 2,
                        size=(K_UE, N_sym_per_tb, bits_per_sym),
                        dtype=np.int8
                    )
                    s_tx = np.zeros((N_sym_per_tb, K_UE), dtype=complex)
                    for k in range(K_UE):
                        s_tx[:, k] = qam_mod(bits_tx[k], M_qam)

                    # 2) Channel (block fading per TB)
                    H = generate_channel(mode, M_AP, K_UE, layout=layout)
                    V = uplink_mmse_combiner(H, noise_var)

                    # === 新增：计算每个用户的等效增益 g_k ===
                    g = np.zeros(K_UE, dtype=complex)
                    for k in range(K_UE):
                        # vdot 会对第一个参数做共轭，所以相当于 v_k^H h_k
                        g[k] = np.vdot(V[:, k], H[:, k])
                        # 避免极端数值问题
                        if np.abs(g[k]) < 1e-12:
                            g[k] = 1e-12 + 0j

                    tb_error = np.zeros(n_methods, dtype=bool)

                    # 3) Transmit each symbol time in the TB
                    for t in range(N_sym_per_tb):
                        s = s_tx[t]  # (K,)
                        noise = np.sqrt(noise_var / 2) * (
                            rng.standard_normal(M_AP) +
                            1j * rng.standard_normal(M_AP)
                        )
                        y = H @ s + noise  # (M,)

                        for imeth, method in enumerate(methods):
                            if tb_error[imeth]:
                                continue

                            # Apply fronthaul compression at RU side
                            if method == "Uncompressed":
                                y_comp = y
                            elif method == "BFP":
                                y_comp, _ = compress_bfp_block(y, bw, mod_tag)
                            elif method == "BlockScaling":
                                y_comp, _ = compress_bsc_block(y, bw, mod_tag)
                            elif method == "MuLaw":
                                y_comp, _ = compress_mulaw_block(y, bw, mod_tag)
                            else:
                                raise ValueError("Unknown method.")

                            # CPU side combining
                            r = V.conj().T @ y_comp  # (K,)

                            # === 新增：对每个用户做幅度归一化后再解调 ===
                            any_error = False
                            for k in range(K_UE):
                                s_eff = r[k] / g[k]          # equalized symbol
                                bits_hat = qam_demod(
                                    np.array([s_eff]),
                                    M_qam
                                )[0]
                                if np.any(bits_hat != bits_tx[k, t, :]):
                                    any_error = True
                                    break

                            if any_error:
                                tb_error[imeth] = True

                    blk_err += tb_error.astype(int)

                BLER[:, imod, ibw, isnr] = blk_err / Ntrials

    return methods, mod_order_tags, bitwidths, snr_db_vec, BLER


# ============================================================
#  Plotting BLER Only
# ============================================================

def plot_bler_results(methods, mod_order_tags, bitwidths, snr_db_vec, BLER,
                      save_prefix=None):
    """
    Plot BLER vs SNR for different methods, for each (mod_tag, bitwidth) pair.
    Only BLER curves are plotted (no CR/EVM).

    支持 methods 中包含 "Uncompressed" 基线。
    """
    n_methods = len(methods)
    snr_db_vec = np.asarray(snr_db_vec, dtype=float)

    for imod, mod_tag in enumerate(mod_order_tags):
        for ibw, bw in enumerate(bitwidths):
            plt.figure()
            for imeth, name in enumerate(methods):
                plt.semilogy(
                    snr_db_vec,
                    BLER[imeth, imod, ibw, :],
                    marker='o',
                    label=name
                )

            if mod_tag == 2:
                mod_str = "QPSK"
            else:
                mod_str = f"{mod_tag}-QAM"

            plt.xlabel("SNR (dB)")
            plt.ylabel("BLER")
            plt.title(f"BLER vs SNR, Mod={mod_str}, bitWidth={bw}")
            plt.grid(True, which="both")
            plt.legend()
            plt.tight_layout()

            if save_prefix is not None:
                fname = f"{save_prefix}_mod{mod_tag}_bw{bw}.png"
                # plt.savefig(fname, dpi=150)
            # plt.show()  # 由主程序统一 plt.show()


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    # 模式选择: "mimo" 或 "cellfree"
    MODE = "cellfree"   # 改成 "mimo" 可切换集中式 MIMO

    # ========= 这里可以是单场景 =========
    # 例如：只验证 QPSK + 4bit 压缩：
    MOD_ORDER_TAGS = 2
    BITWIDTHS = 4

    # 当前示例：16 / 64 QAM + 8 / 10 bit
    # MOD_ORDER_TAGS = (16, 64)     # 2->QPSK, 16-QAM, 64-QAM
    # BITWIDTHS = (8, 10)

    SNR_DB_VEC = np.arange(60, 81, 4)

    NTRIALS = 500       # 每个 (mod, bw, snr) 的 TB 次数
    N_SYM_PER_TB = 50   # 每个 TB 内的符号数

    print("Starting BLER simulation ...")
    methods, mod_tags, bws, snrs, BLER = simulate_bler_all(
        mode=MODE,
        M_AP=8,
        K_UE=2,
        mod_order_tags=MOD_ORDER_TAGS,
        bitwidths=BITWIDTHS,
        snr_db_vec=SNR_DB_VEC,
        Ntrials=NTRIALS,
        N_sym_per_tb=N_SYM_PER_TB,
    )
    print("Simulation finished.")

    # 只画 BLER 曲线
    plot_bler_results(methods, mod_tags, bws, snrs, BLER,
                      save_prefix=f"bler_{MODE}")

    plt.show()
