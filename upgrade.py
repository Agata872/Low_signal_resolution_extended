#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORAN Fronthaul Compression BLER Simulation

Methods:
- BFP (Block Floating Point)
- BlockScaling (Block Scaling)
- MuLaw (μ-law based compression)
- Modulation (lossless placeholder)

Channel modes:
- "mimo": centralized MIMO, i.i.d. Rayleigh
- "cellfree": cell-free with geometry-based large-scale fading

Author: (adapted for Tianzheng)
"""
import matplotlib.pyplot as plt
from tool import *


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
    Simulate BLER for four compression methods:
    BFP, BlockScaling, MuLaw, Modulation (lossless).

    Returns:
        methods: list of method names
        mod_order_tags: tuple of modulation tags (2,16,64)
        bitwidths: tuple of bit widths
        snr_db_vec: SNR grid
        BLER: array shape (n_methods, n_mod, n_bw, n_snr)
    """
    methods = ["BFP", "BlockScaling", "MuLaw", "Modulation"]
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
                            if method == "BFP":
                                y_comp, _ = compress_bfp_block(y, bw, mod_tag)
                            elif method == "BlockScaling":
                                y_comp, _ = compress_bsc_block(y, bw, mod_tag)
                            elif method == "MuLaw":
                                y_comp, _ = compress_mulaw_block(y, bw, mod_tag)
                            elif method == "Modulation":
                                y_comp, _ = modulation_compression(y, bw, mod_tag)
                            else:
                                raise ValueError("Unknown method.")

                            # CPU side combining
                            r = V.conj().T @ y_comp  # (K,)

                            # Demodulate each user
                            any_error = False
                            for k in range(K_UE):
                                bits_hat = qam_demod(np.array([r[k]]), M_qam)[0]
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
    """
    n_methods = len(methods)
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
            # 如果不想自动弹窗可以注释掉 show
            # plt.show()


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    # 模式选择: "mimo" 或 "cellfree"
    MODE = "cellfree"   # 改成 "mimo" 可切换集中式 MIMO

    # 基本参数
    M_AP = 8
    K_UE = 2
    MOD_ORDER_TAGS = (16, 64)   # 2->QPSK, 16-QAM, 64-QAM
    # BITWIDTHS = (8, 10, 12, 14)
    BITWIDTHS = (8, 10)
    SNR_DB_VEC = np.arange(80, 101, 4)  # 0,4,8,...,20 dB

    NTRIALS = 500       # 每个 (mod, bw, snr) 的 TB 次数
    N_SYM_PER_TB = 50   # 每个 TB 内的符号数

    print("Starting BLER simulation ...")
    methods, mod_tags, bws, snrs, BLER = simulate_bler_all(
        mode=MODE,
        M_AP=M_AP,
        K_UE=K_UE,
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
