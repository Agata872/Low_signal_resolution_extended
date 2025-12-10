#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
仿真 DD-MIMO 上行场景中 O-RAN 压缩算法对 BLER 的影响

包含算法：
1) 无压缩（baseline）
2) Block Scaling (BS)
3) Block Floating Point (BFP)
4) μ-law 压缩
5) Uniform 量化

依赖: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# 固定随机种子，便于复现
rng = np.random.default_rng(42)


# ================= QAM 调制 / 解调 ================= #

def qam_constellation(M: int) -> np.ndarray:
    """
    生成正方形 M-QAM 星座，平均功率归一化为 1
    使用自然二进制映射（不是 Gray 码）
    """
    m_side = int(np.sqrt(M))
    if m_side ** 2 != M:
        raise ValueError("Only square QAM is supported, e.g., 4,16,64,256.")

    # 一维坐标: -(m_side-1),..., -1,1,...,(m_side-1)
    re_im_vals = np.arange(-(m_side - 1), m_side, 2)
    xv, yv = np.meshgrid(re_im_vals, re_im_vals)
    const = xv + 1j * yv  # 笛卡尔积

    const = const.flatten()  # 大小 M 的一维数组
    # 归一化平均功率为 1
    const_power = np.mean(np.abs(const) ** 2)
    const /= np.sqrt(const_power)
    return const


def bits_to_int(bits: np.ndarray) -> np.ndarray:
    """
    将形状 [..., n_bits] 的比特向量转换为整数（MSB 优先）
    """
    n_bits = bits.shape[-1]
    weights = 1 << np.arange(n_bits - 1, -1, -1)
    return np.sum(bits * weights, axis=-1)


def int_to_bits(vals: np.ndarray, n_bits: int) -> np.ndarray:
    """
    将整数数组 vals 转为比特数组，形状 [len(vals), n_bits]，MSB 优先
    """
    bits = (((vals[:, None] & (1 << np.arange(n_bits - 1, -1, -1))) > 0)
            .astype(np.int8))
    return bits


def qam_mod(bits: np.ndarray, M: int) -> np.ndarray:
    """
    QAM 调制
    bits: shape [N_bits,]
    返回: shape [N_sym,] 的复数符号
    """
    bits = np.asarray(bits).astype(np.int8)
    bits_per_sym = int(np.log2(M))
    if bits.size % bits_per_sym != 0:
        raise ValueError("Number of bits must be multiple of log2(M).")

    const = qam_constellation(M)
    bit_groups = bits.reshape(-1, bits_per_sym)
    idx = bits_to_int(bit_groups)
    syms = const[idx]
    return syms


def qam_demod(syms: np.ndarray, M: int) -> np.ndarray:
    """
    QAM 解调（最小距离硬判决）
    syms: shape [N_sym,]
    返回: shape [N_bits,] 的比特
    """
    syms = np.asarray(syms)
    const = qam_constellation(M)
    bits_per_sym = int(np.log2(M))

    # 逐符号找最近星座点
    # 可以向量化：dist_matrix [N_sym, M]
    dist2 = np.abs(syms[:, None] - const[None, :]) ** 2
    idx_hat = np.argmin(dist2, axis=1)
    bits_block = int_to_bits(idx_hat, bits_per_sym)
    return bits_block.reshape(-1)


# =============== 压缩算法实现（逐 block） =============== #

def compress_bs_block(y: np.ndarray, qbits: int) -> np.ndarray:
    """
    Block Scaling 压缩，实部和虚部共享一个 scale
    y: 复数向量 [N,]
    qbits: 量化比特数（对实部、虚部分别量化）
    """
    y = np.asarray(y)
    y_hat = np.zeros_like(y, dtype=np.complex128)
    if y.size == 0:
        return y_hat

    re = np.real(y)
    im = np.imag(y)
    max_val = np.max(np.abs(np.concatenate([re, im])))
    if max_val == 0:
        return y_hat

    max_int = (1 << (qbits - 1)) - 1  # 2^(qbits-1)-1
    S = max_val / max_int  # y/S ≈ 整数范围

    # 实部
    re_n = re / S
    re_q = np.round(re_n)
    re_q = np.clip(re_q, -max_int - 1, max_int)
    re_hat = re_q * S

    # 虚部
    im_n = im / S
    im_q = np.round(im_n)
    im_q = np.clip(im_q, -max_int - 1, max_int)
    im_hat = im_q * S

    y_hat = re_hat + 1j * im_hat
    return y_hat


def compress_bfp_block(y: np.ndarray, qbits: int) -> np.ndarray:
    """
    Block Floating Point 压缩
    使用 block-level exponent e，mantissa 在 [-1, 1] 上量化
    """
    y = np.asarray(y)
    y_hat = np.zeros_like(y, dtype=np.complex128)
    if y.size == 0:
        return y_hat

    re = np.real(y)
    im = np.imag(y)
    max_val = np.max(np.abs(np.concatenate([re, im])))
    if max_val == 0:
        return y_hat

    # exponent e
    e = np.floor(np.log2(max_val))
    scale = 2.0 ** e

    # mantissa in [-1,1]
    re_m = re / scale
    im_m = im / scale

    max_int = (1 << (qbits - 1)) - 1

    # 实部 mantissa 量化
    re_q = np.round(re_m * max_int)
    re_q = np.clip(re_q, -max_int - 1, max_int)
    re_hat = (re_q / max_int) * scale

    # 虚部 mantissa 量化
    im_q = np.round(im_m * max_int)
    im_q = np.clip(im_q, -max_int - 1, max_int)
    im_hat = (im_q / max_int) * scale

    y_hat = re_hat + 1j * im_hat
    return y_hat


def compress_mulaw_block(y: np.ndarray, qbits: int, mu: float = 8.0) -> np.ndarray:
    """
    μ-law 压缩（按块归一化到 [-1,1]，再 compand & expand）
    """
    y = np.asarray(y)
    y_hat = np.zeros_like(y, dtype=np.complex128)
    if y.size == 0:
        return y_hat

    re = np.real(y)
    im = np.imag(y)
    max_val = np.max(np.abs(np.concatenate([re, im])))
    if max_val == 0:
        return y_hat

    # 归一化到 [-1,1]
    re_n = re / max_val
    im_n = im / max_val

    # μ-law companding
    re_s = np.sign(re_n)
    im_s = np.sign(im_n)
    re_a = np.abs(re_n)
    im_a = np.abs(im_n)

    re_c = re_s * (np.log(1 + mu * re_a) / np.log(1 + mu))
    im_c = im_s * (np.log(1 + mu * im_a) / np.log(1 + mu))

    # 量化 companded 值
    max_int = (1 << (qbits - 1)) - 1
    re_q = np.round(re_c * max_int)
    re_q = np.clip(re_q, -max_int - 1, max_int)
    im_q = np.round(im_c * max_int)
    im_q = np.clip(im_q, -max_int - 1, max_int)

    re_c_hat = re_q / max_int
    im_c_hat = im_q / max_int

    # μ-law expand
    re_a_hat = ((1 + mu) ** np.abs(re_c_hat) - 1) / mu
    im_a_hat = ((1 + mu) ** np.abs(im_c_hat) - 1) / mu

    re_hat = np.sign(re_c_hat) * re_a_hat * max_val
    im_hat = np.sign(im_c_hat) * im_a_hat * max_val

    y_hat = re_hat + 1j * im_hat
    return y_hat


def compress_uniform_block(y: np.ndarray, qbits: int, fullscale: float = 4.0) -> np.ndarray:
    """
    Uniform 量化，使用固定 fullscale 范围 [-fullscale, fullscale]
    """
    y = np.asarray(y)
    y_hat = np.zeros_like(y, dtype=np.complex128)
    if y.size == 0:
        return y_hat

    re = np.real(y)
    im = np.imag(y)

    max_int = (1 << (qbits - 1)) - 1

    # 实部
    re_n = re / fullscale
    re_q = np.round(re_n * max_int)
    re_q = np.clip(re_q, -max_int - 1, max_int)
    re_hat = (re_q / max_int) * fullscale

    # 虚部
    im_n = im / fullscale
    im_q = np.round(im_n * max_int)
    im_q = np.clip(im_q, -max_int - 1, max_int)
    im_hat = (im_q / max_int) * fullscale

    y_hat = re_hat + 1j * im_hat
    return y_hat


# ===================== 主仿真函数 ===================== #

def main():
    # ---------- 场景参数 ---------- #
    M_RU = 8      # RU 数量
    K_UE = 2      # UE 数量
    N_sym_per_TB = 144  # 每个 UE 每个 TB 的符号数（可看作 12 PRB*12 RE）

    # 选择 MCS（决定 QAM 阶数 & 压缩比特）
    # 1=QPSK, 2=16QAM, 3=64QAM, 4=256QAM
    mcs = 4

    if mcs == 1:
        M_mod = 4
        name_mod = "QPSK"
        CW_bits = 2
    elif mcs == 2:
        M_mod = 16
        name_mod = "16QAM"
        CW_bits = 4
    elif mcs == 3:
        M_mod = 64
        name_mod = "64QAM"
        CW_bits = 5
    elif mcs == 4:
        M_mod = 256
        name_mod = "256QAM"
        CW_bits = 6
    else:
        raise ValueError("Unknown MCS value")

    bits_per_sym = int(np.log2(M_mod))
    bits_per_TB = N_sym_per_TB * bits_per_sym

    # SNR 扫描范围（dB）
    SNRdB_vec = np.arange(0, 29, 4)  # 0,4,...,28
    N_SNR = len(SNRdB_vec)

    # 每个 SNR 下的 Monte-Carlo 次数（可以调大）
    Ntrials = 300

    tx_symbol_power = 1.0  # 调制归一化后为 1

    algo_names = [
        "No compression",
        "Block Scaling (BS)",
        "Block Floating Point (BFP)",
        "mu-law",
        "Uniform"
    ]
    N_algo = len(algo_names)

    BLER = np.zeros((N_algo, N_SNR), dtype=float)

    print(f"Simulating {name_mod}, CW={CW_bits} bits, M={M_RU}, K={K_UE} ...")

    # ---------- 遍历 SNR ---------- #
    for iSNR, SNRdB in enumerate(SNRdB_vec):
        SNRlin = 10 ** (SNRdB / 10.0)
        noise_var = tx_symbol_power / SNRlin

        blk_err_cnt = np.zeros(N_algo, dtype=int)

        for _ in range(Ntrials):
            # ------ 1. 生成发送比特 & 调制 ------ #
            # bits_tx shape: [K_UE, bits_per_TB]
            bits_tx = rng.integers(0, 2, size=(K_UE, bits_per_TB), dtype=np.int8)
            # s_tx shape: [N_sym, K_UE]
            s_tx = np.zeros((N_sym_per_TB, K_UE), dtype=complex)
            for k in range(K_UE):
                syms_k = qam_mod(bits_tx[k], M_mod)
                s_tx[:, k] = syms_k

            # ------ 2. 信道 & 噪声 ------ #
            # H shape: [N_sym, M_RU, K_UE]
            H = (rng.normal(size=(N_sym_per_TB, M_RU, K_UE)) +
                 1j * rng.normal(size=(N_sym_per_TB, M_RU, K_UE))) / np.sqrt(2)

            # 接收信号 Y, shape: [N_sym, M_RU]
            Y = np.zeros((N_sym_per_TB, M_RU), dtype=complex)
            for n in range(N_sym_per_TB):
                g_n = H[n]          # [M_RU, K_UE]
                s_n = s_tx[n]       # [K_UE,]
                y_n = g_n @ s_n
                noise = (rng.normal(size=M_RU) + 1j * rng.normal(size=M_RU)) \
                    * np.sqrt(noise_var / 2)
                Y[n] = y_n + noise

            # 转置方便逐 RU 压缩: [M_RU, N_sym]
            Y_T = Y.T

            # ------ 3. 各种压缩算法 ------ #
            Yhat_list = [None] * N_algo

            # (1) 无压缩
            Yhat_list[0] = Y_T.copy()

            # (2) BS
            Yhat_bs = np.zeros_like(Y_T)
            for m in range(M_RU):
                Yhat_bs[m] = compress_bs_block(Y_T[m], CW_bits)
            Yhat_list[1] = Yhat_bs

            # (3) BFP
            Yhat_bfp = np.zeros_like(Y_T)
            for m in range(M_RU):
                Yhat_bfp[m] = compress_bfp_block(Y_T[m], CW_bits)
            Yhat_list[2] = Yhat_bfp

            # (4) mu-law
            Yhat_mu = np.zeros_like(Y_T)
            for m in range(M_RU):
                Yhat_mu[m] = compress_mulaw_block(Y_T[m], CW_bits, mu=8.0)
            Yhat_list[3] = Yhat_mu

            # (5) Uniform
            Yhat_uni = np.zeros_like(Y_T)
            for m in range(M_RU):
                Yhat_uni[m] = compress_uniform_block(Y_T[m], CW_bits, fullscale=4.0)
            Yhat_list[4] = Yhat_uni

            # ------ 4. DU 侧检测 & BLER 统计 ------ #
            for ia in range(N_algo):
                Y_use_T = Yhat_list[ia]      # [M_RU, N_sym]
                bits_hat = np.zeros((K_UE, bits_per_TB), dtype=np.int8)

                for n in range(N_sym_per_TB):
                    g_n = H[n]              # [M_RU, K_UE]
                    y_n = Y_use_T[:, n]     # [M_RU,]

                    # MMSE 合并: R = G G^H + sigma^2 I, W = R^{-1} G
                    R = g_n @ g_n.conj().T + noise_var * np.eye(M_RU)
                    W = np.linalg.solve(R, g_n)    # [M_RU, K_UE]
                    x_hat = W.conj().T @ y_n      # [K_UE,]

                    # QAM 解调
                    for k in range(K_UE):
                        sym_hat = x_hat[k]
                        bits_demod_k = qam_demod(np.array([sym_hat]), M_mod)
                        # 写入 TB 比特流
                        start = n * bits_per_sym
                        end = start + bits_per_sym
                        bits_hat[k, start:end] = bits_demod_k

                # 统计 block error（一个 UE 的 TB 只要有一位错就算 1 个 block error）
                for k in range(K_UE):
                    n_err = np.sum(bits_hat[k] != bits_tx[k])
                    if n_err > 0:
                        blk_err_cnt[ia] += 1

        total_TB = Ntrials * K_UE
        BLER[:, iSNR] = blk_err_cnt / total_TB

        print(f"SNR = {SNRdB:2d} dB | "
              f"{algo_names[0]}={BLER[0, iSNR]:.3f}, "
              f"BS={BLER[1, iSNR]:.3f}, "
              f"BFP={BLER[2, iSNR]:.3f}, "
              f"muL={BLER[3, iSNR]:.3f}, "
              f"Uni={BLER[4, iSNR]:.3f}")

    # ---------- 作图 ---------- #
    plt.figure()
    markers = ['o', 's', 'd', '^', 'v']
    for i in range(N_algo):
        plt.semilogy(SNRdB_vec, BLER[i], '-' + markers[i],
                     linewidth=1.5, label=algo_names[i])
    plt.grid(True, which='both')
    plt.xlabel("SNR (dB)")
    plt.ylabel("BLER")
    plt.ylim(1e-3, 1)
    plt.xlim(SNRdB_vec[0], SNRdB_vec[-1])
    plt.title(f"DD-MIMO, {name_mod}, CW={CW_bits} bits, M={M_RU}, K={K_UE}")
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
