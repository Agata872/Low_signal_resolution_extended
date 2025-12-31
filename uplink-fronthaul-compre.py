#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compression Analysis for 5G/O-RAN Fronthaul (Python version, no Deep Learning)

This script is a faithful Python counterpart of the MATLAB-style workflow you shared,
but with the critical modulation-label fix applied:

- BPSK  -> modOrder = 2
- QPSK  -> modOrder = 4
- 16-QAM-> modOrder = 16
- 64-QAM-> modOrder = 64

It generates the SAME types of figures as your MATLAB code (excluding deep learning):
1) Subplots (one per method): CR vs EVM
2) Combined plot: CR vs EVM for all methods
3) Bar chart: Average EVM per method
4) Subplots (one per method): CR vs Bitwidth

Block-size selection note (important for fairness and O-RAN consistency):
- In O-RAN/NR fronthaul compression, quantization parameters (scale/exponent/shift)
  are commonly shared over a PRB-level granularity. A PRB spans 12 subcarriers per
  OFDM symbol. Therefore we set blockSize = 12 for ALL modulation orders.
- This decouples block granularity from modulation format and ensures fair CR–EVM
  comparisons across compression schemes.

Dependencies: numpy, matplotlib
Run: python compression_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Reproducibility
# -----------------------------
rng = np.random.default_rng(42)


# -----------------------------
# QAM/BPSK Modulation (UnitAveragePower style)
# -----------------------------
def constellation_unit_power(M: int) -> np.ndarray:
    """
    Returns constellation points with unit average power.
    Supports:
      - M=2 (BPSK)
      - Square QAM: 4,16,64,256,...
    """
    if M == 2:
        # BPSK: {+1, -1}, already unit power
        return np.array([1 + 0j, -1 + 0j], dtype=np.complex128)

    m_side = int(np.sqrt(M))
    if m_side * m_side != M:
        raise ValueError("Only BPSK (M=2) and square QAM (4,16,64,...) are supported.")

    levels = np.arange(-(m_side - 1), m_side, 2)
    xv, yv = np.meshgrid(levels, levels)
    const = (xv + 1j * yv).flatten().astype(np.complex128)

    # UnitAveragePower normalize
    const /= np.sqrt(np.mean(np.abs(const) ** 2))
    return const


def qammod_from_integers(data: np.ndarray, M: int) -> np.ndarray:
    """
    MATLAB-like: qammod(data, M, 'UnitAveragePower', true)
    data: integers in [0, M-1]
    """
    data = np.asarray(data).astype(int).ravel()
    const = constellation_unit_power(M)
    if np.any(data < 0) or np.any(data >= M):
        raise ValueError(f"Data symbols must be in [0, {M-1}].")
    return const[data]


# -----------------------------
# EVM (match your MATLAB compute_evm behavior)
# -----------------------------
def compute_evm_percent(original: np.ndarray, compressed: np.ndarray, modOrder: int, methodName: str) -> float:
    """
    Matches the MATLAB logic you posted:
    - If methodName == "Modulation": EVM=0 (lossless)
    - Else: EVM = 100 * sqrt(errorPower/signalPower) * scalingFactor
      scalingFactor: 2->1, 4->1, 16->1.5, 64->2.0 (consistent with your MATLAB intent)
    """
    original = np.asarray(original).reshape(-1)
    compressed = np.asarray(compressed).reshape(-1)

    if methodName == "Modulation":
        return 0.0

    signalPower = np.mean(np.abs(original) ** 2) + np.finfo(float).eps
    errorPower = np.mean(np.abs(original - compressed) ** 2)

    if modOrder in (2, 4):      # BPSK/QPSK
        scalingFactor = 1.0
    elif modOrder == 16:
        scalingFactor = 1.5
    elif modOrder == 64:
        scalingFactor = 2.0
    else:
        scalingFactor = 1.0

    evm = max(0.0, 100.0 * np.sqrt(errorPower / signalPower) * scalingFactor)
    return float(evm)


# -----------------------------
# Compression: BFP (faithful to your MATLAB structure)
# -----------------------------
def _process_component_bfp(data: np.ndarray, blockSize: int, exponentBitWidth: int, mantissaBitWidth: int):
    data = np.asarray(data).reshape(-1)
    numBlocks = int(np.ceil(len(data) / blockSize))
    compressed = np.zeros_like(data, dtype=np.complex128)

    metadataBits = numBlocks * exponentBitWidth
    maxLevel = 2 ** (mantissaBitWidth - 1) - 1
    threshold = 2 ** (-mantissaBitWidth)

    for blk in range(numBlocks):
        s = blk * blockSize
        e = min((blk + 1) * blockSize, len(data))
        block = data[s:e]

        blockMax = np.max(np.abs(block))
        if blockMax < np.finfo(float).eps:
            compressed[s:e] = 0
            continue

        blockExponent = np.ceil(np.log2(blockMax + np.finfo(float).eps))
        blockExponent = np.clip(
            blockExponent,
            -2 ** (exponentBitWidth - 1),
            2 ** (exponentBitWidth - 1) - 1
        )

        scaleFactor = 2 ** (-blockExponent)

        quantizedBlock = np.round((block * scaleFactor) * maxLevel) / maxLevel
        quantizedBlock[np.abs(quantizedBlock) < threshold] = 0

        compressed[s:e] = quantizedBlock / scaleFactor

    return compressed, int(metadataBits)


def bfp_compression(data: np.ndarray, bitWidth: int, modOrder: int, blockSize: int = 12):
    """
    BFP compression with exponent+mantissa split (same as MATLAB: exponent=ceil(0.3*bitWidth)).
    Note: blockSize is fixed to 12 (PRB-level) for ALL modulations for fairness and O-RAN consistency.
    """
    exponentBitWidth = int(np.ceil(0.3 * bitWidth))
    mantissaBitWidth = bitWidth - exponentBitWidth

    compressed, metadataBits = _process_component_bfp(data, blockSize, exponentBitWidth, mantissaBitWidth)

    originalBitWidth = 32
    originalSize = data.size * originalBitWidth
    compressedSize = compressed.size * bitWidth + metadataBits
    CR = originalSize / compressedSize
    return compressed, float(CR)


# -----------------------------
# Compression: Block Scaling (faithful to your MATLAB structure)
# -----------------------------
def _process_block_scaling(data: np.ndarray, blockSize: int, exponentBitWidth: int, mantissaBitWidth: int):
    data = np.asarray(data).reshape(-1)
    numBlocks = int(np.ceil(len(data) / blockSize))
    out = np.zeros_like(data, dtype=float)
    metadataBits = numBlocks * exponentBitWidth

    maxLevel = 2 ** (mantissaBitWidth - 1) - 1
    threshold = 2 ** (-mantissaBitWidth)

    for blk in range(numBlocks):
        s = blk * blockSize
        e = min((blk + 1) * blockSize, len(data))
        block = data[s:e]

        blockMax = np.max(np.abs(block))
        if blockMax < np.finfo(float).eps:
            out[s:e] = 0
            continue

        blockExponent = np.ceil(np.log2(blockMax + np.finfo(float).eps))
        blockExponent = np.clip(
            blockExponent,
            -2 ** (exponentBitWidth - 1),
            2 ** (exponentBitWidth - 1) - 1
        )

        scaleFactor = 2 ** (-blockExponent)

        quantizedBlock = np.round((block * scaleFactor) * maxLevel) / maxLevel
        quantizedBlock[np.abs(quantizedBlock) < threshold] = 0

        out[s:e] = quantizedBlock / scaleFactor

    return out, int(metadataBits)


def bsc_compression(data: np.ndarray, bitWidth: int, modOrder: int, blockSize: int = 12):
    """
    Block Scaling compression. In your MATLAB you used exponentBitWidth depending on modulation.
    We'll keep that same rule, but fix blockSize = 12 for all modulations.
    """
    if modOrder in (2, 4):  # BPSK/QPSK
        exponentBitWidth = int(np.ceil(0.2 * bitWidth))
    elif modOrder == 16:
        exponentBitWidth = int(np.ceil(0.3 * bitWidth))
    elif modOrder == 64:
        exponentBitWidth = int(np.ceil(0.4 * bitWidth))
    else:
        raise ValueError("Unsupported modulation order for this demo.")

    mantissaBitWidth = bitWidth - exponentBitWidth

    isComplexData = np.iscomplexobj(data)
    if isComplexData:
        real_part = np.real(data)
        imag_part = np.imag(data)
        cr, meta_r = _process_block_scaling(real_part, blockSize, exponentBitWidth, mantissaBitWidth)
        ci, meta_i = _process_block_scaling(imag_part, blockSize, exponentBitWidth, mantissaBitWidth)
        compressed = cr + 1j * ci
        metadataBits = meta_r + meta_i
    else:
        compressed, metadataBits = _process_block_scaling(data, blockSize, exponentBitWidth, mantissaBitWidth)

    # Keep MATLAB-like CR formula (including sparsity counting)
    originalBitWidth = 32
    originalSize = data.size * originalBitWidth * (1 + int(isComplexData))
    nonzeros = np.count_nonzero(compressed)
    compressedSize = (nonzeros * bitWidth + metadataBits) * (1 + int(isComplexData))
    CR = originalSize / compressedSize if compressedSize > 0 else 1.0
    return compressed.astype(np.complex128), float(CR)


# -----------------------------
# Compression: μ-law (faithful to your MATLAB function)
# -----------------------------
def mu_law_compression(data: np.ndarray, bitWidth: int, _modOrder_unused: int):
    mu = 200
    eps = np.finfo(float).eps

    data = np.asarray(data).reshape(-1).astype(np.complex128)
    maxVal = np.max(np.abs(data))
    if maxVal == 0:
        return np.zeros_like(data), 1.0

    normalizedData = data / maxVal

    # MATLAB-like complex sign: sign(z)=z/abs(z), sign(0)=0
    abs_z = np.abs(normalizedData)
    sign_z = np.zeros_like(normalizedData)
    nz = abs_z > 0
    sign_z[nz] = normalizedData[nz] / abs_z[nz]

    # μ-law companding on magnitude, keep phase
    compressedData = sign_z * (np.log(1 + mu * abs_z) / np.log(1 + mu))

    quantizationLevels = 2 ** (bitWidth - 2)
    stepSize = 1.0 / (quantizationLevels - 2)

    noiseMultiplier_map = {8: 0.25, 9: 0.20, 10: 0.18, 12: 0.15, 14: 0.10}
    noiseMultiplier = noiseMultiplier_map.get(bitWidth, 0.25)

    # MATLAB randn -> real Gaussian noise
    noise = stepSize * noiseMultiplier * rng.standard_normal(size=compressedData.shape)

    qscale = (quantizationLevels - 2)
    quantizedData = np.round((compressedData + noise) * qscale) / qscale

    # expand
    decompressionScaling = 1 + 0.03 * noiseMultiplier
    abs_q = np.abs(quantizedData)

    # sign for quantized complex
    abs_q0 = abs_q > 0
    sign_q = np.zeros_like(quantizedData)
    sign_q[abs_q0] = quantizedData[abs_q0] / abs_q[abs_q0]

    decompressedData = sign_q * (1 / mu) * ((1 + mu) ** abs_q - 1)
    decompressedData = decompressedData * decompressionScaling

    compressed = decompressedData * maxVal

    originalBitWidth = 32
    compressedSize = quantizedData.size * bitWidth
    originalSize = data.size * originalBitWidth
    CR = originalSize / compressedSize

    return compressed.astype(np.complex128), float(CR)



# -----------------------------
# Compression: Modulation (lossless, CR formula like MATLAB)
# -----------------------------
def modulation_compression(data: np.ndarray, bitWidth: int, modOrder: int):
    data = np.asarray(data).reshape(-1).astype(np.complex128)
    maxValue = np.max(np.abs(data))
    if maxValue == 0:
        return {"compressed": np.zeros_like(data), "CR": 1.0, "EVM": 0.0}

    normalizedData = data / maxValue
    compressed = normalizedData * maxValue  # lossless

    originalBitWidth = 32
    numSymbols = data.size
    originalSize = numSymbols * originalBitWidth
    compressedSize = numSymbols * (bitWidth + np.log2(modOrder))
    CR = originalSize / compressedSize

    return {"compressed": compressed, "CR": float(CR), "EVM": 0.0}


# -----------------------------
# Plot: CR vs Bitwidth (subplots, like MATLAB helper)
# -----------------------------
def plot_cr_vs_bitwidth_all_techniques(results, modulationOrders, mod_labels):
    numTechniques = len(results)
    numRows = int(np.ceil(np.sqrt(numTechniques)))
    numCols = int(np.ceil(numTechniques / numRows))

    plt.figure()
    for i in range(numTechniques):
        ax = plt.subplot(numRows, numCols, i + 1)
        ax.grid(True)
        ax.set_xlabel("Bitwidth (bits)")
        ax.set_ylabel("Compression Ratio (CR)")
        ax.set_title(f"CR vs Bitwidth - {results[i]['methodName']}")

        colors = plt.cm.tab10(np.linspace(0, 1, len(modulationOrders)))
        markers = ['o', 's', 'd', '^']  # extended for 4 modulations

        for j, modOrder in enumerate(modulationOrders):
            valid = (np.array(results[i]["modulationOrder"]) == modOrder)
            bitWidths = np.array(results[i]["bitWidth"])[valid]
            CRs = np.array(results[i]["CR"])[valid]

            sortIdx = np.argsort(bitWidths)
            ax.plot(bitWidths[sortIdx], CRs[sortIdx],
                    marker=markers[j % len(markers)],
                    linewidth=1.5,
                    color=colors[j],
                    label=mod_labels[modOrder])

        ax.legend(loc="best")

    plt.suptitle("CR vs Bitwidth for All Compression Techniques")
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])


# -----------------------------
# Main: MATLAB-like analysis loop (no DL)
# -----------------------------
def run_compression_analysis():
    numSamples = int(1e5)
    originalBitWidth = 32

    # ✅ Corrected modulation orders
    modulationOrders = [2, 4, 16, 64]  # BPSK, QPSK, 16-QAM, 64-QAM
    mod_labels = {2: "BPSK", 4: "QPSK", 16: "16-QAM", 64: "64-QAM"}

    bitWidths = [8, 9, 10, 12, 14]

    # ===== Block-size selection (O-RAN/NR-consistent) =====
    # In O-RAN fronthaul compression, quantization parameters (scale/exponent/shift)
    # are commonly shared over a PRB-level granularity. Since a PRB spans 12 subcarriers
    # per OFDM symbol, we set blockSize = 12 for all modulation orders. This avoids
    # coupling block granularity with the modulation format and enables fair CR–EVM
    # comparisons across compression schemes.
    blockSize = 12

    compressionMethods = [
        bfp_compression,
        bsc_compression,
        mu_law_compression,
        modulation_compression
    ]
    methodNames = ["BFP", "BlockScaling", "MuLaw", "Modulation"]

    results = []
    for name in methodNames:
        results.append({
            "methodName": name,
            "bitWidth": [],
            "CR": [],
            "EVM": [],
            "originalSize": [],
            "compressedSize": [],
            "modulationOrder": []
        })

    print("Starting Compression Analysis...")

    for mIdx, methodName in enumerate(methodNames):
        print(f"\nAnalyzing Method: {methodName}")

        for modOrder in modulationOrders:
            dataBits = rng.integers(0, modOrder, size=numSamples, dtype=np.int32)
            IQ_samples = qammod_from_integers(dataBits, modOrder)

            for bitWidth in bitWidths:
                print(f"  Processing {mod_labels[modOrder]}, Bitwidth: {bitWidth}")

                if methodName == "Modulation":
                    compRes = modulation_compression(IQ_samples, bitWidth, modOrder)
                    compressed = compRes["compressed"]
                    CR = compRes["CR"]
                    EVM = compRes["EVM"]
                elif methodName == "BFP":
                    compressed, CR = bfp_compression(IQ_samples, bitWidth, modOrder, blockSize=blockSize)
                    EVM = compute_evm_percent(IQ_samples, compressed, modOrder, methodName)
                elif methodName == "BlockScaling":
                    compressed, CR = bsc_compression(IQ_samples, bitWidth, modOrder, blockSize=blockSize)
                    EVM = compute_evm_percent(IQ_samples, compressed, modOrder, methodName)
                elif methodName == "MuLaw":
                    compressed, CR = mu_law_compression(IQ_samples, bitWidth, modOrder)
                    EVM = compute_evm_percent(IQ_samples, compressed, modOrder, methodName)
                else:
                    raise ValueError("Unknown method")

                originalSize = IQ_samples.size * originalBitWidth
                compressedSize = IQ_samples.size * bitWidth

                results[mIdx]["bitWidth"].append(bitWidth)
                results[mIdx]["CR"].append(CR)
                results[mIdx]["EVM"].append(EVM)
                results[mIdx]["originalSize"].append(originalSize / 1e3)
                results[mIdx]["compressedSize"].append(compressedSize / 1e3)
                results[mIdx]["modulationOrder"].append(modOrder)

                print(f"    CR: {CR:.2f}, EVM: {EVM:.2f}%, "
                      f"Original: {originalSize/1e3:.2f} kb, Compressed: {compressedSize/1e3:.2f} kb")

    print("\nCompression Analysis Completed.\n")
    return results, modulationOrders, mod_labels


def plot_all_figures(results, modulationOrders, mod_labels):
    # =========================
    # 1) CR vs EVM for each method separately (subplots)
    # =========================
    numTechniques = len(results)
    numRows = int(np.ceil(np.sqrt(numTechniques)))
    numCols = int(np.ceil(numTechniques / numRows))

    plt.figure()
    for i in range(numTechniques):
        ax = plt.subplot(numRows, numCols, i + 1)
        ax.grid(True)
        ax.set_xlabel("Compression Ratio (CR)")
        ax.set_ylabel("EVM (%)")
        ax.set_title(f"CR vs EVM - {results[i]['methodName']}")

        colors = plt.cm.tab10(np.linspace(0, 1, len(modulationOrders)))
        markers = ['o', 's', 'd', '^']

        for j, modOrder in enumerate(modulationOrders):
            valid = (np.array(results[i]["modulationOrder"]) == modOrder)
            CR_values = np.array(results[i]["CR"])[valid]
            EVM_values = np.array(results[i]["EVM"])[valid]

            sortIdx = np.argsort(CR_values)
            ax.plot(CR_values[sortIdx], EVM_values[sortIdx],
                    marker=markers[j % len(markers)],
                    linewidth=1.5,
                    color=colors[j],
                    label=mod_labels[modOrder])

        ax.legend(loc="best")

    plt.suptitle("CR vs EVM for All Compression Techniques")
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # =========================
    # 2) Combined Plot: CR vs EVM for all methods together
    # =========================
    plt.figure()
    plt.grid(True)
    plt.xlabel("Compression Ratio (CR)")
    plt.ylabel("EVM (%)")
    plt.title("CR vs EVM for All Compression Methods")

    colors = plt.cm.tab10(np.linspace(0, 1, numTechniques))
    markers = ['o', 's', 'd', '^', 'v', 'p', 'h', 'x']

    for i in range(numTechniques):
        methodName = results[i]["methodName"]
        for modOrder in modulationOrders:
            valid = (np.array(results[i]["modulationOrder"]) == modOrder)

            if "MuLaw" in methodName:
                lineStyle = '--'
                markerStyle = 'd'
            else:
                lineStyle = '-'
                markerStyle = markers[(modOrder % len(markers))]

            plt.plot(np.array(results[i]["CR"])[valid],
                     np.array(results[i]["EVM"])[valid],
                     lineStyle,
                     marker=markerStyle,
                     linewidth=1.5,
                     markersize=6,
                     color=colors[i],
                     label=f"{methodName} - {mod_labels[modOrder]}")

    plt.legend(loc="best")

    # =========================
    # 3) Bar Chart: Average EVM per method
    # =========================
    plt.figure()
    avgEVMs = [float(np.mean(r["EVM"])) for r in results]
    x = np.arange(len(avgEVMs))

    plt.bar(x, avgEVMs)
    plt.xticks(x, [r["methodName"] for r in results])
    plt.xlabel("Compression Methods")
    plt.ylabel("Average EVM (%)")
    plt.title("Average EVM Comparison")
    plt.grid(True, axis='y')

    for i, v in enumerate(avgEVMs):
        plt.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")

    # =========================
    # 4) CR vs Bitwidth for each method separately (subplots)
    # =========================
    plot_cr_vs_bitwidth_all_techniques(results, modulationOrders, mod_labels)

    plt.show()


def main():
    results, modulationOrders, mod_labels = run_compression_analysis()
    plot_all_figures(results, modulationOrders, mod_labels)


if __name__ == "__main__":
    main()
