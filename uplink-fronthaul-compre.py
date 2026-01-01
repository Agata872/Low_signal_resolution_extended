#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tikzplotlib as tikz

rng = np.random.default_rng(42)

TEX_DIR = "tex_figures"
os.makedirs(TEX_DIR, exist_ok=True)


def _sanitize_for_tikz(fig: plt.Figure):
    """
    Work around tikzplotlib/matplotlib private API incompatibilities:
    - Force all Line2D objects to have solid linestyle to avoid dashSeq access.
    """
    for ax in fig.get_axes():
        # Lines in the axes
        for line in ax.get_lines():
            line.set_linestyle("-")  # avoid dashed patterns that trigger _us_dashSeq
        # Legend lines (sometimes stored separately)
        leg = ax.get_legend()
        if leg is not None:
            for legline in leg.get_lines():
                legline.set_linestyle("-")


def save_tex(fig: plt.Figure, name: str):
    path = os.path.join(TEX_DIR, f"{name}.tex")
    _sanitize_for_tikz(fig)
    tikz.save(path, figure=fig)
    print(f"[Saved TEX] {path}")


# -----------------------------
# QAM/BPSK Modulation (UnitAveragePower style)
# -----------------------------
def constellation_unit_power(M: int) -> np.ndarray:
    if M == 2:
        return np.array([1 + 0j, -1 + 0j], dtype=np.complex128)

    m_side = int(np.sqrt(M))
    if m_side * m_side != M:
        raise ValueError("Only BPSK (M=2) and square QAM (4,16,64,...) are supported.")

    levels = np.arange(-(m_side - 1), m_side, 2)
    xv, yv = np.meshgrid(levels, levels)
    const = (xv + 1j * yv).flatten().astype(np.complex128)
    const /= np.sqrt(np.mean(np.abs(const) ** 2))
    return const


def qammod_from_integers(data: np.ndarray, M: int) -> np.ndarray:
    data = np.asarray(data).astype(int).ravel()
    const = constellation_unit_power(M)
    if np.any(data < 0) or np.any(data >= M):
        raise ValueError(f"Data symbols must be in [0, {M-1}].")
    return const[data]


def compute_evm_percent(original: np.ndarray, compressed: np.ndarray, modOrder: int, methodName: str) -> float:
    if methodName == "Modulation":
        return 0.0

    original = np.asarray(original).reshape(-1)
    compressed = np.asarray(compressed).reshape(-1)

    signalPower = np.mean(np.abs(original) ** 2) + np.finfo(float).eps
    errorPower = np.mean(np.abs(original - compressed) ** 2)

    if modOrder in (2, 4):
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
# Compression: BFP
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
        blockExponent = np.clip(blockExponent,
                                -2 ** (exponentBitWidth - 1),
                                2 ** (exponentBitWidth - 1) - 1)

        scaleFactor = 2 ** (-blockExponent)
        quantizedBlock = np.round((block * scaleFactor) * maxLevel) / maxLevel
        quantizedBlock[np.abs(quantizedBlock) < threshold] = 0
        compressed[s:e] = quantizedBlock / scaleFactor

    return compressed, int(metadataBits)


def bfp_compression(data: np.ndarray, bitWidth: int, modOrder: int, blockSize: int = 12):
    exponentBitWidth = int(np.ceil(0.3 * bitWidth))
    mantissaBitWidth = bitWidth - exponentBitWidth
    compressed, metadataBits = _process_component_bfp(data, blockSize, exponentBitWidth, mantissaBitWidth)

    originalBitWidth = 32
    originalSize = data.size * originalBitWidth
    compressedSize = compressed.size * bitWidth + metadataBits
    CR = originalSize / compressedSize
    return compressed, float(CR)


# -----------------------------
# Compression: Block Scaling
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
        blockExponent = np.clip(blockExponent,
                                -2 ** (exponentBitWidth - 1),
                                2 ** (exponentBitWidth - 1) - 1)

        scaleFactor = 2 ** (-blockExponent)
        quantizedBlock = np.round((block * scaleFactor) * maxLevel) / maxLevel
        quantizedBlock[np.abs(quantizedBlock) < threshold] = 0
        out[s:e] = quantizedBlock / scaleFactor

    return out, int(metadataBits)


def bsc_compression(data: np.ndarray, bitWidth: int, modOrder: int, blockSize: int = 12):
    if modOrder in (2, 4):
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

    originalBitWidth = 32
    originalSize = data.size * originalBitWidth * (1 + int(isComplexData))
    nonzeros = np.count_nonzero(compressed)
    compressedSize = (nonzeros * bitWidth + metadataBits) * (1 + int(isComplexData))
    CR = originalSize / compressedSize if compressedSize > 0 else 1.0
    return compressed.astype(np.complex128), float(CR)


# -----------------------------
# Compression: μ-law
# -----------------------------
def mu_law_compression(data: np.ndarray, bitWidth: int, _modOrder_unused: int):
    mu = 200
    data = np.asarray(data).reshape(-1).astype(np.complex128)

    maxVal = np.max(np.abs(data))
    if maxVal == 0:
        return np.zeros_like(data), 1.0

    normalizedData = data / maxVal

    abs_z = np.abs(normalizedData)
    sign_z = np.zeros_like(normalizedData)
    nz = abs_z > 0
    sign_z[nz] = normalizedData[nz] / abs_z[nz]

    compressedData = sign_z * (np.log(1 + mu * abs_z) / np.log(1 + mu))

    quantizationLevels = 2 ** (bitWidth - 2)
    stepSize = 1.0 / (quantizationLevels - 2)

    noiseMultiplier_map = {8: 0.25, 9: 0.20, 10: 0.18, 12: 0.15, 14: 0.10}
    noiseMultiplier = noiseMultiplier_map.get(bitWidth, 0.25)

    noise = stepSize * noiseMultiplier * rng.standard_normal(size=compressedData.shape)
    qscale = (quantizationLevels - 2)
    quantizedData = np.round((compressedData + noise) * qscale) / qscale

    decompressionScaling = 1 + 0.03 * noiseMultiplier
    abs_q = np.abs(quantizedData)

    sign_q = np.zeros_like(quantizedData)
    nzq = abs_q > 0
    sign_q[nzq] = quantizedData[nzq] / abs_q[nzq]

    decompressedData = sign_q * (1 / mu) * ((1 + mu) ** abs_q - 1)
    decompressedData = decompressedData * decompressionScaling
    compressed = decompressedData * maxVal

    originalBitWidth = 32
    compressedSize = quantizedData.size * bitWidth
    originalSize = data.size * originalBitWidth
    CR = originalSize / compressedSize

    return compressed.astype(np.complex128), float(CR)


def modulation_compression(data: np.ndarray, bitWidth: int, modOrder: int):
    data = np.asarray(data).reshape(-1).astype(np.complex128)
    maxValue = np.max(np.abs(data))
    if maxValue == 0:
        return {"compressed": np.zeros_like(data), "CR": 1.0, "EVM": 0.0}

    normalizedData = data / maxValue
    compressed = normalizedData * maxValue

    originalBitWidth = 32
    numSymbols = data.size
    originalSize = numSymbols * originalBitWidth
    compressedSize = numSymbols * (bitWidth + np.log2(modOrder))
    CR = originalSize / compressedSize

    return {"compressed": compressed, "CR": float(CR), "EVM": 0.0}


def plot_cr_vs_bitwidth_all_techniques(results, modulationOrders, mod_labels):
    numTechniques = len(results)
    numRows = int(np.ceil(np.sqrt(numTechniques)))
    numCols = int(np.ceil(numTechniques / numRows))

    fig = plt.figure()
    for i in range(numTechniques):
        ax = plt.subplot(numRows, numCols, i + 1)
        ax.grid(True)
        ax.set_xlabel("Bitwidth (bits)")
        ax.set_ylabel("Compression Ratio (CR)")
        ax.set_title(f"CR vs Bitwidth - {results[i]['methodName']}")

        colors = plt.cm.tab10(np.linspace(0, 1, len(modulationOrders)))
        markers = ['o', 's', 'd', '^']

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

    fig.suptitle("CR vs Bitwidth for All Compression Techniques")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    save_tex(fig, "CR_vs_Bitwidth_subplots")
    return fig


def run_compression_analysis():
    numSamples = int(1e5)
    originalBitWidth = 32

    modulationOrders = [2, 4, 16, 64]
    mod_labels = {2: "BPSK", 4: "QPSK", 16: "16-QAM", 64: "64-QAM"}
    bitWidths = [8, 9, 10, 12, 14]
    blockSize = 12

    methodNames = ["BFP", "BlockScaling", "MuLaw", "Modulation"]
    results = [{
        "methodName": name,
        "bitWidth": [],
        "CR": [],
        "EVM": [],
        "originalSize": [],
        "compressedSize": [],
        "modulationOrder": []
    } for name in methodNames]

    print("Starting Compression Analysis...")

    for methodName in methodNames:
        print(f"\nAnalyzing Method: {methodName}")
        mIdx = methodNames.index(methodName)

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

                print(f"    CR: {CR:.2f}, EVM: {EVM:.2f}%")

    print("\nCompression Analysis Completed.\n")
    return results, modulationOrders, mod_labels


def plot_all_figures(results, modulationOrders, mod_labels):
    numTechniques = len(results)
    numRows = int(np.ceil(np.sqrt(numTechniques)))
    numCols = int(np.ceil(numTechniques / numRows))

    # 1) CR vs EVM subplots
    fig1 = plt.figure()
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

    fig1.suptitle("CR vs EVM for All Compression Techniques")
    fig1.tight_layout(rect=[0, 0.02, 1, 0.95])
    save_tex(fig1, "CR_vs_EVM_subplots")

    # 2) Combined plot (manual legends)
    fig2, ax = plt.subplots()
    ax.grid(True)
    ax.set_xlabel("Compression Ratio (CR)")
    ax.set_ylabel("EVM (%)")
    ax.set_title("CR vs EVM for All Compression Methods")

    method_style = {
        "BFP":          {"color": "tab:blue", "linestyle": "-"},
        "BlockScaling": {"color": "tab:red",  "linestyle": "-"},
        "MuLaw":        {"color": "tab:pink", "linestyle": "--"},  # will be sanitized to '-'
        "Modulation":   {"color": "tab:cyan", "linestyle": "-"},
    }
    marker_map = {2: 'x', 4: 'o', 16: 's', 64: 'd'}

    for i in range(numTechniques):
        methodName = results[i]["methodName"]
        sty = method_style[methodName]

        for modOrder in modulationOrders:
            valid = (np.array(results[i]["modulationOrder"]) == modOrder)
            CR_values = np.array(results[i]["CR"])[valid]
            EVM_values = np.array(results[i]["EVM"])[valid]

            ax.plot(CR_values, EVM_values,
                    linestyle=sty["linestyle"],
                    color=sty["color"],
                    marker=marker_map[modOrder],
                    linewidth=1.5,
                    markersize=6)

    mod_handles = [
        Line2D([0], [0], color="black", marker=marker_map[M], linestyle="None", markersize=7, label=mod_labels[M])
        for M in modulationOrders
    ]
    method_handles = [
        Line2D([0], [0], color=sty["color"], linestyle=sty["linestyle"], linewidth=2.0, label=mn)
        for mn, sty in method_style.items()
    ]

    leg1 = ax.legend(handles=mod_handles, title="Modulation", loc="upper left", frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=method_handles, title="Compression Method", loc="lower left", frameon=True)

    fig2.tight_layout()
    save_tex(fig2, "CR_vs_EVM_combined")

    # 3) Bar chart
    fig3 = plt.figure()
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

    fig3.tight_layout()
    save_tex(fig3, "Average_EVM_bar")

    # 4) CR vs Bitwidth subplots
    plot_cr_vs_bitwidth_all_techniques(results, modulationOrders, mod_labels)

    plt.show()


def main():
    results, modulationOrders, mod_labels = run_compression_analysis()
    plot_all_figures(results, modulationOrders, mod_labels)


if __name__ == "__main__":
    main()
