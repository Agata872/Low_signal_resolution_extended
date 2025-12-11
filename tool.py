import numpy as np
rng = np.random.default_rng(1232)
# ============================================================
#  QAM Modulation / Demodulation
# ============================================================

def qam_mod(bits, M):
    """
    QAM modulation with unit average power.

    bits: shape (..., bits_per_sym) with values {0,1}
    M: QAM order (4, 16, 64, ...)
    return: complex symbols, shape (...)
    """
    k = int(np.log2(M))
    bits = np.asarray(bits, dtype=np.int8)
    bits = bits.reshape(-1, k)

    # Convert bits to integers 0..M-1
    ints = np.zeros(bits.shape[0], dtype=np.int64)
    for i in range(k):
        ints = (ints << 1) | bits[:, i]

    if M == 4:
        # QPSK (Gray mapping)
        # 00 -> 1+1j, 01 -> -1+1j, 11 -> -1-1j, 10 -> 1-1j
        mapping = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=complex)
        syms = mapping[ints]
    else:
        # Square QAM, Gray coding on I/Q
        m_side = int(np.sqrt(M))
        i = ints % m_side
        q = ints // m_side

        # Gray decode
        def gray_to_bin(x):
            return x ^ (x >> 1)

        i_bin = gray_to_bin(i)
        q_bin = gray_to_bin(q)

        levels = np.arange(-(m_side - 1), m_side, 2)
        re = levels[i_bin]
        im = levels[q_bin]
        syms = re + 1j * im

    # Normalize to unit average power
    syms = syms / np.sqrt(np.mean(np.abs(syms) ** 2))
    return syms.reshape(bits.shape[:-1])


def qam_demod(syms, M):
    """
    ML detection for QAM, returning bits.

    syms: (...,) complex
    M: QAM order (4, 16, 64, ...)
    return: bits, shape (..., log2(M))
    """
    k = int(np.log2(M))
    syms = np.asarray(syms, dtype=complex).reshape(-1)

    # Build full constellation
    all_bits = np.array(
        [list(np.binary_repr(i, width=k)) for i in range(M)],
        dtype=np.int8
    )
    ref_syms = qam_mod(all_bits, M).reshape(M)

    # Nearest neighbor
    d2 = np.abs(syms[:, None] - ref_syms[None, :]) ** 2
    idx = np.argmin(d2, axis=1)
    bits_hat = all_bits[idx]
    return bits_hat.reshape(*syms.shape, k)


# ============================================================
#  Compression Methods
# ============================================================

# ---------- 1) BFP (Block Floating Point) ----------

def _process_component_bfp(data, block_size, exponent_bits, mantissa_bits):
    """
    Component-wise BFP processing for 1D complex data.

    data: 1D complex
    return: reconstructed_data, metadata_bits
    """
    data = np.asarray(data, dtype=complex)
    N = data.size
    num_blocks = int(np.ceil(N / block_size))
    out = np.zeros_like(data, dtype=complex)
    metadata_bits = num_blocks * exponent_bits

    max_level = 2 ** (mantissa_bits - 1) - 1
    threshold = 2 ** (-mantissa_bits)

    eps = np.finfo(float).eps

    for b in range(num_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, N)
        block = data[start:end]

        block_max = np.max(np.abs(block))
        if block_max < eps:
            out[start:end] = 0
            continue

        block_exponent = np.ceil(np.log2(block_max + eps))

        exp_min = -2 ** (exponent_bits - 1)
        exp_max = 2 ** (exponent_bits - 1) - 1
        block_exponent = np.clip(block_exponent, exp_min, exp_max)

        scale = 2.0 ** (-block_exponent)

        scaled = block * scale
        # Quantize real and imag separately
        q_re = np.round(scaled.real * max_level) / max_level
        q_im = np.round(scaled.imag * max_level) / max_level
        quant = q_re + 1j * q_im

        # Thresholding
        mask = np.abs(quant) < threshold
        quant[mask] = 0.0

        out[start:end] = quant / scale

    return out, metadata_bits


def compress_bfp_block(data, bit_width, mod_order_tag):
    """
    BFP compression + decompression (effective) for one complex vector.

    data: 1D complex
    bit_width: total bits per complex sample for mantissa+exponent budget (conceptual)
    mod_order_tag: 2 (QPSK), 16, 64 (used for selecting block size)
    """
    if mod_order_tag == 2:        # QPSK
        block_size = 64
    elif mod_order_tag == 16:     # 16-QAM
        block_size = 32
    elif mod_order_tag == 64:     # 64-QAM
        block_size = 16
    else:
        raise ValueError("Unsupported modulation order tag for BFP.")

    exponent_bits = int(np.ceil(0.3 * bit_width))
    mantissa_bits = bit_width - exponent_bits

    recon, meta_bits = _process_component_bfp(
        data, block_size, exponent_bits, mantissa_bits
    )
    return recon, meta_bits


# ---------- 2) Block Scaling Compression ----------

def _process_block_scaling_real(data, block_size, exponent_bits, mantissa_bits):
    """
    Block scaling for 1D real data.
    """
    data = np.asarray(data, dtype=float)
    N = data.size
    num_blocks = int(np.ceil(N / block_size))
    out = np.zeros_like(data)
    metadata_bits = num_blocks * exponent_bits

    max_level = 2 ** (mantissa_bits - 1) - 1
    threshold = 2 ** (-mantissa_bits)

    eps = np.finfo(float).eps

    for b in range(num_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, N)
        block = data[start:end]

        block_max = np.max(np.abs(block))
        if block_max < eps:
            out[start:end] = 0.0
            continue

        block_exponent = np.ceil(np.log2(block_max + eps))

        exp_min = -2 ** (exponent_bits - 1)
        exp_max = 2 ** (exponent_bits - 1) - 1
        block_exponent = np.clip(block_exponent, exp_min, exp_max)

        scale = 2.0 ** (-block_exponent)

        scaled = block * scale
        quant = np.round(scaled * max_level) / max_level
        quant[np.abs(quant) < threshold] = 0.0

        out[start:end] = quant / scale

    return out, metadata_bits


def compress_bsc_block(data, bit_width, mod_order_tag):
    """
    Block Scaling compression for complex data.

    data: 1D complex
    bit_width: total bits per real dimension (conceptual)
    mod_order_tag: 2, 16, 64
    """
    block_size = 64  # fixed in the MATLAB code

    if mod_order_tag == 2:
        exponent_bits = int(np.ceil(0.2 * bit_width))
    elif mod_order_tag == 16:
        exponent_bits = int(np.ceil(0.3 * bit_width))
    elif mod_order_tag == 64:
        exponent_bits = int(np.ceil(0.4 * bit_width))
    else:
        raise ValueError("Unsupported modulation order tag for BSC.")

    mantissa_bits = bit_width - exponent_bits

    real_part = np.real(data)
    imag_part = np.imag(data)

    comp_r, meta_r = _process_block_scaling_real(
        real_part, block_size, exponent_bits, mantissa_bits
    )
    comp_i, meta_i = _process_block_scaling_real(
        imag_part, block_size, exponent_bits, mantissa_bits
    )

    recon = comp_r + 1j * comp_i
    meta_bits = meta_r + meta_i
    return recon, meta_bits


# ---------- 3) μ-Law Compression ----------

def compress_mulaw_block(data, bit_width, mod_order_tag=None, mu=200.0):
    """
    μ-law compression + quantization + decompression for complex data.

    Treat amplitude with μ-law, preserve phase.
    """
    data = np.asarray(data, dtype=complex)
    max_val = np.max(np.abs(data))
    if max_val == 0:
        return np.zeros_like(data), 0

    norm = data / max_val
    amp = np.abs(norm)
    phase = np.exp(1j * np.angle(norm))

    # μ-law companding
    comp_amp = np.log(1 + mu * amp) / np.log(1 + mu)

    # Quantization
    q_levels = 2 ** (bit_width - 2)
    q_levels = max(q_levels, 4)  # just in case
    step = 1.0 / (q_levels - 2)

    # noise multiplier (mirroring MATLAB logic)
    if bit_width == 8:
        noise_mul = 0.25
    elif bit_width == 9:
        noise_mul = 0.20
    elif bit_width == 10:
        noise_mul = 0.18
    elif bit_width == 12:
        noise_mul = 0.15
    elif bit_width == 14:
        noise_mul = 0.10
    else:
        noise_mul = 0.25

    noise = step * noise_mul * rng.standard_normal(size=comp_amp.shape)
    noisy = comp_amp + noise

    quant = np.round(noisy * (q_levels - 2)) / (q_levels - 2)
    quant = np.clip(quant, 0.0, 1.0)

    # μ-law decompanding
    decomp_amp = (1.0 / mu) * ((1 + mu) ** np.abs(quant) - 1.0)
    decomp_amp *= (1 + 0.03 * noise_mul)

    recon = phase * decomp_amp * max_val
    meta_bits = 0  # ignore metadata overhead here
    return recon, meta_bits


# ============================================================
#  Channel Generation: MIMO / Cell-Free
# ============================================================

def generate_layout_cellfree(M_AP, K_UE, cell_radius=100.0):
    """
    Random AP and UE positions in a disk, used for cell-free pathloss.
    """
    # AP positions
    angles_ap = 2 * np.pi * rng.random(M_AP)
    radii_ap = cell_radius * np.sqrt(rng.random(M_AP))
    ap_x = radii_ap * np.cos(angles_ap)
    ap_y = radii_ap * np.sin(angles_ap)

    # UE positions
    angles_ue = 2 * np.pi * rng.random(K_UE)
    radii_ue = cell_radius * np.sqrt(rng.random(K_UE))
    ue_x = radii_ue * np.cos(angles_ue)
    ue_y = radii_ue * np.sin(angles_ue)

    return (ap_x, ap_y), (ue_x, ue_y)


def generate_channel(mode, M_AP, K_UE, layout=None,
                     pathloss_exp=3.7, d0=1.0):
    """
    Generate channel matrix H (M_AP x K_UE).

    mode: "mimo" or "cellfree"
    """
    if mode == "mimo":
        # i.i.d. Rayleigh
        H = (rng.standard_normal((M_AP, K_UE)) +
             1j * rng.standard_normal((M_AP, K_UE))) / np.sqrt(2.0)
        return H
    elif mode == "cellfree":
        assert layout is not None, "Cell-free mode requires layout."
        (ap_x, ap_y), (ue_x, ue_y) = layout
        H = np.zeros((M_AP, K_UE), dtype=complex)

        for m in range(M_AP):
            for k in range(K_UE):
                dx = ap_x[m] - ue_x[k]
                dy = ap_y[m] - ue_y[k]
                d = np.sqrt(dx * dx + dy * dy)
                d = max(d, d0)
                beta = (d / d0) ** (-pathloss_exp)
                h_small = (rng.standard_normal() +
                           1j * rng.standard_normal()) / np.sqrt(2.0)
                H[m, k] = np.sqrt(beta) * h_small

        return H
    else:
        raise ValueError("mode must be 'mimo' or 'cellfree'")


# ============================================================
#  Uplink MMSE Combiner
# ============================================================

def uplink_mmse_combiner(H, noise_var):
    """
    H: (M_AP, K_UE)
    Return V: (M_AP, K_UE), whose columns are combiner vectors for each UE.
    Detection: r = V^H y
    """
    M, K = H.shape
    A = H @ H.conj().T + noise_var * np.eye(M)
    V = np.linalg.solve(A, H)  # A^{-1} H
    return V