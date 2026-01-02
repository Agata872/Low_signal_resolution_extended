import numpy as np
# -----------------------------
# Compression: BFP (theory-aligned)
# -----------------------------
def _bfp_exponent_from_block(block_iq: np.ndarray) -> float:
    """
    Compute shared exponent e for a block using:
        e = ceil(log2(max(|x|)))
    so that max(|x| * 2^{-e}) <= 1, avoiding overflow without extra clip.
    """
    block_max = np.max(np.abs(block_iq))
    if block_max <= np.finfo(float).eps:
        return -np.inf  # mark as all-zero block
    return float(np.ceil(np.log2(block_max)))


def _process_component_bfp(
    data: np.ndarray,
    blockSize: int,
    exponentBitWidth: int,
    mantissaBitWidth: int
):
    """
    Theory-aligned BFP:
    - Per block (PRB), compute shared exponent e from max magnitude of complex samples.
    - Normalize by 2^{-e}.
    - Uniformly quantize I and Q separately to mantissaBitWidth signed integers.
    - Reconstruct by multiplying by 2^{e}.

    Notes:
    - No extra thresholding (not part of standard BFP definition).
    - If exponent is clipped due to limited exponentBitWidth, we saturate normalized values to [-1,1)
      to match a bounded uniform quantizer and avoid overflow.
    """
    x = np.asarray(data, dtype=np.complex128).reshape(-1)
    n = x.size
    numBlocks = int(np.ceil(n / blockSize))

    out = np.zeros_like(x, dtype=np.complex128)

    # exponent range if exponent is stored as signed integer with exponentBitWidth
    e_min = - (2 ** (exponentBitWidth - 1))
    e_max = (2 ** (exponentBitWidth - 1)) - 1

    # mantissa quantizer range (signed)
    max_int = (1 << (mantissaBitWidth - 1)) - 1  # e.g. 7 bits -> 63
    min_int = -max_int - 1

    metadataBits = numBlocks * exponentBitWidth

    for blk in range(numBlocks):
        s = blk * blockSize
        e = min((blk + 1) * blockSize, n)
        block = x[s:e]

        exp_val = _bfp_exponent_from_block(block)
        if not np.isfinite(exp_val):
            out[s:e] = 0
            continue

        # Clip exponent to representable range (what will be transmitted)
        exp_tx = float(np.clip(exp_val, e_min, e_max))

        # Normalize: x_norm = x * 2^{-e}
        scale = 2.0 ** (-exp_tx)
        x_norm = block * scale

        # If exponent was clipped smaller than needed, x_norm may exceed 1.
        # A bounded uniform quantizer must saturate.
        i_norm = np.clip(x_norm.real, -1.0, 1.0 - np.finfo(float).eps)
        q_norm = np.clip(x_norm.imag, -1.0, 1.0 - np.finfo(float).eps)

        # Uniform quantization in [-1,1): map to signed integers then back to float
        i_int = np.round(i_norm * max_int).astype(np.int64)
        q_int = np.round(q_norm * max_int).astype(np.int64)
        i_int = np.clip(i_int, min_int, max_int)
        q_int = np.clip(q_int, min_int, max_int)

        i_q = (i_int.astype(np.float64) / max_int)
        q_q = (q_int.astype(np.float64) / max_int)

        # Reconstruct: x_hat = (i_q + j q_q) * 2^{e}
        out[s:e] = (i_q + 1j * q_q) / scale

    return out, int(metadataBits)


def bfp_compression(data: np.ndarray, bitWidth: int, modOrder: int, blockSize: int = 12):
    exponentBitWidth = 4            # 论文/O-RAN常见固定4bit
    mantissaBitWidth = bitWidth     # bitWidth就是mantissa宽度
    compressed, metadataBits = _process_component_bfp(
        data=data,
        blockSize=blockSize,
        exponentBitWidth=exponentBitWidth,
        mantissaBitWidth=mantissaBitWidth
    )

    # 原始：float32 I + float32 Q = 64 bits/complex
    originalBitsPerComplex = 64
    originalSize = data.size * originalBitsPerComplex

    # 压缩payload：I/Q各mantissaBitWidth
    compressedPayloadBitsPerComplex = 2 * mantissaBitWidth

    # metadata：每block exponentBitWidth（假设I/Q共享同一个exponent）
    compressedSize = compressed.size * compressedPayloadBitsPerComplex + metadataBits

    CR = originalSize / compressedSize
    return compressed, float(CR)


# -----------------------------
# Compression: Block Scaling
# -----------------------------
def _float_to_int_fs(x: np.ndarray, iq_in_bitwidth: int, fs: float) -> np.ndarray:
    """
    Map float to signed integer range using a full-scale fs (peak amplitude).
    No hard clipping to [-1,1) except final integer saturation.
    """
    x = np.asarray(x, dtype=np.float64)
    max_int = (1 << (iq_in_bitwidth - 1)) - 1

    if fs <= np.finfo(float).eps:
        return np.zeros_like(x, dtype=np.int64)

    x_scaled = x / fs  # now roughly in [-1,1] but not forced
    x_int = np.round(x_scaled * max_int).astype(np.int64)
    x_int = np.clip(x_int, -max_int - 1, max_int)
    return x_int


def _int_to_float_fs(x_int: np.ndarray, iq_in_bitwidth: int, fs: float) -> np.ndarray:
    """Map signed integers back to float using the same full-scale fs."""
    max_int = (1 << (iq_in_bitwidth - 1)) - 1
    return (np.asarray(x_int, dtype=np.float64) / max_int) * fs


def _process_block_scaling_bsc(
    data_real: np.ndarray,
    data_imag: np.ndarray | None,
    blockSize: int,
    wiq_out: int,
    iq_in_bitwidth: int = 16,
    fs: float = 1.0,
):
    r = np.asarray(data_real).reshape(-1)
    i = None if data_imag is None else np.asarray(data_imag).reshape(-1)

    n = r.size
    numBlocks = int(np.ceil(n / blockSize))
    metadataBits = numBlocks * 8  # 1 scaler per PRB (shared I/Q)

    # float -> int with full-scale (no [-1,1) clipping)
    r_int = _float_to_int_fs(r, iq_in_bitwidth, fs)
    i_int = None if i is None else _float_to_int_fs(i, iq_in_bitwidth, fs)

    q_r = np.zeros(n, dtype=np.int64)
    q_i = None if i_int is None else np.zeros(n, dtype=np.int64)

    shift = max(iq_in_bitwidth - wiq_out, 0)
    qmax = (1 << (wiq_out - 1)) - 1
    qmin = -(1 << (wiq_out - 1))

    denom = 1 << max(iq_in_bitwidth - 8, 0)  # 2^(WIQIN-8)

    # store scaler per block so decompression uses the *same* scaler
    scaler_list = np.ones(numBlocks, dtype=np.int64)

    for blk in range(numBlocks):
        s = blk * blockSize
        e = min((blk + 1) * blockSize, n)

        br = r_int[s:e]
        if i_int is None:
            maxValue = int(np.max(np.abs(br)))
        else:
            bi = i_int[s:e]
            maxValue = int(max(np.max(np.abs(br)), np.max(np.abs(bi))))

        if maxValue == 0:
            scaler_list[blk] = 1
            q_r[s:e] = 0
            if q_i is not None:
                q_i[s:e] = 0
            continue

        scaler = int(np.ceil(maxValue / denom))
        scaler = int(np.clip(scaler, 1, 255))
        scaler_list[blk] = scaler

        # inv_q17 ~ (128/scaler) in Q1.7 => store as integer with extra *128
        inv_q17 = int(np.round(16384 / scaler))  # = round((128/scaler)*128)

        # tmp ≈ x*(128/scaler)
        tmp_r = (br * inv_q17) >> 7
        if q_i is not None:
            tmp_i = (bi * inv_q17) >> 7

        # divide by 2^(WIQIN-WIQOUT) to fit WIQout
        if shift > 0:
            rnd = 1 << (shift - 1)
            tmp_r = (tmp_r + np.sign(tmp_r) * rnd) >> shift
            if q_i is not None:
                tmp_i = (tmp_i + np.sign(tmp_i) * rnd) >> shift

        q_r[s:e] = np.clip(tmp_r, qmin, qmax).astype(np.int64)
        if q_i is not None:
            q_i[s:e] = np.clip(tmp_i, qmin, qmax).astype(np.int64)

    # -------- Decompression (MUST divide by 128 i.e. >>7) --------
    xhat_r_int = np.zeros(n, dtype=np.int64)
    xhat_i_int = None if q_i is None else np.zeros(n, dtype=np.int64)

    for blk in range(numBlocks):
        s = blk * blockSize
        e = min((blk + 1) * blockSize, n)
        scaler = int(scaler_list[blk])

        qr = q_r[s:e].astype(np.int64)
        qi = None if q_i is None else q_i[s:e].astype(np.int64)

        if shift > 0:
            qr = qr << shift
            if qi is not None:
                qi = qi << shift

        # recon ≈ (q * 2^shift * scaler) / 128  => >>7
        # rounding for >>7
        rnd7 = 1 << 6
        xhat_r_int[s:e] = (qr * scaler + np.sign(qr) * rnd7) >> 7
        if qi is not None:
            xhat_i_int[s:e] = (qi * scaler + np.sign(qi) * rnd7) >> 7

    out_r = _int_to_float_fs(xhat_r_int, iq_in_bitwidth, fs)
    out_i = None if xhat_i_int is None else _int_to_float_fs(xhat_i_int, iq_in_bitwidth, fs)

    return out_r, out_i, int(metadataBits)


def bsc_compression(
    data: np.ndarray,
    bitWidth: int,
    modOrder: int,
    blockSize: int = 12,
    iq_in_bitwidth: int = 16,
    originalBitWidth: int = 32,
    fs: float | None = None
):
    data = np.asarray(data)
    isComplex = np.iscomplexobj(data)

    if fs is None:
        # fallback: still allow running, but not recommended for thesis alignment
        fs = float(np.max(np.abs(data))) + np.finfo(float).eps
    if isComplex:
        r = np.real(data)
        i = np.imag(data)
        out_r, out_i, metadataBits = _process_block_scaling_bsc(
            r, i, blockSize, wiq_out=bitWidth, iq_in_bitwidth=iq_in_bitwidth, fs=fs
        )
        compressed = out_r + 1j * out_i

        numSamples = data.size
        compressedBits = (numSamples * 2 * bitWidth) + metadataBits
        originalBits = numSamples * 2 * originalBitWidth
    else:
        out_r, _, metadataBits = _process_block_scaling_bsc(
            data, None, blockSize, wiq_out=bitWidth, iq_in_bitwidth=iq_in_bitwidth, fs=fs
        )
        compressed = out_r.astype(np.float64)

        numSamples = data.size
        compressedBits = (numSamples * bitWidth) + metadataBits
        originalBits = numSamples * originalBitWidth

    CR = (originalBits / compressedBits) if compressedBits > 0 else 1.0
    return compressed.astype(np.complex128 if isComplex else np.float64), float(CR)



# -----------------------------
# Compression: μ-law
# -----------------------------
def mu_law_compression(
    data: np.ndarray,
    bitWidth: int,
    _modOrder_unused: int,
    blockSize: int = 12,
    shiftBitWidth: int = 4
):
    """
    Theory-aligned µ-law compression (O-RAN style, per thesis Section 3.2.3):
      1) Per block (PRB), compute a shared power-of-two shift (4-bit metadata).
      2) Apply shift to normalize samples near [-1, 1).
      3) Apply µ-law companding (µ=8) per I and Q component:
           F(y)=sgn(y)*ln(1+µ|y|)/ln(1+µ)
      4) Uniformly quantize F(y) with bitWidth (signed) bits (per component).
      5) Dequantize and apply inverse µ-law:
           y_hat = sgn(Fq)*((1+µ)^{|Fq|}-1)/µ
      6) Undo the block shift to reconstruct.

    Returns:
      compressed (reconstructed complex IQ, float),
      CR (compression ratio, with metadata bits counted).
    """

    mu = 8.0  # per thesis/O-RAN setting
    x = np.asarray(data, dtype=np.complex128).reshape(-1)
    n = x.size
    if n == 0:
        return x.copy(), 1.0

    numBlocks = int(np.ceil(n / blockSize))

    # shift stored as signed integer with shiftBitWidth bits (consistent with exponent-like metadata)
    shift_min = -(2 ** (shiftBitWidth - 1))
    shift_max = (2 ** (shiftBitWidth - 1)) - 1

    # signed uniform quantizer in [-1, 1) with bitWidth bits (per component)
    max_int = (1 << (bitWidth - 1)) - 1
    min_int = -max_int - 1

    out = np.zeros_like(x, dtype=np.complex128)

    metadataBits = numBlocks * shiftBitWidth  # per block 4-bit shift metadata

    ln_denom = np.log(1.0 + mu)

    for blk in range(numBlocks):
        s = blk * blockSize
        e = min((blk + 1) * blockSize, n)
        block = x[s:e]

        # --- block-level shift to normalize near [-1,1) ---
        # Use max over I and Q magnitudes (component-wise) to guarantee both fit.
        blockMax = max(np.max(np.abs(block.real)), np.max(np.abs(block.imag)))
        if blockMax <= np.finfo(float).eps:
            out[s:e] = 0
            continue

        # Choose shift so that blockMax * 2^shift <= 1  (i.e., shift = -ceil(log2(blockMax)))
        shift_val = float(-np.ceil(np.log2(blockMax)))
        shift_tx = float(np.clip(shift_val, shift_min, shift_max))

        scale = 2.0 ** (shift_tx)
        i_norm = block.real * scale
        q_norm = block.imag * scale

        # Saturate to quantizer input range [-1, 1)
        i_norm = np.clip(i_norm, -1.0, 1.0 - np.finfo(float).eps)
        q_norm = np.clip(q_norm, -1.0, 1.0 - np.finfo(float).eps)

        # --- µ-law companding per component (Eq. 11) ---
        def mu_compand(y):
            ay = np.abs(y)
            sy = np.sign(y)
            return sy * (np.log(1.0 + mu * ay) / ln_denom)

        i_c = mu_compand(i_norm)
        q_c = mu_compand(q_norm)

        # --- uniform quantization of companded values Q(F(y)) ---
        i_int = np.round(i_c * max_int).astype(np.int64)
        q_int = np.round(q_c * max_int).astype(np.int64)
        i_int = np.clip(i_int, min_int, max_int)
        q_int = np.clip(q_int, min_int, max_int)

        i_q = i_int.astype(np.float64) / max_int
        q_q = q_int.astype(np.float64) / max_int

        # --- inverse µ-law expansion (Eq. 12) ---
        def mu_expand(fq):
            af = np.abs(fq)
            sf = np.sign(fq)
            return sf * (((1.0 + mu) ** af - 1.0) / mu)

        i_hat_norm = mu_expand(i_q)
        q_hat_norm = mu_expand(q_q)

        # --- undo shift ---
        inv_scale = 2.0 ** (-shift_tx)  # since we multiplied by 2^shift_tx
        out[s:e] = (i_hat_norm + 1j * q_hat_norm) * inv_scale

    # ---- CR accounting ----
    # Original: float32 I + float32 Q => 64 bits per complex sample
    originalBitsPerComplex = 64
    originalSizeBits = x.size * originalBitsPerComplex

    # Compressed payload: per complex, I uses bitWidth and Q uses bitWidth
    compressedPayloadBitsPerComplex = 2 * bitWidth
    compressedSizeBits = out.size * compressedPayloadBitsPerComplex + metadataBits

    CR = originalSizeBits / compressedSizeBits
    return out.astype(np.complex128), float(CR)



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