import numpy as np
import pmt
from gnuradio import gr

class blk(gr.basic_block):
    def __init__(self, Block_Len=12, Mantissa_Bits=8, Exponent_Bits=4, Tag_Key="bfp_exp", Eps=1e-12):
        gr.basic_block.__init__(
            self,
            name="bfp_compress",
            in_sig=[np.complex64],
            out_sig=[np.int16, np.int16],
        )

        self.block_len = int(Block_Len)
        self.mantissa_bits = int(Mantissa_Bits)
        self.exponent_bits = int(Exponent_Bits)
        self.eps = float(Eps)

        if self.block_len <= 0:
            raise ValueError("Block_Len must be > 0")
        if self.mantissa_bits < 2:
            raise ValueError("Mantissa_Bits must be >= 2")
        if self.exponent_bits <= 0:
            raise ValueError("Exponent_Bits must be > 0")

        # Mantissa range (two's complement style)
        self.m_min = -(2 ** (self.mantissa_bits - 1))
        self.m_max =  (2 ** (self.mantissa_bits - 1)) - 1

        # Exponent range
        self.e_min = 0
        self.e_max = (2 ** self.exponent_bits) - 1

        self.tag_key = pmt.intern(str(Tag_Key))

        # block-aligned scheduling hint
        self.set_output_multiple(self.block_len)

        self._buf = np.empty(0, dtype=np.complex64)

    def _compute_exponent(self, block: np.ndarray) -> int:
        # max over I/Q (component-wise), consistent with typical fixed-point thinking
        max_val = float(np.max(np.maximum(np.abs(block.real), np.abs(block.imag))))
        if max_val <= self.eps:
            return 0

        # Choose exponent such that after scaling by 2^{-e}, values fit into [-m_max, m_max]
        # Using ceil ensures max_val / 2^e <= m_max
        e = int(np.ceil(np.log2(max_val / float(self.m_max))))
        if e < self.e_min: e = self.e_min
        if e > self.e_max: e = self.e_max
        return e

    @staticmethod
    def _minv_protect(x_int: np.ndarray, m_min: int) -> np.ndarray:
        # Protect the most-negative two's complement value (minV) to avoid edge overflow asymmetry
        # Map minV -> minV+1
        if x_int.size == 0:
            return x_int
        mask = (x_int == m_min)
        if np.any(mask):
            x_int = x_int.copy()
            x_int[mask] = m_min + 1
        return x_int

    def general_work(self, input_items, output_items):
        x_in = input_items[0]
        yI = output_items[0]
        yQ = output_items[1]

        # buffer input
        if x_in.size > 0:
            self._buf = np.concatenate([self._buf, x_in.astype(np.complex64, copy=False)])

        n_blocks_avail = self._buf.size // self.block_len
        if n_blocks_avail <= 0:
            # consume what we received; produce nothing yet
            self.consume(0, x_in.size)
            return 0

        max_blocks_by_out = min(yI.size // self.block_len, yQ.size // self.block_len)
        n_blocks = min(n_blocks_avail, max_blocks_by_out)
        if n_blocks <= 0:
            self.consume(0, x_in.size)
            return 0

        n_items = n_blocks * self.block_len
        buf_use = self._buf[:n_items]
        self._buf = self._buf[n_items:]

        abs_out_start = self.nitems_written(0)

        for b in range(n_blocks):
            s = b * self.block_len
            blk = buf_use[s:s + self.block_len]

            e = self._compute_exponent(blk)
            scale = 2.0 ** (-e)

            I_scaled = np.round(blk.real * scale)
            Q_scaled = np.round(blk.imag * scale)

            I_m = np.clip(I_scaled, self.m_min, self.m_max).astype(np.int16, copy=False)
            Q_m = np.clip(Q_scaled, self.m_min, self.m_max).astype(np.int16, copy=False)

            # --- minV protection (two's complement min value) ---
            I_m = self._minv_protect(I_m, self.m_min).astype(np.int16, copy=False)
            Q_m = self._minv_protect(Q_m, self.m_min).astype(np.int16, copy=False)

            yI[s:s + self.block_len] = I_m
            yQ[s:s + self.block_len] = Q_m

            # tag exponent at the start of each block
            tag_off = abs_out_start + s
            tag_val = pmt.from_long(int(e))
            self.add_item_tag(0, tag_off, self.tag_key, tag_val)
            self.add_item_tag(1, tag_off, self.tag_key, tag_val)

        # consume the new input chunk (already buffered)
        self.consume(0, x_in.size)
        return int(n_items)
