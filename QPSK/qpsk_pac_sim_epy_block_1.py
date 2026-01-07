import numpy as np
import pmt
from gnuradio import gr

class blk(gr.basic_block):
    def __init__(self, Block_Len=12, Tag_Key="bfp_exp", Default_Exp=0):
        gr.basic_block.__init__(
            self,
            name="bfp_decompress",
            in_sig=[np.int16, np.int16],
            out_sig=[np.complex64],
        )

        self.block_len = int(Block_Len)
        if self.block_len <= 0:
            raise ValueError("Block_Len must be > 0")

        self.tag_key = pmt.intern(str(Tag_Key))
        self.default_exp = int(Default_Exp)

        self.set_output_multiple(self.block_len)

        self._buf_I = np.empty(0, dtype=np.int16)
        self._buf_Q = np.empty(0, dtype=np.int16)

        # Absolute input offset corresponding to _buf_I[0] / _buf_Q[0]
        # This is CRITICAL to read the correct exponent tags after buffering.
        self._abs_buf0 = 0
        self._buf0_valid = False

    def _read_exp(self, abs_in_offset: int) -> int:
        # Read tag at this absolute input offset (port 0 is sufficient)
        tags = self.get_tags_in_range(0, abs_in_offset, abs_in_offset + 1, self.tag_key)
        if not tags:
            return self.default_exp

        t = tags[-1]
        try:
            return int(pmt.to_long(t.value))
        except Exception:
            return self.default_exp

    def general_work(self, input_items, output_items):
        I_in = input_items[0]
        Q_in = input_items[1]
        y = output_items[0]

        # If buffer is empty and we are about to append new samples,
        # latch the absolute input offset that corresponds to the first buffered sample.
        # IMPORTANT: do this BEFORE consume() changes nitems_read().
        if (not self._buf0_valid) and (I_in.size > 0 or Q_in.size > 0):
            self._abs_buf0 = self.nitems_read(0)
            self._buf0_valid = True

        # Append incoming to internal buffers
        if I_in.size > 0:
            self._buf_I = np.concatenate([self._buf_I, I_in.astype(np.int16, copy=False)])
        if Q_in.size > 0:
            self._buf_Q = np.concatenate([self._buf_Q, Q_in.astype(np.int16, copy=False)])

        n_avail = min(self._buf_I.size, self._buf_Q.size)
        if n_avail < self.block_len:
            # We already buffered input, so we can consume all of it.
            self.consume(0, I_in.size)
            self.consume(1, Q_in.size)
            return 0

        n_blocks_avail = n_avail // self.block_len
        max_blocks_by_out = y.size // self.block_len
        n_blocks = min(n_blocks_avail, max_blocks_by_out)
        if n_blocks <= 0:
            self.consume(0, I_in.size)
            self.consume(1, Q_in.size)
            return 0

        n_items = n_blocks * self.block_len

        # The absolute offset of the start of the first block in our buffer
        abs_block0 = self._abs_buf0

        for b in range(n_blocks):
            s = b * self.block_len

            # Correct absolute offset for this buffered block start
            abs_off = abs_block0 + s

            e = self._read_exp(abs_off)
            scale = 2.0 ** (e)

            I_blk = self._buf_I[s:s + self.block_len].astype(np.float32) * scale
            Q_blk = self._buf_Q[s:s + self.block_len].astype(np.float32) * scale
            y[s:s + self.block_len] = (I_blk + 1j * Q_blk).astype(np.complex64)

        # Drop consumed-from-buffer samples
        self._buf_I = self._buf_I[n_items:]
        self._buf_Q = self._buf_Q[n_items:]

        # Advance the absolute offset to match the new buffer head
        self._abs_buf0 += n_items
        if self._buf_I.size == 0 or self._buf_Q.size == 0:
            # buffer empty (or one side empty): mark invalid so next append relatches abs offset
            self._buf0_valid = False

        # Consume ALL newly received samples (we buffered them already)
        self.consume(0, I_in.size)
        self.consume(1, Q_in.size)
        return int(n_items)
