import os
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))

tx = np.fromfile(os.path.join(base_dir, "tx.bin"), dtype=np.uint8)
rx = np.fromfile(os.path.join(base_dir, "rx.bin"), dtype=np.uint8)

min_len = min(len(tx), len(rx))
tx = tx[:min_len]
rx = rx[:min_len]

tx_bits = np.unpackbits(tx)
rx_bits = np.unpackbits(rx)

ber = np.mean(tx_bits != rx_bits)
print("BER =", ber)
