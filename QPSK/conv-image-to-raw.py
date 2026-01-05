from PIL import Image
import numpy as np

H = W = 512

# 1) 读图 + 灰度 + resize
img = Image.open("shannon.png").convert("L").resize((W, H))

# 2) 转成连续 bytes（row-major）
payload = np.array(img, dtype=np.uint8).flatten()

# 3) 加同步头（32 bytes）
preamble = bytes([0xDE, 0xAD, 0xBE, 0xEF]) * 8

# 4) 写出：preamble + payload
with open("tx_raw.bin", "wb") as f:
    f.write(preamble)
    f.write(payload.tobytes())

print("tx.bin bytes =", len(preamble) + payload.size)
