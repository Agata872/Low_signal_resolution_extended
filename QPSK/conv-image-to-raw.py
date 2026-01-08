import numpy as np
from PIL import Image
import os

H = W = 1024
BLOCK_LEN = 12

img = Image.open("shannon.png").convert("L").resize((W, H))
pix = np.array(img, dtype=np.uint8)

# 可逆映射到 int16 域（Q=0）
I16 = ((pix.astype(np.int32) - 128) << 8).astype(np.int16)

# 转成 float IQ（你现有压缩脚本读 complex64）
I = I16.astype(np.float32) / 32768.0
x = (I.flatten() + 1j*np.zeros(I.size, dtype=np.float32)).astype(np.complex64)

# pad 到 BLOCK_LEN 对齐（关键！）
need = H*W
pad = (-need) % BLOCK_LEN
if pad:
    x = np.pad(x, (0, pad), mode="constant", constant_values=0.0 + 0.0j)

os.makedirs("bin", exist_ok=True)
x.tofile("tx/pre-comp.bin")

print("samples:", x.size, "pad:", pad)
