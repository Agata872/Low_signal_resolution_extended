import numpy as np
from PIL import Image

H = W = 1024
INFILE = "mulaw/rx_wiqout12.bin"  # 或 bfp/...
OUTPNG = "rx.png"

x = np.fromfile(INFILE, dtype=np.complex64)
I = x.real.astype(np.float32)

need = H*W
I = I[:need]  # 只取原始长度，忽略 padding

# float -> int16 域
I16_hat = np.rint(I * 32768.0).astype(np.int32)
I16_hat = np.clip(I16_hat, -32768, 32767).astype(np.int32)

# 逆变换回像素（与编码严格匹配）
pix_hat = (I16_hat >> 8) + 128
pix_hat = np.clip(pix_hat, 0, 255).astype(np.uint8)

img = pix_hat.reshape(H, W)
Image.fromarray(img, mode="L").save(OUTPNG)
print("[OK] saved", OUTPNG)
