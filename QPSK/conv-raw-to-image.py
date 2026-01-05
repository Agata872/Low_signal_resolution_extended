import numpy as np
from PIL import Image

H = W = 512
preamble = bytes([0xDE, 0xAD, 0xBE, 0xEF]) * 8

data = open("rx.bin", "rb").read()
pos = data.find(preamble)
if pos == -1:
    raise RuntimeError("没找到 preamble：可能接收端没收到完整同步头，或你TX没发带preamble的文件")

start = pos + len(preamble)
img_bytes = data[start:start + H*W]

# 不足就 pad，保证能 reshape
if len(img_bytes) < H*W:
    img_bytes += bytes([0]) * (H*W - len(img_bytes))

img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(H, W)
Image.fromarray(img).save("rx.png")
print("saved rx.png")
