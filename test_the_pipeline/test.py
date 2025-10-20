import torch
path = "/home/ika/yzlm/Reid_Inference_Pipeline/test_the_pipeline/reid_ltcc.pth"
with open(path, "rb") as f:
    header = f.read(20)
print(header)

ckpt = torch.load(path, map_location="cpu", weights_only=False)
print(type(ckpt))

import torch
ckpt = torch.load(path, map_location="cpu", weights_only=False)

for k, v in ckpt.items():
    print(k, type(v))
