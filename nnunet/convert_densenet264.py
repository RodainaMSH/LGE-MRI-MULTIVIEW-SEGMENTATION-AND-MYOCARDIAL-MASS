"""Convert PaddleClas DenseNet-264 (ImageNet-1k) -> torchvision DenseNet264 state_dict,
then SEMANTICALLY verify (ImageNet top-1 on the canonical Samoyed test image).
Output: weights/densenet264/densenet264_in1k_torch.pth (features.* + classifier.*).
"""
import sys
from pathlib import Path
import numpy as np, torch
from torchvision.models import DenseNet

ROOT = Path("/home/youssef/projects/research/CMR-MULTI")
PD = ROOT / "weights/densenet264/DenseNet264_pretrained.pdparams"
OUT = ROOT / "weights/densenet264/densenet264_in1k_torch.pth"

# 1) torchvision DenseNet-264 skeleton
tv = DenseNet(growth_rate=32, block_config=(6, 12, 64, 48), num_init_features=64, num_classes=1000)
tv_sd = tv.state_dict()
tv_keys = list(tv_sd.keys())
tv_core = [k for k in tv_keys if not k.endswith("num_batches_tracked")]

# 2) paddle params (ordered)
import paddle
pd = paddle.load(str(PD))
pd_items = [(k, np.array(v)) for k, v in pd.items()]
print(f"torch core keys: {len(tv_core)}   paddle params: {len(pd_items)}")
assert len(tv_core) == len(pd_items), f"COUNT MISMATCH {len(tv_core)} vs {len(pd_items)}"

# 3) ordered remap (transpose 2D Linear weight: paddle [in,out] -> torch [out,in])
new_sd, mism = {}, []
for k, (pk, v) in zip(tv_core, pd_items):
    t = torch.from_numpy(v.copy())
    tgt = tuple(tv_sd[k].shape)
    if t.ndim == 2 and tuple(t.shape) == tgt[::-1]:
        t = t.t().contiguous()
    if tuple(t.shape) != tgt:
        mism.append((k, pk, tuple(t.shape), tgt))
    new_sd[k] = t
for k in tv_keys:
    if k.endswith("num_batches_tracked"):
        new_sd[k] = tv_sd[k]
if mism:
    print("SHAPE MISMATCHES (order is wrong):")
    for m in mism[:10]:
        print("  ", m)
    sys.exit(1)
tv.load_state_dict(new_sd, strict=True)
print("load_state_dict(strict=True): OK  (all shapes aligned)")

# 4) SEMANTIC check — Samoyed on the canonical dog.jpg
img_p = ROOT / "weights/densenet264/dog.jpg"
if not img_p.exists():
    import urllib.request
    urllib.request.urlretrieve("https://github.com/pytorch/hub/raw/master/images/dog.jpg", str(img_p))
from PIL import Image
import torchvision.transforms as T
tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
x = tf(Image.open(img_p).convert("RGB")).unsqueeze(0)
tv.eval()
with torch.no_grad():
    p = torch.softmax(tv(x), 1)[0]
top = torch.topk(p, 5)
print("TOP-5 ImageNet:", [(int(i), round(float(v), 3)) for v, i in zip(top.values, top.indices)])
print("  (class 258 = Samoyed; ~0.7+ confirms the conversion is semantically correct)")

torch.save(new_sd, str(OUT))
print(f"SAVED -> {OUT}  ({OUT.stat().st_size/1e6:.0f} MB)")
