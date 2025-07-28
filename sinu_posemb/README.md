# Sinusodial Positional Embedding.

This kernel is a drop-in replacement for difvid.head.time_mlp[0] module.
Since it is optimized for thi specific usage here are its assumptions

### Input Assumptions:
- Dim of 1 (A vector)
- Float32
- ...

```py
import torch
from infer_utils import load_diffusionvid
torch.random.manual_seed(42)

if __name__ == "__main__":
    model_path = "models/DiffusionVID_R101.pth"
    model = load_diffusionvid(model_path)
    t = torch.randn([120])
    time = model.head.time_mlp[0](t)
    print("Time embedding for t=120:", time.shape)
```