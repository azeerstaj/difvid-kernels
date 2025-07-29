## Benchmark Results
```txt
Found 3382 different elements. Showing first 5:
------------------------------------------------------------
Element 1 at index (0, 0):
  a[(0, 0)] = -0.9026293158531189
  b[(0, 0)] = -0.9026293754577637
  Difference: 5.960464477539063e-08

Element 2 at index (0, 1):
  a[(0, 1)] = -0.865967333316803
  b[(0, 1)] = -0.8659674525260925
  Difference: 1.1920928955078125e-07

Element 3 at index (0, 2):
  a[(0, 2)] = -0.8270463347434998
  b[(0, 2)] = -0.8270464539527893
  Difference: 1.1920928955078125e-07

Element 4 at index (0, 3):
  a[(0, 3)] = -0.786862313747406
  b[(0, 3)] = -0.7868626117706299
  Difference: 2.980232238769531e-07

Element 5 at index (0, 4):
  a[(0, 4)] = -0.7462092638015747
  b[(0, 4)] = -0.7462093234062195
  Difference: 5.960464477539063e-08
```

## Groundtruth Script:
```py
import torch
from infer_utils import load_diffusionvid

if __name__ == "__main__":
    model_path = "models/DiffusionVID_R101.pth"
    model = load_diffusionvid(model_path)

    batch_size = 1
    dim = 16
    t = torch.load("calib_inputs/sinu_emb_calib_input.pt").cuda()

    time = model.head.time_mlp[0](t)
    torch.save(time, "tmp.pt")
```