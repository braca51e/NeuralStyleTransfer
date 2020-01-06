from __future__ import print_function
import torch

x = torch.rand(6, 6)
print(x)
print("\nGPU TEST:" + str(torch.cuda.is_available()))
