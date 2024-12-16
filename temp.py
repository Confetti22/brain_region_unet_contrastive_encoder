#%%

from torch.nn import functional as F
import torch
x=torch.randn(size=(2,96,16,16,16))
features=torch.randn(size=(2,32,32,32,32))
size=features.shape[2:]
size=tuple([32,32,32])
x=F.interpolate(x, size,mode='nearest')
print(x.shape)
print(f"finished")
# %%
for i in range(1,5):
    print(i)

# %%
from torch.nn import Conv2d, ConvTranspose2d
import torch

transconv1=ConvTranspose2d(in_channels=1,out_channels=1,stride=2,kernel_size=3,padding=1,output_padding=1)
with torch.no_grad():
    transconv1.weight.fill_(1)
    transconv1.bias.fill_(-9)

input=torch.tensor([[1.0,2.0],[3.0,4.0]])
input = input.unsqueeze(0)
print(input.shape)
output = transconv1(input)
print(f"output{output}")
print(f"output_shape {output.shape}")




# %%
