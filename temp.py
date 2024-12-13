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
for i in range(5,0,-1):
    print(i)

# %%
