#%%
from lib.arch.unet import build_unet_model
from config import load_cfg
import torch
import argparse
import sys
import numpy as np
from torchsummary import summary


activation = {}
def get_activation(name):
    def hook(model, input, output):
        #check for whether registered at last layer of classifier
        activation[name] = output.detach()
    return hook




cfg = load_cfg("config/brain_region_unet.yaml")

device = 'cuda'

model = build_unet_model(cfg)
model.to(device)
model.eval()

summary(model,(1,128,128,128))
print(model)


test_input = torch.zeros(size=(1,1,128,128,128))
test_input[:,:,64,64,64] = 1
test_input = test_input.to(device)

#register forward hook at the destination layer
extract_layer_name ='conv_in'
hook1 = model.down_layers[-1][2].act.register_forward_hook(get_activation(extract_layer_name))

out = model(test_input)
#check the shape of feats acquired by forward hook
feats = activation[extract_layer_name].cpu().detach().numpy()
feats=np.squeeze(feats)

#visulize the activation map via summary operation like std,mean
feats_std=np.std(feats,axis=0)
print(f"shape of feats is {feats.shape}")

feats_2d_midz=feats_std[int(feats_std.shape[0]//2),:,:]
feats_2d_midy=feats_std[:,int(feats_std.shape[0]//2),:]
feats_2d_midx=feats_std[:,:,int(feats_std.shape[0]//2)]

import matplotlib.pyplot as plt
plt.figure(figsize=(27,9))

# fig,axes=plt.subplots(1,3)
# axes[0].imshow(feats_2d_midz)
# axes[0].set_title('midz')

# axes[1].imshow(feats_2d_midy)
# axes[1].set_title('midy')

# axes[2].imshow(feats_2d_midx)
# axes[2].set_title('midx')
plt.imshow(feats_2d_midx)

hook1.remove()






# %%

# %%
