import yaml
from torchsummary import summary

cfg_pth="/home/confetti/e5_workspace/brain_region_unet_contrastive_encoder/config/brain_region_unet.yaml"
with open(cfg_pth,"r") as file:
    cfg=yaml.safe_load(file)


from lib.arch.autoencoder import get_model
model=get_model(cfg['model']).to('cuda')
print(f"model established")

summary(model,(1,128,128,128))
print(f"finished")