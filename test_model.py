import yaml
from torchsummary import summary

cfg_pth="/home/confetti/e5_workspace/brain_region_unet_contrastive_encoder/config/3dunet_brain_region_contrastive_learing.yaml"
with open(cfg_pth,"r") as file:
    cfg=yaml.safe_load(file)


from lib.arch.unet import get_model
model=get_model(cfg['model']).to('cuda')
print(f"model established")

summary(model,(1,64,64,64))
print(f"finished")