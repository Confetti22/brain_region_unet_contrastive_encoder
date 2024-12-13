from config import load_cfg
from lib.datasets.visor_3d_dataset import get_dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import tifffile as tif

cfg = load_cfg ('config/brain_region_unet.yaml')
dataset = get_dataset(cfg)
batch_size =4
loader = DataLoader(dataset,batch_size,shuffle=False)
save_dir = 'out/recon_img'
os.makedirs(save_dir,exist_ok=True)
clip_low = cfg.DATASET.clip_low
clip_high = cfg.DATASET.clip_high


#B*C*D*H*W
for idx, normed in enumerate(loader):

    resotred = normed * (clip_high - clip_low) + clip_low

    num = normed.shape[0]
    for id in range(num):
        y = normed[id]
        y = y.numpy()
        y = np.squeeze(y)

        
        re_x = resotred[id]
        re_x = re_x.numpy()
        re_x= np.squeeze(re_x)

        re_x_name = f"{idx*batch_size + id:04d}_re_x.tif"
        y_name = f"{idx*batch_size + id:04d}_y.tif"

        tif.imwrite(os.path.join(save_dir,re_x_name) , re_x)
        tif.imwrite(os.path.join(save_dir,y_name) , y)
    exit(0)


    







