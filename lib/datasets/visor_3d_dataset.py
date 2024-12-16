import sys
sys.path.append("/home/confetti/e5_workspace/brain_region_unet_contrastive_encoder")

from lib.datasets.read_ims import Ims_Image
import numpy as np
import tifffile as tif
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
from confettii.entropy_helper import entropy_filter
from confettii.rescale_helper import get_hr_mask
import os

class T11_Dataset(Dataset):
    def __init__(self, cfg):
        """
        amount : control the amount of data for training
        """
        
        #filepath,trans,evalue_img=None,evalue_mode=False,amount=0.5
        self.e5 = cfg.SYSTEM.e5
        self.data_cfg=cfg.DATASET
        self.input_shape=self.data_cfg.input_size
        self.data_path = self.data_cfg.data_path_dir
        self.e5_data_path = self.data_cfg.e5_data_path_dir
        self.clip_low = self.data_cfg.clip_low
        self.clip_high = self.data_cfg.clip_high
        self.is_norm = cfg.PREPROCESS.NORM
        #totoal data amount used for training

        print(f"######init visor_3d_dataset#####")
        if self.e5:
            self.files = [os.path.join(self.e5_data_path,fname) for fname in os.listdir(self.e5_data_path) if fname.endswith('.tif')] 
        else:
            self.files = [os.path.join(self.data_path,fname) for fname in os.listdir(self.data_path) if fname.endswith('.tif')] 



    def __len__(self):
 
        return len(self.files) 


    def __getitem__(self,idx) :

        """
        randomly sample a 3d image cube, get the corresponding upsample maskl
        then decide whether use this cube to train 

        then preprocess it and transform it
        """
        roi = tif.imread(self.files[idx])
        roi = np.array(roi).astype(np.float32) 

        if self.is_norm:
            roi = self.clip_norm(roi,clip_low=self.clip_low,clip_high=self.clip_high)
        else:
            roi =self.clip(roi,clip_low=self.clip_low,clip_high=self.clip_high)
        roi=self.tran2tensor(roi)
        roi=torch.unsqueeze(roi,0)

        return roi, 1
    

    @staticmethod
    def clip_norm(img,clip_low = 0 ,clip_high=3000):
        """
        first clip the image to percentiles [clip_low, clip_high]
        second min_max normalize the image to [0,1]
        """
        # input img nparray [0,65535]
        # output img tensor [0,1]
        clipped_arr = np.clip(img,clip_low,clip_high) 
        min_value = clip_low 
        max_value = clip_high 

        img = (clipped_arr-min_value)/(max_value-min_value)

        img = img.astype(np.float32)

        return img
    
    @staticmethod
    def clip(img,clip_low = 0 ,clip_high=3000):
        """
        clip the image to percentiles [clip_low, clip_high]
        """
        img = np.clip(img,clip_low,clip_high) 
        img = img.astype(np.float32)
        return img

    @staticmethod 
    def tran2tensor(img):
        #using no augmentation at all
        trans=v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale = False),
            ]
        )
        transed_img=trans(img)
        return transed_img




def get_dataset(args):

    # === Get Dataset === #
    train_dataset = T11_Dataset(args)

    return train_dataset

if __name__ =="__main__":
    import yaml
    cfg_pth="/home/confetti/e5_workspace/brain_region_unet_contrastive_encoder/config/3dunet_brain_region_contrastive_learing.yaml"
    with open(cfg_pth,"r") as file:
        cfg=yaml.safe_load(file)

    tran_dataset=get_dataset(cfg)
    print(tran_dataset.input_shape)
    print(f"success")
