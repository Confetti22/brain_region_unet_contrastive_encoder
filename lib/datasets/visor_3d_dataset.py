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

class T11_Dataset(Dataset):
    def __init__(self, args):
        """
        amount : control the amount of data for training
        """
        
        #filepath,trans,evalue_img=None,evalue_mode=False,amount=0.5
        data_cfg=args.__dict__['DATASET']
        self.data_cfg=args.__dict__['DATASET']
        self.input_shape=self.data_cfg['input_size']
        self.raw_img_pth=data_cfg['raw_internal_path']
        self.mask_pth=data_cfg['label_internal_path']
        self.channel=data_cfg['channel']
        self.level=data_cfg['level']
        self.lr_mask=tif.imread(self.mask_pth)
        self.zoom_factor=data_cfg['zoom_factor']

        #totoal data amount used for training
        self.amount = data_cfg.get('amount', 10)



    def __len__(self):
 
        return self.amount


    def __getitem__(self,idx) :

        """
        randomly sample a 3d image cube, get the corresponding upsample maskl
        then decide whether use this cube to train 
        criteri: 1:foreground(entropy >1.8)  2.at leat two label

        then preprocess it and transform it
        """
        #move self.ims_vol h5py objects (like those used to handle HDF5 files) cannot be pickled, which is required for torch.utils.data.DataLoader when it is configured to use multiple worker processes (num_workers > 0).
        self.ims_vol=Ims_Image(self.raw_img_pth,channel=self.channel)
        valid_roi=False
        roi=None
        mask=None
        while not valid_roi:
            roi,indexs=self.ims_vol.get_random_roi(filter=entropy_filter(thres=1.8),roi_size=self.input_shape,level=self.level)
            mask=get_hr_mask(self.lr_mask,indexs,self.input_shape,self.zoom_factor)
            #TODO should include the propotion check(each class no less than thes) here
            valid_roi= len(np.unique(mask)) >= 2

        roi = np.array(roi).astype(np.float32)  
        roi=self.preprocess(roi)
        roi=self.transform(roi)
        roi=torch.unsqueeze(roi,0)
        mask=torch.tensor(mask)

        return roi , mask 
    

    @staticmethod
    def preprocess(img,percentiles=[0.1,0.999],num_channel=1):
        """
        first clip the image to percentiles [0.1,0.999]
        second min_max normalize the image to [0,1]
        if num_channel=3, repeat the image to 3 channel
        """
        # input img nparray [0,65535]
        # output img tensor [0,1]
        flattened_arr = np.sort(img.flatten())
        clip_low = int(percentiles[0] * len(flattened_arr))
        clip_high = int(percentiles[1] * len(flattened_arr))-1
        # if flattened_arr[clip_high]<self.bg_thres:
        #     return None
        clipped_arr = np.clip(img, flattened_arr[clip_low], flattened_arr[clip_high])
        min_value = np.min(clipped_arr)
        max_value = np.max(clipped_arr)
        if max_value==min_value:
            # print(f"max_vale{max_value}=min_value{min_value}\n")
            max_value=max_value+1
        filtered = clipped_arr
        img = (filtered-min_value)/(max_value-min_value)

        img = img.astype(np.float32)
        # img = torch.from_numpy(img)
        # # img = img.unsqueeze(0).unsqueeze(0)
        # img = img.unsqueeze(0)
        # img=img.repeat(num_channel,1,1)
        return img

    @staticmethod 
    def transform(img):
        #using no augmentation at all
        trans=v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
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
