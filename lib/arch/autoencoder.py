from __future__ import print_function, division
from typing import Optional, List

import torch.nn as nn
import torch.nn.functional as F

from lib.arch.block import *
from lib.arch.utils import model_init
from lib.arch.decoder import build_decoder_model


def center_crop_embdding(tensor,ratio=4):
    B, C, D, H, W = tensor.shape
    # Calculate the new size for the crop
    new_D = D // ratio 
    new_H = H // ratio
    new_W = W // ratio
    
    # Calculate the center crop start indices
    start_D = (D - new_D) // 2
    start_H = (H - new_H) // 2
    start_W = (W - new_W) // 2
    
    # Perform the crop using slicing
    cropped_tensor = tensor[:, :, start_D:start_D + new_D, start_H:start_H + new_H, start_W:start_W + new_W]
    
    return cropped_tensor


class AutoEncoder3D_s(nn.Module):


    def __init__(self,
                 in_channel: int = 1,
                 out_channel: int = 1,
                 input_shape: int = [128,128,128],
                 filters: List[int] = [32,64,96,128,160],
                 pad_mode: str = 'zeros',
                 act_mode: str = 'elu',
                 norm_mode: str = 'gn',
                 init_mode: str = 'orthogonal',
                 contrastive_mode: bool = False,
                 decode_mode: bool = True,
                 **kwargs):
        super().__init__()
        self.contrastive_mode = contrastive_mode
        self.decode_mode = decode_mode

        self.min_spatio_len = int( input_shape[0] /( 2**(len(filters))) )

        self.flattened_dim = filters[-1]*(self.min_spatio_len)**3

        self.emb_dim = 256 
        self.filters = filters
        self.depth = len(filters)


        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}

        # input and output layers
        self.conv_in = conv3d_norm_act(in_channel, filters[0], kernel_size=(5,5,5),
                                       stride=(2,2,2),padding=(2,2,2), **self.shared_kwargs)
        self.conv_out = nn.ConvTranspose3d(filters[0], out_channel, kernel_size=3,stride=2, padding=1,output_padding=1)

        # encoding path
        self.down_layers = nn.ModuleList()
        for i in range(self.depth -1):
            kernel_size = 3
            stride = 2
            padding = 1
            next = min(self.depth, i+1)
            layer =  conv3d_norm_act(filters[i], filters[next], kernel_size,
                                stride=stride, padding=padding, **self.shared_kwargs)
            self.down_layers.append(layer)

        #linear projection for embdding
        self.last_encoder_conv=nn.Conv3d(self.filters[-1],self.filters[-1],kernel_size=1,stride=1)
        
        self.fc1 = nn.Linear(self.flattened_dim, self.emb_dim)        
        self.fc2= nn.Linear(self.emb_dim, self.flattened_dim)

        # decoding path
        self.up_layers = nn.ModuleList()
        for i in range(self.depth -1 ,0,-1):
            kernel_size = 3
            stride = 2
            padding = 1
            previous = i-1
            layer = transconv3d_norm_act(filters[i],filters[previous],kernel_size=(3,3,3),
                                     stride=2, padding=1, **self.shared_kwargs)
            self.up_layers.append(layer)

       
        # initialization
        model_init(self, mode=init_mode)

    def forward(self, x):
        
        #encoder path
        x = self.conv_in(x)
        for i in range(self.depth-1):
            x = self.down_layers[i](x)
        x = self.last_encoder_conv(x) 

        batch_size = x.shape[0]
        self.shape_before_flattening = x.shape[1:]

        flattened_x = x.view(batch_size,-1) 
        embbding = self.fc1(flattened_x) 

        flattened_x2=self.fc2(embbding)
        x2 = flattened_x2.view(batch_size,*self.shape_before_flattening)

        #decoder path
        for i in range( self.depth -1):
            x2 = self.up_layers[i](x2)
        x2 = self.conv_out(x2)

        return x2


  

class AutoEncoder3D(nn.Module):

    block_dict = {
        'residual': BasicBlock3d,
        'residual_pa': BasicBlock3dPA,
        #se for squeeze and extration
        'residual_se': BasicBlock3dSE,
        'residual_se_pa': BasicBlock3dPASE,
    }

    def __init__(self,
                 block_type='residual',
                 in_channel: int = 1,
                 out_channel: int = 1,
                 input_shape: int = [128,128,128],
                 filters: List[int] = [32,64,96,128,160],
                 is_isotropic: bool = False,
                 isotropy: List[bool] = [False, False, False, True, True],
                 pad_mode: str = 'zeros',
                 act_mode: str = 'elu',
                 norm_mode: str = 'gn',
                 init_mode: str = 'orthogonal',
                 pooling: bool = False,
                 blurpool: bool = False,
                 contrastive_mode: bool = False,
                 decode_mode: bool = True,
                 **kwargs):
        super().__init__()
        assert len(filters) == len(isotropy)
        self.contrastive_mode = contrastive_mode
        self.decode_mode = decode_mode

        self.min_spatio_len = int( input_shape[0] /( 2**(len(filters))) )

        self.flattened_dim = filters[-1]*(self.min_spatio_len)**3

        self.emb_dim = 1024
        self.filters = filters
        self.depth = len(filters)

        if is_isotropic:
            isotropy = [True] * self.depth
        block = self.block_dict[block_type]

        self.pooling, self.blurpool = pooling, blurpool
        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}

        # input and output layers
        kernel_size_io, padding_io = self._get_kernal_size(
            is_isotropic, io_layer=True)
        self.conv_in = conv3d_norm_act(in_channel, filters[0], kernel_size=(5,5,5),
                                       stride=(2,2,2),padding=(2,2,2), **self.shared_kwargs)
        self.conv_out = nn.ConvTranspose3d(filters[0], out_channel, kernel_size=3,stride=2, padding=1,output_padding=1)

        # encoding path
        self.down_layers = nn.ModuleList()
        for i in range(self.depth -1):
            kernel_size = 3
            stride = 2
            padding = 1
            next = min(self.depth, i+1)
            layer = nn.Sequential(
                conv3d_norm_act(filters[i], filters[next], kernel_size,
                                stride=stride, padding=padding, **self.shared_kwargs),
                BasicBlock3d(filters[next], filters[next], **self.shared_kwargs,isotropic=True))
            self.down_layers.append(layer)

        #linear projection for embdding
        self.last_encoder_conv=nn.Conv3d(self.filters[-1],self.filters[-1],kernel_size=1,stride=1)
        
        self.fc1 = nn.Linear(self.flattened_dim, self.emb_dim)        
        self.fc2= nn.Linear(self.emb_dim, self.flattened_dim)

        # decoding path
        self.up_layers = nn.ModuleList()
        for i in range(self.depth -1 ,0,-1):
            kernel_size = 3
            stride = 2
            padding = 1
            previous = i-1
            layer= nn.Sequential(
                transconv3d_norm_act(filters[i],filters[previous],kernel_size=(3,3,3),
                                     stride=2, padding=1, **self.shared_kwargs),
                BasicBlock3d(filters[previous], filters[previous], **self.shared_kwargs,isotropic= True))
            self.up_layers.append(layer)

       
        # initialization
        model_init(self, mode=init_mode)

    def forward(self, x):
        
        #encoder path
        x = self.conv_in(x)
        for i in range(self.depth-1):
            x = self.down_layers[i](x)
        x = self.last_encoder_conv(x) 

        batch_size = x.shape[0]
        self.shape_before_flattening = x.shape[1:]
        flattened_x = x.view(batch_size,-1) 
        embbding = self.fc1(flattened_x) 

        flattened_x2=self.fc2(embbding)
        x2 = flattened_x2.view(batch_size,*self.shape_before_flattening)
        for i in range( self.depth -1):
            x2 = self.up_layers[i](x2)
        x2 = self.conv_out(x2)

        return x2



    def _get_kernal_size(self, is_isotropic, io_layer=False):
        if io_layer:  # kernel and padding size of I/O layers
            if is_isotropic:
                return (5, 5, 5), (2, 2, 2)
            return (1, 5, 5), (0, 2, 2)

        if is_isotropic:
            return (3, 3, 3), (1, 1, 1)
        return (1, 3, 3), (0, 1, 1)

    def _get_stride(self, is_isotropic, previous, i):
        if self.pooling or previous == i:
            return 1

        return self._get_downsample(is_isotropic)

    def _get_downsample(self, is_isotropic):
        if not is_isotropic:
            return (1, 2, 2)
        return 2

    def _make_pooling_layer(self, is_isotropic, previous, i):
        if self.pooling and previous != i:
            kernel_size = stride = self._get_downsample(is_isotropic)
            return nn.MaxPool3d(kernel_size, stride)

        return nn.Identity()

    
MODEL_MAP = {
    'autoencoder': AutoEncoder3D,
    'autoencoder_s': AutoEncoder3D_s,
}

from box import Box
def build_autoencoder_model(cfg):
    cfg=Box(cfg)

    model_arch = cfg.MODEL.ARCHITECTURE
    assert model_arch in MODEL_MAP.keys()
    kwargs = {
        'block_type': cfg.MODEL.BLOCK_TYPE,
        'in_channel': cfg.MODEL.IN_PLANES,
        'out_channel': cfg.MODEL.OUT_PLANES,
        'filters': cfg.MODEL.FILTERS,
        'ks': cfg.MODEL.KERNEL_SIZES,
        'blocks': cfg.MODEL.BLOCKS,
        'attn': cfg.MODEL.ATTENTION,
        'is_isotropic': cfg.DATASET.IS_ISOTROPIC,
        'isotropy': cfg.MODEL.ISOTROPY,
        'pad_mode': cfg.MODEL.PAD_MODE,
        'act_mode': cfg.MODEL.ACT_MODE,
        'norm_mode': cfg.MODEL.NORM_MODE,
        'pooling': cfg.MODEL.POOLING_LAYER,
        'input_size': cfg.MODEL.INPUT_SIZE if cfg.MODEL.MORPH_INPUT_SIZE is None else cfg.MODEL.MORPH_INPUT_SIZE,
        'train_mode': cfg.MODEL.train_mode,
    }


    model = MODEL_MAP[cfg.MODEL.ARCHITECTURE](**kwargs)
    print('model: ', model.__class__.__name__)

    return model
 

