from __future__ import print_function, division
from typing import Optional, List

import torch.nn as nn
import torch.nn.functional as F

from lib.arch.block import *
from lib.arch.utils import model_init


class AutoEncoder3D(nn.Module):

    block_dict = {
        'single': SingleConv3d,
        'double': DoubleConv3d, 
        'residual':BasicBlock3d,
    }

    def __init__(self,
                 in_channel: int = 1,
                 out_channel: int = 1,
                 input_shape: int = [128,128,128],
                 filters: List[int] = [32,64,96,128,160],
                 pad_mode: str = 'reflect',
                 act_mode: str = 'elu',
                 norm_mode: str = 'gn',
                 init_mode: str = 'orthogonal',
                 block_type: str = 'single',
                 upsample_interp :bool = True,
                 pooling: bool = False,
                 **kwargs):
        super().__init__()
        self.pooling  =pooling
        self.upsample_interp = upsample_interp

        self.min_spatio_len = int( input_shape[0] /( 2**(len(filters))) )

        self.flattened_dim = filters[-1]*(self.min_spatio_len)**3

        self.emb_dim = 1024 
        self.filters = filters
        self.depth = len(filters)


        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}

        # input and output layers
        self.conv_in = conv3d_norm_act(in_channel, filters[0], kernel_size=(5,5,5),
                                       stride=(2,2,2),padding=(2,2,2), **self.shared_kwargs)
        if self.upsample_interp:
            self.conv_out = nn.Conv3d(filters[0], out_channel, kernel_size=3,stride=1, padding=1)
        else:
            self.conv_out = nn.ConvTranspose3d(filters[0], out_channel, kernel_size=3,stride=2, padding=1,output_padding=1)

        # encoding path
        self.down_layers = nn.ModuleList()
        for i in range(self.depth -1):
            kernel_size = 3
            stride = 2
            padding = 1
            next = min(self.depth, i+1)

            if block_type == 'single':
                self.down_layers.append(
                  nn.Sequential(
                      conv3d_norm_act(filters[i],filters[next],kernel_size=kernel_size,stride = stride, padding=padding,**self.shared_kwargs),
                      conv3d_norm_act(filters[next],filters[next],kernel_size=kernel_size,stride = 1, padding=padding,**self.shared_kwargs)
                     )
                )

            elif block_type == 'double':
                self.down_layers.append(
                    nn.Sequential(
                        conv3d_norm_act(filters[i],filters[next],kernel_size=kernel_size,stride = stride, padding=padding,**self.shared_kwargs),
                        conv3d_norm_act(filters[next],filters[next],kernel_size=kernel_size,stride = 1, padding=padding,**self.shared_kwargs),
                        conv3d_norm_act(filters[next],filters[next],kernel_size=kernel_size,stride = 1, padding=padding,**self.shared_kwargs)
                         )
                )
            elif block_type == 'residual' :
                self.down_layers.append(
                    nn.Sequential(
                        conv3d_norm_act(filters[i],filters[next],kernel_size=kernel_size,stride = stride, padding=padding,**self.shared_kwargs),
                        ResidualBlock3d(filters[next],filters[next],kernel_size=kernel_size,stride = 1, padding=padding,**self.shared_kwargs)
                          )
                )
            else:
                self.down_layers.append(
                    nn.Sequential(
                        conv3d_norm_act(filters[i],filters[next],kernel_size=kernel_size,stride = stride, padding=padding,**self.shared_kwargs),
                          )
                )




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
            if self.upsample_interp:
                stride = 1
                trans = False 
            else:
                stride = 2
                trans = True

            if block_type == 'single':
                self.up_layers.append(
                  nn.Sequential(
                      conv3d_norm_act(filters[i],filters[previous],kernel_size=kernel_size,stride = stride, padding=padding,trans=trans,**self.shared_kwargs),
                      conv3d_norm_act(filters[previous],filters[previous],kernel_size=kernel_size,stride = 1, padding=padding,**self.shared_kwargs)
                     )
                )

            elif block_type == 'double':
                self.up_layers.append(
                    nn.Sequential(
                        conv3d_norm_act(filters[i],filters[previous],kernel_size=kernel_size,stride = stride, padding=padding,trans=trans,**self.shared_kwargs),
                        conv3d_norm_act(filters[previous],filters[previous],kernel_size=kernel_size,stride = 1, padding=padding,**self.shared_kwargs),
                        conv3d_norm_act(filters[previous],filters[previous],kernel_size=kernel_size,stride = 1, padding=padding,**self.shared_kwargs)
                         )
                )
            elif block_type == 'residual':
                self.up_layers.append(
                    nn.Sequential(
                        conv3d_norm_act(filters[i],filters[previous],kernel_size=kernel_size,stride = stride, padding=padding,trans=trans,**self.shared_kwargs),
                        ResidualBlock3d(filters[previous],filters[previous],kernel_size=kernel_size,stride = 1, padding=padding,**self.shared_kwargs)
                          )
                )
            else:
                self.up_layers.append(
                    nn.Sequential(
                        conv3d_norm_act(filters[i],filters[previous],kernel_size=kernel_size,stride = stride, padding=padding,trans=trans,**self.shared_kwargs),
                          )
                )


     
       
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
            if self.upsample_interp:
                align_corners = False if self.pooling else True
                x2 = F.interpolate(x2, scale_factor=(2,2,2),mode='trilinear',align_corners=align_corners)
                x2 = self.up_layers[i](x2)
            else:
                x2 = self.up_layers[i](x2)
        
        if self.upsample_interp:
            align_corners = False if self.pooling else True
            x2 = F.interpolate(x2, scale_factor=(2,2,2),mode='trilinear',align_corners=align_corners)
            x2 = self.conv_out(x2)
        else:
            x2 = self.conv_out(x2)

        return x2



    
MODEL_MAP = {
    'autoencoder': AutoEncoder3D,
}

def build_autoencoder_model(cfg):

    model_arch = cfg.MODEL.ARCHITECTURE
    assert model_arch in MODEL_MAP.keys()
    kwargs = {
        'block_type': cfg.MODEL.BLOCK_TYPE,
        'in_channel': cfg.MODEL.IN_PLANES,
        'out_channel': cfg.MODEL.OUT_PLANES,
        'filters': cfg.MODEL.FILTERS,
        'blocks': cfg.MODEL.BLOCKS,
        'pad_mode': cfg.MODEL.PAD_MODE,
        'act_mode': cfg.MODEL.ACT_MODE,
        'norm_mode': cfg.MODEL.NORM_MODE,
        'input_size': cfg.MODEL.INPUT_SIZE if cfg.MODEL.MORPH_INPUT_SIZE is None else cfg.MODEL.MORPH_INPUT_SIZE,
        'upsample_interp':cfg.MODEL.UPSAMPLE_INTERP
    }


    model = MODEL_MAP[cfg.MODEL.ARCHITECTURE](**kwargs)
    print('model: ', model.__class__.__name__)

    return model
 

