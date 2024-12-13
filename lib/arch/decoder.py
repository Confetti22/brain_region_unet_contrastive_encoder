import torch.nn as nn
from lib.arch.block import *

from lib.arch.utils import model_init

class CNNDecoder(nn.Module):
    def __init__(
            self,
            in_channel: int =128,
            out_channel: int =1,
            filters: list[int] = [64,32,16],
            is_isotropic: bool = True,
            pad_mode: str = 'zeros',
            act_mode: str = 'elu',
            norm_mode: str = 'gn',
            init_mode: str = 'orthogonal',
            output_logit: bool = True,
            ):
        super().__init__()
        self.depth = len(filters)
        self.output_logit= output_logit
        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}

        self.conv_in = transconv3d_norm_act(in_channel,filters[0],kernel_size=(5,5,5),
                                            stride=2,padding=2,**self.shared_kwargs)       
        self.conv_out= nn.Conv3d(filters[-1],out_channel,kernel_size=3,stride=1,padding=1)

 
        self.decode_layers = nn.ModuleList()
        for i in range(self.depth):
            previous = max(0 ,i-1)
            layer= nn.Sequential(
                transconv3d_norm_act(filters[previous],filters[i],kernel_size=(3,3,3),
                                     stride=2, padding=1, **self.shared_kwargs),
                BasicBlock3d(filters[i], filters[i], **self.shared_kwargs,isotropic=is_isotropic ))
            self.decode_layers.append(layer)
        
        model_init(self, mode=init_mode)
    
    def forward(self, x):
        x = self.conv_in(x)
        for i in range(self.depth):
            x=self.decode_layers[i](x)
        
        x = self.conv_out(x)

        return x

MODEL_MAP = {
    'cnn3d': CNNDecoder,
}
from box import Box
def build_decoder_model(in_channel: int =128,
            out_channel: int =1,
            filters: list[int] = [64,32,16],
            is_isotropic: bool = True,
            pad_mode: str = 'zeros',
            act_mode: str = 'elu',
            norm_mode: str = 'bn',
            init_mode: str = 'orthogonal',):
    
    kwargs ={
            'in_channel': in_channel,
            'out_channel': out_channel,
            'filters':  filters,
            'is_isotropic': is_isotropic,
            'pad_mode': pad_mode ,
            'act_mode':  act_mode,
            'norm_mode': norm_mode,
            'init_mode': init_mode,
    }

    model = CNNDecoder(**kwargs) 

    return model


        
        