import torch
import MinkowskiEngine as ME
import numpy as np
from MinkowskiEngine.MinkowskiConvolution import MinkowskiConvolution
from typing import cast
from models.coder_module import one_SOPA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
Official Implementation for OLC [AAAI 2025]
"""
class pcgcv3_sopa(torch.nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.channels = channels
        self.kernel_size = 3
        self.one_sopa = one_SOPA(1,self.channels,self.kernel_size)
    
    def forward(self, x, th,training=True):
        x1 = downsample(x)
        x1 = ME.SparseTensor(
            features=torch.ones(x1.C.shape[0],1,device = x.device),
            coordinates=x1.C,
            coordinate_manager=x1.coordinate_manager,
            tensor_stride=x1.tensor_stride
        )
        x_dec,probs = self.one_sopa(x1,th,training)
        return probs,x

downsample = MinkowskiConvolution(
            in_channels=1,
            out_channels=1,
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3).to(device)
downsample.kernel.requires_grad = False