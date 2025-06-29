import torch
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiConvolution import MinkowskiConvolution
from models.coder_module import SLNE_Factorized_SingleRate,SLNE_Factorized_MultiRate,SLNE_Factorized_MultiRate_Slim,SLNE_Factorized_MultiRate_Slim_g
from models.coder_module import one_SOPA,one_SOPA_Slim,one_SOPA_Slim_g
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Official Implementation for vrcpcgc [AAAI 2025]
"""

class pcgcv3_sopa_slne_stage1(torch.nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.channels = channels
        self.kernel_size = 3
        self.one_sopa_slne = one_SOPA(self.channels,self.channels,self.kernel_size)
        self.slne = SLNE_Factorized_SingleRate(1,self.channels,self.kernel_size)

    def forward(self, x, th,training=True):
        x1 = downsample(x)
        x1 = ME.SparseTensor(
            features=torch.ones(x1.C.shape[0],1,device = x.device),
            coordinates=x1.C,
            coordinate_manager=x1.coordinate_manager,
            tensor_stride=x1.tensor_stride
        )
        slne_dec,likelihood,x_cls = self.slne(x,x1,th,training)
        x_dec,probs = self.one_sopa_slne(slne_dec,th,training)
        out_cls = [x_cls,probs]
        truth = [x1,x]
        return out_cls[0:],truth[0:],likelihood



class pcgcv3_sopa_slne(torch.nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.channels = channels
        self.kernel_size = 3
        self.one_sopa_slne = one_SOPA(self.channels,self.channels,self.kernel_size)
        self.slne = SLNE_Factorized_SingleRate(1,self.channels,self.kernel_size)

    def forward(self, x, th,training=True):
        x1 = downsample(x)
        x1 = ME.SparseTensor(
            features=torch.ones(x1.C.shape[0],1,device = x.device),
            coordinates=x1.C,
            coordinate_manager=x1.coordinate_manager,
            tensor_stride=x1.tensor_stride
        )
        slne_dec,likelihood,x_cls = self.slne(x,x1,th,training)
        x_dec,probs = self.one_sopa_slne(slne_dec,th,training)
        out_cls = [x_cls,probs]
        truth = [x1,x]
        return out_cls[1:],truth[1:],likelihood


class pcgcv3_sopa_slne_multi_stage1(torch.nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.channels = channels
        self.kernel_size = 3
        self.one_sopa_slne = one_SOPA(self.channels,self.channels,self.kernel_size)
        self.slne = SLNE_Factorized_MultiRate(1,self.channels,self.kernel_size)
    
    def forward(self, x, th,training=True):
        x1 = downsample(x)
        x1 = ME.SparseTensor(
            features=torch.ones(x1.C.shape[0],1,device = x.device),
            coordinates=x1.C,
            coordinate_manager=x1.coordinate_manager,
            tensor_stride=x1.tensor_stride
        )
        slne_dec,likelihood,x_cls = self.slne(x,x1,th,training)
        x_dec,probs = self.one_sopa_slne(slne_dec,th,training)
        out_cls = [x_cls,probs]
        truth = [x1,x]
        return out_cls[0:],truth[0:],likelihood


class pcgcv3_sopa_slne_multi(torch.nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.channels = channels
        self.kernel_size = 3
        self.one_sopa_slne = one_SOPA(self.channels,self.channels,self.kernel_size)
        self.slne = SLNE_Factorized_MultiRate(1,self.channels,self.kernel_size)
    
    def forward(self, x, th,training=True):
        x1 = downsample(x)
        x1 = ME.SparseTensor(
            features=torch.ones(x1.C.shape[0],1,device = x.device),
            coordinates=x1.C,
            coordinate_manager=x1.coordinate_manager,
            tensor_stride=x1.tensor_stride
        )
        slne_dec,likelihood,x_cls = self.slne(x,x1,th,training)
        x_dec,probs = self.one_sopa_slne(slne_dec,th,training)
        out_cls = [x_cls,probs]
        truth = [x1,x]
        return out_cls[1:],truth[1:],likelihood


class pcgcv3_sopa_slne_multi2_stage1(torch.nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.channels = channels
        self.kernel_size = 3
        self.one_sopa_slne = one_SOPA_Slim(self.channels,self.channels,self.kernel_size)
        self.slne = SLNE_Factorized_MultiRate_Slim(1,self.channels,self.kernel_size)
        

    def forward(self, x, th,training=True):
        x1 = downsample(x)
        x1 = ME.SparseTensor(
            features=torch.ones(x1.C.shape[0],1,device = x.device),
            coordinates=x1.C,
            coordinate_manager=x1.coordinate_manager,
            tensor_stride=x1.tensor_stride
        )
        slne_dec,likelihood,x_cls = self.slne(x,x1,th,training)
        x_dec,probs = self.one_sopa_slne(slne_dec,th,training)
        out_cls = [x_cls,probs]
        truth = [x1,x]
        return out_cls[0:],truth[0:],likelihood


class pcgcv3_sopa_slne_multi2(torch.nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.channels = channels
        self.kernel_size = 3
        self.one_sopa_slne = one_SOPA_Slim(self.channels,self.channels,self.kernel_size)
        self.slne = SLNE_Factorized_MultiRate_Slim(1,self.channels,self.kernel_size)
        

    def forward(self, x, th,training=True):
        x1 = downsample(x)
        x1 = ME.SparseTensor(
            features=torch.ones(x1.C.shape[0],1,device = x.device),
            coordinates=x1.C,
            coordinate_manager=x1.coordinate_manager,
            tensor_stride=x1.tensor_stride
        )
        slne_dec,likelihood,x_cls = self.slne(x,x1,th,training)
        x_dec,probs = self.one_sopa_slne(slne_dec,th,training)
        out_cls = [x_cls,probs]
        truth = [x1,x]
        return out_cls[1:],truth[1:],likelihood


downsample = MinkowskiConvolution(
            in_channels=1,
            out_channels=1,
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3).to(device)
downsample.kernel.requires_grad = False
