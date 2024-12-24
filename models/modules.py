import torch 
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiConvolution import MinkowskiConvolution

"""
2024-12-19 Simplify the code
"""
class IRN(torch.nn.Module):
    """
    Inception Residual Networks , Inspired by PCGCv2: https://github.com/NJUVISION/PCGCv2
    """
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.conv0_0 = MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels//2,
            kernel_size = kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        self.conv0_1 = MinkowskiConvolution(
            in_channels=channels//2,
            out_channels=channels//2,
            kernel_size = kernel_size,
            stride=1,
            bias=True,
            dimension=3)

        self.conv1_0 = MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels//2,
            kernel_size= 1,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_1 = MinkowskiConvolution(
            in_channels=channels//2,
            out_channels=channels//2,
            kernel_size = kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_2 = MinkowskiConvolution(
            in_channels=channels//2,
            out_channels=channels//2,
            kernel_size= 1,
            stride=1,
            bias=True,
            dimension=3)

        self.relu0_0 = ME.MinkowskiReLU(inplace=False)
        self.relu1_0 = ME.MinkowskiReLU(inplace=False)
        self.relu1_1 = ME.MinkowskiReLU(inplace=False)
        
    def forward(self, x):
        out0 = self.conv0_1(self.relu0_0(self.conv0_0(x)))
        out1 = self.conv1_2(self.relu1_1(self.conv1_1(self.relu1_0(self.conv1_0(x)))))
        out = ME.cat(out0, out1) + x
        return out
    

class FEL(torch.nn.Module):
    """
    Feature Extraction Layer (FEL) used to characterize and embed information from spatial neighbors within the receptive field
    Inspired by SparsePCGC: https://github.com/NJUVISION/SparsePCGC
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        
        self.conv0 = MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size = kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        
        self.relu0 = ME.MinkowskiReLU(inplace=False)
        
        self.IRNs = torch.nn.Sequential(IRN(out_channels, kernel_size),
                                        IRN(out_channels, kernel_size),
                                        IRN(out_channels, kernel_size))       
        
        self.conv1 = MinkowskiConvolution(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size = kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        
    
    def forward(self, x):
        out0 = self.relu0(self.conv0(x))
        out1 = self.IRNs(out0)
        out = self.conv1(out0+out1)
        return out


class ProbCoder(torch.nn.Module):
    
    def __init__(self,inchannels,channels,kernel_size):
        
        super().__init__()
        self.DFA1_1 = FEL(inchannels,channels,kernel_size)
        self.OPG = torch.nn.Sequential(
            MinkowskiConvolution(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                bias=True,
                dimension=3
               
            ),
            ME.MinkowskiReLU(),
            MinkowskiConvolution(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                bias=True,
                dimension=3
            ),
            ME.MinkowskiReLU(),
            MinkowskiConvolution(
                in_channels=channels,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=True,
                dimension=3
            ),
        )
        
    def forward(self, x):
        x = self.DFA1_1(x)
        x_anchor_occupied_code_prob = self.OPG(x)
        return x_anchor_occupied_code_prob

