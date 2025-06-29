import torch
import MinkowskiEngine as ME
from models.entropy_model import EntropyBottleneck
from MinkowskiEngine.MinkowskiConvolution import MinkowskiConvolution, MinkowskiGenerativeConvolutionTranspose
from data_utils import isin, sort_spare_tensor
from models.slim_ops import sc_conv,scg_conv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Downsample(torch.nn.Module):
    def __init__(self,  inchannels,channels,kernel_size, *args):
        super().__init__()
        self.conv1 = MinkowskiConvolution(
            in_channels=inchannels,
            out_channels=channels,
            kernel_size= kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        self.conv2 = MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        self.down = MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.block = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.down(x)
        out = self.block(x)+x
        out = self.conv2(out)
        
        return out



class slne_encoder(torch.nn.Module):
    def __init__(self, inchannels,channels,kernel_size):
        super().__init__()
        self.dfa1 = DFA(inchannels,channels,kernel_size)
        self.dfa2 = DFA_Slim(channels,channels,kernel_size)
        self.dfa3 = DFA(channels,channels,kernel_size)
        self.down1 = MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3)
        self.down2 = MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3)
    def forward(self,x,th):
        x = self.dfa1(x)
        x = self.down1(x)
        x = self.dfa2(x,th)
        x = self.down2(x)
        x = self.dfa3(x)
        return x



class slne_encoder2(torch.nn.Module):
    def __init__(self, inchannels,channels,kernel_size):
        super().__init__()
        self.dfa1 = DFA(inchannels,channels,kernel_size)
        self.dfa2 = DFA_Slim_g(channels,channels,kernel_size)
        self.dfa3 = DFA(channels,channels,kernel_size)
        self.down1 = MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3)
        self.down2 = MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3)
    def forward(self,x,th):
        x = self.dfa1(x)
        x = self.down1(x)
        x = self.dfa2(x,th)
        x = self.down2(x)
        x = self.dfa3(x)
        return x

class slne_decoder(torch.nn.Module):
    def __init__(self, inchannels,channels,kernel_size):
        super().__init__()
        self.dfa1 = DFA_Slim(inchannels,channels,kernel_size)
        
        self.up1 = MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3)
        self.dfa2 = MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
    def forward(self,x,th):
        x = self.up1(x)
        x = self.dfa1(x,th)
        x = self.dfa2(x)
        return x
        
class slne_decoder2(torch.nn.Module):
    def __init__(self, inchannels,channels,kernel_size):
        super().__init__()
        self.dfa1 = DFA_Slim_g(inchannels,channels,kernel_size)
        
        self.up1 = MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3)
        self.dfa2 = MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
    def forward(self,x,th):
        x = self.up1(x)
        x = self.dfa1(x,th)
        x = self.dfa2(x)
        return x

class bm(torch.nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.fc1 = torch.nn.Linear(1,channels)
        self.fc2 = torch.nn.Linear(channels,channels)
        self.relu = torch.nn.ReLU()
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = torch.exp(x)
        return x
    
class SLNE_Factorized_SingleRate(torch.nn.Module):
    def __init__(self, inchannels,channels,kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        
        self.encoder =  torch.nn.Sequential(
            DFA(inchannels,channels,self.kernel_size),
            MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3),
            DFA(channels,channels,self.kernel_size),
            MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3),
            DFA(channels,channels,self.kernel_size)
            )
        self.decoder = torch.nn.Sequential(
            MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3),
            DFA(channels,channels,kernel_size),
            MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        )
        self.cls = MinkowskiConvolution(
            in_channels=channels,
            out_channels=1,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        
        self.entropy_bottleneck = EntropyBottleneck(channels)
        self.pruning = ME.MinkowskiPruning()

    def get_likelihood(self, data,quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F, 
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood

    def forward(self, x,pov ,th,training=True):
        x_down = self.encoder(x)
        x_down_hat, likelihood = self.get_likelihood(x_down, 
           quantize_mode="noise" if training else "symbols")
        x_down_hat = self.decoder(x_down_hat)
        x_cls = self.cls(x_down_hat)
        x_down_hat = sort_spare_tensor(x_down_hat)
        pov = sort_spare_tensor(pov)
        mask_true = isin(x_down_hat.C, pov.C)
        x_down_hat = self.pruning(x_down_hat, mask_true.to(x_down_hat.device))

        return x_down_hat,likelihood,x_cls
    






class SLNE_Factorized_MultiRate(torch.nn.Module):
    def __init__(self, inchannels,channels,kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.bm_module = bm(channels)
        self.encoder =  torch.nn.Sequential(
            DFA(inchannels,channels,self.kernel_size),
            MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3),
            DFA(channels,channels,self.kernel_size),
            MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3),
            DFA(channels,channels,self.kernel_size)
            )
        self.decoder = torch.nn.Sequential(
            MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3),
            DFA(channels,channels,kernel_size),
            MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        )
        self.cls = MinkowskiConvolution(
            in_channels=channels,
            out_channels=1,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        
        self.entropy_bottleneck = EntropyBottleneck(channels)
        self.pruning = ME.MinkowskiPruning()

    def get_likelihood(self, data,quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F, 
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood

    def forward(self, x,pov ,th,training=True):
        x_down = self.encoder(x)
        
        th = torch.tensor([th],device = x.device).float()
        bm_data = self.bm_module(th)
        x_down_F = x_down.F*bm_data
        x_down = ME.SparseTensor(
            features=x_down_F, 
            coordinate_map_key=x_down.coordinate_map_key, 
            coordinate_manager=x_down.coordinate_manager, 
            device=x_down.device)
        x_down_hat, likelihood = self.get_likelihood(x_down, 
           quantize_mode="noise" if training else "symbols")
        x_down_hat_F = x_down_hat.F/bm_data
        x_down_hat = ME.SparseTensor(
            features=x_down_hat_F, 
            coordinate_map_key=x_down_hat.coordinate_map_key, 
            coordinate_manager=x_down_hat.coordinate_manager, 
            device=x_down_hat.device)

        x_down_hat = self.decoder(x_down_hat)
        
        x_down_hat = sort_spare_tensor(x_down_hat)
        x_cls = self.cls(x_down_hat)
        pov = sort_spare_tensor(pov)
        mask_true = isin(x_down_hat.C, pov.C)
        x_down_hat = mask_sparse_tensor(x_down_hat,mask_true)

        return x_down_hat,likelihood,x_cls



class SLNE_Factorized_MultiRate_Slim(torch.nn.Module):
    def __init__(self, inchannels,channels,kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.bm_module = bm(channels)
        self.encoder =  slne_encoder(inchannels,channels,kernel_size)
        self.decoder = slne_decoder(channels,channels,kernel_size)
        self.cls = MinkowskiConvolution(
            in_channels=channels,
            out_channels=1,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        
        self.entropy_bottleneck = EntropyBottleneck(channels)
        self.pruning = ME.MinkowskiPruning()

    def get_likelihood(self, data,quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F, 
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood

    def forward(self, x,pov ,th,training=True):
        x_down = self.encoder(x,th)
        
        th = torch.tensor([th],device = x.device).float()
        bm_data = self.bm_module(th)
        x_down_F = x_down.F*bm_data
        x_down = ME.SparseTensor(
            features=x_down_F, 
            coordinate_map_key=x_down.coordinate_map_key, 
            coordinate_manager=x_down.coordinate_manager, 
            device=x_down.device)
        x_down_hat, likelihood = self.get_likelihood(x_down, 
           quantize_mode="noise" if training else "symbols")
        x_down_hat_F = x_down_hat.F/bm_data
        x_down_hat = ME.SparseTensor(
            features=x_down_hat_F, 
            coordinate_map_key=x_down_hat.coordinate_map_key, 
            coordinate_manager=x_down_hat.coordinate_manager, 
            device=x_down_hat.device)

        x_down_hat = self.decoder(x_down_hat,th)
        
        x_down_hat = sort_spare_tensor(x_down_hat)
        x_cls = self.cls(x_down_hat)
        pov = sort_spare_tensor(pov)
        mask_true = isin(x_down_hat.C, pov.C)
        x_down_hat = mask_sparse_tensor(x_down_hat,mask_true)

        return x_down_hat,likelihood,x_cls


class SLNE_Factorized_MultiRate_Slim_g(torch.nn.Module):
    def __init__(self, inchannels,channels,kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.bm_module = bm(channels)
        self.encoder =  slne_encoder2(inchannels,channels,kernel_size)
        self.decoder = slne_decoder2(channels,channels,kernel_size)
        self.cls = MinkowskiConvolution(
            in_channels=channels,
            out_channels=1,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        
        self.entropy_bottleneck = EntropyBottleneck(channels)
        self.pruning = ME.MinkowskiPruning()

    def get_likelihood(self, data,quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F, 
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood

    def forward(self, x,pov ,th,training=True):
        x_down = self.encoder(x,th)
        
        th = torch.tensor([th],device = x.device).float()
        bm_data = self.bm_module(th)
        x_down_F = x_down.F*bm_data
        x_down = ME.SparseTensor(
            features=x_down_F, 
            coordinate_map_key=x_down.coordinate_map_key, 
            coordinate_manager=x_down.coordinate_manager, 
            device=x_down.device)
        x_down_hat, likelihood = self.get_likelihood(x_down, 
           quantize_mode="noise" if training else "symbols")
        x_down_hat_F = x_down_hat.F/bm_data
        x_down_hat = ME.SparseTensor(
            features=x_down_hat_F, 
            coordinate_map_key=x_down_hat.coordinate_map_key, 
            coordinate_manager=x_down_hat.coordinate_manager, 
            device=x_down_hat.device)

        x_down_hat = self.decoder(x_down_hat,th)
        
        x_down_hat = sort_spare_tensor(x_down_hat)
        x_cls = self.cls(x_down_hat)
        pov = sort_spare_tensor(pov)
        mask_true = isin(x_down_hat.C, pov.C)
        x_down_hat = mask_sparse_tensor(x_down_hat,mask_true)

        return x_down_hat,likelihood,x_cls

    
   

class one_SOPA(torch.nn.Module):
    def __init__(self, inchannels,channels,kernel_size, *args):
        super().__init__()
        self.dfa1 = DFA(inchannels,channels,kernel_size)
        self.dfa2 = DFA(channels,channels,kernel_size)
        self.vsl = MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3)
        self.ool = OOL(channels)
    def forward(self, x,th,training):
        x = self.dfa1(x)
        x = self.vsl(x)
        x = self.dfa2(x)
        likelihood = self.ool(x,training) #likelihood

        return x, likelihood # x为32channel，likelihood为1channel

class one_SOPA_Slim(torch.nn.Module):
    def __init__(self, inchannels,channels,kernel_size, *args):
        super().__init__()
        self.dfa1 = DFA(inchannels,channels,kernel_size)
        self.dfa2 = DFA_Slim(channels,channels,kernel_size)
        self.vsl = MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3)
        self.ool = OOL(channels)
    def forward(self, x,th,training):
        x = self.dfa1(x)
        x = self.vsl(x)
        if(th<0.01):
            th = 0.01
        x = self.dfa2(x,th)
        likelihood = self.ool(x,training) #likelihood

        return x, likelihood # x为32channel，likelihood为1channel
        
class one_SOPA_Slim_g(torch.nn.Module):
    def __init__(self, inchannels,channels,kernel_size, *args):
        super().__init__()
        self.dfa1 = DFA(inchannels,channels,kernel_size)
        self.dfa2 = DFA_Slim_g(channels,channels,kernel_size)
        self.vsl = MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3)
        self.ool = OOL(channels)
    def forward(self, x,th,training):
        x = self.dfa1(x)
        x = self.vsl(x)
        if(th<0.01):
            th = 0.01
        x = self.dfa2(x,th)
        likelihood = self.ool(x,training) #likelihood

        return x, likelihood # x为32channel，likelihood为1channel
        

class DFA_Slim(torch.nn.Module):
    def __init__(self,  inchannels,channels,kernel_size, *args):
        super().__init__()
        self.conv1 = MinkowskiConvolution(
            in_channels=inchannels,
            out_channels=channels,
            kernel_size= kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        self.conv2 = sc_conv(
            in_channels=channels,
            out_channels=channels,
            kernel_size= kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.block = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels)
        self.conv3 = MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= kernel_size,
            stride=1,
            bias=True,
            dimension=3)
    def forward(self, x,th):
        x = self.relu(self.conv1(x))
        out = self.block(x)+x
        out = self.conv2(out,th)
        out = self.conv3(out)
        return out

class DFA_Slim_g(torch.nn.Module):
    def __init__(self,  inchannels,channels,kernel_size, *args):
        super().__init__()
        self.conv1 = MinkowskiConvolution(
            in_channels=inchannels,
            out_channels=channels,
            kernel_size= kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        self.conv2 = scg_conv(
            in_channels=channels,
            out_channels=channels,
            kernel_size= kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.block = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels)
        self.conv3 = MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= kernel_size,
            stride=1,
            bias=True,
            dimension=3)
    def forward(self, x,th):
        x = self.relu(self.conv1(x))
        out = self.block(x)+x
        out = self.conv2(out,th)
        out = self.conv3(out)
        return out


class DFA(torch.nn.Module):
    def __init__(self,  inchannels,channels,kernel_size, *args):
        super().__init__()
        self.conv1 = MinkowskiConvolution(
            in_channels=inchannels,
            out_channels=channels,
            kernel_size= kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        self.conv2 = MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.block = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        out = self.block(x)+x
        out = self.conv2(out)
        return out

    
class OOL(torch.nn.Module):
    """
    概率估计
    """
    def __init__(self, channels):
        super().__init__()
        self.conv0 = MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 1,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1 = MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 1,
            stride=1,
            bias=True,
            dimension=3)
        self.conv2 = MinkowskiConvolution(
            in_channels=channels,
            out_channels=1,
            kernel_size= 1,
            stride=1,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)
        
    def forward(self, x,is_train=True):
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        out = self.conv2(x)
        #out = self.sigmoid(out)
        return out

class InceptionResNet(torch.nn.Module):
    """Inception Residual Network
    """
    
    def __init__(self, channels,kernel_size):
        super().__init__()
        self.conv0_0 = MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels//2,
            kernel_size= kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        self.conv0_1 = MinkowskiConvolution(
            in_channels=channels//2,
            out_channels=channels//2,
            kernel_size= kernel_size,
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
            kernel_size= kernel_size,
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

        self.relu = ME.MinkowskiReLU(inplace=True)
        
    def forward(self, x):
        out0 = self.conv0_1(self.relu(self.conv0_0(x)))
        out1 = self.conv1_2(self.relu(self.conv1_1(self.relu(self.conv1_0(x)))))
        out = ME.cat(out0, out1) + x

        return out

def make_layer(block, block_layers, channels):
    """make stacked InceptionResNet layers.
    """
    layers = []
    for i in range(block_layers):
        layers.append(block(channels=channels,kernel_size=3))
        
    return torch.nn.Sequential(*layers)


def mask_sparse_tensor(sparse_tensor, mask, C = None, tensor_stride=None):
        if isinstance(sparse_tensor, torch.Tensor)  :
            assert C is not None
            F = sparse_tensor[mask]
            newC = C 
            return ME.SparseTensor(
            features=F, coordinates=C, tensor_stride=tensor_stride
            )
        else:
            newC = C if C is not None else sparse_tensor.C[mask] 
            newF = sparse_tensor.F[mask]
            new_tensor_stride = tensor_stride if tensor_stride is not None else sparse_tensor.tensor_stride
            return ME.SparseTensor(
                features=newF, coordinates=newC, tensor_stride=new_tensor_stride
            )


