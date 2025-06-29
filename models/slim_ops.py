import torch
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiConvolution import MinkowskiConvolution

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

"""
Some Dynamic Ops
"""

class sc_conv(torch.nn.Module):
    """dynamic_conv_with_space and channel adaptive compute  with_attention
    """
    def __init__(self, in_channels, out_channels, kernel_size,stride,bias,dimension):
        super().__init__()
        # self.channels = in_channels//2
        self.channel_conv = Channel_code(
            in_channels=in_channels,
            out_channels = out_channels,
            kernel_size= kernel_size,
            stride=stride,
            bias=bias,
            dimension=3)
        self.space_conv = Adaconv(
            in_channels=out_channels,
            out_channels = out_channels,
            kernel_size= kernel_size,
            stride=stride,
            bias=bias,
            dimension=3)
    def forward(self, x,th):
        x = self.channel_conv(x)
        out = self.space_conv(x,th)
        out = out + x
        return x

class scg_conv(torch.nn.Module):
    """dynamic_conv_with_space and channel adaptive compute  with_attention
    """
    def __init__(self, in_channels, out_channels, kernel_size,stride,bias,dimension):
        super().__init__()
        # self.channels = in_channels//2
        self.channel_conv = Channel_code(
            in_channels=in_channels,
            out_channels = out_channels,
            kernel_size= kernel_size,
            stride=stride,
            bias=bias,
            dimension=3)
        self.space_conv = Adaconv_g(
            in_channels=out_channels,
            out_channels = out_channels,
            kernel_size= kernel_size,
            stride=stride,
            bias=bias,
            dimension=3)
    def forward(self, x,th):
        x = self.channel_conv(x)
        out = self.space_conv(x,th)
        out = out + x
        return x

class Channel_code(torch.nn.Module):
    """dynamic_channel_conv  
    """
    def __init__(self, in_channels, out_channels, kernel_size,stride,bias,dimension):
        super().__init__()
        self.conv = MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size= kernel_size,
            stride=stride,
            bias=True,
            dimension=3)
        
    def forward(self, x):
        out = self.conv(x)
        return out

class Adaconv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride,bias,dimension):
        super().__init__()
        self.channels = in_channels
       # self.th = th/2.5
        self.dw = MinkowskiConvolution(
            in_channels=in_channels,
            out_channels = out_channels,
            kernel_size= kernel_size,
            stride=stride,
            bias=bias,
            dimension=3)
        self.union = ME.MinkowskiUnion()
    
    def mask_sparse_tensor(self, sparse_tensor, mask, C = None, tensor_stride=None):
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
    
    def forward(self, x,th):
        corrtran = MinkowskiConvolution(
            in_channels=self.channels,
            out_channels=1,
            kernel_size= 3,
            stride=1,
            bias=False,
            dimension=3)
        corrtran_one = MinkowskiConvolution(
            in_channels=self.channels,
            out_channels=1,
            kernel_size= 1,
            stride=1,
            bias=False,
            dimension=3)
        corrtran.kernel.data = torch.ones_like(corrtran.kernel.data)
        corrtran.kernel.data[13,:,:] = 0.0
        corrtran.kernel.data = corrtran.kernel.data.cuda()
        for param in corrtran.parameters():
            param.requires_grad = False
        corrtran_one.kernel.data = torch.ones_like(corrtran_one.kernel.data)
        corrtran_one.kernel.data = corrtran_one.kernel.data.cuda()
        for param in corrtran_one.parameters():
            param.requires_grad = False
      
        corr = corrtran(x)
        corr_norm = corrtran_one(x*x)
        corr = corr/corr_norm
        batch_ids = torch.unique(x.C[:, 0])
        corr_index = []
        for i in batch_ids:
            batch_y = corr.F[corr.C[:, 0] == i]
            _, indices = torch.topk(batch_y, k=int(batch_y.shape[0]*th/3.21), dim=0)
            topk_mask = torch.full_like(batch_y, fill_value=False, dtype=torch.bool)
            topk_mask[indices[:, 0], 0] = True
            corr_index.append(topk_mask)
        corr_index = torch.cat(corr_index , dim=0)
        corr_index = corr_index.squeeze()
        #corr_mask = (corr_index>self.th).squeeze()
        # print(torch.sum(corr_index)/corr_index.shape[0])
        # 计算
        strong_point = self.mask_sparse_tensor(x,corr_index)
        weak_point = self.mask_sparse_tensor(x,~corr_index)
        strong_point = self.dw(strong_point)
        weak_point = ME.SparseTensor(
            features=weak_point.F,
            coordinates=weak_point.C,
            coordinate_manager=strong_point.coordinate_manager,
            tensor_stride=strong_point.tensor_stride
        )
        all_point = self.union(strong_point,weak_point)
        all_point = ME.SparseTensor(
            features=all_point.F,
            coordinates=all_point.C,
            coordinate_manager=x.coordinate_manager,
            tensor_stride=all_point.tensor_stride
        )
        return all_point
        
class Adaconv_g(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride,bias,dimension):
        super().__init__()
        self.channels = in_channels
       # self.th = th/2.5
        self.dw = MinkowskiConvolution(
            in_channels=in_channels,
            out_channels = out_channels,
            kernel_size= kernel_size,
            stride=stride,
            bias=bias,
            dimension=3)
        self.cls = MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=2,
            kernel_size= 1,
            stride=stride,
            bias=bias,
            dimension=3)
        self.union = ME.MinkowskiUnion()
    
    def mask_sparse_tensor(self, sparse_tensor, mask, C = None, tensor_stride=None):
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
    
    def forward(self, x,th):
        out = self.cls(x)
        torch.manual_seed(42)
        out_F = torch.nn.functional.gumbel_softmax(out.F,tau=0.1, hard=True, eps=1e-10, dim=-1)
        
        strong_point_F = x.F*out_F[:,0].unsqueeze(1)
        weak_point_F = x.F*out_F[:,1].unsqueeze(1)
        strong_point = ME.SparseTensor(
            features= strong_point_F,
            coordinates=x.C,
            tensor_stride=x.tensor_stride
        )
        weak_point = ME.SparseTensor(
            features= weak_point_F ,
            coordinates=x.C,
            tensor_stride=x.tensor_stride
        )
        strong_point = self.mask_sparse_tensor(strong_point, out_F[:,0].bool())
        weak_point = self.mask_sparse_tensor(weak_point, out_F[:,1].bool())
        
        
        strong_point = self.dw(strong_point)
        weak_point = ME.SparseTensor(
            features=weak_point.F,
            coordinates=weak_point.C,
            coordinate_manager=strong_point.coordinate_manager,
            tensor_stride=strong_point.tensor_stride
        )
        all_point = self.union(strong_point,weak_point)
        # out = self.pw(all_point)
        all_point = ME.SparseTensor(
            features=all_point.F,
            coordinates=all_point.C,
            coordinate_manager=x.coordinate_manager,
            tensor_stride=all_point.tensor_stride
        )
        return all_point
        
class Dychan_conv(torch.nn.Module):
    """dynamic_channel_conv  
    """
    def __init__(self, in_channels, out_channels, kernel_size,stride,bias,dimension):
        super().__init__()
        self.channels = in_channels
        self.conv = MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size= kernel_size,
            stride=stride,
            bias=bias,
            dimension=3)
        self.cls  = MinkowskiConvolution(
            in_channels=out_channels,
            out_channels=out_channels*2,
            kernel_size= 1,
            stride=stride,
            bias=bias,
            dimension=3)
        self.avg = ME.MinkowskiGlobalAvgPooling()
        
        
    def forward(self, x):
        out = self.conv(x)
        mask = self.avg(out)
        mask = self.cls(mask)
#         torch.manual_seed(42)
        channel_cls = mask.F.reshape(mask.shape[0],2,self.channels)
        channel_cls = torch.nn.functional.gumbel_softmax(channel_cls ,tau=1.0, hard=True, eps=1e-10, dim=1)
        channel_cls = channel_cls[:,0,:].squeeze(1)
#         channel_cls = ME.MinkowskiFunctional.sigmoid(mask)
        batch_ids = torch.unique(out.C[:, 0])
        new_y = []
        for i in batch_ids:
            x_i = x.F[x.C[:, 0] == i]
            batch_y = out.F[out.C[:, 0] == i]
            new_batch_y = batch_y * channel_cls[i]
            x_mean = x_i.mean(dim=1, keepdim=True)
            new_batch_y[:,channel_cls[i]==0.0] = x_mean
            new_y.append(new_batch_y)
        new_F = torch.cat(new_y , dim=0)
        out = ME.SparseTensor(
            features=new_F, 
            coordinates=out.C, 
            tensor_stride = out.tensor_stride, 
            device=out.device)
        return out
    
if __name__ == '__main__':
    import numpy as np
    origin_pc1 = 100 * np.random.uniform(0, 1, (10, 3))
    feat1 = np.ones((10, 3), dtype=np.float32)
    origin_pc1 = torch.from_numpy(origin_pc1)
    feat1 = torch.from_numpy(feat1)
    zeros = torch.zeros(origin_pc1.size(0), 1)
    origin_pc1 = torch.cat((zeros, origin_pc1), dim=1)
    x = ME.SparseTensor(feat1, coordinates=origin_pc1)

