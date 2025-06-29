
import os, time
import numpy as np
import torch
import MinkowskiEngine as ME
from entropy.io import write_body, read_body
from data_utils import array2vector, istopk, sort_spare_tensor, load_sparse_tensor
from data_utils import write_ply_ascii_geo, read_ply_ascii_geo
from MinkowskiEngine.MinkowskiConvolution import MinkowskiConvolution, MinkowskiGenerativeConvolutionTranspose
from gpcc import gpcc_encode, gpcc_decode
from pc_error import pc_error
from data_utils import isin, istopk
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# from entropy.utils import draw_y

downsample = MinkowskiConvolution(
            in_channels=1,
            out_channels=1,
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3).to(device)
class CoordinateCoder():
    """encode/decode coordinates using gpcc
    """
    def __init__(self, filename):
        self.filename = filename
        self.ply_filename = filename + '.ply'

    def encode(self, coords, postfix=''):
        coords = coords.numpy().astype('int')
        write_ply_ascii_geo(filedir=self.ply_filename, coords=coords) #没有处理，仅仅是把坐标作为ply文件写入
        gpcc_encode(self.ply_filename, self.filename+postfix+'_C.bin') #用gpcc工具转成bin
        os.system('rm '+self.ply_filename)
        
        return 

    def decode(self, postfix=''):
        gpcc_decode(self.filename+postfix+'_C.bin', self.ply_filename) #从bin回到ply，这不是重点
        coords = read_ply_ascii_geo(self.ply_filename) #读ply文件，没有处理
        os.system('rm '+self.ply_filename)
        
        return coords


class FeatureCoder():
    """encode/decode feature using learned entropy model
    """
    def __init__(self, filename, rate_reduction_module):
        self.filename = filename
        self.rate_reduction_module = rate_reduction_module# .cpu()


    def encode(self, feats, postfix=''):
        # self.checkerboard.gaussian_conditional.update()
        # print(feats)
        res_strings, res_shape = self.rate_reduction_module.compress(feats)
        index = ['y', 'z']
        for idx_str in index:
            # 求anchor
            strings = res_strings[idx_str]
            assert isinstance(strings, list)
            # if 'y' == idx_str : assert len(strings) == 2
            if 'z' == idx_str : assert len(strings) == 1 # because 
            shape = res_shape[idx_str] 
            print(idx_str)
            with open(self.filename+postfix+"_"+idx_str+'_F.bin', 'wb') as fout:
                bits = write_body(fout, shape, strings)
            # print(bits)
        return 
    
   

    def decode(self,tensor_stride_y, postfix='', y_C=None, z_C=None ):

        index = ['y', 'z']
        strings = {}
        shapes = {}
        for idx_str in index:
            with open(self.filename+postfix+"_"+idx_str+'_F.bin', 'rb') as fin:
                lstrings, shape = read_body(fin)
                strings[idx_str] = lstrings
                shapes[idx_str] = shape

        feats = self.rate_reduction_module.decompress(strings, shapes, y_C, z_C,tensor_stride_y)
        
        return feats


class Coder():
    def __init__(self,sopa,sopa_slne,filename):
        self.sopa = sopa 
        self.sopa_slne = sopa_slne
        self.filename = filename
        self.coordinate_coder = CoordinateCoder(filename)
        self.feature_coder = FeatureCoder(self.filename, self.sopa_slne.slne.rate_reduction_module)
        self.pruning = ME.MinkowskiPruning()
        self.feature_coder.rate_reduction_module.entropy_bottleneck.to("cpu")
        self.feature_coder.rate_reduction_module.gaussian_conditional.to("cpu")

    @torch.no_grad()
    def prune_voxel(self, data, data_cls, nums, ground_truth, training):
        
        mask_topk = istopk(data_cls, nums)
        if training: 
            assert not ground_truth is None
            mask_true = isin(data_cls.C, ground_truth.C) #判断data_cls.C的数是否在ground_truth.C中出现过
            mask =  mask_topk + mask_true
        else: 
            mask = mask_topk
        data_pruned = self.pruning(data, mask.to(data.device))

        return data_pruned

    @torch.no_grad()
    def encode(self,x,th,scale,if_slne,postfix=''):
        # Encoder
        # scale->下采样的次数,scale=2/3/4
        print(x.shape)
        x1 = downsample(x)
        ground_truth_list = [x]
        x1 = ME.SparseTensor(
            features=torch.ones(x1.C.shape[0],1,device = x.device,requires_grad = False),
            coordinates=x1.C,
            coordinate_manager=x1.coordinate_manager,
            tensor_stride=x1.tensor_stride)
        return_x = x1
        if(scale>=3):
            x2 = downsample(x1)
            x2 = ME.SparseTensor(
            features=torch.ones(x2.C.shape[0],1,device = x.device,requires_grad = False),
            coordinates=x2.C,
            coordinate_manager=x2.coordinate_manager,
            tensor_stride=x2.tensor_stride)
            ground_truth_list = [x1,x]
            return_x = x2
        if(scale>=4):
            x3 = downsample(x2)
            x3 = ME.SparseTensor(
            features=torch.ones(x3.C.shape[0],1,device = x.device,requires_grad = False),
            coordinates=x3.C,
            coordinate_manager=x3.coordinate_manager,
            tensor_stride=x3.tensor_stride)
            ground_truth_list = [x2,x1,x]
            return_x = x3
        #ground_truth_list = reversed(ground_truth_list)
        #print(ground_truth_list)
        num_points = [[len(C) for C in ground_truth.decomposed_coordinates] \
            for ground_truth in ground_truth_list]
        print(num_points)
        with open(self.filename+postfix+'_num_points.bin', 'wb') as f:
            f.write(np.array(num_points, dtype=np.int32).tobytes())
            
        #return_zC = downsample(return_x)
        th = torch.tensor([th],device = x.device).float()
        if(if_slne):
            x_enc = self.sopa_slne.slne.encoder(ground_truth_list[0])
            x_enc = sort_spare_tensor(x_enc)
            # 主编码器得到x_enc
            z = self.sopa_slne.slne.rate_reduction_module.hyper_encoder(x_enc)
            z = sort_spare_tensor(z)
            bts = self.feature_coder.encode(x_enc, postfix)
            #self.feature_coder.encode(x_enc.F, postfix=postfix)
            # self.coordinate_coder.encode((x_enc.C//x_enc.tensor_stride[0]).detach().cpu()[:,1:], postfix=postfix+"_y") #y.tensor_stride[0]=8 y.C.shape[13849, 4]
            # self.coordinate_coder.encode((z.C//z.tensor_stride[0]).detach().cpu()[:,1:], postfix=postfix+"_z") #y.tensor_stride[0]=8 y.C.shape[13849, 4]
        
            return x_enc,return_x
        else:
            return return_x,return_x

    @torch.no_grad()
    def decode(self,pov,th,scale,if_slne,rho=1, postfix=''):
        training = False
        
        with open(self.filename+postfix+'_num_points.bin', 'rb') as fin:
                num_points = np.frombuffer(fin.read(4*3), dtype=np.int32).tolist()
                num_points[-1] = int(rho * num_points[-1])# update
                num_points = [[num] for num in num_points]
        if(scale==2):
            if(if_slne):
                
                pov_down = downsample(pov)
                pov_down = sort_spare_tensor(pov_down)
                # y_C = self.coordinate_coder.decode(postfix=postfix+"_y") #返回ply的coords
                # y_C = torch.cat((torch.zeros((len(y_C),1)).int(), torch.tensor(y_C).int()), dim=-1)
                # indices_sort = np.argsort(array2vector(y_C, y_C.max()+1)) #y_C.shape: torch.Size([13849, 4]) y_C[0]: tensor([ 0, 35,  1, 25], dtype=torch.int32)
                # y_C = y_C[indices_sort]

                # z_C = self.coordinate_coder.decode(postfix=postfix+"_z") #返回ply的coords
                # z_C = torch.cat((torch.zeros((len(z_C),1)).int(), torch.tensor(z_C).int()), dim=-1)
                # indices_sort = np.argsort(array2vector(z_C, z_C.max()+1)) #y_C.shape: torch.Size([13849, 4]) y_C[0]: tensor([ 0, 35,  1, 25], dtype=torch.int32)
                # z_C = z_C[indices_sort]
                y_C = pov_down.C
                z_down = downsample(pov_down)
                z_down = sort_spare_tensor(z_down)
                z_C = z_down.C
                print(pov_down.tensor_stride)
                print(z_down.tensor_stride)
                print(y_C.shape)
                y = self.feature_coder.decode(pov_down.tensor_stride[0],postfix=postfix, y_C=y_C, z_C=z_C)
                # y = ME.SparseTensor(features=y_F, coordinates=y_C,
                #                 tensor_stride=pov_down.tensor_stride, device=device)
                x_down_hat = self.sopa_slne.slne.decoder(y)
                x_down_hat = sort_spare_tensor(x_down_hat)
                pov = sort_spare_tensor(pov)
                mask_true = isin(x_down_hat.C, pov.C)
                x_down_hat = mask_sparse_tensor(x_down_hat,mask_true)
                x2_dec,likelihoodx2 = self.sopa_slne.one_sopa_slne(x_down_hat,th,training=False)
            else:
                y = ME.SparseTensor(features=torch.ones(pov.shape[0],1), coordinates=pov.C,
                                tensor_stride=pov.tensor_stride, device=device)
                x2_dec,likelihoodx2 = self.sopa.one_sopa(y,th,training=False)
            out = self.prune_voxel(x2_dec, likelihoodx2,  num_points[0], None , training=False)
        if(scale==3):
            if(if_slne):
                pov_down = downsample(pov)
                pov_down = sort_spare_tensor(pov_down)
                y_C = pov_down.C
                z_down = downsample(pov_down)
                z_down = sort_spare_tensor(z_down)
                z_C = z_down.C
                print(y_C.shape)
                y = self.feature_coder.decode(pov_down.tensor_stride[0],postfix=postfix, y_C=y_C, z_C=z_C)
                x_down_hat = self.sopa_slne.slne.decoder(y)
                x_down_hat = sort_spare_tensor(x_down_hat)
                pov = sort_spare_tensor(pov)
                mask_true = isin(x_down_hat.C, pov.C)
                x_down_hat = mask_sparse_tensor(x_down_hat,mask_true)
                x2_dec,likelihoodx2 = self.sopa_slne.one_sopa_slne(x_down_hat,th,training=False)
                out = self.prune_voxel(likelihoodx2,likelihoodx2,num_points[0],None,training=False)
                out = ME.SparseTensor(features=torch.ones(out.shape[0],1), coordinates=out.C,
                                tensor_stride=out.tensor_stride, device=device)
                x2_dec,likelihoodx2 = self.sopa.one_sopa(out,th,training=False)


            else:
                y = ME.SparseTensor(features=torch.ones(pov.shape[0],1), coordinates=pov.C,
                                tensor_stride=pov.tensor_stride, device=device)
                x2_dec,likelihoodx2 = self.sopa.one_sopa(y,th,training=False)
                out = self.prune_voxel(likelihoodx2,likelihoodx2,num_points[0],None,training=False)
                out = ME.SparseTensor(features=torch.ones(out.shape[0],1), coordinates=out.C,
                                tensor_stride=out.tensor_stride, device=device)
                x2_dec,likelihoodx2 = self.sopa.one_sopa(out,th,training=False)
            out = self.prune_voxel(x2_dec, likelihoodx2,  num_points[1], None , training=False)        

        if(scale==4):
            if(if_slne):
                pov_down = downsample(pov)
                pov_down = sort_spare_tensor(pov_down)
                y_C = pov_down.C
                z_down = downsample(pov_down)
                z_down = sort_spare_tensor(z_down)
                z_C = z_down.C
                print(y_C.shape)
                y = self.feature_coder.decode(pov_down.tensor_stride[0],postfix=postfix, y_C=y_C, z_C=z_C)
                x_down_hat = self.sopa_slne.slne.decoder(y)
                x_down_hat = sort_spare_tensor(x_down_hat)
                pov = sort_spare_tensor(pov)
                mask_true = isin(x_down_hat.C, pov.C)
                x_down_hat = mask_sparse_tensor(x_down_hat,mask_true)
                x2_dec,likelihoodx2 = self.sopa_slne.one_sopa_slne(x_down_hat,th,training=False)
                out = self.prune_voxel(likelihoodx2,likelihoodx2,num_points[0],None,training=False)
                out = ME.SparseTensor(features=torch.ones(out.shape[0],1), coordinates=out.C,
                                tensor_stride=out.tensor_stride, device=device)
                x2_dec,likelihoodx2 = self.sopa.one_sopa(out,th,training=False)
                out = self.prune_voxel(likelihoodx2,likelihoodx2,num_points[1],None,training=False)
                out = ME.SparseTensor(features=torch.ones(out.shape[0],1), coordinates=out.C,
                                tensor_stride=out.tensor_stride, device=device)
                x2_dec,likelihoodx2 = self.sopa.one_sopa(out,th,training=False)           



            else:
                y = ME.SparseTensor(features=torch.ones(pov.shape[0],1), coordinates=pov.C,
                                tensor_stride=pov.tensor_stride, device=device)
                x2_dec,likelihoodx2 = self.sopa.one_sopa(y,th,training=False)
                out = self.prune_voxel(likelihoodx2,likelihoodx2,num_points[0],None,training=False)
                out = ME.SparseTensor(features=torch.ones(out.shape[0],1), coordinates=out.C,
                                tensor_stride=out.tensor_stride, device=device)
                x2_dec,likelihoodx2 = self.sopa.one_sopa(out,th,training=False)
                out = self.prune_voxel(likelihoodx2,likelihoodx2,num_points[1],None,training=False)
                out = ME.SparseTensor(features=torch.ones(out.shape[0],1), coordinates=out.C,
                                tensor_stride=out.tensor_stride, device=device)
                x2_dec,likelihoodx2 = self.sopa.one_sopa(out,th,training=False)

            out = self.prune_voxel(x2_dec, likelihoodx2,  num_points[2], None , training=False)

            
            print(out.shape)
        return out

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