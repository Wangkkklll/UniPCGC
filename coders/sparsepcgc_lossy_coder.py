# 下采样两次的编解码器
# SparsePCGC最高两个码率点的编码器->测试后拓展到全码率点 （待测试）


import os, time
import numpy as np
import torch
import MinkowskiEngine as ME

from data_utils import array2vector, istopk, sort_spare_tensor, load_sparse_tensor
from data_utils import write_ply_ascii_geo, read_ply_ascii_geo
from MinkowskiEngine.MinkowskiConvolution import MinkowskiConvolution, MinkowskiGenerativeConvolutionTranspose
from gpcc import gpcc_encode, gpcc_decode
from pc_error import pc_error
from data_utils import isin, istopk
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        write_ply_ascii_geo(filedir=self.ply_filename, coords=coords)
        gpcc_encode(self.ply_filename, self.filename+postfix+'_C.bin')
        os.system('rm '+self.ply_filename)
        
        return 

    def decode(self, postfix=''):
        gpcc_decode(self.filename+postfix+'_C.bin', self.ply_filename)
        coords = read_ply_ascii_geo(self.ply_filename)
        os.system('rm '+self.ply_filename)
        
        return coords


class FeatureCoder():
    """encode/decode feature using learned entropy model
    """
    def __init__(self, filename, entropy_model):
        self.filename = filename
        self.entropy_model = entropy_model.cpu()

    def encode(self, feats, postfix=''):
        #assert 1==2
        strings, min_v, max_v = self.entropy_model.compress(feats.cpu())
        #a = self.entropy_model.compress(feats.cpu())
        shape = feats.shape
        with open(self.filename+postfix+'_F.bin', 'wb') as fout:
            fout.write(strings)
        with open(self.filename+postfix+'_H.bin', 'wb') as fout:
            fout.write(np.array(shape, dtype=np.int32).tobytes())
            fout.write(np.array(len(min_v), dtype=np.int8).tobytes())
            fout.write(np.array(min_v, dtype=np.float32).tobytes())
            fout.write(np.array(max_v, dtype=np.float32).tobytes())
            
        return 

    def decode(self, postfix=''):
        with open(self.filename+postfix+'_F.bin', 'rb') as fin:
            strings = fin.read()
        with open(self.filename+postfix+'_H.bin', 'rb') as fin:
            shape = np.frombuffer(fin.read(4*2), dtype=np.int32)
            len_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]
            min_v = np.frombuffer(fin.read(4*len_min_v), dtype=np.float32)[0]
            max_v = np.frombuffer(fin.read(4*len_min_v), dtype=np.float32)[0]
            
        feats = self.entropy_model.decompress(strings, min_v, max_v, shape, channels=shape[-1])
        
        return feats


class Lossy_Coder():
    def __init__(self, sopa,sopa_slne, filename):
        self.sopa = sopa 
        self.sopa_slne = sopa_slne
        self.filename = filename
        self.feature_coder = FeatureCoder(self.filename, self.sopa_slne.slne.entropy_bottleneck)
        self.pruning = ME.MinkowskiPruning()

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
    def encode(self,x,th,scale,if_slne,if_mutil,postfix=''):
        # Encoder
        # scale->下采样的次数,scale=2/3/4
        th = torch.tensor([th],device = x.device).float()
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
            
            
        th = torch.tensor([th],device = x.device).float()
        if(if_slne):
            if(if_mutil==True):
                bm_data = self.sopa_slne.slne.bm_module(th)
                x_enc = self.sopa_slne.slne.encoder(ground_truth_list[0])
                x_enc = sort_spare_tensor(x_enc)
                x_enc_F = x_enc.F*bm_data
                self.feature_coder.encode(x_enc_F, postfix=postfix)
            else:
                x_enc = self.sopa_slne.slne.encoder(ground_truth_list[0])
                x_enc = sort_spare_tensor(x_enc)
                self.feature_coder.encode(x_enc.F, postfix=postfix)
            return x_enc,return_x
        else:
            return return_x,return_x
        
        

    @torch.no_grad()
    def decode(self,pov,th,scale,if_slne,if_mutil,rho=1, postfix=''):
        training = False
        th = torch.tensor([th],device = device).float()
        with open(self.filename+postfix+'_num_points.bin', 'rb') as fin:
                num_points = np.frombuffer(fin.read(4*3), dtype=np.int32).tolist()
                num_points[-1] = int(rho * num_points[-1])# update
                num_points = [[num] for num in num_points]
        if(scale==2):
            if(if_slne):
                y_F = self.feature_coder.decode(postfix=postfix)
                pov_down = downsample(pov)
                pov_down = sort_spare_tensor(pov_down)
                y_C = pov_down.C
                print(y_F.shape)
                print(y_C.shape)
                if(if_mutil==True):
                    bm_data = self.sopa_slne.slne.bm_module(th)
                    y_F = y_F.to(device)
                    y_F = y_F/bm_data
                y = ME.SparseTensor(features=y_F, coordinates=y_C,
                                tensor_stride=pov_down.tensor_stride, device=device)
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
                y_F = self.feature_coder.decode(postfix=postfix)
                pov_down = downsample(pov)
                pov_down = sort_spare_tensor(pov_down)
                y_C = pov_down.C
                if(if_mutil==True):
                    bm_data = self.sopa_slne.slne.bm_module(th)
                    y_F = y_F.to(device)
                    y_F = y_F/bm_data
                y = ME.SparseTensor(features=y_F, coordinates=y_C,
                                tensor_stride=pov_down.tensor_stride, device=device)
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
                y_F = self.feature_coder.decode(postfix=postfix)
                pov_down = downsample(pov)
                pov_down = sort_spare_tensor(pov_down)
                y_C = pov_down.C
                if(if_mutil==True):
                    bm_data = self.sopa_slne.slne.bm_module(th)
                    y_F = y_F.to(device)
                    y_F = y_F/bm_data
                y = ME.SparseTensor(features=y_F, coordinates=y_C,
                                tensor_stride=pov_down.tensor_stride, device=device)
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

            
        print("xdec:",out.shape)
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
