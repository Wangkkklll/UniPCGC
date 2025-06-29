# 下采样两次的编解码器
# SparsePCGC最高两个码率点的编码器->测试后拓展到全码率点 （待测试）


import os, time
import numpy as np
import torch
import MinkowskiEngine as ME

from data_utils import array2vector, istopk, sort_spare_tensor, load_sparse_tensor,MK_our_union
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
        self.union = MK_our_union()

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
    # 编码端知道哪个会编码错误，将编码错误的point以无损的方式进行编码->增大psnr
    def encode(self,x,th,scale,if_slne,if_mutil,postfix=''):
        # Encoder
        th = torch.tensor([th],device = x.device).float()
        print(x.shape)
        if(scale>=2):
            x1 = downsample(x)
            ground_truth_list = [x]
            x1 = ME.SparseTensor(
                features=torch.ones(x1.C.shape[0],1,device = x.device,requires_grad = False),
                coordinates=x1.C,
                coordinate_manager=x1.coordinate_manager,
                tensor_stride=x1.tensor_stride)
            return_x = x1
            return_y = x

            
            
        if(scale>=3):
            x2 = downsample(x1)
            x2 = ME.SparseTensor(
            features=torch.ones(x2.C.shape[0],1,device = x.device,requires_grad = False),
            coordinates=x2.C,
            coordinate_manager=x2.coordinate_manager,
            tensor_stride=x2.tensor_stride)
            ground_truth_list = [x1,x]
            return_x = x2
            return_y = x1
        if(scale>=4):
            x3 = downsample(x2)
            x3 = ME.SparseTensor(
            features=torch.ones(x3.C.shape[0],1,device = x.device,requires_grad = False),
            coordinates=x3.C,
            coordinate_manager=x3.coordinate_manager,
            tensor_stride=x3.tensor_stride)
            ground_truth_list = [x2,x1,x]
            return_x = x3
            return_y = x2
        #ground_truth_list = reversed(ground_truth_list)
        #print(ground_truth_list)
        num_points = [[len(C) for C in ground_truth.decomposed_coordinates] \
            for ground_truth in ground_truth_list]
        print(num_points)
        with open(self.filename+postfix+'_num_points.bin', 'wb') as f:
            f.write(np.array(num_points, dtype=np.int32).tobytes())
            
        # 解码
        # x2_dec,likelihoodx2 = self.sopa.one_sopa(return_x,th,training=False)
        # lossless_num = int(th*return_y.shape[0])
        # num = [i-lossless_num for i in num_points[0]]
        # dec = self.prune_voxel(likelihoodx2, likelihoodx2,  num, None , training=False)
        # dec_if_true = isin(return_y.C,dec.C)
        # # 取出false的
        # dec_false = mask_sparse_tensor(return_y,~dec_if_true)
        # print(dec_false.shape[0])
        
        # if(lossless_num<dec_false.shape[0]):
        #     #截取部分
        #     dec_false_C = dec_false.C[:lossless_num,:]
        # else:
        #     dec_false_C = dec_false.C
        #th = torch.tensor([th],device = x.device).float()
        # print(dec_false)
        return return_y,return_x,None
        
        

    @torch.no_grad()
    def decode(self,pov,x_lastscale_lossless,prior,th,scale,if_slne,if_mutil,rho=1, postfix=''):
        training = False
        th = torch.tensor([th],device = device).float()
        with open(self.filename+postfix+'_num_points.bin', 'rb') as fin:
                num_points = np.frombuffer(fin.read(4*3), dtype=np.int32).tolist()
                num_points[-1] = int(rho * num_points[-1])# update
                num_points = [[num] for num in num_points]
        if(scale==2):
            y = ME.SparseTensor(features=torch.ones(pov.shape[0],1), coordinates=pov.C,
                                tensor_stride=pov.tensor_stride, device=device)
            mask = isin(y.C,x_lastscale_lossless.C)
            last_scale_lossy = mask_sparse_tensor(y,~mask)
            # x2_dec,no_prior = self.sopa.one_sopa(last_scale_lossy,th,training=False)
            y = self.sopa.one_sopa.dfa1(last_scale_lossy)
            xup = self.sopa.one_sopa.vsl(y)
            prior_num = prior.shape[0]
            prior_c = ME.SparseTensor(features=torch.ones(prior.shape[0],xup.shape[1]), coordinates=prior.C,
                                tensor_stride=prior.tensor_stride, device=device)
            # print(prior_c,xup)
            x_emb = self.union(prior_c,xup)
            no_prior = self.sopa.one_sopa.dfa2(x_emb)
            no_prior = self.sopa.one_sopa.ool(no_prior)
            
            mask_isin = isin(no_prior.C,xup.C)
            no_prior = mask_sparse_tensor(no_prior,mask_isin)
            # no_prior = mask_sparse_tensor(xup,~mask_isin)
            # no_prior = self.sopa.one_sopa.dfa2(no_prior)
            # no_prior = self.sopa.one_sopa.ool(no_prior)
            num = [i-prior_num for i in num_points[0]]
            no_prior_prune = self.prune_voxel(no_prior, no_prior,  num, None , training=False)
            out = self.union(prior,no_prior_prune)
        if(scale==3):
            y = ME.SparseTensor(features=torch.ones(pov.shape[0],1), coordinates=pov.C,
                                tensor_stride=pov.tensor_stride, device=device)
            mask = isin(y.C,x_lastscale_lossless.C)
            last_scale_lossy = mask_sparse_tensor(y,~mask)
            #x2_dec,likelihoodx2 = self.sopa.one_sopa(last_scale_lossy,th,training=False)
            #prior_num = prior.shape[0]
            
            y = self.sopa.one_sopa.dfa1(last_scale_lossy)
            xup = self.sopa.one_sopa.vsl(y)
            prior_num = prior.shape[0]
            prior_c = ME.SparseTensor(features=torch.ones(prior.shape[0],xup.shape[1]), coordinates=prior.C,
                                tensor_stride=prior.tensor_stride, device=device)
            # print(prior_c,xup)
            x_emb = self.union(prior_c,xup)
            no_prior = self.sopa.one_sopa.dfa2(x_emb)
            no_prior = self.sopa.one_sopa.ool(no_prior)
            
            mask_isin = isin(no_prior.C,xup.C)
            likelihoodx2 = mask_sparse_tensor(no_prior,mask_isin)
          
            num = [i-prior_num for i in num_points[0]]
            no_prior_prune = self.prune_voxel(likelihoodx2, likelihoodx2,  num, None , training=False)
            out = self.union(prior,no_prior_prune)
            
            out = ME.SparseTensor(features=torch.ones(out.shape[0],1), coordinates=out.C,
                                tensor_stride=out.tensor_stride, device=device)
            x2_dec,likelihoodx2 = self.sopa.one_sopa(out,th,training=False)
            out = self.prune_voxel(x2_dec, likelihoodx2,  num_points[1], None , training=False)        

        if(scale==4):
            y = ME.SparseTensor(features=torch.ones(pov.shape[0],1), coordinates=pov.C,
                                tensor_stride=pov.tensor_stride, device=device)
            mask = isin(y.C,x_lastscale_lossless.C)
            last_scale_lossy = mask_sparse_tensor(y,~mask)
            #x2_dec,likelihoodx2 = self.sopa.one_sopa(last_scale_lossy,th,training=False)
            #prior_num = prior.shape[0]
            
            y = self.sopa.one_sopa.dfa1(last_scale_lossy)
            xup = self.sopa.one_sopa.vsl(y)
            prior_num = prior.shape[0]
            prior_c = ME.SparseTensor(features=torch.ones(prior.shape[0],xup.shape[1]), coordinates=prior.C,
                                tensor_stride=prior.tensor_stride, device=device)
            # print(prior_c,xup)
            x_emb = self.union(prior_c,xup)
            no_prior = self.sopa.one_sopa.dfa2(x_emb)
            no_prior = self.sopa.one_sopa.ool(no_prior)
            
            mask_isin = isin(no_prior.C,xup.C)
            likelihoodx2 = mask_sparse_tensor(no_prior,mask_isin)
          
            num = [i-prior_num for i in num_points[0]]
            no_prior_prune = self.prune_voxel(likelihoodx2, likelihoodx2,  num, None , training=False)
            out = self.union(prior,no_prior_prune)

            
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
