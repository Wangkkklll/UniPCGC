import os, time
import numpy as np
import torch
import MinkowskiEngine as ME
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckptdir", default='./ckpts/pretrain/sopa_best.pth') #sparsepcgc
    parser.add_argument("--ckptdir_slne", default='/home/wkl/code/UniPCGC/ckpts/one_sopa_slne/epoch_24.pth') #sparsepcgc
    parser.add_argument("--lossless_ckptdir", default='./ckpts/pretrain/lossless_shapenet.pth')
    parser.add_argument("--filedir", default='/public/DATA/wkl/8i5/longdress_vox10_1300.ply')
    parser.add_argument("--scaling_factor", type=float, default=1.0, help='scaling_factor')
    parser.add_argument("--rho", type=float, default=1.0, help='the ratio of the number of output points to the number of input points')
    parser.add_argument("--res", type=int, default=1024, help='resolution')
    parser.add_argument("--muti_lambda", type=float, default=-20, help='mutil-rate')
    parser.add_argument("--if_slne", type=bool, default=True, help="if slne")
    parser.add_argument("--down_scale", type=int, default=2, help="the num of down-scale")
    parser.add_argument("--outdir", default='./output/')
    
    args = parser.parse_args()
    filedir = args.filedir
    #out = np.load("out_c.npy")
    # load data
    start_time = time.time()
    x = load_sparse_tensor(filedir, device)
    print('Loading Time:\t', round(time.time() - start_time, 4), 's')

    outdir = './output'
    if not os.path.exists(outdir): os.makedirs(outdir)
    filename = os.path.split(filedir)[-1].split('.')[0]
    filename = os.path.join(outdir, filename)
    print(filename)
    channels = 32
    # model
    print('='*10, 'Test', '='*10)
    from models.olc import pcgcv3_sopa
    # from models.vrcpcgc import pcgcv3_sopa_slne_multi2
    from models.vrcpcgc import pcgcv3_sopa_slne
    slne = pcgcv3_sopa_slne(channels).to(device)
    sopa = pcgcv3_sopa(channels).to(device)
    from models.uelc import UELC
    uelc = UELC().to(device)
    assert os.path.exists(args.ckptdir)
    ckpt1 = torch.load(args.ckptdir)
    ckpt2 = torch.load(args.ckptdir_slne)
    ckpt3 = torch.load(args.lossless_ckptdir)
    sopa.load_state_dict(ckpt1['model'])
    slne.load_state_dict(ckpt2['model'])
    uelc.load_state_dict(ckpt3['model'])
    print('load checkpoint from \t', args.ckptdir)
    print('load checkpoint from \t', args.ckptdir_slne)
    print('load checkpoint from \t', args.lossless_ckptdir)
    
    # coder
    # from coders.sparsepcgc_lossy_coder_slim import Lossy_Coder
    from coders.sparsepcgc_lossy_coder import Lossy_Coder
    lossy_coder = Lossy_Coder(sopa=sopa,sopa_slne=slne, filename=filename)
    from coders.lossless_coder import Coder
    lossless_coder = Coder(model=uelc, filename=args.outdir,  fake_bpp = False)


    th = args.muti_lambda
    print(th)
    start_time = time.time()
    #有损编码
    _,x1 = lossy_coder.encode(x,th,args.down_scale,args.if_slne,False)
    tensor_stride = 2**(args.down_scale-1)
    # 进行无损编码
    x1_C = x1.C[:,1:]//tensor_stride
    bits_lossless = lossless_coder.encode(x1_C)
    print('Enc Time:\t', round(time.time() - start_time, 3), 's')

    # decode
    start_time = time.time()
    # 进行无损解码
    x1_dec = lossless_coder.decode()
    x1_dec_C = x1_dec.C[:,1:] * tensor_stride
    feats = torch.ones(x1_dec.shape[0],1)
    coords, feats = ME.utils.sparse_collate([x1_dec_C], [feats])
    x1_dec = ME.SparseTensor(features=feats, coordinates=coords,
                                tensor_stride=tensor_stride, device=device)
    x_dec = lossy_coder.decode(x1_dec,th,args.down_scale,args.if_slne,False,rho=args.rho)
    # print(x_dec)
    print('Dec Time:\t', round(time.time() - start_time, 3), 's')

    if(args.if_slne):
        # bitrate
        bits = np.array([os.path.getsize(filename + postfix)*8 \
                                for postfix in ['_F.bin', '_H.bin', '_num_points.bin']])
        bits = np.append(bits, bits_lossless)
        bpps = (bits/len(x)).round(3)
        print('bits:\t', bits, '\nbpps:\t', bpps)
        print('bits:\t', sum(bits), '\nbpps:\t',  sum(bpps).round(3))
    else:
        bpps = (bits_lossless/len(x))
        print('bits:\t', bits_lossless, '\nbpps:\t', round(bpps,3))

    # distortion
    start_time = time.time()
    write_ply_ascii_geo(filename+'_dec.ply', x_dec.C.detach().cpu().numpy()[:,1:])
    print('Write PC Time:\t', round(time.time() - start_time, 3), 's')

    start_time = time.time()
    pc_error_metrics = pc_error(args.filedir, filename+'_dec.ply', res=args.res, show=False)
    print('PC Error Metric Time:\t', round(time.time() - start_time, 3), 's')
    # print('pc_error_metrics:', pc_error_metrics)
    print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])
