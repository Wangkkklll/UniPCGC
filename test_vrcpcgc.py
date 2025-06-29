import torch
import numpy as np
import os
import MinkowskiEngine as ME
import time
from glob import glob
from data_utils import load_sparse_tensor
from data_utils import write_ply_ascii_geo, read_ply_ascii_geo
from pc_error import pc_error
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(filedir, ckptdir, ckptdir_slne,th,if_slne,if_mutil,scale,down_scale,outdir, r,original=None, res=1024):
    # load data
    start_time = time.time()
    reduce_bit = None
    x = load_sparse_tensor(filedir, device, reduce_bit)
    
    print('Loading Time:\t', round(time.time() - start_time, 4), 's')
    # x = sort_spare_tensor(input_data)

    # output filename
    # if not os.path.exists(outdir): os.makedirs(outdir)
    filename = os.path.join(outdir, os.path.split(filedir)[-1].split('.')[0])
    print('output filename:\t', filename)
    channels = 32
    from models.olc import pcgcv3_sopa
    from models.vrcpcgc import pcgcv3_sopa_slne_multi2
    slne = pcgcv3_sopa_slne_multi2(channels).to(device)
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
    
    from coders.sparsepcgc_lossy_coder_slim import Lossy_Coder
    #from coders.sparsepcgc_lossy_coder import Lossy_Coder
    lossy_coder = Lossy_Coder(sopa=sopa,sopa_slne=slne, filename=filename)
    from coders.lossless_coder import Coder
    lossless_coder = Coder(model=uelc, filename=args.outdir,  fake_bpp = False)

    # postfix: rate index
    postfix_idx = ''

    
    print(th)
    start_time = time.time()
    #有损编码
    _,x1 = lossy_coder.encode(x,th,down_scale,args.if_slne,True)
    tensor_stride = 2**(down_scale-1)
    # 进行无损编码
    x1_C = x1.C[:,1:]//tensor_stride
    bits_lossless = lossless_coder.encode(x1_C)
    print('Enc Time:\t', round(time.time() - start_time, 3), 's')
    time_enc = round(time.time() - start_time, 3)
    # decode
    start_time = time.time()
    # 进行无损解码
    x1_dec = lossless_coder.decode()
    x1_dec_C = x1_dec.C[:,1:] * tensor_stride
    feats = torch.ones(x1_dec.shape[0],1)
    coords, feats = ME.utils.sparse_collate([x1_dec_C], [feats])
    x1_dec = ME.SparseTensor(features=feats, coordinates=coords,
                                tensor_stride=tensor_stride, device=device)
    x_dec = lossy_coder.decode(x1_dec,th,down_scale,args.if_slne,True,rho=args.rho)
    # print(x_dec)
    print('Dec Time:\t', round(time.time() - start_time, 3), 's')
    time_dec = round(time.time() - start_time, 3)
    # bitrate
    if(if_slne==False):
        bits = bits_lossless
        bpps = (bits/len(x))
        print('bits:\t', bits, '\nbpps:\t',  round(bpps,3))

        # distortion
        start_time = time.time()
        x_dec_C = x_dec.C.detach().cpu().numpy()[:,1:] * (2 ** reduce_bit) if reduce_bit else x_dec.C.detach().cpu().numpy()[:,1:]
        write_ply_ascii_geo(filename+postfix_idx+'_dec.ply', x_dec_C)
        print('Write PC Time:\t', round(time.time() - start_time, 3), 's')

        start_time = time.time()
        pc_error_metrics = pc_error(filedir, filename+postfix_idx+'_dec.ply', 
                                res=res, normal=True, show=False)
        print('PC Error Metric Time:\t', round(time.time() - start_time, 3), 's')
        print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])

        # save results 2546611
        results = {}

        # 可变码率版本
        results['file'] = filename.split('/')[-1]
        results['R'] = r
        results['D1'] = pc_error_metrics["mseF,PSNR (p2point)"][0]
        results['D2'] = pc_error_metrics["mseF,PSNR (p2plane)"][0]
        # results["num_points(input)"] = len(x)
        # results["num_points(output)"] = len(x_dec)
        # results["resolution"] = res
        # results["bits"] = sum(bits).round(3)
        results['bpp'] = sum(bpps).round(4)
        # results["bpp(coords)"] = bpps[1]+bpps[2]+bpps[3]
        # results["bpp(feats)"] = bpps[0]
        results["time(enc)"] = time_enc
        results["time(dec)"] = time_dec

        return results
    else:
        bits = np.array([os.path.getsize(filename + postfix_idx + postfix)*8 \
                            for postfix in [ '_F.bin', '_H.bin', '_num_points.bin']])
        bits = np.append(bits, bits_lossless)
        bpps = (bits/len(x)).round(3)
        print('bits:\t', sum(bits), '\nbpps:\t',  sum(bpps).round(3))

        # distortion
        start_time = time.time()
        x_dec_C = x_dec.C.detach().cpu().numpy()[:,1:] * (2 ** reduce_bit) if reduce_bit else x_dec.C.detach().cpu().numpy()[:,1:]
        write_ply_ascii_geo(filename+postfix_idx+'_dec.ply', x_dec_C)
        print('Write PC Time:\t', round(time.time() - start_time, 3), 's')

        start_time = time.time()
        pc_error_metrics = pc_error(filedir, filename+postfix_idx+'_dec.ply', 
                                res=res, normal=True, show=False)
        print('PC Error Metric Time:\t', round(time.time() - start_time, 3), 's')
        print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])

        # save results 2546611
        results = {}
        results['file'] = filename.split('/')[-1]
        results['R'] = r
        results['D1'] = pc_error_metrics["mseF,PSNR (p2point)"][0]
        results['D2'] = pc_error_metrics["mseF,PSNR (p2plane)"][0]
        # results["num_points(input)"] = len(x)
        # results["num_points(output)"] = len(x_dec)
        # results["resolution"] = res
        # results["bits"] = sum(bits).round(3)
        results['bpp'] = sum(bpps).round(4)
        # results["bpp(coords)"] = bpps[1]+bpps[2]+bpps[3]
        # results["bpp(feats)"] = bpps[0]
        results["time(enc)"] = time_enc
        results["time(dec)"] = time_dec

        return results

if __name__ == '__main__':
    import argparse
    import glob
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filedir", default='/public/DATA/wkl/8i5')
    parser.add_argument("--outdir_test", default='./vrm/loot/out')
    parser.add_argument("--resultdir", default='./vrm/loot/out')
    #parser.add_argument("--ckpt", default='./ckpts/r2_0.05bpp.pth')
    parser.add_argument("--tag", default='loot')
    parser.add_argument("--res", type=int, default=1024, help='resolution')
    parser.add_argument("--ckptdir", default='./ckpts/pretrain/sopa_best.pth') #sparsepcgc
    parser.add_argument("--ckptdir_slne", default='./ckpts/pretrain/vrcpcgc.pth') #sparsepcgc
    parser.add_argument("--lossless_ckptdir", default='./ckpts/pretrain/lossless_shapenet.pth')
    parser.add_argument("--scaling_factor", type=float, default=1.0, help='scaling_factor')
    parser.add_argument("--rho", type=float, default=1.0, help='the ratio of the number of output points to the number of input points')
    parser.add_argument("--muti_lambda", type=float, default=-20, help='mutil-rate')
    parser.add_argument("--if_slne", type=bool, default=True, help="if slne")
    parser.add_argument("--down_scale", type=int, default=2, help="the num of down-scale")
    parser.add_argument("--if_mutil", type=bool, default=True, help="if mutil rate")
    parser.add_argument("--scale", type=int, default=5, help="the num of down-scale")
    parser.add_argument("--outdir", default='./output/')
    args = parser.parse_args()
    # args.filedir = args.filedir + args.tag
    args.outdir = args.outdir + args.tag
    args.resultdir = args.resultdir + args.tag

    if not os.path.exists(args.outdir): os.makedirs(args.outdir)
    if not os.path.exists(args.resultdir): os.makedirs(args.resultdir)
    files_dir = glob.glob(args.filedir + "/*.ply")
    # files_dir = [args.filedir]
    sum_csv_name = args.resultdir + "_summary.csv"
    all_result = None
    # file loop
    r = 0
    rate_list1 = np.arange(3.0, -1.0, -0.2).tolist()
    rate_list2 = np.arange(-2.0, -20.0, -2.0).tolist()
    rate_list1.extend(rate_list2)
    print("rate scale:")
    print(rate_list1)
    for scale in [2,3,4]:
        for rate in rate_list1:
        # for rate in [2.0,0.0,-20]:
            for id,input_file in tqdm(enumerate(files_dir)):
                input_file = input_file.strip()
                if len(input_file)<5 : 
                    continue 
                print('='*10, id+1, '='*10)
                print(input_file)
                # v_index = input_file.find("vox")
                # res_bit = int(input_file[v_index+3:v_index+5])
                # print(res_bit)
                # r = pow(2,res_bit)
                torch.cuda.empty_cache()
                res = test(input_file, args.ckptdir,args.ckptdir_slne,rate,args.if_slne,args.if_mutil,args.scale,scale, args.outdir,r, res=args.res)
                result = pd.DataFrame(res, index=[id])
                if all_result is None:
                    all_result = result.copy(deep=True)
                else:
                    all_result = all_result._append(result)
                all_result.to_csv(sum_csv_name, index=False)
            #ave_bpp = np.mean(np.array(all_result['bpp'][:]))
            #ave_D1 = np.mean(np.array(all_result['D1'][:]))
            #ave_D2 = np.mean(np.array(all_result['D2'][:]))
            #results = {}
            #results['file'] = 'all_files_ave'
            #results['D1'] = ave_D1
            #results['D2'] = ave_D2
            #results["num_points(input)"] = np.mean(np.array(all_result['num_points(input)'][:]))
            #results["resolution"] = args.res
            #results["bpp"] = ave_bpp
            #results["time(enc)"] = np.mean(np.array(all_result['time(enc)'][:]))
            #results["time(dec)"] = np.mean(np.array(all_result['time(dec)'][:]))
            #results['prefix'] = 'all_prefix_ave'
            #print("ave:",results)
            #all_result = all_result._append(results, ignore_index=True)
            r+=1
    #all_result.to_csv(sum_csv_name, index=False)
    # 计算average



