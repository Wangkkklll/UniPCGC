import os, time
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from data_utils import sort_spare_tensor, load_sparse_tensor
from models.uelc import UELC
from coders.lossless_coder import Coder

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckptdir", default='./ckpts/epoch_6.pth')
    parser.add_argument("--filedir", default='/public/DATA/wkl/8i5/loot_vox10_1200.ply')
    parser.add_argument("--outdir", default='./output/') 
    args = parser.parse_args()
    filedir = args.filedir
    # load data
    start_time = time.time()
    x = load_sparse_tensor(filedir, device)
    print('Loading Time:\t', round(time.time() - start_time, 4), 's')
    outdir = './output'
    if not os.path.exists(outdir): os.makedirs(outdir)
    filename = os.path.split(filedir)[-1].split('.')[0]
    filename = os.path.join(outdir, filename)
    print(filename)
    # model
    print('='*10, 'Test', '='*10)
    model = UELC().to(device)
    if os.path.exists(args.ckptdir):
        ckpt = torch.load(args.ckptdir)
        model.load_state_dict(ckpt['model'])
    print('load checkpoint from \t', args.ckptdir)
    # coder
    coder = Coder(model=model, filename=args.outdir,  fake_bpp = False)
    # encode
    start_time = time.time()
    x_C = x.C[:,1:]
    x_enc,_ = coder.encode(x_C)
    print('Enc Time:\t', round(time.time() - start_time, 4), 's')
    # decode
    start_time = time.time()
    x_dec = coder.decode(x_enc)
    print('Dec Time:\t', round(time.time() - start_time, 3), 's')
    # 验证无损:
    x_dec = sort_spare_tensor(x_dec)
    x = sort_spare_tensor(x)
    if  torch.equal(x_dec.C, x.C):
        print("congrats: lossless!")
    else:
        print("no !")
        print(x_dec.C)
        print(x.C)