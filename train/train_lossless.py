import time, os, sys, glob, argparse
import importlib
import numpy as np
import torch
import MinkowskiEngine as ME
from data_loader import PCDataset, make_data_loader
from models.uelc import UELC
from train.trainer_lossless import Trainer

"""
Training For UELC
"""

def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", default='/public/DATA/yzz/pc_vox8_n100k/')
    parser.add_argument("--filedir_add", default='')
    parser.add_argument("--dataset_num", type=int, default=4e4)
    parser.add_argument("--alpha", type=float, default=1, help="weights for distoration.")
    parser.add_argument("--beta", type=float, default=1., help="weights for bit rate.")
    parser.add_argument("--init_ckpt", default='') 
    parser.add_argument("--pre_loading", default='')
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--aux_lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--check_time", type=float, default=1,  help='frequency for recording state (min).') 
    parser.add_argument("--prefix", type=str, default='nonanchor_enhanced', help="prefix of checkpoints/logger, etc.")
    parser.add_argument("--coding_type", type=str, default='all', help="prefix of checkpoints/logger, etc.")
    args = parser.parse_args()

    return args

class TrainingConfig():
    def __init__(self, logdir, ckptdir, init_ckpt, alpha, beta, lr, aux_lr, check_time, pre_loading, coding_type):
        self.logdir = logdir
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        self.ckptdir = ckptdir
        if not os.path.exists(self.ckptdir): os.makedirs(self.ckptdir)
        self.init_ckpt = init_ckpt
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.aux_lr = aux_lr 
        self.check_time = check_time
        self.pre_loading = pre_loading if len(pre_loading) > 0 else None
        self.coding_type = coding_type

if __name__ == '__main__':
    # log
    args = parse_args()
    training_config = TrainingConfig(
                            logdir=os.path.join('./logs', args.prefix), 
                            ckptdir=os.path.join('./ckpts', args.prefix), 
                            init_ckpt=args.init_ckpt, 
                            alpha=args.alpha, 
                            beta=args.beta, 
                            lr=args.lr, 
                            aux_lr=args.aux_lr,
                            check_time=args.check_time,
                            pre_loading=args.pre_loading,
                            coding_type=args.coding_type,)
    # model
    model = UELC()
    params = list(model.parameters())
    total_params = sum(p.numel() for p in params)
    print("params:",total_params)
    trainer = Trainer(config=training_config, model=model)
    filedirs = sorted(glob.glob(args.dataset+'*/*.h5'))[:int(args.dataset_num)]
    train_dataset = PCDataset(filedirs[round(len(filedirs)/10):])
    train_dataloader = make_data_loader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, repeat=False)
    test_dataset = PCDataset(filedirs[:round(len(filedirs)/10)])
    test_dataloader = make_data_loader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, repeat=False)
    MPEG_filedirs = sorted(glob.glob('/public/DATA/wkl/UELC_test/*.ply'))
    val_dataset = PCDataset(MPEG_filedirs)
    val_dataloader = make_data_loader(dataset=val_dataset, batch_size=1, shuffle=False, repeat=False)

    # training
    for epoch in range(0, args.epoch):
        if epoch > 25: trainer.config.lr =  max(trainer.config.lr/1.2, 3e-5)# update lr 
        else:
            if epoch > 0: trainer.config.lr =  max(trainer.config.lr/2, 1e-4)# update lr 
        trainer.train(train_dataloader)
        trainer.test(test_dataloader, 'Test')
        trainer.test(val_dataloader, 'Valid')
