import time, os, sys, glob, argparse
import importlib
import numpy as np
import torch
import MinkowskiEngine as ME
from data_loader import PCDataset, make_data_loader



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", default='/public/DATA/wkl/pc_vox8_n100k/')
    parser.add_argument("--dataset_num", type=int, default=4e4)

    parser.add_argument("--alpha", type=float, default=1., help="weights for distoration.")
    parser.add_argument("--beta", type=float, default=1., help="weights for bit rate.")
    parser.add_argument("--xita", type=float, default=1.,help="weights for corr.")
    parser.add_argument("--init_ckpt", default='') 
    parser.add_argument("--pre_loading", default='')# 
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--aux_lr", type=float, default=1e-4)
    parser.add_argument("--scale", type=int, default='8', help="the num of down-scale")
    parser.add_argument("--entropy_model", type=str, default='f', help="the entropy model type") # f\c
    parser.add_argument("--muti_lambda", nargs="+", type=float, default=[1.0])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--check_time", type=float, default=10,  help='frequency for recording state (min).') 
    parser.add_argument("--prefix", type=str, default='down2', help="prefix of checkpoints/logger, etc.")
    parser.add_argument("--channel", type=int, default=32, help="prefix of checkpoints/logger, etc.")
    parser.add_argument("--stage", type=str, default="stage1", help="stage of training")
 
    args = parser.parse_args()

    return args

class TrainingConfig():
    def __init__(self, logdir, ckptdir, init_ckpt, alpha, beta, xita,lr, aux_lr, check_time, pre_loading,muti_lambda):
        self.logdir = logdir
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        self.ckptdir = ckptdir
        if not os.path.exists(self.ckptdir): os.makedirs(self.ckptdir)
        self.init_ckpt = init_ckpt
        self.alpha = alpha
        self.beta = beta
        self.xita = xita
        self.lr = lr
        self.aux_lr = aux_lr 
        self.check_time = check_time
        self.muti_lambda = muti_lambda
        self.pre_loading = pre_loading if len(pre_loading) > 0 else None


if __name__ == '__main__':
    # log
    args = parse_args()
    training_config = TrainingConfig(
                            logdir=os.path.join('./logs', args.prefix), 
                            ckptdir=os.path.join('./ckpts', args.prefix), 
                            init_ckpt=args.init_ckpt, 
                            alpha=args.alpha, 
                            beta=args.beta, 
                            xita=args.xita,
                            lr=args.lr, 
                            aux_lr=args.aux_lr,
                            check_time=args.check_time,
                            pre_loading=args.pre_loading,
                            muti_lambda = args.muti_lambda)
    #select model
    channels = args.channel
    if(args.scale ==5):
        # one stage sopa train
        from models.olc import pcgcv3_sopa
        from train.trainer_sopa import Trainer
        model = pcgcv3_sopa(channels)
    if(args.scale ==6 ):
        # one stage sopa&slne train
        if(args.stage == "stage1"):
            from models.vrcpcgc import pcgcv3_sopa_slne_stage1
            from train.trainer_sopa_slne import Trainer
            model = pcgcv3_sopa_slne_stage1(channels)
        else:
            from models.vrcpcgc import pcgcv3_sopa_slne
            from train.trainer_sopa_slne import Trainer
            model = pcgcv3_sopa_slne(channels)
    if(args.scale ==7 ):
        # one stage sopa&slne train
        if(args.stage == "stage1"):
            from models.vrcpcgc import pcgcv3_sopa_slne_multi_stage1
            from train.trainer_sopa_slne_mutil import Trainer
            model = pcgcv3_sopa_slne_multi_stage1(channels)
        else:
            from models.vrcpcgc import pcgcv3_sopa_slne_multi
            from train.trainer_sopa_slne_mutil import Trainer
            model = pcgcv3_sopa_slne_multi(channels)
    if(args.scale ==8 ):
        # one stage sopa&slne train
        if(args.stage == "stage1"):
            from models.vrcpcgc import pcgcv3_sopa_slne_multi2_stage1
            from train.trainer_sopa_slne_mutil import Trainer
            model = pcgcv3_sopa_slne_multi2_stage1(channels)
        else:
            from models.vrcpcgc import pcgcv3_sopa_slne_multi2
            from train.trainer_sopa_slne_mutil import Trainer
            model = pcgcv3_sopa_slne_multi2(channels)
    

    trainer = Trainer(config=training_config, model=model)

    filedirs = sorted(glob.glob(args.dataset+'*/*.h5'))[:int(args.dataset_num)]
    train_dataset = PCDataset(filedirs)
    train_dataset = PCDataset(filedirs[round(len(filedirs)/10):])
    train_dataloader = make_data_loader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, repeat=False)
    test_dataset = PCDataset(filedirs[:round(len(filedirs)/10)])
    test_dataloader = make_data_loader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, repeat=False)
    
    MPEG_filedirs = sorted(glob.glob('/public/DATA/wkl/UELC_test/*.ply'))
    val_dataset = PCDataset(MPEG_filedirs)
    val_dataloader = make_data_loader(dataset=val_dataset, batch_size=1, shuffle=False, repeat=False)

    # training
    for epoch in range(0, args.epoch):
        if epoch>0: trainer.config.lr =  max(trainer.config.lr/1.5, 1e-5)# update lr 
        if epoch>0: trainer.config.aux_lr =  max(trainer.config.aux_lr/1.5, 2e-5)# update lr
        trainer.config.check_time = 1 if epoch==0 else args.check_time

        trainer.train(train_dataloader)
        trainer.test(test_dataloader, 'Test')
        trainer.test(val_dataloader, 'Valid')
