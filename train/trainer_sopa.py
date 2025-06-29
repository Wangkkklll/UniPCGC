import os, sys, time, logging
from tqdm import tqdm
import numpy as np
import torch
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiConvolution import MinkowskiConvolution, MinkowskiConvolutionTranspose
from loss import get_bce, get_bits, get_metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tensorboardX import SummaryWriter
from data_utils import sort_spare_tensor

down = MinkowskiConvolution(
                in_channels=1,
                out_channels=1,
                kernel_size=2,
                stride=2,
                bias=True,
                dimension=3,
        ).to(device)

down.kernel.requires_grad = False
class Trainer():
    def __init__(self, config, model):
        self.config = config
        self.logger = self.getlogger(config.logdir)
        self.writer = SummaryWriter(log_dir=config.logdir)

        self.model = model.to(device)
        # self.logger.info(model)
        self.load_state_dict(config.pre_loading)
        # self.model.load_ckpt(config.anchor1_ckptdir,config.anchor2_ckptdir,config.anchor3_ckptdir,config.anchor4_ckptdir)
        self.epoch = 0
        self.record_set = { 'bpp':[],'scale1':[],'scale2':[],'scale3':[],'scale4':[]}
        self.setting_packing = None

    def getlogger(self, logdir):
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(os.path.join(logdir, 'log.txt'))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d %H:%M:%S')
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(console)

        return logger

    def load_state_dict(self, loading_pre=None):
        """selectively load model
        """
        if self.config.init_ckpt=='':
            self.logger.info('Random initialization.')
            if loading_pre:
                print("loading_pre")
                print(loading_pre)
                ckpt = torch.load(loading_pre)
                model_dict = self.model.state_dict()
                # self.model.load_state_dict(ckpt["model"])
                pretrained_dict = {k: v for k, v in ckpt['model'].items() if k in model_dict.keys()}
                model_dict.update(pretrained_dict) #利用预训练模型的参数，更新模型
                self.model.load_state_dict(model_dict)
        else:            
            ckpt = torch.load(self.config.init_ckpt)
            self.model.load_state_dict(ckpt["model"])
            self.logger.info('Load checkpoint from ' + self.config.init_ckpt)

        return

    def save_model(self):
        torch.save({'model': self.model.state_dict()}, 
            os.path.join(self.config.ckptdir, 'epoch_' + str(self.epoch) + '.pth'))
        return

    def set_optimizer(self):
        params_lr_list = []
        for module_name in self.model._modules.keys():
            params_lr_list.append({"params":self.model._modules[module_name].parameters(), 'lr':self.config.lr})
        optimizer = torch.optim.Adam(params_lr_list, betas=(0.9, 0.999), weight_decay=1e-4)

        return optimizer

    def set_sperate_optimizer(self):
        from compressai.optimizers.net_aux import net_aux_optimizer
        
        conf = {
            "net": {"type": "Adam", "lr": self.config.lr, "weight_decay":1e-4},
            "aux": {"type": "Adam", "lr": self.config.aux_lr,"weight_decay":1e-4},
        }
        optimizer = net_aux_optimizer(self.model, conf)
        return optimizer["net"], optimizer["aux"]

    def freeze_param(self):
        need_frozen_list =['encoder', 'decoder']
        for param in self.model.named_parameters():
            type_param = param[0].split(".")[0]
            if type_param in need_frozen_list:
                param[1].requires_grad = False

    @torch.no_grad()
    def record(self, main_tag, global_step):
        # print record
        self.logger.info('='*10+main_tag + ' Epoch ' + str(self.epoch) + ' Step: ' + str(global_step))
        for k, v in self.record_set.items(): 
            self.record_set[k]=np.mean(np.array(v), axis=0)
        for k, v in self.record_set.items(): 
            self.logger.info(k+': '+str(np.round(v, 4).tolist()))   
            self.writer.add_scalar(main_tag+"_"+k, v, global_step)
        # return zero
        for k in self.record_set.keys(): 
            self.record_set[k] = []  

        return 

    @torch.no_grad()
    def test(self, dataloader, main_tag='Test'):
        self.model.eval()
        self.logger.info('Testing Files length:' + str(len(dataloader)))
        for batch_step, (coords, feats) in enumerate(tqdm(dataloader)):
            x = ME.SparseTensor(features=feats, coordinates=coords, 
            tensor_stride=1,device = device)
            # print(device)
            x1 = down(x)
            x1 = ME.SparseTensor(features=torch.ones_like(x1.F), coordinates=x1.C, 
            tensor_stride=x1.tensor_stride,device = x1.device) 
            x2 = down(x1)
            x2 = ME.SparseTensor(features=torch.ones_like(x2.F), coordinates=x2.C, 
            tensor_stride=x2.tensor_stride,device = x2.device)
            x3 = down(x2)
            x3 = ME.SparseTensor(features=torch.ones_like(x3.F), coordinates=x3.C, 
            tensor_stride=x3.tensor_stride,device = x3.device)
            input = [x,x1,x2,x3]
            bpp_list = []
            bpp = 0
            for j in input:
                bpp_i = 0
                probs, labels  = self.model(j,0)
                bpp_i = get_bce(probs, labels)/x.C.shape[0]
                bpp +=  bpp_i
                bpp_list.append(round(bpp_i.item(),5))
            
            self.record_set['bpp'].append(bpp.item())
            self.record_set['scale1'].append(bpp_list[0])
            self.record_set['scale2'].append(bpp_list[1])
            self.record_set['scale3'].append(bpp_list[2])
            self.record_set['scale4'].append(bpp_list[3])
            torch.cuda.empty_cache()# empty cache.
        self.record(main_tag=main_tag, global_step=self.epoch)

        return 

    def train(self, dataloader):
        self.model.train()
        self.logger.info('='*40+'\n'+'Training Epoch: ' + str(self.epoch))
        # optimizer
        self.optimizer = self.set_optimizer()
        self.logger.info('alpha:' + str(round(self.config.alpha,2)) + '\tbeta:' + str(round(self.config.beta,2)))
        self.logger.info('LR:' + str(np.round([params['lr'] for params in self.optimizer.param_groups], 6).tolist()))
        # dataloader
        self.logger.info('Training Files length:' + str(len(dataloader)))
        start_time = time.time()
        for batch_step, (coords,feats) in enumerate(tqdm(dataloader)):
            self.optimizer.zero_grad()
            x = ME.SparseTensor(features=feats, coordinates=coords, 
            tensor_stride=1,device = device)
            # print(device)
            x1 = down(x)
            x1 = ME.SparseTensor(features=torch.ones_like(x1.F), coordinates=x1.C, 
            tensor_stride=x1.tensor_stride,device = x1.device) 
            x2 = down(x1)
            x2 = ME.SparseTensor(features=torch.ones_like(x2.F), coordinates=x2.C, 
            tensor_stride=x2.tensor_stride,device = x2.device)
            x3 = down(x2)
            x3 = ME.SparseTensor(features=torch.ones_like(x3.F), coordinates=x3.C, 
            tensor_stride=x3.tensor_stride,device = x3.device)
            input = [x,x1,x2,x3]
            bpp_list = []
            bpp = 0
            for j in input:
                bpp_i = 0
                probs, labels  = self.model(j,0)
                bpp_i = get_bce(probs ,labels)/x.C.shape[0]
                bpp +=  bpp_i
                bpp_list.append(round(bpp_i.item(),5))
            bpp.backward()
            self.optimizer.step()       
                   
            with torch.no_grad():
                metrics = []
                
                self.record_set['bpp'].append(bpp.item())
                self.record_set['scale1'].append(bpp_list[0])
                self.record_set['scale2'].append(bpp_list[1])
                self.record_set['scale3'].append(bpp_list[2])
                self.record_set['scale4'].append(bpp_list[3])
                if (time.time() - start_time) > self.config.check_time*60:
                    self.record(main_tag='Train', global_step=self.epoch*len(dataloader)+batch_step)
                    self.save_model()
                    start_time = time.time()
            torch.cuda.empty_cache()# empty cache.
        with torch.no_grad(): self.record(main_tag='Train', global_step=self.epoch*len(dataloader)+batch_step)
        self.save_model()
        self.epoch += 1

        return
    
