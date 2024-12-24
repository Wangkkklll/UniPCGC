import torch
import MinkowskiEngine as ME
from data_utils import  group_mask_gen,group_mask_gen_par,group_mask_gen_8stage,isin,MK_our_union
from MinkowskiEngine.MinkowskiConvolution import MinkowskiConvolution, MinkowskiGenerativeConvolutionTranspose
from models.modules import FEL, ProbCoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
Official Implementation for UELC [AAAI 2025]
"""
class UELC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = 16
        self.kernel = 3
        self.large_kernel = 5
        self.dfa = FEL(1, self.channels,self.kernel)
        self.up = MinkowskiGenerativeConvolutionTranspose(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=2,
                stride=2,
                bias=True,
                dimension=3,
            )
        self.down = MinkowskiConvolution(
                in_channels=1,
                out_channels=1,
                kernel_size=2,
                stride=2,
                bias=True,
                dimension=3,  
            )
        self.down.kernel.requires_grad = False
        self.anchor1_coder = ProbCoder(self.channels,self.channels,self.kernel)
        self.anchor2_coder = ProbCoder(self.channels,self.channels,self.kernel)
        self.anchor3_coder = ProbCoder(self.channels,self.channels,self.kernel)
        self.anchor4_coder = ProbCoder(self.channels,self.channels,self.kernel)
        self.anchor5_coder = ProbCoder(self.channels,self.channels,self.kernel)
        self.anchor6_coder = ProbCoder(self.channels,self.channels,self.kernel)
        self.anchor1_2_coder = ProbCoder(self.channels,self.channels,self.large_kernel)
        self.anchor1_3_coder = ProbCoder(self.channels,self.channels,self.large_kernel)
        self.union = MK_our_union()

    def forward(self, x, partition_again = False, coding='all', attention_enhanced=False):
        '''
            inputs: 
                x : the scale i
            returns:
                probs: the occupied probs of scale i
        '''
        
        # obtain label:
        anchor_list = self.generate_label(x)
        x_down = self.down(x)
        # last layer pov
        last_POV = ME.SparseTensor(
            features=torch.ones_like(x_down.F), coordinates=x_down.C, 
            tensor_stride=x_down.tensor_stride,device = x.device
        ) 
        prob = self.dfa(last_POV)
        prob = self.up(prob)
        anchor_prob_list = group_mask_gen(prob)
        if coding in ['all', 'anchor1']:
            prob_anchor1 = self.mask_sparse_tensor(prob,anchor_prob_list[0])
            anchor1_prob,anchor1_label = self.anchor1_compression(prob_anchor1,anchor_list[0],partition_again=partition_again)
            
        if coding in ['all', 'anchor2']:
            prob_anchor2 = self.mask_sparse_tensor(prob,anchor_prob_list[1])
            dec_anchor1 = isin(prob.C,anchor_list[0].C)
            
            prob_anchor1_dec = self.mask_sparse_tensor(prob,dec_anchor1)
            anchor2_input = self.union(prob_anchor2,prob_anchor1_dec)
            anchor2_prob = self.anchor2_coder(anchor2_input)
            anchor2_prob_isin = group_mask_gen(anchor2_prob)
            anchor2_prob = self.mask_sparse_tensor(anchor2_prob,anchor2_prob_isin[1])
            
        if coding in ['all', 'anchor3']:
            prob_anchor3 = self.mask_sparse_tensor(prob,anchor_prob_list[2])
            dec_anchor2 = isin(prob.C,anchor_list[1].C)
            prob_anchor2_dec = self.mask_sparse_tensor(prob,dec_anchor2)
            anchor12_dec = self.union(prob_anchor1_dec,prob_anchor2_dec)
            anchor3_input = self.union(anchor12_dec,prob_anchor3)
            anchor3_prob = self.anchor3_coder(anchor3_input)
            anchor3_prob_isin = group_mask_gen(anchor3_prob)
            anchor3_prob = self.mask_sparse_tensor(anchor3_prob,anchor3_prob_isin[2])
            
        if coding in ['all', 'anchor4']:
            prob_anchor4 = self.mask_sparse_tensor(prob,anchor_prob_list[3])
            dec_anchor3 = isin(prob.C,anchor_list[2].C)
            prob_anchor3_dec = self.mask_sparse_tensor(prob,dec_anchor3)
            anchor123_dec = self.union(anchor12_dec,prob_anchor3_dec)
            anchor4_input = self.union(anchor123_dec,prob_anchor4)
            anchor4_prob = self.anchor4_coder(anchor4_input)
            anchor4_prob_isin = group_mask_gen(anchor4_prob)
            anchor4_prob = self.mask_sparse_tensor(anchor4_prob,anchor4_prob_isin[3])
            
        if coding in ['all', 'anchor5']:
            prob_anchor5 = self.mask_sparse_tensor(prob,anchor_prob_list[4])
            dec_anchor4 = isin(prob.C,anchor_list[3].C)
            prob_anchor4_dec = self.mask_sparse_tensor(prob,dec_anchor4)
            anchor1234_dec = self.union(anchor123_dec,prob_anchor4_dec)
            anchor5_input = self.union(anchor1234_dec,prob_anchor5)
            anchor5_prob = self.anchor5_coder(anchor5_input)
            anchor5_prob_isin = group_mask_gen(anchor5_prob)
            anchor5_prob = self.mask_sparse_tensor(anchor5_prob,anchor5_prob_isin[4])
            
        if coding in ['all', 'anchor6']:
            prob_anchor6 = self.mask_sparse_tensor(prob,anchor_prob_list[5])
            dec_anchor5 = isin(prob.C,anchor_list[4].C)
            prob_anchor5_dec = self.mask_sparse_tensor(prob,dec_anchor5)
            anchor12345_dec = self.union(anchor1234_dec,prob_anchor5_dec)
            anchor6_input = self.union(anchor12345_dec,prob_anchor6)
            anchor6_prob = self.anchor6_coder(anchor6_input)
            anchor6_prob_isin = group_mask_gen(anchor6_prob)
            anchor6_prob = self.mask_sparse_tensor(anchor6_prob,anchor6_prob_isin[5])
            

        probs = {
                'anchor1_1': anchor1_prob['prob_anchor1'],
                'anchor1_2': anchor1_prob['prob_anchor2'],
                'anchor1_3': anchor1_prob['prob_anchor3'],
                'anchor2': anchor2_prob,
                'anchor3': anchor3_prob,
                'anchor4': anchor4_prob,
                'anchor5': anchor5_prob,
                'anchor6': anchor6_prob,
                
        }
        labels = {
                'anchor1_1': anchor1_label['label_anchor1'],
                'anchor1_2': anchor1_label['label_anchor2'],
                'anchor1_3': anchor1_label['label_anchor3'],
                'anchor2': anchor_list[1],
                'anchor3': anchor_list[2],
                'anchor4': anchor_list[3],
                'anchor5': anchor_list[4],
                'anchor6': anchor_list[5],
        }  

            
        
        return probs, labels 
    

    def forward_8stage(self, x, partition_again = False, coding='all', attention_enhanced=False):
        anchor_list = self.generate_label_8stage(x)
        x_down = self.down(x)
        last_POV = ME.SparseTensor(
            features=torch.ones_like(x_down.F), coordinates=x_down.C, 
            tensor_stride=x_down.tensor_stride,device = x.device
        ) 
        prob = self.dfa(last_POV)
        prob = self.up(prob)
        anchor_prob_list = group_mask_gen_8stage(prob)
        anchor_prob = []
        dec_point = None
        for i in range(len(anchor_list)):
            stage_anchor = self.mask_sparse_tensor(prob,anchor_prob_list[i])
            now_stage_dec = isin(prob.C,anchor_list[i].C)
            prob_now_stage_dec = self.mask_sparse_tensor(prob,now_stage_dec)
            if(dec_point!=None):
                stage_input = self.union(dec_point,stage_anchor)
                dec_point = self.union(dec_point,prob_now_stage_dec)
            else:
                stage_input = stage_anchor
                dec_point = prob_now_stage_dec
            stage_output = self.anchor1_coder(stage_input)
            stage_isin = group_mask_gen_8stage(stage_output)
            stage_anchor_prob = self.mask_sparse_tensor(stage_output,stage_isin[i])
            anchor_prob.append(stage_anchor_prob)
        probs = anchor_prob
        labels = anchor_list
        return probs, labels

    
    def anchor1_compression(self, x, anchor_label_last, partition_again):
        last_POV = x
        ############################re par#####################################
        x_repar = group_mask_gen_par(last_POV)
        label_repar = group_mask_gen_par(anchor_label_last)
        x_repar_anchor1 = self.mask_sparse_tensor(last_POV,x_repar[0])
        x_repar_anchor2 = self.mask_sparse_tensor(last_POV,x_repar[1])
        x_repar_anchor3 = self.mask_sparse_tensor(last_POV,x_repar[2])
        x_label_anchor1 = self.mask_sparse_tensor(anchor_label_last,label_repar[0])
        x_label_anchor2 = self.mask_sparse_tensor(anchor_label_last,label_repar[1])
        x_label_anchor3 = self.mask_sparse_tensor(anchor_label_last,label_repar[2])

        # stage1 encode
        x_repar_anchor1_prob = self.anchor1_coder(x_repar_anchor1)
        # stage1 decode
        dec_anchor1 = isin(last_POV.C,x_label_anchor1.C)
        prob_anchor1_dec = self.mask_sparse_tensor(last_POV,dec_anchor1)
        # stage2 encode
        anchor2_input = self.union(x_repar_anchor2,prob_anchor1_dec)
        x_repar_anchor2_prob = self.anchor1_2_coder(anchor2_input)
        anchor2_prob_mask = group_mask_gen_par(x_repar_anchor2_prob)
        x_repar_anchor2_prob = self.mask_sparse_tensor(x_repar_anchor2_prob,anchor2_prob_mask[1])
        # stage2 decode
        dec_anchor2 = isin(last_POV.C,x_label_anchor2.C)
        prob_anchor2_dec = self.mask_sparse_tensor(last_POV,dec_anchor2)
            
        # stage3 encode
        anchor3_input = self.union(prob_anchor1_dec,prob_anchor2_dec)
        anchor3_input = self.union(anchor3_input,x_repar_anchor3)
        x_repar_anchor3_prob = self.anchor1_3_coder(anchor3_input)
        anchor3_prob_mask = group_mask_gen_par(x_repar_anchor3_prob)
        x_repar_anchor3_prob = self.mask_sparse_tensor(x_repar_anchor3_prob,anchor3_prob_mask[2])
        probs = {
            'prob_anchor1':x_repar_anchor1_prob,
            'prob_anchor2':x_repar_anchor2_prob,
            'prob_anchor3':x_repar_anchor3_prob,
        }
        labels = {
            'label_anchor1':x_label_anchor1,
            'label_anchor2':x_label_anchor2,
            'label_anchor3':x_label_anchor3,       
        }
        return probs,labels
     
    
    def mask_sparse_tensor(self, sparse_tensor, mask, C = None, tensor_stride=None, same_manager=False, coord_manag = None ):
        bool_manag_1 = same_manager and (coord_manag is None)
        bool_manag_2 = not same_manager 
        assert bool_manag_1 or bool_manag_2
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
            if same_manager:
                return ME.SparseTensor(
                    features=newF, coordinates=newC, tensor_stride=new_tensor_stride, coordinate_manager = sparse_tensor.coordinate_manager 
                )
            else:
                return_mk = ME.SparseTensor(
                    features=newF, coordinates=newC, tensor_stride=new_tensor_stride
                ) if coord_manag is None else ME.SparseTensor(
                    features=newF, coordinates=newC, tensor_stride=new_tensor_stride, coordinate_manager = coord_manag
                ) 
                return return_mk

    def generate_label(self, x):
        anchor_list = group_mask_gen(x)
        return_anchor = []
        for i in range(len(anchor_list)):
            return_anchor.append(self.mask_sparse_tensor(x, anchor_list[i]))
        return  return_anchor
    
    def generate_label_8stage(self, x):
        anchor_list = group_mask_gen_8stage(x)
        return_anchor = []
        for i in range(len(anchor_list)):
            return_anchor.append(self.mask_sparse_tensor(x, anchor_list[i]))
        return  return_anchor

if __name__ == '__main__':
    model = UELC()
    print(model)

