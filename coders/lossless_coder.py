import os
import numpy as np
import torch
import MinkowskiEngine as ME
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from data_utils import sort_spare_tensor
from data_utils import group_mask_gen,group_mask_gen_par,group_mask_gen_8stage,isin
from coders.utils import encode_prob_multi, wirte_encoded_binary_file, decode_prob_multi


class Coder():
    def __init__(self, model, filename,  fake_bpp = False):
        self.model = model 
        self.filename = filename
        self.sigmoid = ME.MinkowskiSigmoid()


    @torch.no_grad()
    def model_input_ME(self, coords, tensor_stride=1, feats = None, coord_manag=None ):
        if feats is None:
            feats = torch.ones((len(coords),1)).float()
        feats = feats.float()
        coords, feats = ME.utils.sparse_collate([coords], [feats])
        if coord_manag is None:
            x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=tensor_stride, device=device)
        else:
            x = ME.SparseTensor(features=feats, coordinates=coords, coordinate_manager=coord_manag,
                                tensor_stride=tensor_stride, device=device)
        return x
    
    def encode_according_to_F_and_label(self, F, label, path):    
        sorted_probs = torch.sigmoid(F.F).detach().clone().cpu()
        sorted_probs2 = 1-sorted_probs
        sorted_probs = torch.cat([sorted_probs2,sorted_probs],dim=1)
        sorted_labels = isin(F.C,label.C).int()
        cdf, byte_stream, real_bits = encode_prob_multi(sorted_probs, sorted_labels)
        wirte_encoded_binary_file(path = path, byte_stream = byte_stream)
        return real_bits
            
    def decode_label_according_to_F(self, binpath, F):          
        sorted_probs = torch.sigmoid(F.F).detach().clone().cpu()
        sorted_probs2 = 1-sorted_probs
        sorted_probs = torch.cat([sorted_probs2,sorted_probs],dim=1)
        sym = decode_prob_multi(sorted_probs, binpath)
        return sym
      

    @torch.no_grad()
    def encode(self, x, postfix='test'):
        
        prefix = self.filename + postfix + '/'
        if not os.path.exists (prefix): os.makedirs (prefix)
        assert torch.is_tensor(x)
        number_of_points = len(x)
        print("number_of_points: ",number_of_points)
        total_bits = 0
        T = 0
        while True:
            # encoding_this_scale : x
            print("T:",T)
            model_input = self.model_input_ME(x, 2**T)
            probs, labels = self.model(model_input, partition_again=False, attention_enhanced=True)
            for item in ['anchor1_1', 'anchor1_2','anchor1_3','anchor2','anchor3', 'anchor4','anchor5','anchor6']:
                probs_mk = (probs[item])
                labels_mk = (labels[item])
                sorted_probs_MK =  sort_spare_tensor(probs_mk)
                sorted_labels_MK = sort_spare_tensor(labels_mk)
                real_bits = self.encode_according_to_F_and_label(
                        sorted_probs_MK, sorted_labels_MK, 
                        path = prefix+str(T)+"_"+item+".bin"
                    )
                if T<=-1:
                    total_bits +=0
                else:
                    total_bits +=  real_bits
            x = self.model.down(model_input)
            x = x.C[:,1:]
            
            if len(x) <= 16: break
            T += 1
        T_ue = T
        T += 1
        # low scale # uniform grouping
        while True:
            print("T:",T)
            model_input = self.model_input_ME(x, 2**T)
            partition_again= False
            attention_enhanced = True if T <= 6 else False
            probs, labels = self.model.forward_8stage(model_input, partition_again=partition_again, attention_enhanced=attention_enhanced) 
            k = 0 
            for item in ['anchor1_1', 'anchor1_2','anchor1_3','anchor2','anchor3', 'anchor4','anchor5','anchor6']:
                
                probs_mk = (probs[k])
                labels_mk = (labels[k])
                sorted_probs_MK =  sort_spare_tensor(probs_mk)
                sorted_labels_MK = sort_spare_tensor(labels_mk)
                real_bits = self.encode_according_to_F_and_label(
                        sorted_probs_MK, sorted_labels_MK, 
                        path = prefix+str(T)+"_"+item+".bin"
                    )
                k+=1
                if T<=-1:
                    total_bits +=0
                else:
                    total_bits +=  real_bits
            x = self.model.down(model_input)
            x = x.C[:,1:]
            
            if len(x) <= 4: break
            T += 1
        initial_x = x.cpu().numpy()
        scale = np.array([T,T_ue])
        with open(prefix+'scale.bin', 'wb') as f:
            f.write(np.array(scale, dtype=np.int32).tobytes())
        with open(prefix+'initial_x.bin', 'wb') as f:
            f.write(np.array(initial_x, dtype=np.int32).tobytes())
        total_bits += os.path.getsize(prefix+'scale.bin')*8
        total_bits += os.path.getsize(prefix+'initial_x.bin')*8
        print("bpp: ",total_bits / number_of_points)
        # print(x)
        return total_bits

    @torch.no_grad()
    def decode(self, postfix='test'):
        prefix = self.filename + postfix + '/'
        assert os.path.exists(prefix)
        with open(prefix+'scale.bin', 'rb') as f:
            scale = np.frombuffer(f.read(4*2), dtype=np.int32)
        with open(prefix+'initial_x.bin', 'rb') as f:
            file_content = f.read()
            x_flat = np.frombuffer(file_content, dtype=np.int32)
            if len(x_flat) % 3 == 0:
                x_numpy = x_flat.reshape(-1, 3)
            else:
                x_numpy = x_flat.reshape(-1, 3)  # 或其他形状
            x = torch.tensor(x_numpy).to(device)
            x = torch.cat([torch.zeros(x.shape[0], 1).to(device), x], dim=1)
        # print(x)
      
        T_high = scale[0] 
        T_ue = scale[1]
        T = T_high
        NOW_tensor_stride = 2 **(T+1)
        # print(x.shape)
        # print(x)
        # x = torch.tensor(x).to(device)
        # x = torch.cat([torch.zeros(x.shape[0],1).to(device),x],dim=-1)
        # print(x)
        model_input = ME.SparseTensor(features=torch.ones(x.shape[0],1), coordinates=x, tensor_stride=NOW_tensor_stride, device=device)
        while T >= 0:
            print("T:", T)
            if(T<=T_ue):
                #up
                prob = self.model.dfa(model_input)
                prob = self.model.up(prob)
                anchor_prob_list = group_mask_gen(prob)
                for item in ['anchor1_1', 'anchor1_2','anchor1_3','anchor2','anchor3','anchor4','anchor5','anchor6']:
                    file_name =  prefix+str(T)+"_"+item+".bin"
                    if item == 'anchor1_1':
                        prob_anchor1 = self.model.mask_sparse_tensor(prob,anchor_prob_list[0])
                        anchor1_1 = group_mask_gen_par(prob_anchor1)
                        prob_anchor1_1 =  self.model.mask_sparse_tensor(prob_anchor1,anchor1_1[0])
                        prob_anchor1_1 = self.model.anchor1_coder(prob_anchor1_1)
                        prob_anchor1_1 = sort_spare_tensor(prob_anchor1_1)
                        sym1 = self.decode_label_according_to_F(file_name, prob_anchor1_1)
                        anchor1_label = self.model_input_ME(prob_anchor1_1.C[:,1:], prob.tensor_stride[0], sym1)
                        if sym1[0][0]==1 and anchor1_label.shape[0]==1:
                            print("anchor1_label do not need mask")
                        else:
                            anchor1_label = self.model.mask_sparse_tensor(anchor1_label,sym1.squeeze().bool())
                        
                    elif item =='anchor1_2':
                        prob_anchor2 = self.model.mask_sparse_tensor(prob_anchor1,anchor1_1[1])
                        dec_anchor1 = isin(prob.C,anchor1_label.C)
                        prob_anchor1_dec = self.model.mask_sparse_tensor(prob,dec_anchor1)
                        anchor2_input = self.model.union(prob_anchor2,prob_anchor1_dec)
                        anchor2_prob = self.model.anchor1_2_coder(anchor2_input)
                        anchor2_prob_isin = group_mask_gen_par(anchor2_prob)
                        anchor2_prob = self.model.mask_sparse_tensor(anchor2_prob,anchor2_prob_isin[1])
                        sym2 = self.decode_label_according_to_F(file_name, anchor2_prob)
                        anchor2_label = self.model_input_ME(anchor2_prob.C[:,1:], prob.tensor_stride[0], sym2)
                        anchor2_label = self.model.mask_sparse_tensor(anchor2_label,sym2.squeeze().bool())
                        
                    elif item =='anchor1_3':
                        prob_anchor3 = self.model.mask_sparse_tensor(prob_anchor1,anchor1_1[2])
                        dec_anchor2 = isin(prob.C,anchor2_label.C)
                        prob_anchor2_dec = self.model.mask_sparse_tensor(prob,dec_anchor2)
                        anchor12_dec = self.model.union(prob_anchor1_dec,prob_anchor2_dec)

                        anchor3_input = self.model.union(prob_anchor3,anchor12_dec)
                        anchor3_prob = self.model.anchor1_3_coder(anchor3_input)
                        anchor3_prob_isin = group_mask_gen_par(anchor3_prob)
                        anchor3_prob = self.model.mask_sparse_tensor(anchor3_prob,anchor3_prob_isin[2])
                        sym3 = self.decode_label_according_to_F(file_name, anchor3_prob)
                        anchor3_label = self.model_input_ME(anchor3_prob.C[:,1:], prob.tensor_stride[0], sym3)
                        anchor3_label = self.model.mask_sparse_tensor(anchor3_label,sym3.squeeze().bool())
                        
                    elif item =='anchor2':
                        prob_anchor4 = self.model.mask_sparse_tensor(prob,anchor_prob_list[1])
                        dec_anchor3 = isin(prob.C,anchor3_label.C)
                        prob_anchor3_dec = self.model.mask_sparse_tensor(prob,dec_anchor3)
                        anchor123_dec = self.model.union(anchor12_dec,prob_anchor3_dec)

                        anchor4_input = self.model.union(prob_anchor4,anchor123_dec)
                        anchor4_prob = self.model.anchor2_coder(anchor4_input)
                        anchor4_prob_isin = group_mask_gen(anchor4_prob)
                        anchor4_prob = self.model.mask_sparse_tensor(anchor4_prob,anchor4_prob_isin[1])
                        sym4 = self.decode_label_according_to_F(file_name, anchor4_prob)
                        anchor4_label = self.model_input_ME(anchor4_prob.C[:,1:], prob.tensor_stride[0], sym4)
                        anchor4_label = self.model.mask_sparse_tensor(anchor4_label,sym4.squeeze().bool())
                        
                    elif item =='anchor3':
                        prob_anchor5 = self.model.mask_sparse_tensor(prob,anchor_prob_list[2])
                        dec_anchor4 = isin(prob.C,anchor4_label.C)
                        prob_anchor4_dec = self.model.mask_sparse_tensor(prob,dec_anchor4)
                        anchor1234_dec = self.model.union(anchor123_dec,prob_anchor4_dec)

                        anchor5_input = self.model.union(prob_anchor5,anchor1234_dec)
                        anchor5_prob = self.model.anchor3_coder(anchor5_input)
                        anchor5_prob_isin = group_mask_gen(anchor5_prob)
                        anchor5_prob = self.model.mask_sparse_tensor(anchor5_prob,anchor5_prob_isin[2])
                        sym5 = self.decode_label_according_to_F(file_name, anchor5_prob)
                        anchor5_label = self.model_input_ME(anchor5_prob.C[:,1:], prob.tensor_stride[0], sym5)
                        anchor5_label = self.model.mask_sparse_tensor(anchor5_label,sym5.squeeze().bool())
                    elif item =='anchor4':
                        prob_anchor6 = self.model.mask_sparse_tensor(prob,anchor_prob_list[3])
                        dec_anchor5 = isin(prob.C,anchor5_label.C)
                        prob_anchor5_dec = self.model.mask_sparse_tensor(prob,dec_anchor5)
                        anchor12345_dec = self.model.union(anchor1234_dec,prob_anchor5_dec)

                        anchor6_input = self.model.union(prob_anchor6,anchor12345_dec)
                        anchor6_prob = self.model.anchor4_coder(anchor6_input)
                        anchor6_prob_isin = group_mask_gen(anchor6_prob)
                        anchor6_prob = self.model.mask_sparse_tensor(anchor6_prob,anchor6_prob_isin[3])
                        sym6 = self.decode_label_according_to_F(file_name, anchor6_prob)
                        anchor6_label = self.model_input_ME(anchor6_prob.C[:,1:], prob.tensor_stride[0], sym6)
                        anchor6_label = self.model.mask_sparse_tensor(anchor6_label,sym6.squeeze().bool())
                    elif item =='anchor5':
                        prob_anchor7 = self.model.mask_sparse_tensor(prob,anchor_prob_list[4])
                        dec_anchor6 = isin(prob.C,anchor6_label.C)
                        prob_anchor6_dec = self.model.mask_sparse_tensor(prob,dec_anchor6)
                        anchor123456_dec = self.model.union(anchor12345_dec,prob_anchor6_dec)

                        anchor7_input = self.model.union(prob_anchor7,anchor123456_dec)
                        anchor7_prob = self.model.anchor5_coder(anchor7_input)
                        anchor7_prob_isin = group_mask_gen(anchor7_prob)
                        anchor7_prob = self.model.mask_sparse_tensor(anchor7_prob,anchor7_prob_isin[4])
                        sym7 = self.decode_label_according_to_F(file_name, anchor7_prob)
                        anchor7_label = self.model_input_ME(anchor7_prob.C[:,1:], prob.tensor_stride[0], sym7)
                        anchor7_label = self.model.mask_sparse_tensor(anchor7_label,sym7.squeeze().bool())
                    elif item =='anchor6':
                        prob_anchor8 = self.model.mask_sparse_tensor(prob,anchor_prob_list[5])
                        dec_anchor7 = isin(prob.C,anchor7_label.C)
                        prob_anchor7_dec = self.model.mask_sparse_tensor(prob,dec_anchor7)
                        anchor1234567_dec = self.model.union(anchor123456_dec,prob_anchor7_dec)

                        anchor8_input = self.model.union(prob_anchor8,anchor1234567_dec)
                        #print(anchor8_input)
                        anchor8_prob = self.model.anchor6_coder(anchor8_input)
                        anchor8_prob_isin = group_mask_gen(anchor8_prob)
                        anchor8_prob = self.model.mask_sparse_tensor(anchor8_prob,anchor8_prob_isin[5])
                        sym8 = self.decode_label_according_to_F(file_name, anchor8_prob)
                        anchor8_label = self.model_input_ME(anchor8_prob.C[:,1:], prob.tensor_stride[0], sym8)
                        anchor8_label = self.model.mask_sparse_tensor(anchor8_label,sym8.squeeze().bool())
                        
                this_layer_pov = torch.cat([anchor1_label.C[:,1:], anchor2_label.C[:,1:],anchor3_label.C[:,1:],anchor4_label.C[:,1:],anchor5_label.C[:,1:], anchor6_label.C[:,1:],anchor7_label.C[:,1:],anchor8_label.C[:,1:]], dim=0)
                T -= 1 
                model_input = self.model_input_ME(this_layer_pov, prob.tensor_stride[0])
                model_input = sort_spare_tensor(model_input)

            else:             
                prob = None
                prob = self.model.dfa(model_input)
                prob = self.model.up(prob)
                anchor_prob_list = group_mask_gen_8stage(prob)
                for item in ['anchor1_1', 'anchor1_2','anchor1_3','anchor2','anchor3','anchor4','anchor5','anchor6']:
                    # print(item)
                    file_name =  prefix+str(T)+"_"+item+".bin"
                    
                    if item == 'anchor1_1':
                        prob_anchor1 = self.model.mask_sparse_tensor(prob,anchor_prob_list[0])
                        anchor1_1 = group_mask_gen_8stage(prob_anchor1)
                        prob_anchor1_1 =  self.model.mask_sparse_tensor(prob_anchor1,anchor1_1[0])
                        prob_anchor1_1 = self.model.anchor1_coder(prob_anchor1_1)
                        prob_anchor1_1 = sort_spare_tensor(prob_anchor1_1)
                        sym1 = self.decode_label_according_to_F(file_name, prob_anchor1_1)
                        
                        anchor1_label = self.model_input_ME(prob_anchor1_1.C[:,1:], prob.tensor_stride[0], sym1)
                        anchor1_label = self.model.mask_sparse_tensor(anchor1_label,sym1.squeeze().bool())
                        
                    elif item =='anchor1_2':
                        prob_anchor2 = self.model.mask_sparse_tensor(prob,anchor_prob_list[1])
                        dec_anchor1 = isin(prob.C,anchor1_label.C)
                        prob_anchor1_dec = self.model.mask_sparse_tensor(prob,dec_anchor1)
                        anchor2_input = self.model.union(prob_anchor2,prob_anchor1_dec)
                        anchor2_prob = self.model.anchor1_coder(anchor2_input)
                        anchor2_prob_isin = group_mask_gen_8stage(anchor2_prob)
                        anchor2_prob = self.model.mask_sparse_tensor(anchor2_prob,anchor2_prob_isin[1])
                        sym2 = self.decode_label_according_to_F(file_name, anchor2_prob)
                        anchor2_label = self.model_input_ME(anchor2_prob.C[:,1:], prob.tensor_stride[0], sym2)
                        anchor2_label = self.model.mask_sparse_tensor(anchor2_label,sym2.squeeze().bool())
                        
                    elif item =='anchor1_3':
                        prob_anchor3 = self.model.mask_sparse_tensor(prob,anchor_prob_list[2])
                        dec_anchor2 = isin(prob.C,anchor2_label.C)
                        prob_anchor2_dec = self.model.mask_sparse_tensor(prob,dec_anchor2)
                        anchor12_dec = self.model.union(prob_anchor1_dec,prob_anchor2_dec)

                        anchor3_input = self.model.union(prob_anchor3,anchor12_dec)
                        anchor3_prob = self.model.anchor1_coder(anchor3_input)
                        anchor3_prob_isin = group_mask_gen_8stage(anchor3_prob)
                        anchor3_prob = self.model.mask_sparse_tensor(anchor3_prob,anchor3_prob_isin[2])
                        sym3 = self.decode_label_according_to_F(file_name, anchor3_prob)
                        anchor3_label = self.model_input_ME(anchor3_prob.C[:,1:], prob.tensor_stride[0], sym3)
                        anchor3_label = self.model.mask_sparse_tensor(anchor3_label,sym3.squeeze().bool())
                        
                    elif item =='anchor2':
                        prob_anchor4 = self.model.mask_sparse_tensor(prob,anchor_prob_list[3])
                        dec_anchor3 = isin(prob.C,anchor3_label.C)
                        prob_anchor3_dec = self.model.mask_sparse_tensor(prob,dec_anchor3)
                        anchor123_dec = self.model.union(anchor12_dec,prob_anchor3_dec)

                        anchor4_input = self.model.union(prob_anchor4,anchor123_dec)
                        anchor4_prob = self.model.anchor1_coder(anchor4_input)
                        anchor4_prob_isin = group_mask_gen_8stage(anchor4_prob)
                        anchor4_prob = self.model.mask_sparse_tensor(anchor4_prob,anchor4_prob_isin[3])
                        sym4 = self.decode_label_according_to_F(file_name, anchor4_prob)
                        anchor4_label = self.model_input_ME(anchor4_prob.C[:,1:], prob.tensor_stride[0], sym4)
                        anchor4_label = self.model.mask_sparse_tensor(anchor4_label,sym4.squeeze().bool())
                        
                    elif item =='anchor3':
                        prob_anchor5 = self.model.mask_sparse_tensor(prob,anchor_prob_list[4])
                        dec_anchor4 = isin(prob.C,anchor4_label.C)
                        prob_anchor4_dec = self.model.mask_sparse_tensor(prob,dec_anchor4)
                        anchor1234_dec = self.model.union(anchor123_dec,prob_anchor4_dec)

                        anchor5_input = self.model.union(prob_anchor5,anchor1234_dec)
                        anchor5_prob = self.model.anchor1_coder(anchor5_input)
                        anchor5_prob_isin = group_mask_gen_8stage(anchor5_prob)
                        anchor5_prob = self.model.mask_sparse_tensor(anchor5_prob,anchor5_prob_isin[4])
                        sym5 = self.decode_label_according_to_F(file_name, anchor5_prob)
                        anchor5_label = self.model_input_ME(anchor5_prob.C[:,1:], prob.tensor_stride[0], sym5)
                        anchor5_label = self.model.mask_sparse_tensor(anchor5_label,sym5.squeeze().bool())
                    elif item =='anchor4':
                        prob_anchor6 = self.model.mask_sparse_tensor(prob,anchor_prob_list[5])
                        dec_anchor5 = isin(prob.C,anchor5_label.C)
                        prob_anchor5_dec = self.model.mask_sparse_tensor(prob,dec_anchor5)
                        anchor12345_dec = self.model.union(anchor1234_dec,prob_anchor5_dec)

                        anchor6_input = self.model.union(prob_anchor6,anchor12345_dec)
                        anchor6_prob = self.model.anchor1_coder(anchor6_input)
                        anchor6_prob_isin = group_mask_gen_8stage(anchor6_prob)
                        anchor6_prob = self.model.mask_sparse_tensor(anchor6_prob,anchor6_prob_isin[5])
                        sym6 = self.decode_label_according_to_F(file_name, anchor6_prob)
                        anchor6_label = self.model_input_ME(anchor6_prob.C[:,1:], prob.tensor_stride[0], sym6)
                        anchor6_label = self.model.mask_sparse_tensor(anchor6_label,sym6.squeeze().bool())
                    elif item =='anchor5':
                        prob_anchor7 = self.model.mask_sparse_tensor(prob,anchor_prob_list[6])
                        dec_anchor6 = isin(prob.C,anchor6_label.C)
                        prob_anchor6_dec = self.model.mask_sparse_tensor(prob,dec_anchor6)
                        anchor123456_dec = self.model.union(anchor12345_dec,prob_anchor6_dec)

                        anchor7_input = self.model.union(prob_anchor7,anchor123456_dec)
                        anchor7_prob = self.model.anchor1_coder(anchor7_input)
                        anchor7_prob_isin = group_mask_gen_8stage(anchor7_prob)
                        anchor7_prob = self.model.mask_sparse_tensor(anchor7_prob,anchor7_prob_isin[6])
                        sym7 = self.decode_label_according_to_F(file_name, anchor7_prob)
                        anchor7_label = self.model_input_ME(anchor7_prob.C[:,1:], prob.tensor_stride[0], sym7)
                        anchor7_label = self.model.mask_sparse_tensor(anchor7_label,sym7.squeeze().bool())
                    elif item =='anchor6':
                        prob_anchor8 = self.model.mask_sparse_tensor(prob,anchor_prob_list[7])
                        dec_anchor7 = isin(prob.C,anchor7_label.C)
                        prob_anchor7_dec = self.model.mask_sparse_tensor(prob,dec_anchor7)
                        anchor1234567_dec = self.model.union(anchor123456_dec,prob_anchor7_dec)

                        anchor8_input = self.model.union(prob_anchor8,anchor1234567_dec)
                        anchor8_prob = self.model.anchor1_coder(anchor8_input)
                        anchor8_prob_isin = group_mask_gen_8stage(anchor8_prob)
                        anchor8_prob = self.model.mask_sparse_tensor(anchor8_prob,anchor8_prob_isin[7])
                        sym8 = self.decode_label_according_to_F(file_name, anchor8_prob)
                        anchor8_label = self.model_input_ME(anchor8_prob.C[:,1:], prob.tensor_stride[0], sym8)
                        anchor8_label = self.model.mask_sparse_tensor(anchor8_label,sym8.squeeze().bool())
                        
                this_layer_pov = torch.cat([anchor1_label.C[:,1:], anchor2_label.C[:,1:],anchor3_label.C[:,1:],anchor4_label.C[:,1:],anchor5_label.C[:,1:], anchor6_label.C[:,1:],anchor7_label.C[:,1:],anchor8_label.C[:,1:]], dim=0)
                T -= 1 
                model_input = self.model_input_ME(this_layer_pov, prob.tensor_stride[0])
                model_input = sort_spare_tensor(model_input)

        return model_input
    


if __name__ == '__main__':
    print("coder")