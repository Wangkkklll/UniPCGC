import torchac
import torch
import os

def pmf_to_cdf(pmf):
    cdf = pmf.cumsum(dim=-1)
    spatial_dimensions = pmf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    cdf_with_0 = cdf_with_0.clamp(max=1.)
    return cdf_with_0

def encode_prob_multi(prob, sym):
    device = torch.device('cpu')
    prob = prob.to(device)
    sym = sym.short().to(device)
    cdf = pmf_to_cdf(prob)
    byte_stream = torchac.encode_float_cdf(cdf, sym, check_input_bounds=True)
    real_bits = len(byte_stream) * 8
    return cdf, byte_stream, real_bits

def decode_prob_multi(prob, path):
    device = torch.device('cpu')
    prob = prob.to(device)
    cdf = pmf_to_cdf(prob)
    byte_stream = read_encoded_binary_file(path)
    sym = torchac.decode_float_cdf(cdf, byte_stream)
    if len(sym.shape) < 2:
        sym = sym.unsqueeze(1)
    return sym

def wirte_encoded_binary_file(path, byte_stream):
    if os.path.exists(path):
        os.system("rm "+path)
    with open(path, 'wb') as fout:
        fout.write(byte_stream)
    return

def read_encoded_binary_file(path):
    
    with open(path, 'rb') as fout:
        content = fout.read()
    
    return content 
