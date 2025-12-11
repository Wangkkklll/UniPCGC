import torch
from data_utils import isin, istopk
criterion = torch.nn.BCEWithLogitsLoss()

"""
Thanks for PCGCv2 code: https://github.com/NJUVISION/PCGCv2
"""

def get_bits(probF, mask):
    mask = isin(probF.C,mask.C)
    probF = probF.F
    likelihood = torch.nn.functional.binary_cross_entropy(torch.sigmoid(probF.squeeze()), 
            mask.squeeze().type(probF.dtype), reduction='none')
    likelihood /= torch.log(torch.tensor(2.0)).to(likelihood.device)
    bits = torch.sum(likelihood)
    return bits

def get_bits_lossy(likelihood):
    bits = -torch.sum(torch.log2(likelihood))

    return bits

def get_bce(data, groud_truth):
    """ Input data and ground_truth are sparse tensor.
    """
    mask = isin(data.C, groud_truth.C) 
    bce = criterion(data.F.squeeze(), mask.type(data.F.dtype)) 
    bce /= torch.log(torch.tensor(2.0)).to(bce.device)
    sum_bce = bce * data.shape[0]
    
    return sum_bce


def get_metrics(data, groud_truth):
    mask_real = isin(data.C, groud_truth.C)
    nums = [len(C) for C in groud_truth.decomposed_coordinates]
    mask_pred = istopk(data, nums, rho=1.0)
    metrics = get_cls_metrics(mask_pred, mask_real)

    return metrics[0]

def get_cls_metrics(pred, real):
    TP = (pred * real).cpu().nonzero(as_tuple=False).shape[0]
    FN = (~pred * real).cpu().nonzero(as_tuple=False).shape[0]
    FP = (pred * ~real).cpu().nonzero(as_tuple=False).shape[0]
    TN = (~pred * ~real).cpu().nonzero(as_tuple=False).shape[0]

    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    IoU = TP / (TP + FP + FN + 1e-7)

    return [round(precision, 4), round(recall, 4), round(IoU, 4)]


