import os, sys, glob
import time
from tqdm import tqdm
import numpy as np
import h5py
import torch
import torch.utils.data
from torch.utils.data.sampler import Sampler
import MinkowskiEngine as ME
from data_utils import read_h5_geo, read_ply_ascii_geo, read_bin ,read_ply_float_geo
import random
from quantize import quantize_precision,quantize_resolution,quantize_octree, random_quantize, merge_points

class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)


def collate_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1
    list_data = new_list_data
    if len(list_data) == 0:
        raise ValueError('No data in the batch')
    coords, feats = list(zip(*list_data))
    coords_batch, feats_batch = ME.utils.sparse_collate(coords, feats)

    return coords_batch, feats_batch



class PCDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None, voxel_size=1, resolution=None, qlevel=None, max_num=1e7, augment=False):
        self.files = []
        self.cache = {}
        self.files = files
        self.transforms = transforms
        assert voxel_size is None or resolution is None
        self.voxel_size = voxel_size
        self.resolution = resolution
        self.qlevel = qlevel
        self.max_num = max_num
        self.augment = augment

    def __len__(self):

        return len(self.files)
    
    def __getitem__(self, idx):
        filedir = self.files[idx]
        if filedir.endswith('bin'):
            self.voxel_size = None
            self.resolution = None
            self.qlevel = 12
        if idx in self.cache:
            coords, feats = self.cache[idx]
        else:
            # import time
            # start = time.time()
            coords = read_coords(filedir)
            # coords = quantize_precision(coords, precision=self.voxel_size, quant_mode='round', return_offset=False)
            if self.voxel_size is not None:
                coords = quantize_precision(coords, precision=self.voxel_size, quant_mode='round', return_offset=False)
            elif self.resolution is not None:
                coords, _, _ = quantize_resolution(coords, resolution=self.resolution, quant_mode='round', return_offset=False)
            elif self.qlevel is not None:
                coords, _, _, _ = quantize_octree(coords, qlevel=self.qlevel, quant_mode='round', return_offset=False)
            coords = np.unique(coords.astype('int'), axis=0).astype('int')
            # print('DBG!!! loading time', round(time.time() - start, 4), filedir, len(coords), coords.max() - coords.min())
            # print('DBG!!! loading', len(coords), coords.max() - coords.min())
            if self.augment: 
                coords = random_quantize(coords)
                # print('DBG!!! augment', coords.max() - coords.min())
            if len(coords) > self.max_num:
                print('DBG', len(coords), self.max_num)
                parts = kdtree_partition(coords, max_num=self.max_num)
                coords = random.sample(parts, 1)[0]
                print('DBG!!! partition', len(parts), len(coords))
            # # transform
            # if self.transforms is not None:
            #     for trans in self.transforms:
            #         coords = trans(coords)
            feats = np.ones([len(coords), 1]).astype('bool')
            self.cache[idx] = (coords, feats)
        feats = feats.astype("float32")

        return (coords, feats)


from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def make_data_loader(dataset, batch_size=1, shuffle=True, num_workers=4, repeat=False, 
                    collate_fn=collate_pointcloud_fn):
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': True,
        'drop_last': False
    }
    if repeat:
        args['sampler'] = InfSampler(dataset, shuffle)
    else:
        args['shuffle'] = shuffle
    loader = DataLoaderX(dataset, **args)

    return loader



def read_bin(filedir, dtype="float32"):
    """kitti
    """
    data = np.fromfile(filedir, dtype=dtype).reshape(-1, 4)
    coords = data[:,:3]
    
    return coords

def read_coords(filedir):
    if filedir.endswith('h5'): coords = read_h5_geo(filedir)
    if filedir.endswith('ply'): coords = read_ply_float_geo(filedir)
    if filedir.endswith('bin'): coords = read_bin(filedir)

    return coords






class PCDatasetOffset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None, voxel_size=None, resolution=1023, qlevel=None, max_num=1e7, augment=False):
        self.files = []
        self.cache = {}
        self.files = files
        self.transforms = transforms
        assert voxel_size is None or resolution is None
        self.voxel_size = voxel_size
        self.resolution = resolution
        self.qlevel = qlevel
        self.max_num = max_num
        self.augment = augment

    def __len__(self):

        return len(self.files)
    
    def __getitem__(self, idx):
        filedir = self.files[idx]
        if filedir.endswith('bin'):
            self.voxel_size = None
            # self.resolution = 2**12-1
            self.qlevel = None
        if idx in self.cache:
            coords, feats = self.cache[idx]
        else:
            coords = read_coords(filedir)
            if self.voxel_size is not None:
                coords, offset = quantize_precision(coords, precision=self.voxel_size, quant_mode='round', return_offset=True)
            elif self.resolution is not None:
                coords, _, _, offset = quantize_resolution(coords, resolution=self.resolution, quant_mode='round', return_offset=True)
            elif self.qlevel is not None:
                coords, _, _, _, offset = quantize_octree(coords, qlevel=self.qlevel, quant_mode='round', return_offset=True)
            coords, offset = merge_points(coords, offset)
            feats = offset.astype("float32")
            self.cache[idx] = (coords, feats)

        return (coords, feats)





if __name__ == "__main__":
    # filedirs = sorted(glob.glob('/home/ubuntu/HardDisk2/color_training_datasets/training_dataset/'+'*.h5'))
    filedirs = sorted(glob.glob('/home/ubuntu/HardDisk1/point_cloud_testing_datasets/8i_voxeilzaed_full_bodies/8i/longdress/Ply/'+'*.ply'))
    test_dataset = PCDataset(filedirs[:10])
    test_dataloader = make_data_loader(dataset=test_dataset, batch_size=2, shuffle=True, num_workers=1, repeat=False,
                                        collate_fn=collate_pointcloud_fn)
    for idx, (coords, feats) in enumerate(tqdm(test_dataloader)):
        print("="*20, "check dataset", "="*20, 
            "\ncoords:\n", coords, "\nfeat:\n", feats)

    test_iter = iter(test_dataloader)
    print(test_iter)
    for i in tqdm(range(10)):
        coords, feats = test_iter.next()
        print("="*20, "check dataset", "="*20, 
            "\ncoords:\n", coords, "\nfeat:\n", feats)
