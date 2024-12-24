import os
import numpy as np
import h5py
from plyfile import PlyData, PlyElement
import pandas as pd
import torch
import MinkowskiEngine as ME
from PIL import Image

# import PlyData
def read_h5_geo(filedir):
    pc = h5py.File(filedir, 'r')['data'][:]
    coords = pc[:,0:3].astype('int')

    return coords


def read_bin(filedir, dtype="float32"):
    """kitti
    """
    data = np.fromfile(filedir, dtype=dtype).reshape(-1, 4)
    coords = data[:,:3]
    
    return coords

def write_h5_geo(filedir, coords):
    data = coords.astype('uint8')
    with h5py.File(filedir, 'w') as h:
        h.create_dataset('data', data=data, shape=data.shape)

    return

def read_ply_ascii_geo(filedir):

    plydata = PlyData.read(filedir)
    data = plydata.elements[0].data

    data_pd = pd.DataFrame(data)  
    data_np = np.zeros(data_pd.shape, dtype=np.float64)  # 
    property_names = data[0].dtype.names  #
    for i, name in enumerate(property_names):  # 
        data_np[:, i] = data_pd[name]
    # print(data_np)
    
    coords = (data_np[:,0:3]).astype('int')
    # data_np[:,0:3] = coords
    # print(coords)
    # files = open(filedir)
    # data = []
    # for i, line in enumerate(files):
    #     wordslist = line.split(' ')
    #     try:
    #         line_values = []
    #         for i, v in enumerate(wordslist):
    #             if v == '\n': continue
    #             line_values.append(float(v))
    #     except ValueError: continue
    #     data.append(line_values)
    # data = np.array(data) #data[0]: [255. 39. 291. 0.505101 0.565322 0.652139 137. 62. 46. 255. ] [x y z nx ny nz r g b alpha]
    # coords = data[:,0:3].astype('int')

    return coords

def read_ply_float_geo(filedir):

    plydata = PlyData.read(filedir)
    data = plydata.elements[0].data

    data_pd = pd.DataFrame(data)  # 
    data_np = np.zeros(data_pd.shape, dtype=np.float64)  # 
    property_names = data[0].dtype.names  # 
    for i, name in enumerate(property_names):  # 
        data_np[:, i] = data_pd[name]
    # print(data_np)
    
    coords = (data_np[:,0:3])
    # data_np[:,0:3] = coords
    # print(coords)
    # files = open(filedir)
    # data = []
    # for i, line in enumerate(files):
    #     wordslist = line.split(' ')
    #     try:
    #         line_values = []
    #         for i, v in enumerate(wordslist):
    #             if v == '\n': continue
    #             line_values.append(float(v))
    #     except ValueError: continue
    #     data.append(line_values)
    # data = np.array(data) #data[0]: [255. 39. 291. 0.505101 0.565322 0.652139 137. 62. 46. 255. ] [x y z nx ny nz r g b alpha]
    # coords = data[:,0:3].astype('int')

    return coords

def write_ply_ascii_geo(filedir, coords):
    if os.path.exists(filedir): os.system('rm '+filedir)
    f = open(filedir,'a+')
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(coords.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n'])
    f.write('end_header\n')
    coords = coords.astype('int')
    for p in coords:
        f.writelines([str(p[0]), ' ', str(p[1]), ' ',str(p[2]), '\n'])
    f.close() 

    return

def write_ply_ascii_geo_float(filedir, coords):
    if os.path.exists(filedir): os.system('rm '+filedir)
    f = open(filedir,'a+')
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(coords.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n'])
    f.write('end_header\n')
    # coords = coords.astype('int')
    for p in coords:
        f.writelines([str(p[0]), ' ', str(p[1]), ' ',str(p[2]), '\n'])
    f.close() 

    return

def read_obj_geo_attr(obj_file_path , texture_image_path ):
    # Create dictionaries to store UV indices and RGB colors
    uv_indices = {}
    rgb_colors = {}

    # Parse the OBJ file
    with open(obj_file_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('vt '):
                # Parse UV coordinates
                parts = line.split()
                u, v = map(float, parts[1:])
                uv_indices[len(uv_indices) + 1] = (u, v)

    # Load the texture image
    texture_image = Image.open(texture_image_path)

    # Extract RGB colors based on UV indices
    for vertex_index, (u, v) in uv_indices.items():
        # Convert UV coordinates to pixel coordinates
        width, height = texture_image.size
        pixel_x = int(u * (width - 1))
        pixel_y = int((1 - v) * (height - 1))  # Invert v coordinate as it's often stored top-down

        # Get RGB color at the corresponding pixel
        rgb_color = texture_image.getpixel((pixel_x, pixel_y))

        # Store the RGB color in the dictionary
        rgb_colors[vertex_index] = rgb_color

    # Print RGB colors associated with each vertex
    # for vertex_index, rgb_color in rgb_colors.items():
    #     print(f"Vertex {vertex_index}: RGB Color = {rgb_color}")
    
    # sort according 
    
    return 


###########################################################################################################

import torch
# import MinkowskiEngine as ME
def array2vector(array, step):
    """ravel 2D array with multi-channel to one 1D vector by sum each channel with different step.
    """
    array, step = array.long(), step.long()
    vector = sum([array[:,i]*(step**(array.shape[-1]-i)) for i in range(array.shape[-1])])

    return vector

def isin(data, ground_truth):
    """ Input data and ground_truth are torch tensor of shape [N, D].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is in `ground_truth` and False otherwise.
    """
    assert data.shape[1] == ground_truth.shape[1]
    device = data.device
    if len(ground_truth)==0:
        return torch.zeros([len(data)]).bool().to(device)
    try:
        step = torch.max(data.max(), ground_truth.max()) + 1
    except RuntimeError as e:
        print(data)
        print(ground_truth)
        # step = data.max()
        if(data.max()>ground_truth.max()):
            step = data.max()+1
        else:
            step = ground_truth.max()+1
    data = array2vector(data, step)
    ground_truth = array2vector(ground_truth, step)
    mask = torch.isin(data.to(device), ground_truth.to(device))

    return mask

def istopk(data, nums, rho=1.0):
    """ Input data is sparse tensor and nums is a list of shape [batch_size].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is the top k (=nums*rho) value and False otherwise.
    """
    mask = torch.zeros(len(data), dtype=torch.bool)
    row_indices_per_batch = data._batchwise_row_indices
    for row_indices, N in zip(row_indices_per_batch, nums):
        k = int(min(len(row_indices), N*rho))
        _, indices = torch.topk(data.F[row_indices].squeeze().detach().cpu(), k)# must CPU.
        mask[row_indices[indices]]=True

    return mask.bool().to(data.device)


def create_new_sparse_tensor(coordinates, features, tensor_stride, dimension, device):
    sparse_tensor = ME.SparseTensor(features=features, 
                                coordinates=coordinates,
                                tensor_stride=tensor_stride,
                                device=device)

    return sparse_tensor
def sort_spare_tensor(sparse_tensor):
    """ Sort points in sparse tensor according to their coordinates.
    """
    if(sparse_tensor.shape[0]==0):
        return sparse_tensor
    
    indices = torch.argsort(array2vector(sparse_tensor.C, 
                                           sparse_tensor.C.max()+1))
    sparse_tensor = create_new_sparse_tensor(coordinates=sparse_tensor.C[indices], 
                                            features=sparse_tensor.F[indices], 
                                            tensor_stride=sparse_tensor.tensor_stride, 
                                            dimension=sparse_tensor.D, 
                                            device=sparse_tensor.device)

    return sparse_tensor


def load_sparse_tensor(filedir, device, reduce_bit=None):
    coords = torch.tensor(read_ply_ascii_geo(filedir)).int()
    # 临时
    if reduce_bit is not None :
        coords = coords / (2 ** reduce_bit)
        coords = coords.int()
    feats = torch.ones((len(coords),1)).float()
    # coords, feats = ME.utils.sparse_quantize(coordinates=coords, features=feats, quantization_size=1)
    # coords_min = coords.min(dim=0)[0]
    # coords -= coords_min
    # print(coords)
    coords, feats = ME.utils.sparse_collate([coords], [feats])
    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=device)
    
    return x



class MK_our_union(torch.nn.Module):
    def forward(self, x1, x2):
        assert x1.tensor_stride[0] == x2.tensor_stride[0]
        if(x1.shape[0]==0):
            return x2
        elif(x2.shape[0]==0):
            return x1
        else:
            mask = isin(x1.C, x2.C)
            assert mask.int().sum() == 0
            mask = isin(x2.C, x1.C)
            assert mask.int().sum() == 0
            x_C = torch.cat([x1.C, x2.C], dim=0)
            x_F = torch.cat([x1.F, x2.F], dim=0)
            q = ME.SparseTensor(
                        features=x_F, coordinates=x_C, tensor_stride=x1.tensor_stride[0] 
                    )
            return sort_spare_tensor(q)
    


def group_mask_gen_8stage(x):
    STRIDE = -1
    
    if isinstance(x, ME.SparseTensor): STRIDE = x.tensor_stride
    elif isinstance(x, SparseTensor): STRIDE = x.stride
    else: raise AssertionError("mismatch data type->expect ME.SparseTensor or SparseTensor")
    tmp = x.C[:, 1:]/STRIDE[0]
    # ->group
    c0 = tmp[:,0] % 2
    c1 = tmp[:,1] % 2
    c2 = tmp[:,2] % 2
    summ = 4 * c0 + 2 * c1 + c2
    group_masks = []
    for group, values in {0: [0], 1: [7], 2: [1], 3:[6], 4:[2],5:[5],6:[3],7:[4]}.items(): #1234
        mask = (summ == values[0])
        for value in values[1:]:
            mask |= (summ == value)
        group_masks.append(mask)
    assert group_masks[0].shape[0] == tmp.shape[0], "mask shape mismatch coord shape"
    return group_masks



def group_mask_gen(x):
    STRIDE = -1
    
    if isinstance(x, ME.SparseTensor): STRIDE = x.tensor_stride
    elif isinstance(x, SparseTensor): STRIDE = x.stride
    else: raise AssertionError("mismatch data type->expect ME.SparseTensor or SparseTensor")
    tmp = x.C[:, 1:]/STRIDE[0]
    c0 = tmp[:,0] % 2
    c1 = tmp[:,1] % 2
    c2 = tmp[:,2] % 2
    summ = 4 * c0 + 2 * c1 + c2
    group_masks = []
    for group, values in {0: [0], 1: [7], 2: [1], 3:[6], 4:[2,5],5:[3,4]}.items(): #1234
        mask = (summ == values[0])
        for value in values[1:]:
            mask |= (summ == value)
        group_masks.append(mask)
    assert group_masks[0].shape[0] == tmp.shape[0], "mask shape mismatch coord shape"
    return group_masks

def group_mask_gen_par(x):
    STRIDE = -1
    
    if isinstance(x, ME.SparseTensor): STRIDE = x.tensor_stride
    elif isinstance(x, SparseTensor): STRIDE = x.stride
    else: raise AssertionError("mismatch data type->expect ME.SparseTensor or SparseTensor")
    tmp = x.C[:, 1:]/STRIDE[0]
    tmp = tmp//2
    c0 = tmp[:,0] % 2
    c1 = tmp[:,1] % 2
    c2 = tmp[:,2] % 2
    summ = 4 * c0 + 2 * c1 + c2
    group_masks = []
    for group, values in {0: [0,7], 1: [1,6],2:[2,3,4,5]}.items():
        mask = (summ == values[0])
        for value in values[1:]:
            mask |= (summ == value)
        group_masks.append(mask)
    assert group_masks[0].shape[0] == tmp.shape[0], "mask shape mismatch coord shape"
    return group_masks






import struct

def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]

def write_body(fd, shape, out_strings):
    bytes_cnt = 0
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    for s in out_strings:
        bytes_cnt += write_uints(fd, (len(s[0]),))
        bytes_cnt += write_bytes(fd, s[0])
    return bytes_cnt

def read_body(fd):
    lstrings = []
    shape = read_uints(fd, 2)
    n_strings = read_uints(fd, 1)[0]
    for _ in range(n_strings):
        s = read_bytes(fd, read_uints(fd, 1)[0])
        lstrings.append([s])

    return lstrings, shape










if __name__ == '__main__':
    filedir= '/home/yzz/yzz/119_model_normalized_3.h5'
    pc = h5py.File(filedir, 'r')['data'][:]
    coords = pc[:,0:3].astype('int')
    print(coords)
    attr = pc[:,3:].astype('int')
    print(attr)






