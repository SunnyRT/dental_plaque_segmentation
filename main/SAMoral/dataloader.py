import os
import numpy as np
import torch
from torch.utils.data import Dataset # Note difference: from datasets import Dataset
from PIL import Image
# from .config import device
# from .utils import get_bbox_teeth,find_regions
from utils import get_bbox_teeth,find_regions
import copy
import re
import cv2
import torchvision




class SAMDataset(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, train_images, train_masks,train_mesh,label_mesh):
    self.images = train_images
    self.masks = train_masks

    self.mesh = train_mesh
    self.label_mesh = label_mesh

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):

    image_torch = torch.tensor(self.images[idx]).float()
    label_torch = torch.tensor(self.masks[idx]).float()

    # print(image_ary.shape)
    # print('image_torch', image_torch.shape) #[256, 256, 3]
    # print('label_torch', label_torch.shape) #[256, 256]
    # print('label_torch', label_torch[125])

    image_torch = image_torch.permute(2, 0, 1)  # Assuming original shape is [height, width, channels]
        # Add an additional dimension to mask to match [1, height, width]
    mask = label_torch.unsqueeze(0)  # Expands the mask to [1,1024,1024]


    inputs = {}

    inputs['pixel_values'] = image_torch
    inputs["ground_truth_mask"] = mask
    # print('ground_truth_mask', inputs["ground_truth_mask"].shape) #ground_truth_mask torch.Size([1, 256, 256])
    # inputs['ID'] = id
    
    # SOTA mask of id.png


    inputs['SOTA_mesh'] = np.array(self.mesh[idx]) 
    inputs['label_mesh'] = np.array(self.label_mesh[idx]) 
    # print('inputs[SOTA_mesh]',  inputs['SOTA_mesh'][50:60]) #[8192,10]
    # print('inputs[label_mesh]',  inputs['label_mesh'][50:60]) #[8192,10]

    return inputs






def patchify(image,name, patch_size=256):
    """Manually divides the image into patches."""
    patches_per_dim = (image.shape[0] // patch_size, image.shape[1] // patch_size)
    patches = []
    name_patches = []
    for i in range(patches_per_dim[0]):
        for j in range(patches_per_dim[1]):
            patch = image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patches.append(patch)
            name_patches.append(name[:-4]+"0"+str(i)+"0"+str(j))
    return np.array(patches),np.array(name_patches)



def load_and_patchify_png(base_dir, min_positive_pixels=50):
    """ Load images (both origin & label) from the given directory and patchify them into 256x256 patches. 
    Args:
        base_dir: the directory containing the origin and label folders.
    Returns:
        origin_patches: a numpy array of shape (num_patches, 256, 256, 3) containing the origin images.
        label_patches: a numpy array of shape (num_patches, 256, 256, 1) containing the label images.
    """

    label_dir = os.path.join(base_dir, 'label') #/home/Liting/zhanxch/PlaqueIOS/16000/2D_png/train/label
    origin_dir = os.path.join(base_dir, 'origin')

    SOTA_mesh_dir = os.path.join(base_dir,'SOTA_mesh')
    label_mesh_dir = os.path.join(base_dir,'label_mesh')


    label_patches = []
    origin_patches = []
    

    SOTA_mesh_patches = []
    label_mesh_patches = []

    patch_size = (256, 256)
    stride = 256

    file_names = sorted(os.listdir(label_dir))
    print('length of file_names', len(file_names))
    cnt = 0
    for file_name in file_names:

        label_path = os.path.join(label_dir, file_name) #/home/Liting/zhanxch/PlaqueIOS/16000/2D_png/train/label/000102_0.png
        origin_path = os.path.join(origin_dir, file_name)

        SOTA_mesh_path = os.path.join(SOTA_mesh_dir,file_name[:-6]+".npz")
        label_mesh_path = os.path.join(label_mesh_dir,file_name[:-6]+".npz")


        # Load images
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_img = (label_img > 127).astype(np.uint8)  # label_img is binary
        # print('label_img', label_img.shape)

        origin_img = cv2.imread(origin_path, cv2.IMREAD_COLOR)
        # Convert BGR to RGB
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB) # TIPS: preprocess will rescale it

        # Load mesh
        SOTA_npz = np.load(SOTA_mesh_path, allow_pickle=True)
        label_npz = np.load(label_mesh_path, allow_pickle=True)

        # SOTA_mesh_patches_ary = load_npz(SOTA_npz,file_name[-5])
        # label_mesh_patches_ary = load_npz(label_npz,file_name[-5])
        try:
            # Patchify images
            label_patches_ary,name_patches_ary = patchify(label_img,file_name)
            origin_patches_ary,_ = patchify(origin_img,file_name)

            SOTA_mesh_patches_ary = load_npz(SOTA_npz,file_name[-5])
            label_mesh_patches_ary = load_npz(label_npz,file_name[-5])



        except ValueError as e:
            print(f"Error patchifying {file_name}: {e}")
            continue

        # Filter patches based on the number of positive pixels
        for label_patch, origin_patch,mesh_patch,label_mesh_patch in zip(label_patches_ary, origin_patches_ary,SOTA_mesh_patches_ary,label_mesh_patches_ary):
            if np.count_nonzero(label_patch) >= min_positive_pixels:
                label_patches.append(label_patch)
                origin_patches.append(origin_patch)
                mesh_patch[:, -1] = 0.5
                SOTA_mesh_patches.append(mesh_patch)
                label_mesh_patches.append(label_mesh_patch)
            else:
                cnt += 1

    # print('cnt', cnt) #1321
    # Convert list to numpy array
    label_patches = np.array(label_patches)
    origin_patches = np.array(origin_patches)
    # print('origin_patches', origin_patches.shape) #(4839, 256, 256, 3)


    SOTA_mesh_patches = np.array(SOTA_mesh_patches)
    label_mesh_patches = np.array(label_mesh_patches)

    # print(f"Origin patches shape: {origin_patches.shape}") #(4839,256,256,3)
    # print(f"Label patches shape: {label_patches.shape}") #(4839,256,256)
    # print('SOTA_mesh_patches', SOTA_mesh_patches.shape) #(4839, 8192, 10)
    # print('label_mesh_patches', label_mesh_patches.shape) #(4839, 8192, 10)
    # print('SOTA_mesh_patches', SOTA_mesh_patches[0][:10][:]) #[0.  0.  0.  0.  0.  0.  0.  0.  0.  0.5]
    # print('label_mesh_patches', label_mesh_patches[0][:10][:]) #[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]


    return origin_patches, label_patches,SOTA_mesh_patches,label_mesh_patches

def load_npz(SOTA_npz,pos):

    output_shape = (8192, 10) # 6000 is ok
    if pos == "0":
        SOTA_mesh_patches_ary = pad(SOTA_npz['up'],output_shape)
    if pos == "1":
        SOTA_mesh_patches_ary = pad(SOTA_npz['in'],output_shape)
    if pos == "2":
        SOTA_mesh_patches_ary = pad(SOTA_npz['out'],output_shape)

    return SOTA_mesh_patches_ary

def pad(data,output_shape):
    SOTA_mesh_patches_ary = []
    # print('data', len(data)) #6，8，8每个角度


    # Iterate over each matrix in the list and pad it with zeros
    for i in range(len(data)):
        matrix = data[i]
        # print(matrix.shape)
        if matrix.shape[0] == 0:
            SOTA_mesh_patches_ary.append(np.zeros(output_shape))
            continue
        pad_rows = output_shape[0] - matrix.shape[0]
        pad_cols = output_shape[1] - matrix.shape[1]

        # 在前面补零
        padded_matrix = np.pad(matrix, [(pad_rows, 0), (pad_cols, 0)], mode='constant', constant_values=0)
        # padded_matrix = np.pad(matrix, [(0, output_shape[0] - matrix.shape[0]), (0, output_shape[1] - matrix.shape[1])], mode='constant')
        # print(padded_matrix.shape)
        SOTA_mesh_patches_ary.append(padded_matrix)
    return np.array(SOTA_mesh_patches_ary)

def load_and_patchify_png_permesh(base_dir, mesh_name, min_positive_pixels=-1):
    origin_patches = []
    label_patches = []

    SOTA_mesh_patches = []
    label_mesh_patches = []

    preserved_patches_idx = []
    idx = 0
    name_patches = []
    for i in range(3):
        # print(mesh_name)
        origin_path = os.path.join(base_dir, "origin", f"{mesh_name}_{i}.png")
        label_path = os.path.join(base_dir, "label", f"{mesh_name}_{i}.png")

        SOTA_mesh_path = os.path.join(base_dir,'SOTA_mesh', f"{mesh_name}.npz")
        label_mesh_path = os.path.join(base_dir,'label_mesh', f"{mesh_name}.npz")

        # Load images
        # print('label_path', label_path)
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        # print(f"label_img 类型: {type(label_img)}")
        # print(f"label_img 值: {label_img}")

        # label_img = (label_img > 0).astype(np.uint8) # label_img is binary
        label_img = (label_img > 127).astype(np.uint8) 
        # print('label_img', label_img.shape) #(512, 768),(256, 2048),

        origin_img = cv2.imread(origin_path, cv2.IMREAD_COLOR)
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        

        
        SOTA_npz = np.load(SOTA_mesh_path, allow_pickle=True)               
        SOTA_mesh_patches_ary = load_npz(SOTA_npz,str(i))
        # print('SOTA_mesh_patches_ary', SOTA_mesh_patches_ary.shape) #(6, 8192, 10),(8,8192,10),(8,8192,10)
        
        label_npz = np.load(label_mesh_path, allow_pickle=True)               
        label_mesh_patches_ary = load_npz(label_npz,str(i))


        try:
            # Patchify images
            label_patches_ary,name_patches_ary = patchify(label_img,f"{mesh_name}_{i}.png")
            origin_patches_ary,_ = patchify(origin_img,f"{mesh_name}_{i}.png")
            # print('origin_patches_ary', origin_patches_ary.shape) #(6, 256, 256, 3),(8, 256, 256, 3), (8, 256, 256, 3)

        except ValueError as e:
            print(f"Error patchifying {mesh_name}: {e}")
            continue

        # Filter patches based on the number of positive pixels
        for label_patch, origin_patch,mesh_patch,label_mesh_patch in zip(label_patches_ary, origin_patches_ary,SOTA_mesh_patches_ary,label_mesh_patches_ary):

            if np.count_nonzero(label_patch) >= min_positive_pixels:
                label_patches.append(label_patch)
                origin_patches.append(origin_patch)
                preserved_patches_idx.append(idx)
                mesh_patch[:, -1] = 0.5
                SOTA_mesh_patches.append(mesh_patch)
                label_mesh_patches.append(label_mesh_patch)

            idx += 1


    # Convert list to numpy array
    label_patches = np.array(label_patches) # shape (num_patches=22, 256, 256), where 22 = 6(up)+8(in)+8(out)
    origin_patches = np.array(origin_patches) # shape (num_patches=22, 256, 256, 3), where 22 = 6(up)+8(in)+8(out)

    SOTA_mesh_patches = np.array(SOTA_mesh_patches)
    label_mesh_patches = np.array(label_mesh_patches)
    # print('SOTA_mesh_patches', SOTA_mesh_patches.shape) #(22, 8192, 10)
    # print('label_mesh_patches', label_mesh_patches.shape) #(22, 8192, 10)

    # print(f"Origin patches shape: {origin_patches.shape}")
    # print(f"Label patches shape: {label_patches.shape}")
    
    return origin_patches, label_patches, preserved_patches_idx,SOTA_mesh_patches,label_mesh_patches