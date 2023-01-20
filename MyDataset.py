import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from pathlib import Path

class MyDataSet(Dataset):
    def __init__(self, mask_name, data_dir, data_name, kdata_input=True, mode="train", transform=False):
        mask_name, data_dir = map(Path, (mask_name, data_dir))
        self.mask_name = mask_name
        self.kdata_input = kdata_input
        mask_key = [v for k, v in {"radial_256_256_20": "Umask", "radial_256_256_30":"mask_matrix", "uniform":"x", "1DGV":"x"}.items() if k in self.mask_name.name][0]
        self.mask = loadmat(self.mask_name)[mask_key].astype(float)
        self.data_file = data_dir / data_name
        data = loadmat(str(self.data_file))
        self.imageLabel = data["label"].astype(float)

        self.transform_flag = transform
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5)
            ])
        
    def __len__(self):
        return self.imageLabel.shape[0]

    def fft2(self, image):
        return np.fft.fftshift(np.fft.fft2(image))

    def ifft2(self, kspace_cplx):
        return np.absolute(np.fft.ifft2(np.fft.fftshift(kspace_cplx)))
        
    def __getitem__(self, i):
        image = self.imageLabel[i,:,:].copy()
        unmask = self.mask
        if self.transform_flag:
            trans = self.train_transform(image=image)
            image = trans["image"]
        kspace = self.fft2(image)
        masked_k = kspace * self.mask
        if self.kdata_input:
            kdata = np.expand_dims(masked_k, axis=0)
            mask = np.expand_dims(unmask, axis=0)
            label = np.expand_dims(image, axis=0)
    
            # seperate complex data to two channels data(real and imaginary)
            kdata_real = kdata.real
            kdata_imag = kdata.imag
            kdata = np.concatenate((kdata_real, kdata_imag), axis=0)
            return torch.from_numpy(label), torch.from_numpy(kdata), torch.from_numpy(mask)
        else:
            fold_image = self.ifft2(masked_k)
            return torch.from_numpy(image[None]), torch.from_numpy(fold_image[None]), torch.from_numpy(unmask[None])
        
