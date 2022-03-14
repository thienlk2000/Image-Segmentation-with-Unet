from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class CarSegment(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_file = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_file)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_file[idx])
        mask_path = os.path.join(self.mask_dir, self.img_file[idx].replace('.jpg', '_mask.gif'))
        img = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask == 255] = 1.0
        if self.transform is not None:
            augment = self.transform(image=img, mask=mask)
            img = augment['image']
            mask = augment['mask']
        return img, mask