from torch.utils.data import Dataset
import os
from PIL import Image

class SolarPanelDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        # Pretpostavljamo da su slike u poddirektoriju 'images' a maske u 'masks'
        self.image_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        self.filenames = [f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)  # Pretpostavljamo da maske imaju isti naziv
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Pretpostavljamo da su maske u grayscale

        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return image, mask, img_name  # Vraćamo i ime datoteke
