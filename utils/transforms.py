import albumentations as A
from albumentations.pytorch import ToTensorV2


# Transforms for images and masks
def get_transforms(train=True):
    if train:
        return A.Compose([
            #A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.1, shift_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
    else:
        return A.Compose([
            #A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})