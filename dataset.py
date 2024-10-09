import os
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class SegmentDataset(Dataset):
    def __init__(self, dirPath=r'/car-segmentation', imageDir='images', masksDir='masks', train=True):
        self.train = train
        self.imgDirPath = os.path.join(dirPath, imageDir)
        self.nameImgFile = sorted([f for f in os.listdir(self.imgDirPath) if f.endswith('.png')])

        if train:
            self.maskDirPath = os.path.join(dirPath, masksDir)
            self.nameMaskFile = sorted([f for f in os.listdir(self.maskDirPath) if f.endswith('.png')])

    def __len__(self):
        return len(self.nameImgFile)

    def __getitem__(self, index):
        imgPath = os.path.join(self.imgDirPath, self.nameImgFile[index])
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        width = img.shape[0]
        height = img.shape[1]

        train_transform = A.Compose(
            [
                A.Resize(1024, 1024),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

        if self.train:
            maskPath = os.path.join(self.maskDirPath, self.nameMaskFile[index])
            mask = cv2.imread(maskPath, cv2.IMREAD_UNCHANGED)
        else:
            mask = img
        
        dt = train_transform(image=img, mask=mask)
        return dt['image'], dt['mask'], width, height, self.nameImgFile[index]