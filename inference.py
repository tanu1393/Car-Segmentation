import os
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np
import argparse
import segmentation_models_pytorch as smp

from dataset import SegmentDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

denorm = A.Normalize(
    mean=[-m / s for m, s in zip(mean, std)],
    std=[1.0 / s for s in std],
    always_apply=True,
    max_pixel_value=1.0
)
tt = ToTensorV2()

classes = {0: 'Background', 1: 'car', 2: 'wheels', 3: 'lights', 4: 'windows'}
class_colors = {
    0: 'black', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'orange'
}

def show(imgs, fileName=None):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
    plt.savefig(fileName)
    plt.show()


def inference(model_path, dirPath='/content/car-segmentation', image_dir='test_images'):
    if not os.path.isdir(os.path.join(dirPath, 'results')):
        os.makedirs(os.path.join(dirPath, 'results'))

    # trained_model = torch.load(model_path, map_location=torch.device(device))
    trained_model = smp.from_pretrained(model_path)
    trained_model.to(device)

    car_data = SegmentDataset(dirPath = dirPath, imageDir=image_dir, train=False)
    car_dataloader = DataLoader(car_data, batch_size=1)

    with tqdm(car_dataloader, desc=f"Inference process") as tepoch:
        for infer_batch in tepoch:
            img = infer_batch[0].to(device)
            width = infer_batch[2]
            height = infer_batch[3]
            fileName = infer_batch[4]

            output = trained_model(img)

            output_mask = output.argmax(axis=1)
            # We get the unique colors, as these would be the object ids.
            obj_ids = torch.unique(output_mask)

            # first id is the background, so remove it.
            obj_ids = obj_ids[1:]

            # split the color-encoded mask into a set of boolean masks.
            # Note that this snippet would work as well if the masks were float values instead of ints.
            masks = output_mask == obj_ids[:, None, None]

            re_size = A.Resize(width.item(), height.item())

            drawn_masks = []
            resize_masks = []
            for mask in masks:
                #inverse transform
                a = (denorm(image=img[0].permute(1,2,0).cpu().numpy())["image"]*255).astype(np.uint8)
                a = re_size(image=a, mask=mask.type(torch.uint8).cpu().numpy())
                a = tt(image=a['image'], mask=a['mask'])

                drawn_masks.append(draw_segmentation_masks(a['image'], a['mask'].bool(), alpha=0.8, colors="blue"))
                resize_masks.append(a['mask'])

            show(drawn_masks, os.path.join(dirPath, 'results', fileName[0].replace('.png', '') + '_infer_segMask.png'))

            boxes = masks_to_boxes(torch.stack(resize_masks))
            labels = [classes[i.item()] for i in obj_ids]
            colors = [class_colors[i.item()] for i in obj_ids]

            drawn_segs = draw_segmentation_masks(a['image'], torch.stack([mask.bool() for mask in resize_masks]), alpha=0.9, colors=colors)
            drawn_boxes = draw_bounding_boxes(drawn_segs, boxes, colors="red", labels=labels, width=3, font=os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf'), font_size=20)

            show(drawn_boxes, os.path.join(dirPath, 'results', fileName[0].replace('.png', '') + '_infer_segBox.png'))

    print(f"Results stored at {os.path.join(dirPath, 'results')}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Segmentation inference')
    parser.add_argument('--model_path', default='/trained_model', help='Trained segmentation model weight path')
    parser.add_argument('--dirPath', default='/car-segmentation', help='Parent directory')
    parser.add_argument('--image_dir', default='test_images', help='Images directory name')
    args = parser.parse_args()

    inference(args.model_path, args.dirPath, args.image_dir)