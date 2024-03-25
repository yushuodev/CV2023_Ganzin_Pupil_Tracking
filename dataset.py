import os
import json
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np


def get_dataloader(dataset_dir, batch_size=1, split='test'):
    ###############################
    # TODO:                       #
    # Define your own transforms. #
    ###############################
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            ##### TODO: Data Augmentation Begin #####
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ##### TODO: Data Augmentation End #####
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
    else:  # 'val' or 'test'
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            # we usually don't apply data augmentation on test or val data
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    dataset = PupilDataset(dataset_dir, split=split, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(
        split == 'train'), num_workers=0, pin_memory=True, drop_last=(split == 'train'))

    return dataloader


class PupilDataset(Dataset):  # 在as的.json檔案裡
    def __init__(self, dataset_dir, split='test', transform=None):
        super(PupilDataset).__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform

        if self.split == 'train':
            with open(os.path.join(self.dataset_dir, 'train.json'), 'r') as f:
                train_data = json.load(f)
            self.image_names = train_data['filenames']
            self.labels = train_data['labels']
        elif self.split == 'val':
            with open(os.path.join(self.dataset_dir, 'val.json'), 'r') as f:
                val_data = json.load(f)
            self.image_names = val_data['filenames']
            self.labels = val_data['labels']
        else:
            with open(os.path.join(self.dataset_dir, 'output.json'), 'r') as f:
                test_data = json.load(f)
            self.image_names = test_data['filenames']

        print(f'Number of {self.split} images is {len(self.image_names)}')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):  # test?
        
        if self.split == 'test':
            img = self.image_names[index]
            image = Image.open(os.path.join(self.dataset_dir, img))
            if self.transform:
                grayscale_transform1 = transforms.Grayscale(num_output_channels=1)
                image = grayscale_transform1(image)
                image = np.array(image)

                clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(24, 24))
                med = clahe.apply(image).astype(np.float32)/255
                gamma = (1/2.2)
                med = med ** gamma
                med = np.clip(med * 255, 0, 255).astype(np.uint8)
                med = cv2.medianBlur(med, 17)
                image = Image.fromarray(med)

                grayscale_transform = transforms.Grayscale(num_output_channels=3)
                image = grayscale_transform(image)
                image = self.transform(image)
            return {'images': image}
        else:

            img = self.image_names[index]
            image = Image.open(img)
            if self.transform:
                grayscale_transform = transforms.Grayscale(num_output_channels=3)
                image = grayscale_transform(image)
                image = self.transform(image)
            label = self.labels[index]
            return {
                'images': image,
                'labels': label
            }
