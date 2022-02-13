# coding: utf-8
from torch.utils.data import Dataset
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        txt_f = open(txt_path, "r")
        images = []
        for line in txt_f:
            line = line.rstrip()
            image_path = line.split()[0]
            label = int(line.split()[1])
            images.append((image_path, label))

        self.images = images      
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image_path, label = self.images[index]
        image = Image.open(image_path).convert("RGB")  

        if self.transform is not None:
            image = self.transform(image) 

        return image, label

    def __len__(self):
        return len(self.images)

