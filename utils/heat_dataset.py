import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch


class HeatDataset(Dataset):
    """dataset for heat sensor."""

    def __init__(self, root_dir, transform=None, size=64, return_type="yolo"):
        self.return_type = return_type
        self.size = size
        self.root_dir = root_dir
        self.transform = transform
        self.heat_data_labels = {}

        heat_files = os.listdir(os.path.join(root_dir))
        for heat_file in heat_files:
            if not heat_file.split(".")[-1] == "txt":
                continue
            name = heat_file[:-4]
            label_folder = name + "_small_images_label_indexed"
            if not os.path.exists(os.path.join(root_dir, label_folder)):
                continue
            self.heat_data_labels[name] = os.listdir(
                os.path.join(root_dir, label_folder))

    def load_heat_data(self, path, index):
        line = open(path).readlines()[index]
        heat_data = line.split()[-1].split(",")[-self.size:]
        heat_data = np.array(heat_data, dtype=int)
        w_h = int(np.sqrt(self.size))
        heat_data = heat_data.reshape((w_h, w_h))
        heat_data = np.rot90(heat_data)
        return heat_data

    def load_label(self, path):
        mask = np.asarray(Image.open(path))
        mask = np.equal.outer(np.unique(mask), mask).view(
            'i1')
        # Skip first "class" which is nothing
        return mask[1:]

    def get_subsets(self):
        return [*self.heat_data_labels.keys()]

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        imgs = torch.stack(imgs)
        return paths, imgs, targets

    def __len__(self):
        return sum([len(entries) for entries in self.heat_data_labels.values()])

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range.")
        acc_counter = 0
        for name_id, files in self.heat_data_labels.items():
            if idx < (acc_counter + len(files)):
                data = self.load_heat_data(os.path.join(
                    self.root_dir, name_id + ".txt"), idx-acc_counter)
                file_name = files[idx-acc_counter]
                label = self.load_label(os.path.join(
                    self.root_dir, name_id + "_small_images_label_indexed", file_name))
                break
            acc_counter += len(files)

        if self.transform:
            data = self.transform(data)
            label = self.transform(label)
        else:
            data = torch.as_tensor(data.copy(), dtype=torch.float32)
            label = torch.as_tensor(label, dtype=torch.uint8)

        # Get bounding box
        areas = [np.where(layer == 1) for layer in label]
        if self.return_type == "yolo":
            boxes = torch.zeros(len(areas), 6)
            for ix, a in enumerate(areas):
                x_size = (np.max(a[1])+1 - np.min(a[1])) / np.sqrt(self.size)
                y_size = (np.max(a[0])+1 - np.min(a[0])) / np.sqrt(self.size)
                x_center = (x_size / 2) + (np.min(a[1]) / np.sqrt(self.size))
                y_center = (y_size / 2) + (np.min(a[0]) / np.sqrt(self.size))
                boxes[ix, :] = torch.tensor(
                    [0, 0, x_center, y_center, x_size, y_size])

            return (
                os.path.join(self.root_dir, name_id, file_name),
                data.unsqueeze(-1).permute(2,0,1),
                boxes
            )

        if self.return_type == "Mask R-CNN":
            boxes = [[np.min(a[1]), np.min(a[0]), np.max(
                a[1])+1, np.max(a[0])+1] for a in areas]
            boxes = torch.FloatTensor(boxes)

            return (data, {
                "image_id": torch.tensor(idx, dtype=torch.int64),
                "boxes": boxes,
                "labels": torch.tensor([1 for _ in label], dtype=torch.int64),
                "masks": label,
            })
