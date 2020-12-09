import os
import numpy as np
import torch
from PIL import Image
import json
import cv2
import time


class Labelme_Dataset(object):
    def __init__(self, root, cls2id, transforms=None):
        self.root = root
        self.transforms = transforms
        self.cls2id = cls2id
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "img"))))
        self.label = list(sorted(os.listdir(os.path.join(root, "label"))))

    def __getitem__(self, idx):
        img, target = self.get_item_from_labelme_format_json(idx)
        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_item_from_labelme_format_json(self, idx):
        img_path = os.path.join(self.root, "img", self.imgs[idx])
        label_path = os.path.join(self.root, "label", self.label[idx])
        img = Image.open(img_path).convert("RGB")
        label_info = json.load(open(label_path), encoding='utf-8')
        labels = []
        masks = []
        shapes = label_info['shapes']
        for shape in shapes:
            mask = np.zeros(img.size[::-1], dtype="uint8")
            points = shape['points']
            classID = self.cls2id[shape['label']]
            cv2.polylines(mask, np.int32([points]), True, 1)
            cv2.fillPoly(mask, np.int32([points]), 1)
            labels.append(classID)
            masks.append(mask)
        # get bounding box coordinates for each mask
        num_objs = len(labels)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # masks = torch.tensor(masks, dtype=torch.uint8)
        # from_numpy 比 as_tensor 快很多
        masks = torch.from_numpy(np.array(masks))

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target