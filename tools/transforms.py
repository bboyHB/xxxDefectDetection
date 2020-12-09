import random
import torch

from torchvision.transforms import functional as F

def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomFlipAndRotate(object):
    def __init__(self, prob):
        self.prob = prob
        self.trans = ['90', '180', '270', 'vertical_flip', 'horizontal_flip', 'vertical_90', 'horizontal_90']

    def __call__(self, image, target):
        if random.random() < self.prob:
            randomInt = random.randint(0, 6)
            if self.trans[randomInt] in ['90', '180', '270']:
                angle = int(self.trans[randomInt])
                image, target = self.rotate(image, target, angle)
            elif self.trans[randomInt] in ['vertical_flip', 'horizontal_flip', 'vertical_90', 'horizontal_90']:
                image, target = getattr(self, self.trans[randomInt])(image, target)
            else:
                print("随机数错误")
                exit(-1)
        return image, target

    def rotate(self, image, target, angle):
        # 逆时针旋转90, 180, 270
        height, width = image.shape[-2:]
        image = image.rot90(angle // 90, dims=(-2, -1))
        bbox = target["boxes"]
        if angle // 90 == 1:
            bbox[:, [0, 2]], bbox[:, [1, 3]] = bbox[:, [1, 3]], width - bbox[:, [2, 0]]
        elif angle // 90 == 2:
            bbox[:, [0, 2]], bbox[:, [1, 3]] = width - bbox[:, [2, 0]], height - bbox[:, [3, 1]]
        elif angle // 90 == 3:
            bbox[:, [0, 2]], bbox[:, [1, 3]] = height - bbox[:, [3, 1]], bbox[:, [0, 2]]
        else:
            print("角度错误")
            exit(-1)
        target["boxes"] = bbox
        if "masks" in target:
            target["masks"] = target["masks"].rot90(angle // 90, dims=(-2, -1))
        if "keypoints" in target:
            print("尚未实现keypoints")
            exit(-1)
        return image, target

    def horizontal_flip(self, image, target):
        # Horizontal
        height, width = image.shape[-2:]
        image = image.flip(-1)
        bbox = target["boxes"]
        bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
        target["boxes"] = bbox
        if "masks" in target:
            target["masks"] = target["masks"].flip(-1)
        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = _flip_coco_person_keypoints(keypoints, width)
            target["keypoints"] = keypoints
        return image, target

    def vertical_flip(self, image, target):
        # Vertical
        height, width = image.shape[-2:]
        image = image.flip(-2)
        bbox = target["boxes"]
        # bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
        bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
        target["boxes"] = bbox
        if "masks" in target:
            target["masks"] = target["masks"].flip(-2)
        if "keypoints" in target:
            print("尚未实现keypoints")
            exit(-1)
            # keypoints = target["keypoints"]
            # keypoints = _flip_coco_person_keypoints(keypoints, width)
            # target["keypoints"] = keypoints
        return image, target

    def horizontal_90(self, image, target):
        image, target = self.horizontal_flip(image, target)
        image, target = self.rotate(image, target, 90)
        return image, target

    def vertical_90(self, image, target):
        image, target = self.vertical_flip(image, target)
        image, target = self.rotate(image, target, 90)
        return image, target


class RandomResize(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            randomRatio = random.uniform(0.5, 1.5)
            image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                                    scale_factor=randomRatio,
                                                    mode='bilinear',
                                                    align_corners=False,
                                                    recompute_scale_factor=False).squeeze(0)
            target["boxes"] = target["boxes"] * randomRatio
            if "masks" in target:
                # 考虑到mask为二值图，所以mode采用最邻近插值，防止值被修改成0和1之外的数
                target["masks"] = torch.nn.functional.interpolate(target["masks"].unsqueeze(0),
                                                                  scale_factor=randomRatio,
                                                                  mode='nearest',
                                                                  recompute_scale_factor=False).squeeze(0)
            if "keypoints" in target:
                print("尚未实现keypoints")
                exit(-1)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
