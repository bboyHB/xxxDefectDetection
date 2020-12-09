import torch
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from eval import custom_eval_sanan_when_train
from relationNet.maskrcnn_with_relation import get_relation_model
from tools.dataset import Labelme_Dataset
from tools.engine import train_one_epoch, evaluate
from tools import utils, transforms as T

"""
本代码需要调用第三方库pycocotools
windows若无法安装pycocotools
请输入以下指令安装： 
pip(或conda) install pycocotools-windows
"""

num_classes = 17
data_path = 'data/NP_pick_together'
model_save_path = 'models'
get_model = get_relation_model
batch_size = 1
num_workers = 4
learning_rate = 0.001
momentum = 0.9
weight_decay = 0.0005
num_epochs = 60
# lr_decay_step = 30
# lr_decay_gamma = 0.1
train_set_ratio = 0.7
random_flip_rotate_prob = 0.7
random_resize_prob = 0.7

# 0为背景
cls2id_sanan202012 = {"GS_EPI": 1, "GS_PACO": 2, "GS_MURA": 3,
                      "TP_EPI": 4, "TP_GT_BLACKPIONT": 5, "TP_BP": 6,
                      "TP_PACO": 7, "TP_DUANSHAN": 8, "TP_SCAR": 9,
                      "V0_EPI": 10, "V0_GT_BLACKPOINT": 11, "V0_GT": 12,
                      "V0_GT_SBYS": 13, "V0_PACO": 14, "VO_DUANSHAN": 15,
                      "V0_SCAR": 16}
cls2id = cls2id_sanan202012

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        # 自定义组合一些图像变换方法，随机图像翻转旋转以及缩放，增强模型泛化性
        transforms.append(T.RandomFlipAndRotate(random_flip_rotate_prob))
        transforms.append(T.RandomResize(random_resize_prob))
    return T.Compose(transforms)


def train():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    time_stamp = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    stamp = "relation" + '_b' + str(batch_size) + '_e' + str(num_epochs) + '_lr' + str(learning_rate) + '_tr' + str(
        train_set_ratio)
    writer = SummaryWriter(comment=stamp)

    # use our dataset and defined transformations
    dataset = Labelme_Dataset(data_path, cls2id=cls2id, transforms=get_transform(train=True))
    dataset_test = Labelme_Dataset(data_path, cls2id=cls2id, transforms=get_transform(train=False))

    # split the dataset in train and test set
    num_train_set = round(train_set_ratio * len(dataset))
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:num_train_set])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[num_train_set:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate,
                                momentum=momentum, weight_decay=weight_decay)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10)

    max_AP = 0
    max_acc = 0
    cm_when_max_acc = 0
    ps_when_max_acc = 0
    rc_when_max_acc = 0
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        writer.add_scalar('epoch-loss', logger.loss.total, epoch)
        # evaluate on the test dataset
        evaler, pick = evaluate(model, data_loader_test, device=device)
        # 在TensorBoard中显示参数和测试结果
        writer.add_scalar('mAP', evaler.coco_eval['bbox'].stats[0], epoch)
        writer.add_scalar('AP50', evaler.coco_eval['bbox'].stats[1], epoch)
        writer.add_scalar('AP75', evaler.coco_eval['bbox'].stats[2], epoch)
        bboxes = pick[1][0]['boxes']    # 如果输出是X,Y,W,H 就要多一行代码： bboxes[:, 2:] += bboxes[:, :2]
        preds = ["background"] + list(cls2id.keys())
        labels = pick[1][0]['labels']
        scores = pick[1][0]['scores']
        bboxes_str = []
        for i in range(len(labels)):
            bboxes_str.append("{} {:.2%}".format(preds[int(labels[i])], float(scores[i])))
        # tensorboard中输出mask，若需要可取消注释
        # masks = pick[1][0]['masks']
        # final_mask = torch.zeros(masks.shape[-2:], dtype=torch.bool)
        # for i in masks > 0.5:
        #     final_mask = torch.bitwise_or(i[0], final_mask)
        # writer.add_image('test_mask', final_mask, epoch, dataformats='HW')

        writer.add_image('real_img', pick[0][0], epoch)
        writer.add_image_with_boxes('test_img', pick[0][0], bboxes, epoch, labels=bboxes_str)

        # 在测试集上的AP50以及准确率作为评测指标来选取最佳模型，保存最后一个最佳模型（AP50或准确率）
        if max_AP < evaler.coco_eval['bbox'].stats[1]:
            max_AP = evaler.coco_eval['bbox'].stats[1]
            torch.save(model, os.path.join(model_save_path, stamp + '_' + time_stamp + '.pth'))
        acc, cm, ps, rc = custom_eval_sanan_when_train(model, data_loader_test, device=device)
        writer.add_scalar('acc', acc, epoch)
        if max_acc < acc:
            max_acc = acc
            cm_when_max_acc = cm
            ps_when_max_acc = ps
            rc_when_max_acc = rc
            torch.save(model, os.path.join(model_save_path, stamp + '_' + time_stamp + '.pth'))
        # update the learning rate
        writer.add_scalar('lr', logger.lr.value, epoch)
        lr_scheduler.step(evaler.coco_eval['bbox'].stats[1])  # AP50连续patience个epoch不超过当前最大值，就降低学习率
    print("That's it!")
    print('acc:{:.2%}'.format(max_acc))
    print(cm_when_max_acc)
    print('precision:', ['{:.2%}'.format(x) for x in ps_when_max_acc])
    print('recall:', ['{:.2%}'.format(x) for x in rc_when_max_acc])


if __name__ == '__main__':
    train()