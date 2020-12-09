import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from .ROIHeads import get_ROIHeads
from .CustomBackbone import get_DCN_Resnet

def get_relation_model(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=5)
    # DCN可变卷积网络
    # model = get_DCN_Resnet(num_classes)

    # replace the pre-trained head with a new one
    model.roi_heads = get_ROIHeads(num_classes)

    return model


