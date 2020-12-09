import torch
import numpy as np
import os
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import json

def tensor_filter(t, indexes):
    '''
    tensor根据indexes筛选留下的部分
    :param t: 要进行筛选的tensor，比如有这些元素： [A, B, C, D, E, F]
    :param indexes: 表示需要留下的部分的索引，比如[1,3,4,5]
    :return: 筛选完的, 根据上面的假设，会返回： [B, D, E, F]
    '''
    tensor2array = t.cpu().numpy()
    new_array = []
    for index in indexes:
        new_array.append(tensor2array[index])
    new_array = torch.from_numpy(np.asarray(new_array))
    return new_array


def determine_final_class_sanan(predict):
    '''
    三安科技
    根据分数最高来判定一张图片最后的预测类别（属于哪一类瑕疵）
    :param predict: 输入一个预测集合，一张图片中预测的很多个矩形框及其类别和分数信息
    :return: 只返回分数最高的那个瑕疵框的类别
    '''
    get_pred_classes = predict["labels"].cpu().numpy()
    pred_classes_index = []
    pred_classes = []
    for c in range(len(get_pred_classes)):
        pred_classes_index.append(c)
        pred_classes.append(get_pred_classes[c])
    if len(pred_classes) == 0:
        return 17
    scores = tensor_filter(predict["scores"], pred_classes_index).numpy().tolist()
    maxscore = 0
    maxindex = 0
    for i in range(len(scores)):
        if scores[i] > maxscore:
            maxscore = scores[i]
            maxindex = i
    return pred_classes[maxindex]


def draw_detection_sanan(predict, img):
    '''
    三安科技
    根据分数最高来判定一张图片最后的预测类别（属于哪一类瑕疵），同时画出最高置信度的检测框并输出图片
    :param predict: 输入一个预测集合，一张图片中预测的很多个矩形框及其类别和分数信息
    :param img: 输入原图
    :return: 只返回分数最高的那个瑕疵框的类别以及画有检测框的图片
    '''
    pred_classes = predict["labels"].tolist()
    if len(pred_classes) == 0:
        return 17, img
    scores = predict["scores"].tolist()
    bbox = predict["boxes"].tolist()
    maxscore = 0
    maxindex = 0
    for i in range(len(scores)):
        if scores[i] > maxscore:
            maxscore = scores[i]
            maxindex = i
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox[maxindex])
    return pred_classes[maxindex], img


def eval_sanan():
    test_path = 'sanan/img'
    test_label = 'sanan/label'
    model_path = 'models/relation_b1_e60_lr0.001_tr0.7_2020-12-08T16-28-22.pth'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # 如果保存的是模型参数，则用下面两行
    # model = get_model_instance_segmentation(8)
    # model.load_state_dict(torch.load(model_path))
    model = torch.load(model_path)
    model.to(device)
    model.eval()

    predict_list = []
    real_list = []
    predict_list2 = []
    real_com_cls = []
    pred_com_cls = []
    count_step = 0
    count_right = 0
    cls = ["background", "DECP1", "DECP2", "DECP3", "TARE", "PDCR", "PACO", 'FALSE']
    with torch.no_grad():
        for img in os.listdir(test_path):
            img_path = os.path.join(test_path, img)
            x = Image.open(img_path).convert("RGB")
            x = F.to_tensor(x).unsqueeze(0).to(device)
            predictions = model(x)
            pred = determine_final_class_sanan(predictions[0])

            predict_list.append(pred)
            # predict_list2.append(contain_class(predictions[0], cls.index(c)))
            label_path = os.path.join(test_label, img[:-3] + 'json')
            label_json = json.load(open(label_path))
            label_cls = label_json['shapes'][0]['label']
            real_list.append(cls.index(label_cls))

            right = pred == cls.index(label_cls)
            if right:
                count_right += 1
            count_step += 1
            acc = count_right / count_step
            acc = '{:.2%}'.format(acc)
            print(f'{right}, real: {label_cls}, pred:{cls[pred]}, acc:{acc}')
    cm = confusion_matrix(real_list, predict_list)
    ps = precision_score(real_list, predict_list, average=None)
    rc = recall_score(real_list, predict_list, average=None)
    # cm2 = confusion_matrix(real_list, predict_list2)
    # ps2 = precision_score(real_list, predict_list2, average=None)
    # rc2 = recall_score(real_list, predict_list2, average=None)
    print(r'统计')
    print(cm)
    print('precision:', ['{:.2%}'.format(x) for x in ps])
    print('recall:', ['{:.2%}'.format(x) for x in rc])
    # print(r'非正常统计')
    # print(cm2)
    # print('precision:', ['{:.2%}'.format(x) for x in ps2])
    # print('recall:', ['{:.2%}'.format(x) for x in rc2])


def custom_eval_sanan_when_train(model, data_loader_test, device):
    print('custom eval...')
    model.to(device)
    model.eval()

    predict_list = []
    real_list = []
    count_step = 0
    count_right = 0
    with torch.no_grad():
        for img, label in data_loader_test:
            # 默认batchsize为1
            img = img[0].unsqueeze(0).to(device)
            label = label[0]
            predictions = model(img)
            pred = determine_final_class_sanan(predictions[0])

            predict_list.append(pred)

            label_cls = label["labels"].tolist()[0]
            real_list.append(label_cls)

    #         right = pred == label_cls
    #         if right:
    #             count_right += 1
    #         count_step += 1
    # acc = count_right / count_step
    # acc = '{:.2%}'.format(acc)

    acc = accuracy_score(real_list, predict_list)
    cm = confusion_matrix(real_list, predict_list)
    ps = precision_score(real_list, predict_list, average=None)
    rc = recall_score(real_list, predict_list, average=None)
    print('acc:{:.2%}'.format(acc))
    print(cm)
    print('precision:', ['{:.2%}'.format(x) for x in ps])
    print('recall:', ['{:.2%}'.format(x) for x in rc])
    model.train()
    return acc, cm, ps, rc


def eval_unknown_data():
    """
    遍历文件夹及其子文件夹中的所有图片进行测试，保存输出结果
    :return:
    """
    from train_relation import cls2id
    test_path = 'C:/Users/DeepLearning/Desktop/repickNP/验证照片'
    model_path = 'models/relation_b1_e60_lr0.001_tr0.7_2020-12-08T16-28-22.pth'
    save_path = 'preds_output'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # 如果保存的是模型参数，则用下面两行
    # model = get_model_instance_segmentation(8)
    # model.load_state_dict(torch.load(model_path))
    model = torch.load(model_path)
    model.to(device)
    model.eval()

    class_list = list(cls2id.keys()) + ['False']
    print(class_list)
    if not os.path.exists(save_path):
        for cls in class_list:
            os.makedirs(os.path.join(save_path, cls))

    queue = [test_path]
    while queue:
        temp = queue.pop(0)
        files_dirs = os.listdir(temp)
        for file_dir in files_dirs:
            path_to_check = os.path.join(temp, file_dir)
            if os.path.isdir(path_to_check):
                queue.append(path_to_check)
            elif file_dir.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
                with torch.no_grad():
                    img_path = path_to_check
                    img = Image.open(img_path).convert("RGB")
                    x = F.to_tensor(img).unsqueeze(0).to(device)
                    predictions = model(x)
                    pred, img = draw_detection_sanan(predictions[0], img)
                    img.save(os.path.join(save_path, class_list[pred-1], path_to_check[len(test_path):].replace('\\', '_')))
                    print(os.path.join(save_path, class_list[pred-1], path_to_check[len(test_path):].replace('\\', '_')))
            elif file_dir == 'Thumbs.db':
                continue
            else:
                print("文件格式不支持:", path_to_check)
                continue


if __name__ == '__main__':
    eval_unknown_data()

