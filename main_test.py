# coding:utf-8
# 测试模型的代码，相当于Bob对接收到的图像进行消息提取

import os
import time

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tools.my_dataset import MyDataset
from tools.iresnet import get_model
from tools.margin_softmax import ArcFace
from tools.partial_fc import PartialFC
from tools.compute_error import *


def test_function(image_path_txt, class_num, test_image_num, now, attacked_type, test_model_backbone_path, test_weight_name_path, test_weight_mom_name_path):

    batch_size = 32

    gpu_available = torch.cuda.is_available()

    test_transformer = transforms.Compose([
        transforms.Resize([89, 109]),
        transforms.ToTensor(),
    ])
    
    test_dataset = MyDataset(txt_path=image_path_txt, transform=test_transformer, target_transform=None)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    
    margin_softmax = ArcFace()
    module_partial_fc = PartialFC(
        rank=0, local_rank=0, world_size=1, resume=1,
        batch_size=batch_size, margin_softmax=margin_softmax, num_classes=class_num,
        sample_rate=1, embedding_size=512, prefix="./", test_weight_name_path=test_weight_name_path, test_weight_mom_name_path=test_weight_mom_name_path)

    image_error_num = 0
    error_list = []

    with torch.no_grad():

        test_model_backbone = get_model("r18", dropout=0, fp16=False, num_features=512)
        
        if torch.cuda.is_available():
            test_model_backbone = test_model_backbone.cuda()
    
        test_model_backbone.load_state_dict(torch.load(test_model_backbone_path))
        test_model_backbone.eval()
        
        for batch_id, (image, label) in enumerate(test_dataloader):
            if gpu_available:
                image = Variable(image.cuda())
                label = Variable(label.cuda())
            else:
                image = Variable(image)
                label = Variable(label)
            features = F.normalize(test_model_backbone(image))
            output = module_partial_fc.forward_in_test(features)
            pred = output.argmax(dim=1)
            for index in range(0, len(pred)):
                if pred[index].item() != label[index].item():
                    image_error_num += 1
                    error_list.append({"label": int(label[index]), "pred": int(pred[index])})

    image_error_rate = compute_image_error_rate(image_error_num, test_image_num)  # 计算图片错误率（码字错误率）
    bit_error_sum = compute_bit_error_num(str(error_list))  # 计算比特错误数
    bit_error_rate = compute_bit_error_rate(bit_error_sum, test_image_num)  # 计算比特错误率

    with open("./test_result.txt", "a+") as file_test_result:

        file_test_result.write(
            "Time: {} \t Attack Type: {} \t Images (Or Classes) Number: {} \n".format(now, attacked_type, test_image_num))
        file_test_result.write(
            "Bit Error Rate: {} \t Bit Correct Rate: {} \n".format(bit_error_rate, 1 - bit_error_rate))
        file_test_result.write(
            "Image Error Number: {} \t Image Error Rate: {} \t Image Correct Rate: {} \n".format(
                image_error_num, image_error_rate, 1 - image_error_rate))
        file_test_result.write("Error List: {} \n".format(error_list))
        file_test_result.write("\n")

        file_test_result.close()

    print("Attack Type: {} \t Bit Error Rate: {} \t Image Error Rate: {}".format(attacked_type, bit_error_rate, image_error_rate))
    # print("Error List: {}".format(error_list))
    

if __name__ == "__main__":

    # 模型信息，与main_train.py中的信息一致
    # dataset_name = "celeba"
    dataset_name = "glint"
    # dataset_name = "ijbc"
    class_num = 2048
    train_date = "0212"
    index = 1
    test_model_backbone_path = "./arcface_models/{}/{}_{}_{}_iresnet18_{}.pkl".format(dataset_name, dataset_name, class_num, train_date, index)
    test_weight_name_path = "./arcface_models/{}/{}_{}_{}_softmax_weight_{}.pt".format(dataset_name, dataset_name, class_num, train_date, index)
    test_weight_mom_name_path = "./arcface_models/{}/{}_{}_{}_softmax_weight_mom_{}.pt".format(dataset_name, dataset_name, class_num, train_date, index)
    
    image_num = 2048

    # 受攻击图片的txt的文件夹
    # attacked_text_folder_path = "./texts/attacked_celeba_2048_1_20210726"
    attacked_text_folder_path = "./texts/attacked_glint_2048_1_20211124"
    # attacked_text_folder_path = "./texts/attacked_ijbc_2048_1_20211124"

    attacked_txt_list = os.listdir(attacked_text_folder_path)

    now = time.strftime("%Y%m%d%H%M")
    print(now)

    for attacked_txt in sorted(attacked_txt_list):
        attacked_type = attacked_txt.replace(".txt", "")
        print(attacked_type)
        image_path_txt = os.path.join(attacked_text_folder_path, attacked_txt)
        test_function(image_path_txt, class_num, image_num, now, attacked_type,
                    test_model_backbone_path, test_weight_name_path, test_weight_mom_name_path)
    print("finished")

