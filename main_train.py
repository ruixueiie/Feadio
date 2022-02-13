# coding:utf-8
# 训练模型的代码

from time import *

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tools.my_dataset import MyDataset
from tools.iresnet import get_model
from tools.margin_softmax import ArcFace
from tools.partial_fc import PartialFC


def lr_func(epoch):
    return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
        [m for m in [11, 17, 22] if m - 1 <= epoch]
    )


def train_function(image_path_txt, class_num, save_backbone_model_name, save_weight_name_path, save_weight_mom_name_path):

    batch_size = 32

    train_transformer = transforms.Compose([
        transforms.Resize([89, 109]),
        transforms.ToTensor(),
    ])
    train_dataset = MyDataset(txt_path=image_path_txt, transform=train_transformer, target_transform=None)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    backbone = get_model("r18", dropout=0, fp16=False, num_features=512)
    backbone.train()

    if torch.cuda.is_available():
        backbone = backbone.cuda()

    margin_softmax = ArcFace()
    module_partial_fc = PartialFC(
        rank=0, local_rank=0, world_size=1, resume=0,
        batch_size=batch_size, margin_softmax=margin_softmax, num_classes=class_num,
        sample_rate=1, embedding_size=512, prefix="./")

    opt_backbone = torch.optim.Adam(
        params=backbone.parameters(),
        lr=0.0001)

    opt_pfc = torch.optim.SGD(
        params=[{'params': module_partial_fc.parameters()}],
        lr=0.0001,
        momentum=0.9, weight_decay=5e-4)

    scheduler_backbone = torch.optim.lr_scheduler.StepLR(opt_backbone, step_size=50, gamma=0.1)
    scheduler_pfc = torch.optim.lr_scheduler.StepLR(opt_pfc, step_size=50, gamma=0.1)

    start = time()
    for epoch in range(20000):
        error_num = 0
        for batch_id, (image, label) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                image = Variable(image.cuda())
                label = Variable(label.cuda())
            else:
                image = Variable(image)
                label = Variable(label)

            features = F.normalize(backbone(image))
            x_grad, loss_v, output = module_partial_fc.forward_backward(label, features, opt_pfc)
            features.backward(x_grad)
            opt_backbone.step()
            opt_pfc.step()
            module_partial_fc.update()
            
            opt_backbone.zero_grad()
            opt_pfc.zero_grad()

            pred = output.argmax(dim=1)
            correct = torch.eq(label, pred).sum().item()
            # for index in range(0, len(pred)):
            #     if pred[index].item() != label[index].item():
            #         print("pred: {}\t label: {}".format(pred[index], label[index]))
            error_num += len(label) - correct
            
        scheduler_backbone.step()
        scheduler_pfc.step()

        print("epoch: {}\t error_num: {}\t".format(epoch, error_num))

        if error_num == 0:
            end = time()
            torch.save(backbone.state_dict(), save_backbone_model_name)
            module_partial_fc.save_params(save_weight_name_path, save_weight_mom_name_path)
            print("epoch:{}, time:{}, save_backbone_model_name:{}".format(epoch, end - start, save_backbone_model_name))
            break


if __name__ == '__main__':

    # image_path_txt = "./texts/celeba_2048_0630_1.txt"
    image_path_txt = "./texts/glint_2048_0929_1.txt"
    # image_path_txt = "./texts/ijbc_2048_0929_1.txt"

    dataset_name = "glint"
    class_num = 2048
    train_date = "0212"
    index = 1
    save_backbone_model_name = "./arcface_models/{}/{}_{}_{}_iresnet18_{}.pkl".format(dataset_name, dataset_name, class_num, train_date, index)
    save_weight_name_path = "./arcface_models/{}/{}_{}_{}_softmax_weight_{}.pt".format(dataset_name, dataset_name, class_num, train_date, index)
    save_weight_mom_name_path = "./arcface_models/{}/{}_{}_{}_softmax_weight_mom_{}.pt".format(dataset_name, dataset_name, class_num, train_date, index)

    print(image_path_txt, dataset_name, class_num, train_date, index)

    train_function(image_path_txt, class_num, save_backbone_model_name, save_weight_name_path, save_weight_mom_name_path)
