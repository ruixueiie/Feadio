# Feadio

## 简介

+ 一种基于优化神经网络特征嵌入的载体选择式隐写方法Feadio
+ 具有100%的完备性
+ 具有极强的抗攻击性

## 项目结构说明

+ main_train.py: 训练模型的代码
+ main_test.py: 测试模型的代码
+ arcface_models/: 存放训练好的模型
+ attacked_images/: 存放三种数据集对应的受攻击图像，[百度网盘链接]( https://pan.baidu.com/s/1eAtBYWhRrtu017z-JN_GOw )  [ptmd] 
+ images/: 存放CelebA、Glint360K与IJB-C的原始图像，自行下载，结构参考texts/中的位置
+ texts/: 存放原始图像与受攻击图像的标签文本
+ tools/: 存放其他代码

## 依赖环境

```
cuda=10.1
pillow=8.2.0
python=3.8
pytorch=1.6.0
torchvision=0.7.0
```

## 其他
+ OSNA-Face数据集
  + [百度网盘链接](https://pan.baidu.com/s/1ZiCJdFAVUdOwBgzGPyVkvw)  [7m5e]

## 引用

+ 如果论文对您有帮助，请考虑引用我们的文章
> The paper quote format is coming soon ...

