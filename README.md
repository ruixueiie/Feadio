# Feadio

## INTRODUCTOIN

+ Feadio: A selection-based coverless image steganography method based on optimized neural network feature embedding
+ Having 100% completeness
+ Having strong resistance to attack

## PROJECT ARCHITECTURE

+ `main_train.py`: Code for training model.
+ `main_test.py`: Code for testing model. 
+ `tools/`: Folder for storing other codes.
+ `arcface_models/`: Folder for storing trained models.
+ `texts/`: Folder for storing the label text of the original image and the attacked image
+ `images/`: Folder for storing the original images of CelebA, Glint360K and IJB-C. Please download them by yourself, and refer to the location in `texts /` for the structure
+ `attacked_images/`: Folder for storing the attacked images corresponding to the three above datasets. [Baidu Netdisk]( https://pan.baidu.com/s/1eAtBYWhRrtu017z-JN_GOw )  [code: ptmd] 

## DEPENDENCY ENVIRONMENT

```
cuda=10.1
pillow=8.2.0
python=3.8
pytorch=1.6.0
torchvision=0.7.0
```

## ACKNOWLEDGEMENT

+ We thank [ArcFace](https://github.com/deepinsight/insightface) for their work on high-quality face recognition.

## OTHERS

+ OSNA-Face Dataset
  + [Baidu Netdisk](https://pan.baidu.com/s/1ZiCJdFAVUdOwBgzGPyVkvw)  [code: 7m5e]

## BibTeX

+ If this paper/code is helpful to you, please consider quoting our work
> The paper quote format is coming soon ...

