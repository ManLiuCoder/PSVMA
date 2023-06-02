# PSVMA
![](https://img.shields.io/badge/CVPR'23-brightgreen)   [![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/) [![arxiv badge](https://img.shields.io/badge/arxiv-2303.15322-red)](https://arxiv.org/abs/2303.15322)

 - [*Progressive Semantic-Visual Mutual Adaption for Generalized Zero-Shot Learning*](https://arxiv.org/abs/2303.15322)
    This repository contains the reference code for the paper "**Progressive Semantic-Visual Mutual Adaption for Generalized Zero-Shot Learning**" accepted to CVPR 2023.
    
    



## 🌈 Model Architecture
![Model_architecture](framework.png)


## 📚 Dependencies

- ```Python 3.6.7```
- ```PyTorch = 1.7.0```
- All experiments are performed with one RTX 3090Ti GPU.

# ⚡ Prerequisites
- **Dataset**: please download the dataset, i.e., [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [AWA2](https://cvml.ist.ac.at/AwA2/), [SUN](https://groups.csail.mit.edu/vision/SUN/hierarchy.html) to the dataset root path on your machine
- **Data split**: Datasets can be download from [Xian et al. (CVPR2017)](https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip) and take them into dir ```../../datasets/```.
- **Attribute w2v**:```extract_attribute_w2v_CUB.py``` ```extract_attribute_w2v_SUN.py``` ```extract_attribute_w2v_AWA2.py``` should generate and place it in ```w2v/```.
- Download pretranined vision Transformer as the vision encoder.

## 🚀 Train & Eval
Before running commands, you can set the hyperparameters in config on different datasets: 
```
config/cub.yaml       #CUB
config/sun.yaml      #SUN
config/awa2.yaml    #AWA2
```
T rain:
```shell
 python train.py
```
Eval:

```shell
 python test.py
```

You can test our trained model: [CUB](https://drive.google.com/file/d/1bRbb6DzwWccwxhCkREuAadYL73jgVgfe/view?usp=share_link), [AwA2](https://drive.google.com/file/d/1ekXylwVbIY9QAbXmQe-Gwk1vk52qfEby/view?usp=share_link), [SUN](https://drive.google.com/file/d/1BEL_Sth2ZdmNaPBrF01Yub70xnIL6YlR/view?usp=share_link).

## ❗ Cite:
If this work is helpful for you, please cite our paper.

```
@InProceedings{Liu_2023_CVPR,
    author    = {Liu, Man and Li, Feng and Zhang, Chunjie and Wei, Yunchao and Bai, Huihui and Zhao, Yao},
    title     = {Progressive Semantic-Visual Mutual Adaption for Generalized Zero-Shot Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {15337-15346}
}
```

## 📕 Ackowledgement
We thank the following repos providing helpful components in our work.
[GEM-ZSL](https://github.com/osierboy/GEM-ZSL)
