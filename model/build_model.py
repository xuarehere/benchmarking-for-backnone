'''
Author: xuarehere xuarehere@foxmail.com.com
Date: 2025-04-09 15:55:24
LastEditTime: 2025-05-19 19:16:56
LastEditors: xuarehere xuarehere@foxmail.com.com
Description: 
FilePath: /benchmarking-for-backnone/model/build_model.py

'''
# model.py
import timm
import torch
import torch.nn as nn
from model.mobileone import mobileone, reparameterize_model

def build_mobileone(num_classes=3, pretrained=True, inference_mode=False, variant='s0'):
    """
    apple/ml-mobileone: This repository contains the official implementation of the research paper, "An Improved One millisecond Mobile Backbone" CVPR 2023.
    https://github.com/apple/ml-mobileone
    """
   # 加载预训练模型（原始模型）
    model = mobileone(variant=variant, inference_mode=inference_mode)
    if pretrained == True:
        if inference_mode == True:
            checkpoint = torch.load('./mobileone_s0.pth.tar')         # 融合之后的，用于推理测试
        else:
            checkpoint = torch.load('./mobileone_s0_unfused.pth.tar')   # 训练
        model.load_state_dict(checkpoint)    

    # 修改分类器为指定类别数
    model.linear = nn.Linear(model.linear.in_features, num_classes)
    # model = update_classifier(model, num_classes)
    return model
