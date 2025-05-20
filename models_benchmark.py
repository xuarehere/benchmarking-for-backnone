'''
Author: xuarehere xuarehere@foxmail.com.com
Date: 2025-04-01 19:35:30
LastEditTime: 2025-05-20 10:24:58
LastEditors: xuarehere xuarehere@foxmail.com.com
Description:  
    测试模型FP16的推理速度
FilePath: /benchmarking-for-backnone/models_benchmark.py
/root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
'''
import torch
import time
import torchvision.models as models
from torchvision import transforms
from timm.models import create_model
import sys
from model.build_model import build_mobileone
import model

# ------------ 测试配置 ------------ 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mobileone_s0 = build_mobileone(
    num_classes=3, inference_mode=True, pretrained=False, )
models_to_test = {
    'convnext_tiny': (models.convnext_tiny(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                      models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    'convnext_small': (models.convnext_small(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                       models.ConvNeXt_Small_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    'convnext_base': (models.convnext_base(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                      models.ConvNeXt_Base_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    "DenseNet121": (models.densenet121(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                    models.DenseNet121_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    "DenseNet161": (models.densenet161(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                    models.DenseNet161_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    "EfficientNet-B0": (models.efficientnet_b0(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                        models.EfficientNet_B0_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    "EfficientNet-B1": (models.efficientnet_b1(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                        models.EfficientNet_B1_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    "EfficientNet-B2": (models.efficientnet_b2(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                        models.EfficientNet_B2_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    "EfficientNet-B3": (models.efficientnet_b3(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                        models.EfficientNet_B3_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    "EfficientNet-B4": (models.efficientnet_b4(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                        models.EfficientNet_B4_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    "EfficientNet-B5": (models.efficientnet_b5(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                        models.EfficientNet_B5_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    "EfficientNet-B6": (models.efficientnet_b6(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                        models.EfficientNet_B6_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    "EfficientNet-B7": (models.efficientnet_b7(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                        models.EfficientNet_B7_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    'Efficientnet_v2_s': (models.efficientnet_v2_s(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                          models.EfficientNet_V2_S_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    'Efficientnet_v2_m': (models.efficientnet_v2_m(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                          models.EfficientNet_V2_M_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    'Efficientnet_v2_l': (models.efficientnet_v2_l(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                          models.EfficientNet_V2_L_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    "Googlenet": (models.googlenet(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                  models.GoogLeNet_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    "Inception_v3": (models.inception_v3(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                     models.Inception_V3_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    "mnasnet0_5": (models.mnasnet0_5(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                   models.MNASNet0_5_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    "mnasnet0_75": (models.mnasnet0_75(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                    models.MNASNet0_75_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    "mnasnet1_0": (models.mnasnet1_0(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                   models.MNASNet1_0_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    "mnasnet1_3": (models.mnasnet1_3(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                   models.MNASNet1_3_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    "mobilenet_v2": (models.mobilenet_v2(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                     models.MobileNet_V2_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    'mobilenet_v3_small': (models.mobilenet_v3_small(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                            models.MobileNet_V3_Small_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),
    'mobilenet_v3_large': (models.mobilenet_v3_large(pretrained=False).to(device).eval().half().cuda(), (1, 3, 224, 224),
                            models.MobileNet_V3_Large_Weights.IMAGENET1K_V1.meta['_metrics']['ImageNet-1K']['acc@1']),                        
    "mobilenetv4_conv_small_035": (create_model('mobilenetv4_conv_small_035', num_classes=1000).to(device).eval().half().cuda(), (1, 3, 224, 224),
                                  72.3),  # 72.3%
    "mobilenetv4_conv_small_050": (create_model('mobilenetv4_conv_small_050', num_classes=1000).to(device).eval().half().cuda(), (1, 3, 224, 224),
                                  74.1),  # 74.1%
    "mobilenetv4_conv_small": (create_model('mobilenetv4_conv_small', num_classes=1000).to(device).eval().half().cuda(), (1, 3, 224, 224),
                              75.8),  # 75.8%
    "mobileone_s0": (mobileone_s0.to(device).eval().half().cuda(), (1, 3, 224, 224),
                    71.6),  # 71.6%
    "repvit_m0_9": (create_model('repvit_m0_9', num_classes=1000).to(device).eval().half().cuda(), (1, 3, 224, 224),
                   72.1),  # 72.1%
    "fastvit_t8": (create_model('fastvit_t8', num_classes=1000).to(device).eval().half().cuda(), (1, 3, 224, 224),
                  72.8),  # 72.8%
    "fastvit_s12": (create_model('fastvit_s12', num_classes=1000).to(device).eval().half().cuda(), (1, 3, 224, 224),
                   75.2),  # 75.2%
    "fastvit_mci1": (create_model('fastvit_mci1', num_classes=1000).to(device).eval().half().cuda(), (1, 3, 224, 224),
                    76.8),  # 76.8%
    "tiny_vit_5m_224": (create_model('tiny_vit_5m_224', num_classes=1000).to(device).eval().half().cuda(), (1, 3, 224, 224),
                       69.1),  # 69.1%
    "tiny_vit_21m_224": (create_model('tiny_vit_21m_224', num_classes=1000).to(device).eval().half().cuda(), (1, 3, 224, 224),
                        81.2),  # 81.2%
    "tiny_vit_21m_384": (create_model('tiny_vit_21m_384', num_classes=1000).to(device).eval().half().cuda(), (1, 3, 224, 224),
                        83.2),  # 83.2%
}
# ------------ END: 测试配置 ------------ 



def test_inference_time(model, input_size=(1, 3, 224, 224), device='cuda', repetitions=100):
    # 初始化设置
    model = model.to(device).eval()
    input_tensor = torch.randn(*input_size).to(device).half().cuda()

    # 预热GPU（不统计时间）
    with torch.no_grad():
        try:
            _ = model(input_tensor)
        except:
            input_tensor = torch.randn(*input_size).to(device)
            _ = model(input_tensor)
    torch.cuda.synchronize()

    # 正式测试
    total_time = 0.0
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            _ = model(input_tensor)

        torch.cuda.synchronize()
        end_time = time.time()
        total_time += (end_time - start_time)

    # 计算结果
    avg_time = total_time / repetitions * 1000  # 转换为毫秒
    fps = 1000 / avg_time
    return avg_time, fps

def run(models_to_test, device='cuda', repetitions=1):
    # 执行测试
    print(f"{'Model Name':<15} | {'Avg Time(ms)':>12} | {'FPS':>8} | {'Acc Top-1 (%)':>8}")
    # print(f"{'Model Name':<15} | {'Avg Time(ms)':>12} | {'FPS':>8}")
    print("-"*55)
    for name, (model, input_size, acc) in models_to_test.items():
        avg_time, fps = test_inference_time(model, input_size, device, repetitions=repetitions)
        print(f"{name:<15} | {avg_time:>10.2f} ms | {fps:>7.1f} | {acc:>7.2f}")



def get_model_info(model_name):
    """获取模型信息用于Web界面"""
    if model_name not in models_to_test:
        return None
    model, input_size = models_to_test[model_name]
    return {
        'name': model_name,
        'input_size': input_size,
        'has_cuda': torch.cuda.is_available()
    }

def get_all_model_names():
    """获取所有模型名称用于Web界面"""
    return list(models_to_test.keys())

if __name__ == "__main__":
    run(models_to_test, device=device, repetitions=1)
    