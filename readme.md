<!--
 * @Author: xuarehere xuarehere@foxmail.com.com
 * @Date: 2025-05-19 11:37:03
 * @LastEditTime: 2025-05-20 10:25:28
 * @LastEditors: xuarehere xuarehere@sutpc.com
 * @Description: 
 * @FilePath: /benchmarking-for-backnone/readme.md
 * 
-->
# 说明

```
models_benchmark.py 是一个用于测试各种模型的推理速度的脚本。
它使用了 PyTorch 提供的模型库，包括 ResNet、EfficientNet、Inception、MobileNet、RegNet、ConvNext 和 Swin Transformer 等。
脚本会加载每个模型，并使用随机输入数据进行推理，然后计算平均推理时间和每秒帧数（FPS）。
```
通过修改 `models_benchmark.py` 中的 `model_names` 列表，您可以选择要测试的模型。

# 环境  
```
python==3.10.12
pytorch==2.0.1
torchvision==0.15.2
transformers==4.31.0
```
GPU:V100
# 运行
```
python models_benchmark.py 
```


# 结果
| Model Name                  | Avg Time(ms) |    FPS | Acc Top-1 (%) |
|:----------------------------|-------------:|-------:|--------------:|
| convnext_tiny               |         8.95 |  111.7 |         82.52 |
| convnext_small              |        16.96 |   59.0 |         83.62 |
| convnext_base               |        16.46 |   60.7 |         84.06 |
| DenseNet121                 |        27.94 |   35.8 |         74.43 |
| DenseNet161                 |        36.74 |   27.2 |         77.14 |
| EfficientNet-B0             |        15.12 |   66.1 |         77.69 |
| EfficientNet-B1             |        21.12 |   47.4 |         78.64 |
| EfficientNet-B2             |        21.61 |   46.3 |         80.61 |
| EfficientNet-B3             |        23.40 |   42.7 |         82.01 |
| EfficientNet-B4             |        28.85 |   34.7 |         83.38 |
| EfficientNet-B5             |        35.95 |   27.8 |         83.44 |
| EfficientNet-B6             |        40.31 |   24.8 |         84.01 |
| EfficientNet-B7             |        48.33 |   20.7 |         84.12 |
| Efficientnet_v2_s           |        30.79 |   32.5 |         84.23 |
| Efficientnet_v2_m           |        50.11 |   20.0 |         85.11 |
| Efficientnet_v2_l           |        62.41 |   16.0 |         85.81 |
| Googlenet                   |        12.78 |   78.3 |         69.78 |
| Inception_v3                |        19.90 |   50.3 |         77.29 |
| mnasnet0_5                  |         8.80 |  113.7 |         67.73 |
| mnasnet0_75                 |         9.42 |  106.1 |         71.18 |
| mnasnet1_0                  |         9.56 |  104.6 |         73.46 |
| mnasnet1_3                  |         9.52 |  105.0 |         76.51 |
| mobilenet_v2                |        10.26 |   97.4 |         71.88 |
| mobilenet_v3_small          |         8.22 |  121.6 |         67.67 |
| mobilenet_v3_large          |        11.51 |   86.9 |         74.04 |
| mobilenetv4_conv_small_035  |         8.12 |  123.1 |         72.30 |
| mobilenetv4_conv_small_050  |         8.17 |  122.4 |         74.10 |
| mobilenetv4_conv_small      |        11.21 |   89.2 |         75.80 |
| mobileone_s0                |         5.44 |  183.9 |         71.60 |
| repvit_m0_9                 |        23.80 |   42.0 |         72.10 |
| fastvit_t8                  |        20.05 |   49.9 |         72.80 |
| fastvit_s12                 |        23.54 |   42.5 |         75.20 |
| fastvit_mci1                |        57.80 |   17.3 |         76.80 |
| tiny_vit_5m_224             |        16.28 |   61.4 |         69.10 |
| tiny_vit_21m_224            |        16.07 |   62.2 |         81.20 |
| tiny_vit_21m_384            |        16.41 |   60.9 |         83.20 |