from flask import Flask, request, jsonify
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import torchvision.datasets
app = Flask(__name__)

# 模型缓存
models_cache = {}

# 加载模型
def load_model(model_name):
    if model_name in models_cache:
        return models_cache[model_name]
    
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name == 'mobilenetv3':
        model = models.mobilenet_v3_large(pretrained=True)
    else:
        return None
    
    model.eval()
    models_cache[model_name] = model
    return model

# 图像预处理
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# 加载标签
def load_labels():
    with open('imagenet_classes.json') as f:
        return json.load(f)

labels = load_labels()

# 推理API
@app.route('/api/inference', methods=['POST'])
def inference():
    try:
        # 获取请求参数
        model_name = request.form['model']
        top_k = int(request.form.get('top_k', 5))
        
        # 检查文件是否存在
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        
        # 检查文件类型
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # 加载模型
        model = load_model(model_name)
        if model is None:
            return jsonify({'error': 'Model not supported'}), 400
        
        # 处理图像
        image = Image.open(image_file).convert('RGB')
        input_tensor = preprocess_image(image)
        
        # 进行推理
        with torch.no_grad():
            output = model(input_tensor)
        
        # 获取预测结果
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, top_k)
        
        # 准备返回结果
        results = []
        for i in range(top_k):
            results.append({
                'label': labels[top_catid[i].item()],
                'confidence': top_prob[i].item()
            })
        
        return jsonify({
            'success': True,
            'predictions': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='localhost', port=5001, debug=True)