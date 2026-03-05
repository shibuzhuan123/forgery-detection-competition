import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
import json

# 设置随机种子
seed = 8079
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 测试数据转换
val_transform = transforms.Compose([
    transforms.Resize([384, 384]),
    transforms.ToTensor(),
    transforms.Normalize((0.3705, 0.3828, 0.3545), (0.1685, 0.1590, 0.1536))
])

# 加载模型
print("正在加载模型...")
net = models.efficientnet_b1(weights=None)
num_ftrs = net.classifier[1].in_features
net.classifier[1] = nn.Linear(num_ftrs, 2)

# 加载训练好的权重
checkpoint = torch.load('./best_model.pth')
net.load_state_dict(checkpoint['model_state_dict'])
net = net.cuda()
net.eval()

print(f"模型已加载，验证准确率: {checkpoint['val_acc']:.2f}%")

# 测试数据路径
test_dir = './ForgeryAnalysis_Stage_1_Test/Image'

# 获取所有测试图片
test_images = sorted([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
print(f"找到 {len(test_images)} 张测试图片")

# 预测
results = []
print("开始预测...")

with torch.no_grad():
    for img_name in tqdm(test_images):
        img_path = os.path.join(test_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = val_transform(image).unsqueeze(0).cuda()

        output = net(image)
        pred = output.argmax(dim=1).item()

        # 创建结果 (注意: 0=伪造, 1=真实)
        result = {
            'image_name': img_name,
            'label': pred,
            'location': '',  # 可选：伪造区域定位
            'explanation': '真实图片' if pred == 1 else '伪造图片'  # 1=真实, 0=伪造
        }
        results.append(result)

# 保存结果
df_results = pd.DataFrame(results)
output_file = './submission.csv'
df_results.to_csv(output_file, index=False)

print(f"\n预测完成！")
print(f"结果已保存到: {output_file}")
print(f"\n预测统计:")
print(f"伪造图片 (label=0): {sum(1 for r in results if r['label'] == 0)}")
print(f"真实图片 (label=1): {sum(1 for r in results if r['label'] == 1)}")

# 显示前5个预测结果
print(f"\n前5个预测结果:")
for i, r in enumerate(results[:5]):
    print(f"{i+1}. {r['image_name']} -> {r['label']} ({r['explanation']})")
