# 图像伪造检测比赛项目

## 📯 比赛概述

这是一个**图像伪造检测**的二分类任务，目标是判断一张图片是**真实的(0)**还是**伪造的(1)**。

### 任务说明
- **输入**: 图像文件
- **输出**: 二分类标签 (0=真实, 1=伪造)
- **数据集规模**:
  - 训练集: 1000张图片 (800真实 + 200伪造)
  - 测试集: 500张图片

---

## 🏗️ 项目结构

```
/home/fei/competition_1/
├── data/                              # 训练数据
│   ├── 0/                            # 真实图片 (800张) - Black目录
│   └── 1/                            # 伪造图片 (200张) - White目录
├── ForgeryAnalysis_Stage_1_Train/     # 原始训练数据
│   ├── Black/                        # 真实图片源数据
│   │   ├── Image/
│   │   ├── Mask/
│   │   └── Caption/
│   └── White/                        # 伪造图片源数据
│       ├── Image/
│       └── Caption/
├── ForgeryAnalysis_Stage_1_Test/      # 测试数据
│   └── Image/                        # 测试图片 (500张)
├── train.csv                          # 训练标签文件 (1000条)
├── train-cls.py                       # 训练脚本
├── train-csl.csv                      # 原始标签文件
├── submit_example.csv                 # 提交格式示例
└── README.md                          # 本文档
```

---

## 🔬 代码思路详解

### 1. **模型架构** (`train-cls.py`)

#### 核心组件

**① 数据集类 (`ImageDataset`)**
```python
class ImageDataset(Dataset):
    - 从CSV文件读取图片路径和标签
    - 支持自定义数据增强
    - 返回 (image, label) 元组
```

**② 模型选择**
- **EfficientNet-B1**: 轻量级高效卷积神经网络
- 预训练权重: ImageNet
- 输出类别: 2 (二分类)
- 输入尺寸: 384×384×3

**③ 损失函数**
- **当前使用**: `CrossEntropyLoss` (交叉熵损失)
- **可选**: `FocalLoss` (已实现但注释掉，适合处理类别不平衡)

**④ 优化器与调度**
- 优化器: Adam (lr=0.001, weight_decay=1e-4)
- 学习率调度: StepLR (每4个epoch × 0.1)
- 训练轮数: 10 epochs
- Batch Size: 16

**⑤ 数据增强**
```python
train_transform = transforms.Compose([
    transforms.Resize([384, 384]),      # 统一尺寸
    transforms.RandomHorizontalFlip(), # 随机水平翻转
    transforms.ToTensor(),              # 转换为张量
    transforms.Normalize(                # 标准化
        mean=(0.3705, 0.3828, 0.3545),
        std=(0.1685, 0.1590, 0.1536)
    )
])
```

### 2. **训练流程**

```
1. 数据加载
   ↓
2. 前向传播 (图像 → 特征 → 预测)
   ↓
3. 计算损失 (CrossEntropyLoss)
   ↓
4. 反向传播 (计算梯度)
   ↓
5. 参数更新 (Adam优化器)
   ↓
6. 学习率衰减 (StepLR)
   ↓
7. 模型保存 (每个epoch保存一次)
```

### 3. **关键技术点**

| 技术 | 作用 | 实现位置 |
|------|------|----------|
| **数据增强** | 增加训练数据多样性，防止过拟合 | 第42-52行 |
| **预训练模型** | 利用ImageNet特征，加速收敛 | 第125行 |
| **L2正则化** | 防止过拟合 (weight_decay=1e-4) | 第131行 |
| **学习率衰减** | 训练后期降低学习率，稳定收敛 | 第133行 |
| **混合精度训练** | 加速训练，节省显存 | 通过CUDA自动启用 |

---

## ✅ 已完成的工作

### ① 环境准备
- [x] 创建Conda环境 `ForgeryAnalysis` (Python 3.10)
- [x] 安装PyTorch及依赖包 (CUDA 12.4版本)
- [x] 安装必要的Python库 (timm, tqdm, pandas, pillow)

### ② 数据处理
- [x] 解压训练和测试数据集
- [x] 创建符号链接，组织数据到 `data/0/` 和 `data/1/`
- [x] 生成训练标签文件 `train.csv` (1000条记录)
- [x] 验证数据完整性 (800真实 + 200伪造)

### ③ 代码准备
- [x] 提供完整的训练脚本 `train-cls.py`
- [x] 提供依赖安装脚本 `install_deps.sh`
- [x] 创建项目文档 `README.md`

### ④ 数据统计
```
训练数据分布:
- Class 0 (真实): 800张 (80%)
- Class 1 (伪造): 200张 (20%)
- 比例: 4:1 (存在类别不平衡)

测试数据:
- 总计: 500张图片
- 格式: jpg (470张) + png (30张)
```

---

## 🚀 快速开始

### 环境设置

#### 方式1: 自动安装（推荐）
```bash
cd /home/fei/competition_1
chmod +x install_deps.sh
./install_deps.sh
```

#### 方式2: 手动安装
```bash
# 激活环境
conda activate ForgeryAnalysis

# 安装PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖
pip install timm tqdm pandas pillow
```

### 开始训练
```bash
cd /home/fei/competition_1
conda activate ForgeryAnalysis
python train-cls.py
```

### 预期输出
```
========== epoch: [1/10] ==========
Train Epoch: 1 [0/1000 (0%)]	Loss: 0.693147 train acc: 20.000000  lr : 0.001000
Train Epoch: 1 [160/1000 (16%)]	Loss: 0.523456 train acc: 45.500000  lr : 0.001000
...
```

---

## ⚠️ 代码存在的问题与优化建议

### 1. **类别不平衡严重** ⭐⭐⭐
**问题**: 训练数据比例为 4:1 (800 vs 200)
**影响**: 模型会偏向预测多数类（真实图片）
**解决方案**:
- 使用加权损失函数
- 使用FocalLoss (代码已实现)
- 对少数类进行过采样

### 2. **缺少验证集** ⭐⭐⭐
**问题**: 没有验证集来监控模型性能
**影响**: 无法及时发现过拟合，无法选择最佳模型
**解决方案**:
```python
from sklearn.model_selection import train_test_split
# 划分训练集和验证集 (8:2)
train_data, val_data = train_test_split(data, test_size=0.2, stratify=labels)
```

### 3. **训练轮数不足** ⭐⭐
**问题**: 只训练10个epoch可能不够
**建议**: 增加到20-50个epoch

### 4. **数据增强较弱** ⭐
**问题**: 只使用了水平翻转
**建议**: 添加更多增强策略
```python
transforms.Compose([
    transforms.RandomRotation(15),           # 随机旋转
    transforms.RandomHorizontalFlip(),       # 随机水平翻转
    transforms.ColorJitter(0.2, 0.2, 0.2),   # 颜色抖动
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 平移
])
```

### 5. **模型保存策略** ⭐
**问题**: 每个epoch都保存，没有保存最佳模型
**建议**: 只保存验证集上表现最好的模型

### 6. **Batch Size较小** ⭐
**问题**: batch_size=16可能导致训练不稳定
**建议**: 如果显存允许，增加到32或64

### 7. **缺少Early Stopping** ⭐
**问题**: 没有早停机制
**建议**: 监控验证集损失，连续5个epoch不下降则停止

---

## 📊 提交格式说明

### 提交文件格式
测试集预测需要生成CSV文件，包含以下字段：

```csv
image_name,label,location,explanation
956a8325a662463fa5386a99e082dc84.jpg,0,"{""size"": [1024, 768], ""counts"": ""1T2:""}","真实图片"
```

### 字段说明
- **image_name**: 测试图片文件名
- **label**: 预测标签 (0=真实, 1=伪造)
- **location**: 伪造区域的RLE格式编码 (可选)
- **explanation**: 判断依据的文字说明 (可选)

### 评分标准
- 主要指标: 分类准确率
- 可能考虑: 伪造区域的定位精度

---

## 🎯 下一步行动建议

### 立即可做
1. ✅ 运行baseline代码，查看初始效果
2. ✅ 添加验证集，监控训练过程
3. ✅ 实现Early Stopping机制

### 优化提升
1. 🔧 处理类别不平衡问题
2. 🔧 增加数据增强策略
3. 🔧 尝试不同的模型 (EfficientNet-B2/B3, ViT等)
4. 🔧 调整超参数 (learning rate, batch size等)

### 高级技巧
1. ⭐ 使用Test Time Augmentation (TTA)
2. ⭐ 模型集成 (Model Ensemble)
3. ⭐ 伪标签 (Semi-supervised Learning)

---

## 📝 参考资料

- EfficientNet论文: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- Focal Loss论文: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- PyTorch文档: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- timm库文档: [https://rwightman.github.io/pytorch-image-models/](https://rwightman.github.io/pytorch-image-models/)

---

## 🏆 总结

本项目是一个典型的图像二分类任务，使用EfficientNet-B1作为基础模型。当前代码提供了完整的训练流程，但存在类别不平衡、缺少验证集等问题。通过系统性的优化（处理不平衡、添加验证、增强数据、调整超参数），可以显著提升模型性能。

**关键成功因素**:
1. 正确处理类别不平衡
2. 合理的数据增强策略
3. 有效的验证和早停机制
4. 合适的模型选择和超参数调整

祝比赛顺利！🎉
