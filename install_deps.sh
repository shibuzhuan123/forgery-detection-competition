#!/bin/bash
# 激活环境并安装依赖的脚本

echo "激活ForgeryAnalysis环境..."
source /home/fei/miniconda3/etc/profile.d/conda.sh
conda activate ForgeryAnalysis

echo "安装PyTorch和依赖包..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install timm tqdm pandas pillow

echo "安装完成！"
echo "现在可以运行: python train-cls.py"
