import os
import glob
import pandas as pd

# 收集所有图片路径和标签
data = []

# 处理 class 0 (Black/真实图片) - 标签应该是1
black_images = glob.glob('data/0/*.*')
for img_path in sorted(black_images):
    relative_path = img_path
    data.append(f"{relative_path},1")  # 真实图片标签为1

# 处理 class 1 (White/伪造图片) - 标签应该是0
white_images = glob.glob('data/1/*.*')
for img_path in sorted(white_images):
    relative_path = img_path
    data.append(f"{relative_path},0")  # 伪造图片标签为0

# 写入train.csv
with open('train.csv', 'w') as f:
    f.write('Path,Label\n')
    f.write('\n'.join(data))

print(f"✅ 已重新生成 train.csv")
print(f"总样本数: {len(data)}")
print(f"Class 1 (真实/Black): {len(black_images)} 张")
print(f"Class 0 (伪造/White): {len(white_images)} 张")
print(f"\n标签对应关系:")
print(f"  0 = 伪造 (White)")
print(f"  1 = 真实 (Black)")
