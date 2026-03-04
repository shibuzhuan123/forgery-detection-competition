import os
import glob

# 收集所有图片路径和标签
data = []

# 处理 class 0 (Black/真实图片)
black_images = glob.glob('data/0/*.*')
for img_path in sorted(black_images):
    relative_path = img_path  # 保持相对路径
    data.append(f"{relative_path},0")

# 处理 class 1 (White/伪造图片)
white_images = glob.glob('data/1/*.*')
for img_path in sorted(white_images):
    relative_path = img_path
    data.append(f"{relative_path},1")

# 写入train.csv
with open('train.csv', 'w') as f:
    f.write('Path,Label\n')
    f.write('\n'.join(data))

print(f"Generated train.csv with {len(data)} samples")
print(f"Class 0 (Black/Real): {len(black_images)} images")
print(f"Class 1 (White/Fake): {len(white_images)} images")
