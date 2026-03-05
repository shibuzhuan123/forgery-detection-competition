from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torchvision.models as models
import random
from torch.optim import lr_scheduler
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import os

seed = 8079
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class FixedRotation(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        return fixed_rotate(img, self.angles)

def fixed_rotate(img, angles):
    angles = list(angles)
    angles_num = len(angles)
    index = random.randint(0, angles_num - 1)
    return img.rotate(angles[index])

# 训练数据增强 - 温和版
train_transform = transforms.Compose([
    transforms.Resize([384, 384]),
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转±10度（减小角度）
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),  # 颜色抖动（减小强度，去掉hue）
    transforms.ToTensor(),
    transforms.Normalize((0.3705, 0.3828, 0.3545), (0.1685, 0.1590, 0.1536))
])

# 验证数据转换（不增强）
val_transform = transforms.Compose([
    transforms.Resize([384, 384]),
    transforms.ToTensor(),
    transforms.Normalize((0.3705, 0.3828, 0.3545), (0.1685, 0.1590, 0.1536))
])

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        df = pd.read_csv(csv_file)
        self.image_paths = df['Path'].values
        self.labels = df['Label'].values
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        label = int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def validate(model, dataloader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    val_loss /= total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc

def get_cur_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# 读取数据并划分训练集和验证集
print("正在加载数据...")
df = pd.read_csv('./train.csv')
print(f"总数据量: {len(df)}")
print(f"类别分布: {df['Label'].value_counts().to_dict()}")

# 划分训练集和验证集 (90% 训练, 10% 验证)
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['Label'], random_state=seed)
print(f"训练集数量: {len(train_df)}")
print(f"验证集数量: {len(val_df)}")
print(f"训练集类别分布: {train_df['Label'].value_counts().to_dict()}")
print(f"验证集类别分布: {val_df['Label'].value_counts().to_dict()}")

# 保存划分后的数据
train_df.to_csv('./train_split.csv', index=False)
val_df.to_csv('./val_split.csv', index=False)
print("数据划分已保存到 train_split.csv 和 val_split.csv")

# 创建训练集和验证集
train_dataset = ImageDataset(csv_file='./train_split.csv', root_dir='./', transform=train_transform)
val_dataset = ImageDataset(csv_file='./val_split.csv', root_dir='./', transform=val_transform)

# 计算类别权重用于加权采样
train_labels = train_df['Label'].values
class_counts = np.bincount(train_labels)
class_weights = 1. / class_counts
samples_weights = class_weights[train_labels]

print(f"\n类别权重: {class_weights}")
print(f"使用加权采样器平衡类别分布")

# 创建加权采样器
sampler = WeightedRandomSampler(
    weights=samples_weights,
    num_samples=len(samples_weights),
    replacement=True
)

# 创建数据加载器
trainloader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=0)
valloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"训练集batch数: {len(trainloader)}")
print(f"验证集batch数: {len(valloader)}")

# 模型设置 - 使用torchvision的EfficientNet
epoch_num = 20  # 增加训练轮数
print("正在加载EfficientNet-B1模型...")
net = models.efficientnet_b1(weights=None)  # 不加载预训练权重，从头训练

# 修改最后一层为二分类
num_ftrs = net.classifier[1].in_features
net.classifier[1] = nn.Linear(num_ftrs, 2)
net = net.cuda()

print(f"模型已加载，参数量: {sum(p.numel() for p in net.parameters()):,}")
print("注意：使用随机初始化权重（未加载预训练权重）")
print("数据增强：随机裁剪、旋转、翻转、颜色抖动、平移缩放")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

best_val_acc = 0
patience = 5  # 早停耐心值
patience_counter = 0  # 当前等待计数

if __name__ == '__main__':
    print("\n开始训练...")
    for epoch in range(epoch_num):
        print("========== epoch: [{}/{}] ==========".format(epoch + 1, epoch_num))

        # 训练阶段
        net.train()
        correct = 0
        total = 0
        epoch_loss = 0.0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

            if i % 10 == 0:
                train_acc = 100.0 * correct / total
                avg_loss = epoch_loss / (i + 1)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} train acc: {:.6f}  lr : {:.6f}'.format(
                    epoch+1, i * len(inputs), len(trainloader.dataset),
                    100. * i / len(trainloader), avg_loss, train_acc, get_cur_lr(optimizer)))

        # 验证阶段
        val_loss, val_acc = validate(net, valloader, criterion)
        print(f'Validation Loss: {val_loss:.6f}, Validation Accuracy: {val_acc:.6f}')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0  # 重置早停计数
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, './best_model.pth')
            print(f'✅ Best model saved with val_acc: {val_acc:.2f}%')
        else:
            patience_counter += 1
            print(f'⏳ No improvement for {patience_counter} epoch(s) (patience={patience})')

        # 每个epoch保存一次checkpoint
        if epoch % 1 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'val_acc': val_acc,
                'val_loss': val_loss
            }, f'./checkpoint_epoch_{epoch}.pth')

        # 早停检查
        if patience_counter >= patience:
            print(f'\n⚠️ Early stopping triggered! No improvement for {patience} epochs.')
            print(f'Best validation accuracy: {best_val_acc:.2f}% at epoch {epoch - patience + 1}')
            break

        lr_scheduler.step()

    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")
