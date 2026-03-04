from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torch
import random
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import timm
from timm.scheduler.cosine_lr import CosineLRScheduler
import os

train_path = './data'
# test_path = './train'

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

train_transform = transforms.Compose([
   # transforms.RandomRotation(15),
    transforms.Resize([384,384]),
   # transforms.RandomVerticalFlip(),
   transforms.RandomHorizontalFlip(),
   # FixedRotation([0, 180, ]),
    #transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize((0.3705, 0.3828, 0.3545), (0.1685, 0.1590, 0.1536))
   # transforms.Normalize((0.696508, 0.705005, 0.719835), (0.341564, 0.332224, 0.332470))
])
val_transform = transforms.Compose([
    transforms.Resize([384, 384]),
    transforms.ToTensor(),
  transforms.Normalize((0.3705, 0.3828, 0.3545), (0.1685, 0.1590, 0.1536))
   # transforms.Normalize((0.696508, 0.705005, 0.719835), (0.341564, 0.332224, 0.332470))
])

# train_transform2 = transforms.Compose([
#     transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
#     transforms.Resize([256,256]),
#     transforms.ToTensor(),
#     transforms.Normalize((0.696508, 0.705005, 0.719835), (0.341564, 0.332224, 0.332470)),
# ])


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
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
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

train_dataset=ImageDataset(csv_file='./train.csv',root_dir='./',transform=train_transform)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=16,shuffle=True, num_workers=0)

# test_dataset=ImageFolder(test_path,transform=val_transform)
# testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64,shuffle=None, num_workers=0)
model_name = "efficientnet_b1"
epoch_num=10
net=timm.create_model(model_name,pretrained=True,num_classes=2).cuda()
# weight = torch.load('./efficientnet_b0lab_8.pth')
# net.load_state_dict(weight['model_state_dict'])
#criterion=FocalLoss(0.5)
criterion=nn.CrossEntropyLoss()
# weight_decay = 1e-4  # L2 正则化系数
optimizer = torch.optim.Adam(net.parameters(), lr=0.001,weight_decay=1e-4)
#lr_scheduler = CosineLRScheduler(optimizer, t_initial=0.001, lr_min=0.0001)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
PATH = '/'+model_name  #这里也需要注意，你需要先新建一个dir文件夹，一会在这个文件下存放权重
pre_acc = 0
def get_cur_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

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
best_val_acc=0
if __name__ == '__main__':
    correct = 0
    total = 0
    for epoch in range(epoch_num):
        print("========== epoch: [{}/{}] ==========".format(epoch + 1, epoch_num))
        for i, (inputs, labels) in tqdm(enumerate(trainloader)):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            train_acc = 100.0 * correct / total
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} train acc: {:.6f}  lr : {:.6f}'.format(epoch+1, i * len(inputs),
                                                                                                              len(trainloader.dataset),100. * i / len(trainloader),loss.item(),train_acc,get_cur_lr(optimizer)))
        if epoch % 1 ==0:
            torch.save({'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        },'./efficientnet_b1_384'+PATH + "_"+str(epoch)+".pth")
        # print('===================test============================')
        # val_loss, val_acc = validate(net, testloader, criterion)
        # print(f'Validation Loss: {val_loss:.6f}, Validation Accuracy: {val_acc:.6f}')
        
        # Save the model if it has the best accuracy so far
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        # torch.save({'epoch': epoch,
        #             'model_state_dict': net.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),tra
        #             }, './weights2/'+PATH + "_best.pth")
        
        # Save the model checkpoint
        
        
        lr_scheduler.step()