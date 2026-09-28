import torch
import torch.nn as nn
import torch.optim as optim
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader, Dataset
import random
import os
from PIL import Image
from torchvision import transforms


# --- 1. 自定义三元组数据集 ---
class TripletLFW(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.people = [d for d in os.listdir(root_dir) if len(os.listdir(os.path.join(root_dir, d))) >= 2]

    def __len__(self):
        return len(self.people) * 10  # 假设每个 epoch 跑 10 轮循环

    def __getitem__(self, idx):
        # 随机选一个人作为 Anchor 和 Positive
        person_a = random.choice(self.people)
        path_a = os.path.join(self.root_dir, person_a)
        img_names = os.listdir(path_a)

        # 选两张不同的图
        anc_name, pos_name = random.sample(img_names, 2)

        # 随机选另一个不同的人作为 Negative
        person_b = random.choice([p for p in self.people if p != person_a])
        neg_name = random.choice(os.listdir(os.path.join(self.root_dir, person_b)))

        # 读取并转换
        anc = self.transform(Image.open(os.path.join(path_a, anc_name)).convert('RGB'))
        pos = self.transform(Image.open(os.path.join(path_a, pos_name)).convert('RGB'))
        neg = self.transform(Image.open(os.path.join(self.root_dir, person_b, neg_name)).convert('RGB'))

        return anc, pos, neg


# --- 2. 训练配置 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 实例化模型（这里如果不传 pretrained，就是从零开始训练）
model = InceptionResnetV1(classify=False).to(device)

# 三元组损失函数 (Margin 设为 0.2)
triplet_loss = nn.TripletMarginLoss(margin=0.2, p=2)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 数据增强与加载
data_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])
train_ds = TripletLFW(root_dir='./data/lfw_cropped', transform=data_transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# --- 3. 训练循环 ---
model.train()
print("开始三元组训练演示...")
for epoch in range(1):  # 演示只跑 1 个 epoch
    running_loss = 0.0
    for i, (anc, pos, neg) in enumerate(train_loader):
        anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)

        optimizer.zero_grad()

        # 提取三张图的 Embedding
        embed_anc = model(anc)
        embed_pos = model(pos)
        embed_neg = model(neg)

        # 计算 Loss
        loss = triplet_loss(embed_anc, embed_pos, embed_neg)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 0:
            print(f"Batch {i}, Loss: {loss.item():.4f}")

print("训练演示结束。")