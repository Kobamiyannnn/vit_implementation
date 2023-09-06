import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchsummary import summary
from vit import Vit

###################
# データセットの用意 #
###################

# CIFAR-10のダウンロード
train_data = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 128  # ミニバッチ学習のバッチサイズ

# データローダの作成
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# CIFAR-10の全クラス
classes = [
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]
num_classes = len(classes)

##############
# モデルの用意 #
##############
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
vit = Vit(num_classes=num_classes).to(device)
# summary(vit, (3, 32, 32), batch_size=2, device=device)  # モデルの構造確認

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    vit.parameters(),
    betas=[0.9, 0.999],
    weight_decay=0.1,
)

