import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm
import gc

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gc.collect()
torch.cuda.empty_cache()

CFG = {
    'IMG_SIZE': 384,
    'BATCH_SIZE': 16,
    'EPOCHS': 20,
    # 'LEARNING_RATE': 2e-5, -> 성능 저하
    'LEARNING_RATE': 8.301253031045151e-05,
    'SEED': 888,
    'cutmix_prob': 0.5
}

DATA_DIR = 'Hecto_Data/'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
SUBMIT_CSV = os.path.join(DATA_DIR, 'sample_submission.csv')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['SEED'])

class VerticalHalfCrop:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, **kwargs):
        if random.random() > self.p:
            return image  

        h, w, _ = image.shape
        top_or_bottom = random.choice(["top", "bottom"])

        if top_or_bottom == "top":
            cropped = image[:h // 2, :, :]
        else:
            cropped = image[h // 2:, :, :]

        cropped = np.ascontiguousarray(cropped)
        resized = A.Resize(h, w)(image=cropped)['image']
        return resized  


train_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.HorizontalFlip(p=0.5),
    A.Lambda(image=VerticalHalfCrop(p=0.1)), # 추가
    A.ColorJitter(p=0.3), # 추가
    A.GaussianBlur(blur_limit=(3, 7), p=0.1), # 추가
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0), # 추가
    A.RandomGamma(gamma_limit=(80, 120), p=1.0), # 추가
    A.Affine(rotate=(-6, 6), translate_percent=0.05, scale=(0.9, 1.1), p=0.4),
    A.Perspective(scale=(0.04, 0.08), p=0.2),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=10, p=0.2),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, noisy_list=None):
        self.root_dir = root_dir
        self.transform = transform
        self.noisy_list = noisy_list if noisy_list is not None else set()

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []

        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith('.jpg') and fname not in self.noisy_list:
                    img_path = os.path.join(cls_folder, fname)
                    label = self.class_to_idx[cls]
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label

noisy_list = {
    "5시리즈_G60_2024_2025_0010.jpg",
    "6시리즈_GT_G32_2018_2020_0018.jpg",
    "7시리즈_G11_2016_2018_0040.jpg",
    "911_992_2020_2024_0030.jpg",
    "E_클래스_W212_2010_2016_0022.jpg",
    "K5_2세대_2016_2018_0007.jpg",
    "F150_2004_2021_0018.jpg",
    "G_클래스_W463b_2019_2025_0030.jpg",
    "GLE_클래스_W167_2019_2024_0068.jpg",
    "Q5_FY_2021_2024_0032.jpg",
    "Q30_2017_2019_0075.jpg",
    "Q50_2014_2017_0031.jpg",
    "SM7_뉴아트_2008_2011_0053.jpg",
    "X3_G01_2022_2024_0029.jpg",
    "XF_X260_2016_2020_0023.jpg",
    "뉴_ES300h_2013_2015_0000.jpg",
    "뉴_G80_2025_2026_0042.jpg",
    "뉴_G80_2025_2026_0043.jpg",
    "뉴_SM5_임프레션_2008_2010_0033.jpg",
    "더_기아_레이_EV_2024_2025_0078.jpg",
    "더_뉴_K3_2세대_2022_2024_0001.jpg",
    "더_뉴_그랜드_스타렉스_2018_2021_0078.jpg",
    "더_뉴_그랜드_스타렉스_2018_2021_0079.jpg",
    "더_뉴_그랜드_스타렉스_2018_2021_0080.jpg",
    "더_뉴_아반떼_2014_2016_0031.jpg",
    "더_뉴_파사트_2012_2019_0067.jpg",
    "레니게이드_2019_2023_0041.jpg",
    "박스터_718_2017_2024_0011.jpg",
    "싼타페_TM_2019_2020_0009.jpg",
    "아반떼_MD_2011_2014_0081.jpg",
    "아반떼_N_2022_2023_0064.jpg",
    "익스플로러_2016_2017_0072.jpg",
    "콰트로포르테_2017_2022_0074.jpg",
    "프리우스_4세대_2019_2022_0052.jpg",
    "아반떼_N_2022_2023_0035.jpg"
}

full_dataset = CustomImageDataset(TRAIN_DIR, transform=train_transform, noisy_list=noisy_list)
targets = [label for _, label in full_dataset.samples]
class_names = full_dataset.classes

train_idx, val_idx = train_test_split(range(len(targets)), test_size=0.2, stratify=targets, random_state=CFG['SEED'])

train_dataset = Subset(CustomImageDataset(TRAIN_DIR, transform=train_transform, noisy_list=noisy_list), train_idx)
val_dataset = Subset(CustomImageDataset(TRAIN_DIR, transform=val_transform, noisy_list=noisy_list), val_idx)

train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=2, pin_memory=True, persistent_workers=False)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=2, pin_memory=True, persistent_workers=False)

def cutmix(images, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(images.size(0)).to(images.device)
    target_a = labels
    target_b = labels[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
    return images, target_a, target_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

class SEResNeXtModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', pretrained=True)
        self.backbone.reset_classifier(num_classes)

    def forward(self, x):
        return self.backbone(x)

model = SEResNeXtModel(num_classes=len(class_names)).to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.00038159832754892844)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=11)

best_logloss = float('inf')

for epoch in range(CFG['EPOCHS']):
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        if np.random.rand() < CFG['cutmix_prob'] and epoch != 0:
            images, targets_a, targets_b, lam = cutmix(images, labels)
            outputs = model(images)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"[Epoch {epoch+1}] Validation"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    scheduler.step()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    val_logloss = log_loss(all_labels, all_probs, labels=list(range(len(class_names))))

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {val_accuracy:.2f}%")

    if val_logloss < best_logloss:
        best_logloss = val_logloss
        torch.save(model.state_dict(), 'ConvNextV2_OptunaCutMix_Halfcrop_Noise_0ep.pth')
        print(f" Best model saved at epoch {epoch+1} (logloss: {val_logloss:.4f})")
