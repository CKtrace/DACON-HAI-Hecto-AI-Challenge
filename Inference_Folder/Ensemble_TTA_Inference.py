import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
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
    'BATCH_SIZE': 64,
    'SEED': 888,
}

DATA_DIR = 'Hecto_Data/'
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
SUBMIT_CSV = os.path.join(DATA_DIR, 'sample_submission.csv')

SERES_PATH = 'SeResNext_OptunaCutMix_Halfcrop_Noise.pth'
REGNETY_PATH = 'Regnety_OptunaCutMix_Halfcrop_Noise.pth'
EFF_PATH = 'EffNetV2_OptunaCutMix_Halfcrop_Noise.pth'
MAX_PATH = 'MaxVit_OptunaCutMix_Halfcrop_Noise.pth'
CONV_PATH = 'ConvNextV2_OptunaCutMix_Halfcrop_Noise_0ep.pth'

regnety_weight = 0.35
conv_weight = 0.35
eff_weight = 0.1
seres_weight = 0.1
maxvit_weight = 0.1


tta_transforms = [
    A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ]),

    A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ]),

    A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Affine(scale=(0.95, 1.05), rotate=(-3, 3), p=1.0),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ]),

    A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Perspective(scale=(0.03, 0.05), p=1.0),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ]),

    A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.GridDistortion(num_steps=3, distort_limit=0.15, p=1.0),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ]),
]


class TestDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.root_dir, row['img_path'])).convert("RGB")
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)['image']
        return image

class CustomModel(torch.nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False)
        self.backbone.reset_classifier(num_classes)

    def forward(self, x):
        return self.backbone(x)

def load_model(model_name, weight_path, num_classes):
    model = CustomModel(model_name, num_classes)
    state = torch.load(weight_path, map_location=DEVICE)
    if list(state.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            new_state[k.replace('module.', '')] = v
        state = new_state
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    return model

sample = pd.read_csv(SUBMIT_CSV)
num_classes = len(sample.columns) - 1

seres_model = load_model('seresnextaa101d_32x8d.sw_in12k_ft_in1k', SERES_PATH, num_classes)
regenty_model = load_model('regnety_160.swag_ft_in1k', REGNETY_PATH, num_classes)
eff_model = load_model('tf_efficientnetv2_l.in21k_ft_in1k', EFF_PATH, num_classes)
max_model = load_model('maxvit_small_tf_384.in1k', MAX_PATH, num_classes)
conv_model = load_model('convnextv2_base.fcmae_ft_in22k_in1k', CONV_PATH, num_classes)

final_logits_list = []

for tta in tta_transforms:
    dataset = TestDataset(TEST_CSV, DATA_DIR, transform=tta)
    loader = DataLoader(dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4)
    tta_logits = []

    with torch.no_grad():
        for images in tqdm(loader, desc="TTA Ensemble Inference"):
            images = images.to(DEVICE)
            seres_out = seres_weight * seres_model(images)
            regnety_out = regnety_weight * regenty_model(images)
            eff_out = eff_weight * eff_model(images)
            max_out = maxvit_weight * max_model(images)
            conv_out = conv_weight * conv_model(images)


            blended_logits = seres_out + regnety_out + eff_out + max_out + conv_out
            tta_logits.append(blended_logits.cpu().numpy())

    final_logits_list.append(np.concatenate(tta_logits, axis=0))

mean_logits = np.mean(final_logits_list, axis=0)
final_probs = torch.softmax(torch.tensor(mean_logits), dim=1).numpy()


test_ids = pd.read_csv(TEST_CSV)['ID']

probs_df = pd.DataFrame(final_probs.astype(np.float64), columns=sample.columns[1:])

submission = pd.concat([test_ids, probs_df], axis=1)

submission.to_csv('Main_Base_ver4_CH.csv', index=False, float_format='%.10f')
print("✅ Saved: Main_Base.csv")
