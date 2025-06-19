# HAI(하이)! - Hecto AI Challenge : 2025 상반기 헥토 채용 AI 경진대회

```
Final Ranking : 25/748 (Top 3.3%) (인재풀 등록 대상)
```


</br>

## Introduction

</br>

__This repository is a place to share "[HAI(하이)! - Hecto AI Challenge : 2025 상반기 헥토 채용 AI 경진대회](https://dacon.io/competitions/official/236493/overview/description)" solution code.__

</br>

```
주최 : 헥토(Hecto)
주관 : 데이콘
```
<br>

## Development Environment
<br>

```
GPU : NVIDIA GeForce RTX 4090
OS  : Ubuntu22.04
```

</br>

## Key Library Versions

<br>

- Python = 3.10
- Pytorch = 2.1.0 + cu118
- Cuda = 11.8
- pandas = 2.3.0
- Timm = 1.0.15
- Albumentations = 2.0.8
- Numpy = 1.26.4
- Pillow = 9.3.0
- scikit-learn = 1.7.0

<br>

## Repository Structure

<br>

```
├─ README.md
│  
├─ Model_Folder
│       ├─ ConvNextV2_OptunaCutMix_Halfcrop_Noise_0ep.pth
│       ├─ EffNetV2_OptunaCutMix_Halfcrop_Noise.pth
│       ├─ MaxVit_OptunaCutMix_Halfcrop_Noise.pt
│       ├─ Regnety_OptunaCutMix_Halfcrop_Noise.pth
│       └─ SeResNext_OptunaCutMix_Halfcrop_Noise.pth
│
├─ Train_Folder
│       ├─ ConvNextV2_Noise_0ep.py
│       ├─ EffNetV2_Noise.py
│       ├─ MaxVit_Noise.py
│       ├─ Regnety_Noise.py
│       └─ SeResNet_Noise.py
│
└─ Inference_Folder
        ├─ Ensemble_TTA_Inference.py           
```

<br>




