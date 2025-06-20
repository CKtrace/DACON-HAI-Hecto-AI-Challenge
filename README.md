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
        └─ Ensemble_TTA_Inference.py           
```

<br>

## Approach Method


### Noise Image Removal Strategy
The training dataset contained images that were captured from angles that could introduce noise to the training process.

Through empirical experiments, we found that removing such noisy images resulted in lower validation Log Loss compared to retaining them. Based on this observation, we proceeded with excluding these samples from the final training set.

-------

### Mitigating Overconfidence in Model Predictions
Analysis of the prediction outputs revealed that some models were overconfident in specific classes, especially in ambiguous cases.

Since the competition used Log Loss as the evaluation metric, such overconfidence led to significant penalties.

To alleviate this, we employed model ensembling combined with Test Time Augmentation (TTA) to produce softer and more calibrated predictions, which helped improve overall performance.

--------

### Handling Diverse Image Capture Environments
The dataset featured a wide range of image capture conditions—indoors, outdoors, daytime, nighttime, etc.—introducing potential domain shifts that could hinder learning.

To address this, we utilized Optuna to search for optimal augmentation combinations, selected the top-K performing pipelines, and composed a hybrid augmentation strategy. 

This helped the model generalize better across varying environmental conditions.

--------

### Efficient Experimentation Under Time Constraints
Given the need to train multiple models and test various settings, long training times posed a challenge due to the competition deadline.
To overcome this, we adopted a two-stage strategy:

First, we used a small subset of the training set (approximately 20 samples per class) to efficiently explore candidate settings.

Then, we applied the most promising configurations to the full dataset for final training.
This approach significantly reduced experimentation time without compromising model performance.

--------

### Ensemble Weighting Strategy
While assigning ensemble weights based on each model’s validation Log Loss improved performance over single-model predictions, it also led to overreliance on certain models.

To mitigate this, we softened the weight distribution across models, encouraging diversity and reducing the risk of overfitting.

Through this process, we also realized the importance of having a strong base model, and concluded that a soft, yet differentiated weighting strategy yields optimal results.


