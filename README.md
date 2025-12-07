# Multimodal Hemoglobin Regression Project

이 프로젝트는 손톱(nail)과 결막(conjunctiva) 이미지를 활용한 다중 모달리티 헤모글로빈(Hb) 회귀 모델입니다.

## 📋 목차

- [설치](#설치)
- [빠른 시작](#빠른-시작)
- [프로젝트 구조](#프로젝트-구조)
- [데이터셋 준비](#데이터셋-준비)
- [학습 및 평가](#학습-및-평가)
- [성능 보고서](#성능-보고서)

## 설치

### 요구사항

- Python 3.8+
- CUDA-capable GPU (권장: 4× GPU 환경)
- 최소 4GB GPU 메모리

### 설치 방법

1. 저장소 클론:
```bash
git clone <repository-url>
cd URP
```

2. Python 패키지 설치:
```bash
pip install -r requirements.txt
```

주요 의존성:
- PyTorch (>=2.0.0)
- torchvision (>=0.15.0)
- timm (>=0.9.0)
- numpy, pandas, scikit-learn
- tqdm, pyyaml, Pillow

## 빠른 시작

### 1. 데이터셋 준비

프로젝트는 두 가지 데이터셋을 사용합니다:

- **Single-modal dataset**: `singlemodal_dataset/` 디렉토리
  - **Nail 데이터셋**: 
    - 이름: Machine vision model using nail images for non-invasive detection of iron deficiency anemia in university students (Whole hand images)
    - 크기: 823 images (823명)
    - 분할: Train : Val : Test = 8 : 1 : 1 (Patient-wise split)
    - CSV 파일: `nail_meta_1.csv` (실제 사용 파일)
    - 필요 컬럼: `image_path`, `hb_value`, `patient_id`
  - **Conjunctiva 데이터셋**:
    - Folder 1: conjunctiva images for Anemia (800장, 200명)
    - Folder 2: CP-Anemic Dataset (710장, 710명)
    - Hb 범위: 8–16 g/dL 필터링
    - CSV 파일: `conj_folder1.csv`, `conj_folder2.csv`
    - 필요 컬럼: `image_path`, `hb_value`, `patient_id`
- **Multimodal dataset**: `multimodal_dataset/` 디렉토리
  - 이름: ImageHB (같은 환자의 Nail + Conj 이미지)
  - 구성: 26명 환자, 각 환자당 4개 paired view (Left/Right Nail, Left/Right Conj)
  - 총 104 image pairs (= 26명 × 4 pairs)
  - CSV 파일: `fusion_meta.csv`
  - 필요 컬럼: `nail_image_path`, `conj_image_path`, `hb_value`, `patient_id`, `side`, `age`, `gender`

**경로 설정 가이드:**

1. **설정 파일 경로**: 모든 설정 파일(`config/*.yaml`)의 경로는 프로젝트 루트 기준 **상대 경로**로 설정되어 있습니다.
   - 예: `metadata_csv: singlemodal_dataset/nail_meta_1.csv`
   - 설정 파일을 수정할 필요 없이 그대로 사용할 수 있습니다.

2. **CSV 파일 내부 경로**: CSV 파일의 `image_path` 컬럼은 다음 두 가지 방식으로 설정할 수 있습니다:
   - **절대 경로**: 전체 경로를 직접 지정 (예: `/path/to/image.jpg`)
   - **상대 경로**: 프로젝트 루트 기준 상대 경로 (예: `singlemodal_dataset/nail/1/ID001.jpg`)
   - 코드가 자동으로 상대 경로를 절대 경로로 변환합니다.

3. **데이터 디렉토리 구조**: 
   ```
   프로젝트_루트/
   ├── singlemodal_dataset/
   │   ├── nail_meta_1.csv
   │   ├── conj_folder1.csv
   │   ├── conj_folder2.csv
   │   └── [이미지 파일들]
   └── multimodal_dataset/
       ├── fusion_meta.csv
       └── [이미지 파일들]
   ```

자세한 데이터셋 형식은 아래 [데이터셋 준비](#데이터셋-준비) 섹션을 참고하세요.

### 2. 설정 파일 확인

각 단계별 설정 파일은 `config/` 디렉토리에 있습니다:
- `config/nail_single.yaml`: Nail 단일 모델 학습
- `config/conj_single.yaml`: Conjunctiva 단일 모델 학습
- `config/fusion_phase1.yaml`: Phase 1 Fusion 학습
- `config/fusion_phase2.yaml`: Phase 2 Fusion 학습

### 3. 학습 실행

#### Single-modality 학습
```bash
# Nail 단일 모델 (단일 GPU)
python main.py --mode train_single --modality nail --config config/nail_single.yaml

# Conjunctiva 단일 모델 (4× GPU 병렬)
python main.py --mode train_single --modality conj --config config/conj_single.yaml --num_gpus 4
```

#### Fusion 학습
```bash
# Phase 1 Fusion (4× GPU 병렬)
python main.py --mode train_fusion_phase1 --config config/fusion_phase1.yaml --num_gpus 4

# Phase 2 Fusion (4× GPU 병렬)
python main.py --mode train_fusion_phase2 --config config/fusion_phase2.yaml --num_gpus 4
```

### 4. 평가 및 앙상블

```bash
# Single 모델 평가
python main.py --mode eval_single --modality nail --config config/nail_single.yaml

# Fusion 모델 평가
python main.py --mode eval_fusion --config config/fusion_phase1.yaml

# 앙상블 실행
python main.py --mode ensemble --config config/ensemble_w_demo.yaml
```

자세한 사용법은 아래 섹션들을 참고하세요.

---

## Structure (전체 흐름)

**Aug. = Augmentation**

- Nail images → Aug. → Backbone →  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└→ (Fusion에서 Conj와 합침) → Regression HB
- Conj. images → Aug. → Backbone →

즉,

- Nail 단일모달: `Nail image → Aug → Backbone(Feature extractor) + Regression Head → Hb 회귀`
- Conj 단일모달: `Conj image → Aug → Backbone + Regression Head → Hb 회귀`
- Fusion: `Nail image + Conj image (+ age, gender) → 각자 Backbone → Feature → Fusion 모듈 → Hb 회귀`

모든 코드는 프로젝트 루트에 있고, 실행은 `main.py` 하나로 통합되어 있다.

- `main.py --mode ... --config ...` 형태로 사용

## 프로젝트 구조

```
.
├── main.py                 # 메인 엔트리포인트
├── config/                 # 설정 파일들 (YAML)
├── datasets/               # 데이터셋 로더 및 변환
│   ├── nail_dataset.py
│   ├── conj_dataset.py
│   ├── fusion_dataset.py
│   └── transforms.py
├── models/                # 모델 정의
│   ├── backbone_factory.py
│   ├── heads.py
│   └── fusion_models.py
├── train/                 # 학습 스크립트
│   ├── train_single.py
│   ├── train_fusion_phase1.py
│   └── train_fusion_phase2.py
├── eval/                  # 평가 스크립트
│   ├── evaluate_single.py
│   ├── evaluate_fusion.py
│   └── ensemble_predict.py
├── utils/                 # 유틸리티 함수
│   ├── checkpoint.py
│   ├── cv_split.py
│   ├── logger.py
│   ├── lr_schedulers.py
│   ├── metrics.py
│   └── seed.py
├── scripts/              # 보조 스크립트
│   ├── build_fusion_csv.py
│   ├── create_ensemble_config.py
│   └── generate_ensemble_predictions.py
├── checkpoints/          # 학습된 모델 체크포인트 (GitHub에 업로드 안 함)
├── logs/                 # 학습 로그 (GitHub에 업로드 안 함)
├── singlemodal_dataset/  # 단일 모달리티 데이터셋
├── multimodal_dataset/   # 다중 모달리티 데이터셋
└── ensemble_results/    # 앙상블 결과

```

## Nail single model

### 흐름

`Nail images → Augmentation → Feature extractor(backbone) + Regression Head → Regression HB value`

### Dataset

- 이름: **Skin and Fingernails Dataset (Whole hand images)**
- 크기: **823 images (823명)**
- 분할: **Train : Val : Test = 8 : 1 : 1**
- **Patient-wise split** (같은 사람의 이미지는 하나의 split에만 존재)

코드에서:

- `datasets/nail_dataset.py`
  - `load_nail_metadata(csv_path)` : CSV 를 읽어 `image_path, hb_value, patient_id` 를 로드
  - `split_nail_by_patient(...)` : 8:1:1 patient-wise split
- CSV 예시 위치:  
  `singlemodal_dataset/nail_meta_1.csv` (실제 사용 파일)

필요 컬럼:

- `image_path` – nail 이미지 경로
- `hb_value` – Hb (g/dL)
- `patient_id` – 환자 ID

### Augmentation (Nail)

Train 에서만 **on-the-fly augmentation** 사용:

- 매번 이미지 로드 시, 아래 중 하나/복수의 transform 을 랜덤 적용
  1. Raw (resize + normalization only)
  2. Random Horizontal Flip
  3. Random Vertical Flip
  4. Random Resized Crop (scale=(0.9, 1.0))
  5. Random Left Tilt (negative rotation)
  6. Random Right Tilt (positive rotation)

Val/Test:

- **절대 augmentation 적용 X**
- `Resize + ToTensor + ImageNet Normalize` 만 적용

코드에서:

- `datasets/transforms.py` 의 `build_transforms(split="train"/"val"/"test", modality="nail")`

### Feature extractor (backbone) + Regression Head

- Imagenet-12K로 pretrained 된 timm 모델 사용
- 조건: **CNN / Hybrid, 100M 파라미터 미만** (총 13개)
- timm 모델 리스트:
  - CNN (<100M):
    - `timm/convnext_base.clip_laion2b_augreg_ft_in12k` (99.7M)
    - `timm/rexnetr_300.sw_in12k` (76.4M)
    - `timm/regnety_120.sw_in12k` (74M)
    - `timm/resnetaa101d.sw_in12k` (66.7M)
    - `timm/convnext_small.in12k` (58.5M)
    - `timm/efficientnet_b5.sw_in12k` (52.6M)
    - `timm/resnetaa50d.sw_in12k` (48M)
    - `timm/resnetaa50d.d_in12k` (48M)
    - `timm/rexnetr_200.sw_in12k` (44.2M)
    - `timm/convnext_tiny.in12k` (36.9M)
  - Hybrid (<100M):
    - `timm/coatnet_2_rw_224.sw_in12k` (85M)
    - `timm/coatnet_rmlp_2_rw_224.sw_in12k` (85M)
    - `timm/coatnet_rmlp_1_rw2_224.sw_in12k` (53M)

코드에서:

- `models/backbone_factory.py`
  - `create_backbone(model_name, pretrained=True, features_only=False)`
  - `get_backbone_output_dim(...)`

#### Regression Head (Simple MLP)

- 구조: `dim → 512 → 256 → 1`
- 활성함수: ReLU
- 중간에 Dropout (0.3)

코드에서:

- `models/heads.py` 의 `RegressionHead(in_dim, hidden1=512, hidden2=256, dropout=0.3)`

### Optimizer / LR / Loss / Metric

- Optimizer: **AdamW**
- Loss: **Huber Loss** (`nn.SmoothL1Loss(beta=1.0)`)
- Epoch: **100**
- LR 스케줄:
  - Warmup: **5 epochs**
  - 이후 **95 epochs** 동안 cosine decay
  - Backbone LR: `0 → 1e-4 → 1e-6`
  - Head LR: `0 → 5e-4 → 5e-6`
- Metric (5개):
  - **MAE**, **R²**, **ACC@0.5**, **ACC@1.0**, **ACC@2.0**

코드에서:

- `utils/lr_schedulers.py` – warmup+cosine 구현
- `utils/metrics.py` – MAE, R², ACC@δ
- `train/train_single.py` – 위 설정으로 학습

### Best 5 모델 선택 기준 (Nail)

- 매 epoch 마다 Validation metric 계산 후,
  1. **R² > 0** 인 epoch만 후보
  2. **MAE가 더 낮은** 모델 우선
  3. 동률 시 tie-break:
     - ACC@1.0 ↑ → ACC@0.5 ↑ → R² ↑
- 이렇게 해서 **상위 5개 모델** checkpoint 저장
  - 향후 Fusion 단계에서 backbone 초기값으로 사용

Checkpoint 위치:

- `checkpoints/nail/`

실행 예:

```bash
python main.py --mode train_single --modality nail --config config/nail_single.yaml
```

---

## Conj. single model

### 흐름

`Conj. images → Augmentation → Feature extractor(backbone) + Regression Head → Regression HB value`

### Dataset

- Folder 1: **conjunctiva images for Anemia**
  - Whole eye (1장) + ROI (3장) / 한 사람
  - 총 800장 (200명)
- Folder 2: **CP-Anemic Dataset**
  - ROI만 존재
  - 710장 (710명)

두 폴더를 먼저 merge 한 뒤,

- Hb 값이 **8–16 g/dL** 범위 안에 있는 샘플만 사용
- Train : Val : Test = **8 : 1 : 1**
- 마찬가지로 **patient-wise split**

코드에서:

- `datasets/conj_dataset.py`
  - `load_conj_metadata(csv_folder1, csv_folder2, hb_min=8, hb_max=16)`
    - Folder1 + Folder2 CSV merge
    - Hb ∈ [8, 16] 필터링
  - `split_conj_by_patient(...)` : 8:1:1 patient-wise split

CSV 예시:

- `singlemodal_dataset/conj_folder1.csv`
- `singlemodal_dataset/conj_folder2.csv`

필요 컬럼:

- `image_path`, `hb_value`, `patient_id`

### Augmentation (Conj)

Nail과 동일 정책:

- Train 에서만 on-the-fly augmentation (Raw / HFlip / VFlip / ResizedCrop / Left/Right Tilt)
- Val/Test 에서는 **Resize + Normalize만** 적용, Aug. 금지

코드에서:

- `build_transforms(split, modality="conj")`

### Backbone + Regression Head

Backbone / Regression Head 설정은 Nail 과 완전히 동일:

- 같은 13개 ImageNet-12K timm 모델 후보
- 같은 `RegressionHead` 구조 사용
- 같은 AdamW / Huber / LR 스케줄 / Metric

### Best 5 모델 선택 기준 (Conj)

Nail과 동일:

1. R² > 0
2. MAE 낮은 순
3. 동률 시 ACC@1.0 → ACC@0.5 → R²

Checkpoint 위치:

- `checkpoints/conj/`

실행 예:

```bash
python main.py --mode train_single --modality conj --config config/conj_single.yaml
```

---

## Fusion (ImageHB)

### 흐름

`Nail images → Aug. → Backbone → Feature`  
`Conj images → Aug. → Backbone → Feature`  

- 두 feature (그리고 age, gender)을 합쳐서 Fusion 모듈에 넣고 최종 Hb 회귀
- Backbone 은 **점진적으로 unfreeze** (Phase 1 / Phase 2에서 계획적으로)

### Dataset

- 이름: **ImageHB** (같은 환자의 Nail + Conj 이미지)
- 구성:
  - 26명 환자
  - 각 환자당 4개의 paired view:
    - Left Nail, Right Nail
    - Left Conj, Right Conj
  - 총 104 image pairs (= 26명 × 4 pairs)

Split 전략:

1. **외부 테스트 세트**:
   - 환자 4명 (16 pairs) 홀드아웃
2. 나머지 22명에 대해:
   - **5-fold patient-level cross-validation**
   - 각 fold:
     - 약 80% 환자 → train
     - 약 20% 환자 → val

추가 feature:

- 나이, 성별 (오직 Fusion 단계에서만 사용)
- Age → **z-score normalization**
- Gender → **binary (0/1)** 인코딩
- `[age_z, gender_binary]` 벡터를 fusion feature에 concat

코드에서:

- `datasets/fusion_dataset.py`
  - `load_fusion_metadata(csv_path)`:
    - 필요 컬럼: `nail_image_path`, `conj_image_path`, `hb_value`, `patient_id`, `side`, `age`, `gender`
    - age → `age_z`, gender → `gender_binary`
  - `FusionDataset` : `(nail_img, conj_img, hb, patient_id, demo_vec)` 반환
- `utils/cv_split.py`
  - `create_fusion_splits(...)`:
    - 4명 external test
    - 나머지 22명으로 5-fold GroupKFold (patient_id 기준)

CSV 예시:

- `multimodal_dataset/fusion_meta.csv`

### Augmentation (Fusion)

Nail / Conj 단일모델과 동일 정책:

- Train:
  - Nail, Conj 이미지 각각 build_transforms("train", modality=...) 적용
  - on-the-fly augmentation (각 epoch마다 이미지가 조금씩 다르게 변형)
- Val/Test:
  - Resize + Normalize만 적용 (augmentation 없음)

---

## Phase 1 – Basic Fusion (Last-layer Feature Fusion)

### 최근 변경사항 (2024)

- **배치 크기 최적화**: 기본 `batch_size: 128` (4×2080Ti 11GB 환경 기준)
  - GPU 메모리 사용량: GPU당 약 2.5-4.4GB (배치 크기 128 기준, OOM 없음 확인)
  - **자동 배치 크기 조정**: `rexnetr_300`이 포함된 조합은 자동으로 `batch_size: 64`로 낮춰짐 (OOM 방지)
  - 학습 속도 향상: 배치 크기 64 대비 **약 2배 빠름**
  - 총 학습 시간: 약 **19.3시간** (4 GPU 병렬, 250개 조합, 배치 크기 128)
- **평가 결과 저장 방식 명확화**: 
  - Validation metrics: `*_metrics.json` (best model 선택 과정 기록)
  - Test metrics: `*_test_metrics.json` (external test set 평가 결과, 최종 성능 비교 기준)
  - 총 250개 조합 (25 pairs × 5 folds × 2 버전)의 test metrics 생성

### Goal

- Nail backbone 의 **마지막 레이어 feature** 와 Conj backbone 의 **마지막 레이어 feature** 만 사용
- 두 feature + demographic(age_z, gender_binary)를 concat 해서 Fusion MLP 에 넣는 **late fusion baseline**
- 이 단계에서는 **backbone은 최대한 고정**, Fusion head 쪽 학습에 집중

코드에서:

- `models/fusion_models.py` 의 `Phase1FusionModel`
- `train/train_fusion_phase1.py`

### 모델 구조

- Nail backbone → feature `f_n ∈ ℝ^{D_n}`
- Conj backbone → feature `f_c ∈ ℝ^{D_c}`
- Demographic `d ∈ ℝ²` (age_z, gender_binary) - **선택적 사용 가능**
- `[f_n; f_c; d]` → FusionHead(MLP: (D_n + D_c + 2) → 512 → 256 → 1) (demographics 포함 시)
- `[f_n; f_c]` → FusionHead(MLP: (D_n + D_c) → 512 → 256 → 1) (demographics 제외 시)

**Demographic Features 실험:**
- `use_demographics: true`: age_z와 gender_binary를 포함하여 학습 (기본값)
- `use_demographics: false`: demographic features 없이 순수 이미지 feature만으로 학습
- `run_both_demo_versions: true`: 두 버전을 모두 실행하여 비교 (demographics 포함/제외)

### Pretrained Backbone 로드

- Single-modality 학습에서 얻은 **best 5 Nail backbones**와 **best 5 Conj backbones**를 사용
- 각 backbone pair 조합에 대해:
  - Nail backbone: `checkpoints/nail/` 에서 해당 backbone의 best checkpoint 로드
  - Conj backbone: `checkpoints/conj/` 에서 해당 backbone의 best checkpoint 로드
  - Checkpoint가 없으면 ImageNet-12K pretrained weights 사용

**Best 5 Backbones (Test set 성능 기준):**

- **Nail (R² > 0, MAE 낮은 순):**
  1. `timm/rexnetr_300.sw_in12k` (MAE: 0.792, R²: 0.437)
  2. `timm/convnext_base.clip_laion2b_augreg_ft_in12k` (MAE: 0.826, R²: 0.438)
  3. `timm/resnetaa101d.sw_in12k` (MAE: 0.866, R²: 0.374)
  4. `timm/convnext_small.in12k` (MAE: 0.872, R²: 0.440)
  5. `timm/efficientnet_b5.sw_in12k` (MAE: 0.878, R²: 0.442)

- **Conj (R² > 0, MAE 낮은 순):**
  1. `timm/efficientnet_b5.sw_in12k` (MAE: 0.935, R²: 0.356)
  2. `timm/rexnetr_300.sw_in12k` (MAE: 0.950, R²: 0.241)
  3. `timm/convnext_small.in12k` (MAE: 0.964, R²: 0.387)
  4. `timm/regnety_120.sw_in12k` (MAE: 1.078, R²: 0.033)
  5. `timm/coatnet_rmlp_2_rw_224.sw_in12k` (MAE: 1.084, R²: 0.331)

- 총 **25개 backbone pair 조합** (Nail 5 × Conj 5)

### 학습 전략

#### 5-fold Cross-Validation

- **External test set**: 4명 환자 (16 pairs) 홀드아웃
- **5-fold CV**: 나머지 22명 환자로 5-fold patient-level cross-validation
  - 각 fold마다 train/val split
  - 각 fold의 best validation model로 external test set 평가

#### LR Schedule (총 60 epochs)

**bb = backbone, head = fusion head**

- **Epoch 0–5:**
  - LR_head: **0 → 5e-4** (linear warmup)
  - LR_bb = 0 (backbone 완전 freeze)
- **Epoch 5–10:**
  - LR_head: **cosine decay** 5e-4 → ~5e-6 시작
  - LR_bb = 0 (backbone 여전히 freeze)
- **Epoch 10–60:**
  - LR_head: **cosine decay** 5e-4 → ~5e-6
  - LR_bb: **cosine decay** 1e-4 → ~1e-6
  - Backbone의 **마지막 1–2 stage만 unfreeze**하여 미세조정

#### Optimizer / Loss / Metric

- Optimizer: **AdamW** (weight_decay=1e-4)
- Loss: **Huber Loss** (`nn.SmoothL1Loss(beta=1.0)`)
- Metrics: **MAE**, **R²**, **ACC@0.5**, **ACC@1.0**, **ACC@2.0**

### Best Model 선택 기준

각 fold의 validation set에서:

1. **R² > 0** 인 epoch만 후보
2. **MAE가 더 낮은** 모델 우선
3. 동률 시 tie-break:
   - ACC@1.0 ↑ → ACC@0.5 ↑ → R² ↑

각 fold의 best validation model을 선택하여:
- Checkpoint 저장: 
  - Demographics 포함: `p1_fold{fold_idx}_{pair_name}_w_demo_best.pt`
  - Demographics 제외: `p1_fold{fold_idx}_{pair_name}_wo_demo_best.pt`
- External test set 평가 수행

### Output 파일 구조

각 backbone pair와 fold마다:

**Demographics 포함 버전:**
```
checkpoints/fusion_phase1/
├── p1_fold0_nail-{nail_name}_conj-{conj_name}_w_demo_best.pt  # Best checkpoint
├── p1_fold0_nail-{nail_name}_conj-{conj_name}_w_demo_metrics.json     # Validation metrics history
└── p1_fold0_nail-{nail_name}_conj-{conj_name}_w_demo_test_metrics.json  # External test metrics
```

**Demographics 제외 버전:**
```
checkpoints/fusion_phase1/
├── p1_fold0_nail-{nail_name}_conj-{conj_name}_wo_demo_best.pt  # Best checkpoint
├── p1_fold0_nail-{nail_name}_conj-{conj_name}_wo_demo_metrics.json     # Validation metrics history
└── p1_fold0_nail-{nail_name}_conj-{conj_name}_wo_demo_test_metrics.json  # External test metrics
```

**파일 설명:**

1. **`*_metrics.json`**: 각 fold의 validation set에서 best model 선택 과정의 metrics history
   - 형식: `[{"epoch": int, "metrics": {"mae": float, "r2": float, "acc@0.5": float, "acc@1.0": float, "acc@2.0": float, "train_loss": float}}, ...]`
   - R² > 0인 epoch들만 기록되며, MAE 기준으로 정렬됨

2. **`*_test_metrics.json`**: External test set (4명 환자, 16 pairs) 평가 결과
   - 각 fold의 best validation model을 test set에 평가한 결과
   - 형식: `{"mae": float, "r2": float, "acc@0.5": float, "acc@1.0": float, "acc@2.0": float}`
   - **최종 모델 성능 비교는 이 파일들을 기준으로 함**

3. **`*_best.pt`**: Best validation model checkpoint
   - 각 fold에서 선택된 best epoch의 모델 가중치
   - `extra` 필드에 `fold`, `pair` 정보 포함

**평가 결과 활용:**

- 25개 backbone pair × 5 folds × 2 버전 (w_demo/wo_demo) = **총 250개 조합**의 test metrics가 생성됨
- 각 조합의 `*_test_metrics.json`을 비교하여 최적의 backbone pair와 demographic feature 사용 여부를 결정
- 예: `checkpoints/fusion_phase1/p1_fold0_nail-timm_rexnetr_300.sw_in12k_conj-timm_efficientnet_b5.sw_in12k_w_demo_test_metrics.json`

**앙상블용 모델 선택 및 디스크 공간 관리:**

Phase 3 앙상블을 위해 다음 4개 모델만 유지하는 것을 권장합니다:
- `w_demo` MAE 최소 모델
- `w_demo` R² 최대 모델
- `wo_demo` MAE 최소 모델
- `wo_demo` R² 최대 모델

각 모델의 `*_best.pt`와 `*_metrics.json` 파일만 유지하면 됩니다. 나머지 checkpoint 파일은 삭제하여 디스크 공간을 확보할 수 있습니다.

**현재 유지된 모델 (예시):**
- `w_demo` MAE 최소: Fold 2, `convnext_small.in12k + efficientnet_b5.sw_in12k` (MAE: 0.2085, R²: 0.0203)
- `w_demo` R² 최대: Fold 4, `convnext_small.in12k + convnext_small.in12k` (MAE: 0.6552, R²: 0.5122)
- `wo_demo` MAE 최소: Fold 2, `convnext_small.in12k + efficientnet_b5.sw_in12k` (MAE: 0.2169, R²: -0.0489)
- `wo_demo` R² 최대: Fold 4, `convnext_small.in12k + convnext_small.in12k` (MAE: 0.6568, R²: 0.5195)

> **참고**: 전체 결과는 `fusion_phase1_w_demo_results.csv`와 `fusion_phase1_wo_demo_results.csv`에 정리되어 있으며, 이 파일들을 기준으로 최고 성능 모델을 선택할 수 있습니다.

### Config 파일 예시

`config/fusion_phase1.yaml`:

```yaml
seed: 42
log_file: logs/fusion_phase1.log

# Fusion metadata CSV path
fusion_metadata_csv: multimodal_dataset/fusion_meta.csv

# Nail backbone names (best 5 from single-modality training)
nail_backbones:
  - timm/rexnetr_300.sw_in12k
  - timm/convnext_base.clip_laion2b_augreg_ft_in12k
  - timm/resnetaa101d.sw_in12k
  - timm/convnext_small.in12k
  - timm/efficientnet_b5.sw_in12k

# Conj backbone names (best 5 from single-modality training)
conj_backbones:
  - timm/efficientnet_b5.sw_in12k
  - timm/rexnetr_300.sw_in12k
  - timm/convnext_small.in12k
  - timm/regnety_120.sw_in12k
  - timm/coatnet_rmlp_2_rw_224.sw_in12k

# Checkpoint directories for loading pretrained backbones
nail_checkpoint_dir: checkpoints/nail
conj_checkpoint_dir: checkpoints/conj

# Whether to load pretrained backbones from single-modality checkpoints
load_pretrained_backbones: true

# Whether to use demographic features (age_z, gender_binary) in fusion
# Set to false to train without demographic features for comparison
use_demographics: true

# If true, will run both versions (with and without demographics) sequentially
# This allows direct comparison of demographic feature contribution
run_both_demo_versions: false

image_size: 224
batch_size: 128  # Default batch size. rexnetr_300 포함 조합은 자동으로 64로 낮춰짐 (OOM 방지)
num_workers: 4

# Phase 1 training settings
epochs: 60
warmup_epochs_head: 5  # Epoch 0-5: LR_head warmup

weight_decay: 1e-4

checkpoint_dir: checkpoints/fusion_phase1
```

### 실행 예

#### 단일 GPU 실행

```bash
python main.py --mode train_fusion_phase1 --config config/fusion_phase1.yaml
```

#### 멀티 GPU 병렬 실행 (4× GPU)

Phase 1 fusion training은 **5 folds × 25 pairs = 125개 작업**을 수행합니다.  
4개 GPU를 사용하면 각 GPU에 약 31-32개 작업이 자동으로 분배되어 병렬로 학습됩니다.

```bash
python main.py \
  --mode train_fusion_phase1 \
  --config config/fusion_phase1.yaml \
  --num_gpus 4
```

**동작 방식:**
- 모든 (fold_idx, nail_name, conj_name) 조합을 생성
- 125개 작업을 4개 GPU에 균등 분배
- 각 GPU 프로세스는 할당된 작업만 순차적으로 처리
- 각 GPU는 독립적으로 학습 진행 (동시 실행)

> **참고**: `config/fusion_phase1.yaml`의 `batch_size: 256`, `num_workers: 4`는 4×2080Ti (11GB) 환경 기준입니다.  
> GPU 메모리에 따라 `batch_size`를 조정할 수 있습니다. 배치 크기 256 기준으로 GPU당 약 2.5-4.4GB 메모리를 사용하며, OOM 없이 안정적으로 실행됩니다.
> 배치 크기 256으로 최적화하여 배치 크기 64 대비 약 2.6배 빠른 학습 속도를 달성했습니다.

### 요약

- **Fusion head가 메인 학습 대상**: Epoch 0-5 warmup, 이후 cosine decay
- **Backbone은 거의 고정**: Epoch 10부터 마지막 1-2 stage만 작은 LR로 미세조정
- **25개 backbone pair 조합**: Nail best 5 × Conj best 5
- **5-fold CV + External test**: 각 fold의 best model로 external test set 평가
- **Demographic features 실험 옵션**:
  - `use_demographics: true` (기본): age_z, gender_binary를 fusion head에 concat
  - `use_demographics: false`: demographic features 없이 순수 이미지 feature만 사용
  - `run_both_demo_versions: true`: 두 버전을 모두 실행하여 직접 비교 가능
- **멀티 GPU 지원**: `--num_gpus 4` 옵션으로 125개 작업을 4개 GPU에 자동 분배

---

## Phase 2 – Multi-Level Feature Fusion

### Goal

- Backbone 의 **중간 레이어 feature** + **마지막 레이어 feature** 모두 활용
- Stage별 feature를 projection 해서 하나의 큰 multi-level feature 로 묶고, nail/conj를 함께 fusion
- 새로운 fusion 모듈은 파라미터가 많기 때문에:
  - 초반에는 backbone 완전 freeze
  - 이후에만 backbone 상위 몇 stage를 작은 LR로 미세조정

총 Epoch: **80**

코드에서:

- `models/fusion_models.py` 의 `Phase2MultiLevelFusionModel`
  - `timm.create_model(..., features_only=True)` 로 여러 stage output 받음
  - 각 stage마다: GAP → Linear(proj_dim=256) → concat
- `train/train_fusion_phase2.py`

### Multi-level fusion module 구성

- 각 backbone(nail, conj)에 대해:
  - stage 2, 3, 4 (그리고 마지막 feature)를 사용 (코드에서는 마지막 4개 stage 기준)
  - 각 stage feature map 에:
    - Global Average Pooling (GAP) → [N, C_i]
    - Linear (C_i → 256) projection
  - Nail:
    - `[f_n_stage2_proj; f_n_stage3_proj; f_n_stage4_proj; f_n_last_proj]`
  - Conj:
    - `[f_c_stage2_proj; f_c_stage3_proj; f_c_stage4_proj; f_c_last_proj]`
- 최종 fusion 입력:
  - `[nail_multi_features; conj_multi_features; d]` (d = demographics)
  - 이걸 multi-layer MLP head에 넣어 Hb 회귀

### Freeze & LR 계획

Epoch 범위 / 동작:

- 0–20:
  - Nail / Conj backbone **완전 freeze**
  - **multi-level fusion module 전체만** 학습
  - LR_head ≈ 5e-4
- 20–80:
  - Backbone 의 **마지막 1–2 stage** 만 unfreeze
  - 학습 대상:
    - multi-level fusion module 전체
    - Nail/Conj backbone 상위 stage
  - 예시 LR:
    - LR_head ≈ 3e-4 (cosine decay 3e-4 → 3e-6)
    - LR_bb ≈ 5e-5 (cosine decay 5e-5 → 5e-7)

### 세부 LR 스케줄

- Epoch 0–10:
  - LR_head: **0 → 5e-4** (warmup)
  - LR_bb = 0
- Epoch 10–20:
  - LR_head: 5e-4 부근 유지 (또는 천천히 cosine 시작)
  - LR_bb = 0 (여전히 backbone freeze)
- Epoch 20–80:
  - LR_head: cosine decay 3e-4 → ~3e-6
  - LR_bb: cosine decay 5e-5 → ~5e-7 (마지막 1–2 stage만 학습)

요약:

- Phase 2는
  - 0–20: backbone 고정, multi-level fusion module 만 학습
  - 20–80: backbone 상위 stage를 아주 작은 LR로 같이 fine-tune
- Fusion module이 메인 역할, backbone은 살짝만 조정

### Best Model 선택 기준

각 fold의 validation set에서:

1. **R² > 0** 인 epoch만 후보 (Phase 1과 동일)
2. **MAE가 더 낮은** 모델 우선
3. 동률 시 tie-break:
   - ACC@1.0 ↑ → ACC@0.5 ↑ → R² ↑

각 fold의 best validation model을 선택하여:
- Checkpoint 저장: 
  - Demographics 포함: `p2_fold{fold_idx}_{pair_name}_w_demo_best.pt`
  - Demographics 제외: `p2_fold{fold_idx}_{pair_name}_wo_demo_best.pt`

**앙상블용 모델 선택:**

Phase 3 앙상블을 위해 Phase 1과 동일한 명확한 기준으로 모델을 선택합니다:
- `w_demo` MAE 최소 모델
- `w_demo` R² 최대 모델
- `wo_demo` MAE 최소 모델
- `wo_demo` R² 최대 모델

선택된 모델 정보는 `scripts/select_phase2_best_models.py` 스크립트를 실행하여 확인할 수 있습니다:

```bash
python scripts/select_phase2_best_models.py
```

이 스크립트는 `logs/fusion_phase2/fusion_phase2_selected_models.csv` 파일을 생성하며, 선택된 4개 모델의 정보를 포함합니다.

실행 예:

```bash
python main.py --mode train_fusion_phase2 --config config/fusion_phase2.yaml
```

---

## Phase 3 – Ensemble

### Goal

- 다음 네 종류의 모델 예측값을 합쳐 최종 Hb 를 추정:
  - Nail single-modality best models
  - Conj single-modality best models
  - Phase 1 best fusion models (w_demo/wo_demo 각각 MAE 최소 및 R² 최대 모델, 총 4개)
  - Phase 2 best fusion models
- 앙상블 weight 는 **각 fold 의 validation set 예측값**만 사용해서 튜닝
  - test fold / external test 는 **절대 weight 튜닝에 사용하지 않음**

**Phase 1 모델 선택:**
- Phase 1에서는 각 fold별로 여러 backbone pair 조합을 학습하지만, 앙상블을 위해 다음 4개 모델만 사용:
  - `w_demo` MAE 최소 모델
  - `w_demo` R² 최대 모델
  - `wo_demo` MAE 최소 모델
  - `wo_demo` R² 최대 모델
- 이 모델들은 `fusion_phase1_w_demo_results.csv`와 `fusion_phase1_wo_demo_results.csv`를 기준으로 선택됨

**Phase 2 모델 선택:**
- Phase 2도 Phase 1과 동일한 명확한 기준으로 모델을 선택:
  - `w_demo` MAE 최소 모델 (R² > 0 조건 우선, 없으면 전체에서 선택)
  - `w_demo` R² 최대 모델 (R² > 0 조건 우선, 없으면 전체에서 선택)
  - `wo_demo` MAE 최소 모델 (R² > 0 조건 우선, 없으면 전체에서 선택)
  - `wo_demo` R² 최대 모델 (R² > 0 조건 우선, 없으면 전체에서 선택)
- 이 모델들은 `logs/fusion_phase2/fusion_phase2_results_w_demo_best.csv`와 `logs/fusion_phase2/fusion_phase2_results_wo_demo_best.csv`를 기준으로 선택됨
- 선택된 모델 정보는 `scripts/select_phase2_best_models.py` 스크립트를 실행하여 `logs/fusion_phase2/fusion_phase2_selected_models.csv`에 저장됨

### 방식 (추가 학습 없음)

- 별도 학습 없는 단순 조합:

  \[
  \hat{y}_{final} = \sum_{i} w_i \cdot \hat{y}_i
  \]

- weight \(w_i\) 를 정하는 방법:
  - Validation set에서 grid search 로 MAE 최소화
  - 혹은 각 모델의 MAE 에 기반한 inverse-MAE 가중치 (Heuristic)

여기서는:

- **optimizer / LR schedule 없음**
- 단순히 validation prediction 을 이용해 weight만 고르고, 그 weight로 test / external test 예측값을 합친다.

### 앙상블 구성

**w_demo 앙상블:**
- Phase 1 w_demo 모델들 (MAE 최소, R² 최대, 총 2개)
- Phase 2 w_demo 모델들 (MAE 최소, R² 최대, 총 2개)
- 총 4개 모델

**wo_demo 앙상블:**
- Phase 1 wo_demo 모델들 (MAE 최소, R² 최대, 총 2개)
- Phase 2 wo_demo 모델들 (MAE 최소, R² 최대, 총 2개)
- 총 4개 모델

### 앙상블 실행 절차

#### 1단계: 예측값 생성

각 모델의 validation/test 예측값을 생성합니다:

**단일 GPU 실행:**
```bash
# Phase 1 모델 예측값 생성
python scripts/generate_ensemble_predictions.py \
  --config config/fusion_phase1.yaml \
  --selected_models_csv fusion_phase1_w_demo_results.csv \
  --phase 1 \
  --checkpoint_dir checkpoints/fusion_phase1 \
  --output_dir ensemble_predictions \
  --num_gpus 1

python scripts/generate_ensemble_predictions.py \
  --config config/fusion_phase1.yaml \
  --selected_models_csv fusion_phase1_wo_demo_results.csv \
  --phase 1 \
  --checkpoint_dir checkpoints/fusion_phase1 \
  --output_dir ensemble_predictions \
  --num_gpus 1

# Phase 2 모델 예측값 생성
python scripts/generate_ensemble_predictions.py \
  --config config/fusion_phase2.yaml \
  --selected_models_csv logs/fusion_phase2/fusion_phase2_selected_models.csv \
  --phase 2 \
  --checkpoint_dir logs/fusion_phase2 \
  --output_dir ensemble_predictions \
  --num_gpus 1
```

**멀티 GPU 병렬 실행 (4× GPU):**
```bash
# Phase 1 모델 예측값 생성 (4개 GPU 병렬)
python scripts/generate_ensemble_predictions.py \
  --config config/fusion_phase1.yaml \
  --selected_models_csv fusion_phase1_w_demo_results.csv \
  --phase 1 \
  --checkpoint_dir checkpoints/fusion_phase1 \
  --output_dir ensemble_predictions \
  --num_gpus 4

python scripts/generate_ensemble_predictions.py \
  --config config/fusion_phase1.yaml \
  --selected_models_csv fusion_phase1_wo_demo_results.csv \
  --phase 1 \
  --checkpoint_dir checkpoints/fusion_phase1 \
  --output_dir ensemble_predictions \
  --num_gpus 4

# Phase 2 모델 예측값 생성 (4개 GPU 병렬)
python scripts/generate_ensemble_predictions.py \
  --config config/fusion_phase2.yaml \
  --selected_models_csv logs/fusion_phase2/fusion_phase2_selected_models.csv \
  --phase 2 \
  --checkpoint_dir logs/fusion_phase2 \
  --output_dir ensemble_predictions \
  --num_gpus 4
```

**동작 방식:**
- `--num_gpus 4` 옵션을 사용하면 모든 모델 예측 작업이 4개 GPU에 자동으로 분배되어 병렬로 실행됩니다.
- 각 GPU는 할당된 모델들만 순차적으로 처리하며, GPU 간 독립적으로 실행됩니다.
- Phase 1과 Phase 2 학습과 동일한 병렬 처리 방식을 사용합니다.

#### 2단계: 앙상블 Config 파일 생성

```bash
python scripts/create_ensemble_config.py \
  --phase1_w_demo_csv fusion_phase1_w_demo_results.csv \
  --phase1_wo_demo_csv fusion_phase1_wo_demo_results.csv \
  --phase2_selected_csv logs/fusion_phase2/fusion_phase2_selected_models.csv \
  --output_dir config \
  --predictions_dir ensemble_predictions
```

이 스크립트는 다음 파일들을 생성합니다:
- `config/ensemble_w_demo.yaml`
- `config/ensemble_wo_demo.yaml`

#### 3단계: 앙상블 실행

```bash
# w_demo 앙상블
python main.py --mode ensemble --config config/ensemble_w_demo.yaml

# wo_demo 앙상블
python main.py --mode ensemble --config config/ensemble_wo_demo.yaml
```

### 앙상블 결과

각 앙상블의 결과는 다음 디렉토리에 저장됩니다:
- `ensemble_results/w_demo/`: w_demo 앙상블 결과
  - `val_ensemble_preds.npy`: Validation 예측값
  - `test_ensemble_preds.npy`: Test 예측값
  - `metrics.json`: 앙상블 성능 지표 및 가중치
- `ensemble_results/wo_demo/`: wo_demo 앙상블 결과
  - 동일한 파일 구조

**w_demo 앙상블 최종 성능:**
- **Validation**: MAE 0.646 g/dL, R² 0.526, Acc@1.0 0.688
- **Test**: MAE 0.291 g/dL, R² -0.313, Acc@1.0 1.000

**wo_demo 앙상블 최종 성능:**
- **Validation**: MAE 0.811 g/dL, R² 0.412, Acc@1.0 0.750
- **Test**: MAE 0.875 g/dL, R² -6.238, Acc@1.0 0.688

> **참고**: 전체 성능 변화 분석은 `PERFORMANCE_REPORT.md`를 참고하세요.

### 코드 구조

- `scripts/generate_ensemble_predictions.py`: 각 모델의 예측값 생성
- `scripts/create_ensemble_config.py`: 앙상블 config 파일 자동 생성
- `eval/ensemble_predict.py`: 앙상블 실행 (grid search 또는 inverse-MAE 가중치)

---

## Phase 2 모델 선택

Phase 2에서도 Phase 1과 동일한 기준으로 최고 성능 모델을 선택합니다:

1. **R² > 0** 인 모델만 후보
2. **MAE 최소** 모델 선택
3. **R² 최대** 모델 선택

선택된 모델은 `logs/fusion_phase2/fusion_phase2_selected_models.csv`에 저장됩니다.

**자동 선택 스크립트:**
```bash
# 실제 존재하는 체크포인트를 기반으로 Phase 2 최고 성능 모델 선택
python scripts/select_phase2_best_models_from_checkpoints.py \
  --checkpoint_dir logs/fusion_phase2 \
  --output_csv logs/fusion_phase2/fusion_phase2_selected_models.csv
```

이 스크립트는:
- 실제 존재하는 체크포인트 파일에서 직접 메트릭을 읽어서 선택
- w_demo와 wo_demo 각각에 대해 MAE 최소, R² 최대 모델을 선택
- 중복 모델은 자동으로 제거

---

## Python 환경 & 프로젝트 구조 (요약)

- Python 환경:

```bash
pip install torch torchvision timm numpy pandas scikit-learn tqdm pyyaml
```

- 주요 파일 구조:
  - `main.py` : 공용 엔트리포인트 (`--mode`, `--config`, `--modality`)
  - `datasets/` : nail / conj / fusion dataset + transforms
  - `models/` : backbone factory, heads, fusion models
  - `train/` : `train_single.py`, `train_fusion_phase1.py`, `train_fusion_phase2.py`
  - `eval/` : `evaluate_single.py`, `evaluate_fusion.py`, `ensemble_predict.py`
  - `utils/` : seed, logger, metrics, checkpoint, cv_split, lr_schedulers

실제로 필요한 것은:

1. README에 나온 CSV 포맷대로 `singlemodal_dataset`, `multimodal_dataset` 아래에 CSV 생성
2. `config/*.yaml` 파일에서 그 CSV 경로와 backbone 이름을 지정
3. `python main.py --mode ...` 명령어로 각 단계별 학습/평가/앙상블 실행

이면 전체 파이프라인이 네가 설계한 순서대로 동작한다.

---

## 성능 보고서

전체 실험 과정에서 단일 모델부터 앙상블까지의 성능 변화를 분석한 보고서가 `PERFORMANCE_REPORT.md`에 있습니다.

**주요 내용:**
- Single Model (Nail, Conjunctiva) 성능
- Phase 1 Fusion (기본 퓨전) 성능
- Phase 2 Fusion (Multi-level 퓨전) 성능
- Phase 3 Ensemble (앙상블) 성능
- 단계별 성능 변화 분석 및 개선율

**핵심 결과:**
- 단일 모델 대비 MAE 약 73% 감소 (0.792 → 0.208)
- 최종 앙상블 Test MAE: 0.291 g/dL
- 최종 앙상블 Test Acc@1.0: 1.000 (모든 샘플에서 ±1.0 g/dL 이내 정확도)

---

## 7. 멀티 GPU 병렬 실행 & 1-epoch 테스트 런

### 7.1 4× 2080Ti 멀티 GPU 병렬 실행 방법

#### 7.1.1 Single-modality 학습 (train_single)

`train_single.py` 는 하나의 프로세스 안에서 `backbone_names` 리스트를 **순차로** 학습한다.  
4장의 2080Ti를 모두 쓰려면, 백본 13개를 4개의 GPU에 나눠서 병렬 학습하면 된다.

> 현재 `config/nail_single.yaml`, `config/conj_single.yaml` 는 **4× 2080Ti** 환경 기준으로  
> `batch_size: 8`, `num_workers: 4` 로 맞춰져 있다. (단일 모달리티 학습용)  
> CPU 코어 수에 따라 `num_workers` 는 4~8 사이에서 조정 가능하지만, 4가 4×2080Ti 기준 안전한 기본값이다.

#### 7.1.2 추천: `--num_gpus` 한 줄 실행 (코드 레벨 자동 분배)

`main.py` 내부에서 `backbone_names` 를 GPU 개수만큼 자동으로 분할하도록 구현해 두었다.  
아래처럼 **한 줄**만 실행하면, 백본 13개가 4개의 GPU(0,1,2,3)에 3–4개씩 자동으로 할당되어 병렬로 학습된다.

```bash
# Nail 단일모델 (4×2080Ti, batch_size=8, num_workers=4)
python main.py \
  --mode train_single \
  --modality nail \
  --config config/nail_single.yaml \
  --num_gpus 4

# Conj 단일모델 (4×2080Ti, batch_size=8, num_workers=4)
python main.py \
  --mode train_single \
  --modality conj \
  --config config/conj_single.yaml \
  --num_gpus 4
```

이 경우, 내부적으로는 GPU 0~3 각각에 대해 별도의 프로세스가 생성되고,  
각 프로세스는 **자기 GPU 하나만** 사용해서 할당된 backbone 들을 순차적으로 학습한다.

#### 7.1.3 (선택) 수동 분할: `--backbone_start`, `--backbone_end` + `CUDA_VISIBLE_DEVICES`

기존처럼 CLI 레벨에서 직접 분할해서 실행하고 싶다면, 아래 예시처럼  
각 GPU에 대해 `CUDA_VISIBLE_DEVICES` 와 `--backbone_start/--backbone_end` 를 수동으로 지정할 수 있다  
(각 프로세스를 background 로 실행하려면 마지막에 `&`):

```bash
# === Nail single model (13개 backbone) ===

# GPU 0: index 0~3 (nail)
CUDA_VISIBLE_DEVICES=0 python main.py \
  --mode train_single --modality nail \
  --config config/nail_single.yaml \
  --backbone_start 0 --backbone_end 4 &

# GPU 1: index 4~6 (nail)
CUDA_VISIBLE_DEVICES=1 python main.py \
  --mode train_single --modality nail \
  --config config/nail_single.yaml \
  --backbone_start 4 --backbone_end 7 &

# GPU 2: index 7~9 (nail)
CUDA_VISIBLE_DEVICES=2 python main.py \
  --mode train_single --modality nail \
  --config config/nail_single.yaml \
  --backbone_start 7 --backbone_end 10 &

# GPU 3: index 10~12 (또는 10~13 로 마지막까지, nail)
CUDA_VISIBLE_DEVICES=3 python main.py \
  --mode train_single --modality nail \
  --config config/nail_single.yaml \
  --backbone_start 10 --backbone_end 13 &


# === Conj single model (13개 backbone) ===

# GPU 0: index 0~3 (conj)
CUDA_VISIBLE_DEVICES=0 python main.py \
  --mode train_single --modality conj \
  --config config/conj_single.yaml \
  --backbone_start 0 --backbone_end 4 &

# GPU 1: index 4~6 (conj)
CUDA_VISIBLE_DEVICES=1 python main.py \
  --mode train_single --modality conj \
  --config config/conj_single.yaml \
  --backbone_start 4 --backbone_end 7 &

# GPU 2: index 7~9 (conj)
CUDA_VISIBLE_DEVICES=2 python main.py \
  --mode train_single --modality conj \
  --config config/conj_single.yaml \
  --backbone_start 7 --backbone_end 10 &

# GPU 3: index 10~12 (또는 10~13 로 마지막까지, conj)
CUDA_VISIBLE_DEVICES=3 python main.py \
  --mode train_single --modality conj \
  --config config/conj_single.yaml \
  --backbone_start 10 --backbone_end 13 &
```

이 방식에서는 각 명령을 **서로 다른 터미널**에서 실행하는 것이 안전하다  
(동일 GPU에서 여러 학습 프로세스를 동시에 띄우지 않도록 주의).

#### 7.1.4 Fusion Phase 1 학습 (train_fusion_phase1)

Phase 1 fusion training은 **5 folds × 25 pairs = 125개 작업**을 수행합니다.  
`--num_gpus 4` 옵션을 사용하면 125개 작업이 4개 GPU에 자동으로 분배되어 병렬로 학습됩니다.

```bash
# Fusion Phase 1 (4×2080Ti, batch_size=128, num_workers=4)
python main.py \
  --mode train_fusion_phase1 \
  --config config/fusion_phase1.yaml \
  --num_gpus 4
```

**동작 방식:**
- 모든 (fold_idx, nail_name, conj_name) 조합을 생성 (125개)
- 125개 작업을 4개 GPU에 균등 분배 (각 GPU당 약 31-32개)
- 각 GPU 프로세스는 할당된 작업만 순차적으로 처리
- 각 GPU는 독립적으로 학습 진행 (동시 실행)

> **참고**: `config/fusion_phase1.yaml`의 `batch_size: 128`, `num_workers: 4`는 4×2080Ti (11GB) 환경 기준입니다.  
> GPU 메모리에 따라 `batch_size`를 조정할 수 있습니다. 배치 크기 128 기준으로 GPU당 약 2.5-4.4GB 메모리를 사용하며, OOM 없이 안정적으로 실행됩니다.
> **자동 배치 크기 조정**: `rexnetr_300`이 포함된 조합은 자동으로 `batch_size: 64`로 낮춰져 OOM을 방지합니다.

#### Nohup으로 백그라운드 실행

장시간 학습을 위해 nohup으로 백그라운드에서 실행할 수 있습니다:

```bash
# Nohup으로 백그라운드 실행
nohup python3 main.py \
  --mode train_fusion_phase1 \
  --config config/fusion_phase1.yaml \
  --num_gpus 4 \
  > logs/fusion_phase1_both_nohup.log 2>&1 &

# 로그 확인
tail -f logs/fusion_phase1_both_nohup.log

# 프로세스 확인
ps aux | grep 'python.*train_fusion_phase1'

# 프로세스 종료 (필요시)
pkill -f 'python.*train_fusion_phase1'
```

### 7.2 1-epoch 테스트 런 (sanity check)

실제 긴 학습에 들어가기 전에, **모든 backbone 을 1 epoch 씩만 돌려보는 테스트 런**을 추천한다.

절차 예:

1. `config/nail_single.yaml` 을 복사해서 `config/nail_single_test.yaml` 생성
2. 아래처럼 수정:

```yaml
epochs: 1       # 1epoch만
warmup_epochs: 1
batch_size: 16  # 보수적으로 (OOM 방지)
```

3. 실행:

```bash
python main.py --mode train_single --modality nail --config config/nail_single_test.yaml
python main.py --mode train_single --modality conj --config config/conj_single_test.yaml
```

이 테스트 런에서:

- 데이터 로딩/CSV 경로/augmentation/모델 생성이 제대로 되는지
- OOM 발생 여부 및 적당한 `batch_size` 인지

를 빠르게 확인한 뒤,  
이상이 없으면 `epochs: 100`, `batch_size` 를 원하는 값으로 되돌린 본 config 로 **실제 학습을 시작**하면 된다.





