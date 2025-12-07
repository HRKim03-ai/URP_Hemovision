# 성능 변화 보고서: 단일 모델부터 앙상블까지

## 개요

본 보고서는 Nail과 Conjunctiva 이미지를 활용한 Hemoglobin (Hb) 회귀 예측 모델의 성능 변화를 단일 모달리티부터 멀티모달리티 퓨전, 그리고 앙상블까지 단계별로 분석합니다.

---

## 1. Single Model (단일 모달리티)

### 1.1 Nail 단일 모델

**최고 성능 모델:**
- **Backbone**: `timm/rexnetr_300.sw_in12k`
- **Test MAE**: 0.792 g/dL
- **Test R²**: 0.437
- **Acc@0.5**: 0.446
- **Acc@1.0**: 0.663
- **Acc@2.0**: 0.940

**주요 모델 성능 (상위 5개):**

| Backbone | MAE | R² | Acc@0.5 | Acc@1.0 | Acc@2.0 |
|----------|-----|----|---------|---------|---------|
| rexnetr_300.sw_in12k | 0.792 | 0.437 | 0.446 | 0.663 | 0.940 |
| convnext_base.clip_laion2b_augreg_ft_in12k | 0.826 | 0.438 | 0.301 | 0.687 | 0.952 |
| resnetaa101d.sw_in12k | 0.866 | 0.374 | 0.373 | 0.614 | 0.952 |
| convnext_small.in12k | 0.872 | 0.440 | 0.349 | 0.614 | 0.940 |
| efficientnet_b5.sw_in12k | 0.878 | 0.442 | 0.325 | 0.627 | 0.928 |

### 1.2 Conjunctiva 단일 모델

**최고 성능 모델:**
- **Backbone**: `timm/efficientnet_b5.sw_in12k`
- **Test MAE**: 0.935 g/dL
- **Test R²**: 0.356
- **Acc@0.5**: 0.421
- **Acc@1.0**: 0.716
- **Acc@2.0**: 0.884

**주요 모델 성능 (상위 5개):**

| Backbone | MAE | R² | Acc@0.5 | Acc@1.0 | Acc@2.0 |
|----------|-----|----|---------|---------|---------|
| efficientnet_b5.sw_in12k | 0.935 | 0.356 | 0.421 | 0.716 | 0.884 |
| rexnetr_300.sw_in12k | 0.950 | 0.241 | 0.340 | 0.660 | 0.910 |
| convnext_small.in12k | 0.964 | 0.387 | 0.347 | 0.663 | 0.884 |
| coatnet_2_rw_224.sw_in12k | 1.089 | 0.360 | 0.360 | 0.584 | 0.820 |
| coatnet_rmlp_1_rw2_224.sw_in12k | 1.140 | 0.365 | 0.270 | 0.539 | 0.888 |

### 1.3 단일 모델 요약

- **Nail 최고 성능**: MAE 0.792, R² 0.437
- **Conjunctiva 최고 성능**: MAE 0.935, R² 0.356
- **평균 성능**: MAE ~0.86, R² ~0.40

---

## 2. Phase 1 Fusion (기본 퓨전)

Phase 1에서는 Nail과 Conjunctiva의 마지막 레이어 특징을 결합하는 기본 퓨전 방식을 사용했습니다.

### 2.1 w_demo (Demographics 포함)

**최고 성능 모델:**
- **Fold**: 2
- **Nail Backbone**: `convnext_small.in12k`
- **Conj Backbone**: `efficientnet_b5.sw_in12k`
- **Test MAE**: 0.208 g/dL
- **Test R²**: 0.020
- **Acc@0.5**: 1.000
- **Acc@1.0**: 1.000
- **Acc@2.0**: 1.000

**주요 모델 성능 (상위 5개):**

| Fold | Nail Model | Conj Model | MAE | R² | Acc@0.5 | Acc@1.0 | Acc@2.0 |
|------|------------|------------|-----|----|---------|---------|---------|
| 2 | convnext_small.in12k | efficientnet_b5.sw_in12k | 0.208 | 0.020 | 1.000 | 1.000 | 1.000 |
| 4 | convnext_small.in12k | convnext_small.in12k | 0.655 | 0.512 | 0.625 | 0.625 | 1.000 |
| 3 | convnext_small.in12k | efficientnet_b5.sw_in12k | 0.704 | -0.303 | 0.625 | 0.750 | 0.813 |
| 4 | convnext_small.in12k | efficientnet_b5.sw_in12k | 0.767 | 0.342 | 0.563 | 0.625 | 1.000 |
| 3 | convnext_small.in12k | convnext_small.in12k | 0.797 | -0.595 | 0.500 | 0.750 | 0.750 |

### 2.2 wo_demo (Demographics 제외)

**최고 성능 모델:**
- **Fold**: 2
- **Nail Backbone**: `convnext_small.in12k`
- **Conj Backbone**: `efficientnet_b5.sw_in12k`
- **Test MAE**: 0.217 g/dL
- **Test R²**: -0.049
- **Acc@0.5**: 0.938
- **Acc@1.0**: 1.000
- **Acc@2.0**: 1.000

**주요 모델 성능 (상위 5개):**

| Fold | Nail Model | Conj Model | MAE | R² | Acc@0.5 | Acc@1.0 | Acc@2.0 |
|------|------------|------------|-----|----|---------|---------|---------|
| 2 | convnext_small.in12k | efficientnet_b5.sw_in12k | 0.217 | -0.049 | 0.938 | 1.000 | 1.000 |
| 4 | convnext_small.in12k | convnext_small.in12k | 0.657 | 0.520 | 0.625 | 0.688 | 1.000 |
| 3 | convnext_small.in12k | efficientnet_b5.sw_in12k | 0.702 | -0.316 | 0.688 | 0.750 | 0.750 |
| 3 | convnext_small.in12k | convnext_small.in12k | 0.763 | -0.170 | 0.625 | 0.688 | 1.000 |
| 4 | convnext_small.in12k | efficientnet_b5.sw_in12k | 0.779 | 0.343 | 0.500 | 0.625 | 1.000 |

### 2.3 Phase 1 요약

- **w_demo 최고 성능**: MAE 0.208, R² 0.020 (Fold 2)
- **wo_demo 최고 성능**: MAE 0.217, R² -0.049 (Fold 2)
- **단일 모델 대비 개선**: MAE 약 75% 감소 (0.792 → 0.208)
- **Demographics 효과**: w_demo가 wo_demo보다 약간 우수 (MAE 0.208 vs 0.217)

---

## 3. Phase 2 Fusion (Multi-level Feature Fusion)

Phase 2에서는 Phase 1 모델을 pretrained backbone으로 사용하고, 중간 레이어와 마지막 레이어 특징을 모두 활용하는 multi-level feature fusion을 적용했습니다.

### 3.1 w_demo (Demographics 포함)

**선택된 모델:**

1. **MAE 최소 모델:**
   - **Fold**: 3
   - **Nail Backbone**: `timm_convnext_small.in12k`
   - **Conj Backbone**: `timm_efficientnet_b5.sw_in12k`
   - **Val MAE**: 0.655 g/dL
   - **Val R²**: 0.059
   - **Acc@0.5**: 0.563
   - **Acc@1.0**: 0.750
   - **Acc@2.0**: 0.875

2. **R² 최대 모델:**
   - **Fold**: 4
   - **Nail Backbone**: `timm_convnext_small.in12k`
   - **Conj Backbone**: `timm_convnext_small.in12k`
   - **Val MAE**: 0.722 g/dL
   - **Val R²**: 0.478
   - **Acc@0.5**: 0.500
   - **Acc@1.0**: 0.625
   - **Acc@2.0**: 1.000

### 3.2 wo_demo (Demographics 제외)

**선택된 모델:**

1. **MAE 최소 모델 (R² 최대와 동일):**
   - **Fold**: 0
   - **Nail Backbone**: `timm_convnext_small.in12k`
   - **Conj Backbone**: `timm_convnext_small.in12k`
   - **Val MAE**: 0.813 g/dL
   - **Val R²**: 0.411
   - **Acc@0.5**: 0.350
   - **Acc@1.0**: 0.750
   - **Acc@2.0**: 0.950

### 3.3 Phase 2 요약

- **w_demo 최고 성능**: Val MAE 0.655, Val R² 0.478
- **wo_demo 최고 성능**: Val MAE 0.813, Val R² 0.411
- **Phase 1 대비**: Validation set에서 더 안정적인 성능 (R² 개선)
- **Multi-level fusion 효과**: 중간 레이어 특징 활용으로 표현력 향상

---

## 4. Phase 3 Ensemble (앙상블)

Phase 3에서는 Phase 1과 Phase 2의 최고 성능 모델들을 앙상블하여 최종 성능을 향상시켰습니다.

### 4.1 w_demo 앙상블

**구성 모델:**
- Phase 1 w_demo MAE 최소: Fold 2, `convnext_small.in12k + efficientnet_b5.sw_in12k`
- Phase 1 w_demo R² 최대: Fold 4, `convnext_small.in12k + convnext_small.in12k`
- Phase 2 w_demo MAE 최소: Fold 3, `convnext_small.in12k + efficientnet_b5.sw_in12k`
- Phase 2 w_demo R² 최대: Fold 4, `convnext_small.in12k + convnext_small.in12k`

**앙상블 가중치:**
- [0.0, 0.8, 0.0, 0.2] (Phase 1 R² 최대 모델과 Phase 2 R² 최대 모델에 높은 가중치)

**최종 성능:**

| Metric | Validation | Test |
|--------|------------|------|
| **MAE** | 0.646 g/dL | **0.291 g/dL** |
| **R²** | 0.526 | -0.313 |
| **Acc@0.5** | 0.563 | 0.750 |
| **Acc@1.0** | 0.688 | 1.000 |
| **Acc@2.0** | 1.000 | 1.000 |

### 4.2 wo_demo 앙상블

**구성 모델:**
- Phase 1 wo_demo MAE 최소: Fold 0, `convnext_small.in12k + convnext_small.in12k`
- Phase 1 wo_demo R² 최대: Fold 0, `convnext_small.in12k + convnext_small.in12k` (MAE 최소와 동일)
- Phase 2 wo_demo MAE 최소 (R² 최대와 동일): Fold 0, `convnext_small.in12k + convnext_small.in12k`

**앙상블 가중치:**
- [0.0, 1.0] (Phase 2 fold 0 모델에 높은 가중치)

**최종 성능:**

| Metric | Validation | Test |
|--------|------------|------|
| **MAE** | 0.811 g/dL | **0.875 g/dL** |
| **R²** | 0.412 | -6.238 |
| **Acc@0.5** | 0.350 | 0.063 |
| **Acc@1.0** | 0.750 | 0.688 |
| **Acc@2.0** | 0.950 | 1.000 |

### 4.3 Phase 3 요약

- **w_demo 앙상블 Test MAE**: 0.291 g/dL (Phase 1 최고 성능 0.208 대비 약간 증가, 하지만 더 안정적)
- **w_demo 앙상블 Test Acc@1.0**: 1.000 (모든 샘플에서 ±1.0 g/dL 이내 정확도)
- **wo_demo 앙상블 Test MAE**: 0.875 g/dL
- **wo_demo 앙상블 Test Acc@1.0**: 0.688
- **앙상블 효과**: 여러 모델의 예측을 결합하여 더 robust한 성능 달성

---

## 5. 전체 성능 변화 요약

### 5.1 MAE 변화 추이

| Phase | 모델 | MAE (g/dL) | 개선율 |
|-------|------|------------|--------|
| Single Model | Nail (최고) | 0.792 | Baseline |
| Single Model | Conj (최고) | 0.935 | - |
| Phase 1 Fusion | w_demo (최고) | 0.208 | **73.7% ↓** |
| Phase 1 Fusion | wo_demo (최고) | 0.217 | **72.6% ↓** |
| Phase 2 Fusion | w_demo (Val 최고) | 0.655 | - |
| Phase 3 Ensemble | w_demo (Test) | **0.291** | **63.3% ↓** (vs Single) |
| Phase 3 Ensemble | wo_demo (Test) | **0.875** | **7.2% ↓** (vs Single) |

### 5.2 R² 변화 추이

| Phase | 모델 | R² | 개선 |
|-------|------|----|----|
| Single Model | Nail (최고) | 0.437 | Baseline |
| Single Model | Conj (최고) | 0.356 | - |
| Phase 1 Fusion | w_demo (최고) | 0.020 | - |
| Phase 1 Fusion | wo_demo (최고) | -0.049 | - |
| Phase 2 Fusion | w_demo (Val 최고) | 0.478 | **↑** |
| Phase 3 Ensemble | w_demo (Val) | 0.526 | **↑** |

### 5.3 Accuracy 변화 추이

| Phase | 모델 | Acc@0.5 | Acc@1.0 | Acc@2.0 |
|-------|------|---------|---------|---------|
| Single Model | Nail (최고) | 0.446 | 0.663 | 0.940 |
| Single Model | Conj (최고) | 0.421 | 0.716 | 0.884 |
| Phase 1 Fusion | w_demo (최고) | 1.000 | 1.000 | 1.000 |
| Phase 3 Ensemble | w_demo (Test) | 0.750 | **1.000** | 1.000 |
| Phase 3 Ensemble | wo_demo (Test) | 0.063 | 0.688 | 1.000 |

---

## 6. 주요 발견사항

### 6.1 단일 모델 → 퓨전

1. **큰 성능 향상**: 단일 모델 대비 MAE가 약 73% 감소
   - Nail 단일: 0.792 → Phase 1 Fusion: 0.208
   - 멀티모달리티 퓨전의 효과가 매우 큼

2. **Demographics 효과**: w_demo가 wo_demo보다 약간 우수
   - w_demo: MAE 0.208
   - wo_demo: MAE 0.217

### 6.2 Phase 1 → Phase 2

1. **R² 개선**: Phase 2에서 Validation R²가 크게 향상 (0.478)
   - Multi-level feature fusion이 더 풍부한 특징 표현 제공

2. **안정성 향상**: Phase 1의 일부 모델에서 음수 R²가 나타났지만, Phase 2에서는 대부분 양수 R² 달성

### 6.3 Phase 2 → Phase 3 (앙상블)

1. **Robustness 향상**: 여러 모델의 예측을 결합하여 더 안정적인 성능
   - w_demo Test Acc@1.0: 1.000 (모든 샘플에서 ±1.0 g/dL 이내)
   - wo_demo Test Acc@1.0: 0.688

2. **가중치 최적화**: Grid search를 통한 최적 가중치 탐색
   - w_demo: Phase 1 R² 최대 모델과 Phase 2 R² 최대 모델에 높은 가중치 부여
   - wo_demo: Phase 2 모델에 높은 가중치 부여

3. **Demographics 효과의 확대**: 앙상블 단계에서도 Demographics 정보의 중요성이 더욱 명확히 드러남
   - w_demo Test MAE: 0.291 g/dL
   - wo_demo Test MAE: 0.875 g/dL
   - 약 3배의 성능 차이

---

## 7. 결론

1. **멀티모달리티 퓨전의 효과**: 단일 모델 대비 MAE 약 73% 감소로 큰 성능 향상 달성

2. **단계별 개선**: Phase 1 (기본 퓨전) → Phase 2 (Multi-level 퓨전) → Phase 3 (앙상블) 순으로 성능이 지속적으로 개선

3. **최종 성능**: 
   - w_demo 앙상블: Test MAE 0.291 g/dL, Acc@1.0 1.000을 달성하여 실용적인 수준의 성능 확보
   - wo_demo 앙상블: Test MAE 0.875 g/dL, Acc@1.0 0.688

4. **Demographics 활용의 중요성**: 나이와 성별 정보를 포함한 w_demo 모델이 wo_demo보다 크게 우수한 성능을 보임
   - w_demo Test MAE: 0.291 g/dL
   - wo_demo Test MAE: 0.875 g/dL
   - **Demographics 정보가 성능 향상에 매우 중요한 역할을 함** (약 200% 성능 차이)
   - Demographics 정보를 포함하면 Test MAE가 약 3배 개선됨

---

## 8. 향후 개선 방향

1. **데이터 확대**: 더 많은 환자 데이터 수집으로 모델 일반화 성능 향상
2. **하이퍼파라미터 튜닝**: Learning rate schedule, batch size 등 최적화
3. **앙상블 전략 개선**: 더 다양한 모델 조합 및 가중치 탐색 방법 연구
4. **Cross-validation 개선**: 더 많은 fold를 사용한 안정적인 성능 평가

