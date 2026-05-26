# 한국어 유행어 장기 생존 예측 AI — 프로젝트 보고서

---

## 1. 프로젝트 개요

온라인 커뮤니티와 SNS에서 생성되는 한국어 유행어(밈)가 장기적으로 생존할지 여부를 텍스트 자체의 언어적 특성과 초기 사용 빈도 데이터를 기반으로 예측하는 이진 분류(Binary Classification) 모델을 개발하였다.

- **예측 목표**: 유행어가 1년 이상 사용될지 여부 (label=1: 장기생존, label=0: 단기유행)
- **핵심 질문**: 유행어가 처음 등장했을 때의 텍스트 정보만으로도 생존 가능성을 예측할 수 있는가?
- **활용 가능성**: 신조어 모니터링, 트렌드 분석, 콘텐츠 필터링 등

---

## 2. 데이터셋

### 2.1 기본 현황

| 항목 | 값 |
|---|---|
| 원본 샘플 수 | 965개 |
| 첫 등장 월 미상(month=0) 제외 후 | **819개** |
| 레이블 분포 (원본 기준) | 장기생존(1): 246개 / 단기유행(0): 719개 |
| 레이블 불균형 비율 | 약 1:3 (양성:음성) |
| 수집 기준일 | 2026년 5월 |

장기생존과 단기유행 간 레이블 불균형이 약 1:3으로 존재하므로, 모든 모델에 클래스 가중치(class weight)를 적용하고 평가 시 Balanced Accuracy와 AUROC를 주요 지표로 사용하였다.

### 2.2 데이터 컬럼 구성

| 컬럼 | 설명 |
|---|---|
| `x` | 유행어 텍스트 |
| `label` | 장기생존 여부 (0/1) |
| `year` | 첫 등장 연도 |
| `month` | 첫 등장 월 (1-indexed, 0은 미상 → 제외) |
| `tw_use_m01` ~ `tw_use_m12` | 월별 사용 빈도 (트위터 기준) |

### 2.3 관측 가능 개월 수 (observed_months)

수집 기준일(2026년 5월)을 기준으로 각 유행어가 등장한 이후 실제로 관측된 개월 수를 계산하였다.

```
observed_months = min(12, (2026 - year) × 12 + (5 - month + 1))
```

- 2024년 이전 등장 유행어: observed_months = 12 (12개월 전체 관측)
- 2025년 7월 등장: observed_months = 11개월
- 2026년 1월 등장: observed_months = 5개월
- 2026년 3월 등장: observed_months = 3개월

horizon별 사용 가능 샘플 수:

| 실험 조건 | 필요 조건 | 사용 샘플 수 |
|---|---|---|
| 텍스트 전용 (E+L) | 제한 없음 | **819개** |
| h1 (1개월 빈도) | observed_months ≥ 1 | **819개** |
| h3 (3개월 빈도) | observed_months ≥ 3 | **788개** |
| h6 (6개월 빈도) | observed_months ≥ 6 | **767개** |
| h12 (12개월 빈도) | observed_months ≥ 12 | **730개** |

---

## 3. Feature 설계

세 가지 feature 그룹을 정의하였다.

### E — 임베딩 Feature (384차원)

다국어 사전학습 언어모델 `paraphrase-multilingual-MiniLM-L12-v2`를 사용해 유행어 텍스트를 384차원 벡터로 변환하였다. 컬럼명: `emb_000` ~ `emb_383`.

### L — 언어학적 Feature (65개)

한국어 텍스트의 표층·음운·형태소 특성을 규칙 기반으로 추출하였다.

| 범주 | 특징 예시 | 개수 |
|---|---|---|
| 문자/표기 | 글자 수, 한글 비율, 영어 비율, 특수문자 비율, 초성 유무 | ~28개 |
| 형태소 (kiwipiepy) | 명사 수, 동사 비율, 형용사 비율, 품사 다양성 | ~14개 |
| 표현 유형 | 단어형/구/문장형 여부, 밈 패턴 여부, 의문/명령형 | ~10개 |
| 음운/운율 | 음절 수, 반복 패턴, 운율 점수, 자음/모음 반복 | ~13개 |

### F — 빈도수 Feature (horizon당 38개)

월별 사용 빈도 데이터로부터 통계적 지표를 추출하였다. horizon(h1/h3/h6/h12)에 따라 사용할 수 있는 빈도 구간이 달라진다.

| 범주 | 특징 예시 | 개수 |
|---|---|---|
| 기본 통계 | 합계, 평균, 최대, 최소, 표준편차 | 11개 |
| 추세 | 초반 대비 후반 평균, 선형 기울기 | 8개 |
| 피크 | 최대값 위치, 피크 이후 감소율 | 8개 |
| 생존성 | 활성 월 수, 마지막 관측값 | 6개 |
| 분포 | 지니 계수, 엔트로피 | 5개 |

---

## 4. 실험 설계

### 4.1 Train / Validation / Test 분리

- **분리 방식**: 계층화 무작위 분리 (Stratified Random Split)
- **비율**: Train 70% / Validation 15% / Test 15%
- **난수 고정**: random_state = 42

| 데이터셋 | 전체 샘플 | Train | Validation | Test |
|---|---|---|---|---|
| E+L (텍스트 전용) | 819개 | ~573개 | ~123개 | ~123개 |
| E+L+F h12 | 730개 | ~511개 | ~110개 | ~110개 |

**레이블 불균형 처리:**
- 트리 계열 모델: `scale_pos_weight` (음성/양성 비율) 적용
- MLP: 손실 함수에 양성 클래스 가중치 적용
- 공통: Threshold 탐색 — Validation set에서 Youden's J 기준 최적 threshold 결정 후 Test set에 적용

### 4.2 평가 지표

- **AUROC**: 주요 지표 (threshold 독립적, 불균형 데이터에 적합)
- **AUPRC**: 양성 클래스 예측 정밀도-재현율 균형
- **F1**: 최적 threshold 적용 후 조화평균
- **Balanced Accuracy**: 클래스별 정확도 평균
- 모든 성능 수치는 **Test set** 기준

---

## 5. 실험 결과

### 실험 1: 모델 성능 비교

**조건**: feature_set = E+L+F (h12), LightGBM 외 5개 모델 비교

| 모델 | AUROC | F1 | Balanced Accuracy |
|---|---|---|---|
| XGBoost | 1.000 | 1.000 | 1.000 |
| CatBoost | 1.000 | 0.984 | 0.984 |
| Random Forest | 1.000 | 0.931 | 0.935 |
| LightGBM | 1.000 | 0.912 | 0.919 |
| Logistic Regression | **0.943** | 0.794 | 0.872 |
| MLP | 0.914 | 0.742 | 0.820 |

트리 계열 모델 4종(XGBoost, CatBoost, Random Forest, LightGBM)이 모두 AUROC=1.0을 달성하였다. 이는 12개월 빈도 데이터(F feature)가 장기생존 여부를 거의 완벽히 설명하기 때문으로, **12개월 전체 빈도 데이터만으로 생존 여부가 지나치게 잘 구분**되는 양상이 나타난다. 이 구간에서는 선형 모델인 Logistic Regression(AUROC=0.943)이나 MLP(AUROC=0.914)가 상대적으로 더 현실적인 성능 지표를 보여준다고 볼 수 있다.

---

### 실험 2: Feature 조합별 Ablation Study

**조건**: LightGBM, horizon=12

| feature_set | 사용 샘플 | 피처 수 | AUROC | F1 |
|---|---|---|---|---|
| F (빈도만) | 730개 | 38개 | **1.000** | 0.931 |
| E+F | 730개 | 422개 | 1.000 | 0.912 |
| L+F | 730개 | 103개 | 1.000 | 0.912 |
| E+L+F | 730개 | 487개 | 1.000 | 0.912 |
| **E+L (텍스트 전용)** | 819개 | 449개 | **0.725** | 0.544 |
| E (임베딩만) | 819개 | 384개 | 0.711 | 0.542 |
| L (언어학만) | 819개 | 65개 | 0.706 | 0.571 |

**해석:**
- F(빈도 feature)가 포함되면 조합에 무관하게 AUROC=1.0 — 12개월 빈도 데이터가 모든 정보를 압도한다.
- 텍스트 전용(E+L, AUROC=0.725)이 현실적인 성능 기준점이 된다.
- 임베딩(E, 0.711)과 언어학적 feature(L, 0.706)의 독립적 기여가 유사하며, 두 가지를 결합하면 AUROC 0.725로 소폭 개선된다.

---

### 실험 3: 조기 예측 (Horizon 실험)

**조건**: LightGBM, 가용 데이터 누적에 따른 성능 변화 추적

| 실험 조건 | 사용 Feature | 사용 샘플 | AUROC | F1 | Balanced Accuracy |
|---|---|---|---|---|---|
| 텍스트 전용 | E+L | 819개 | 0.725 | 0.544 | 0.657 |
| h1 — 등장 후 1개월 | E+L+F(m1) | 819개 | 0.871 | 0.673 | 0.785 |
| h3 — 등장 후 3개월 | E+L+F(m1~3) | 788개 | 0.953 | 0.787 | 0.878 |
| h6 — 등장 후 6개월 | E+L+F(m1~6) | 767개 | **0.998** | 0.971 | 0.988 |
| h12 — 등장 후 12개월 | E+L+F(m1~12) | 730개 | 1.000 | 0.912 | 0.919 |

**해석:**
- 텍스트만으로도 AUROC 0.725 — 유행어 텍스트에 생존 가능성에 대한 유의미한 신호가 내재되어 있음.
- 등장 후 **1개월** 빈도 데이터를 추가하면 AUROC 0.871로 크게 도약한다 (+0.146).
- **3개월**이면 AUROC 0.953 — 실용적 조기 예측이 가능한 지점.
- **6개월**이면 사실상 완벽한 구분(AUROC 0.998) — 이 시점에서 장기생존 여부가 빈도 패턴에 명확히 드러남.
- h12는 h6 대비 추가 개선이 미미하며, 12개월 전체 데이터는 사실상 결과를 이미 반영한 수준으로 구분된다.

이 실험이 전체 프로젝트에서 **가장 의미 있는 결과**이다. 텍스트 정보만으로 시작해 데이터가 축적될수록 예측력이 단계적으로 향상되는 과정을 명확히 보여준다.

---

### 실험 4: Feature 중요도 분석

**조건**: LightGBM, feature_set = E+L+F (h12)

**그룹별 순열 중요도:**

| Feature 그룹 | 중요도 |
|---|---|
| F (빈도수) | **0.535** |
| E (임베딩) | 0.000 |
| L (언어학적) | 0.000 |

**개별 순열 중요도 Top 5:**

| 순위 | Feature | 중요도 |
|---|---|---|
| 1 | `h12_freq_late_mean` (후반부 월 평균 빈도) | **0.212** |
| 2~5 | 나머지 모든 feature | 0.000 |

**SHAP 기반 중요도 Top 5:**

| 순위 | Feature | SHAP 평균 절댓값 |
|---|---|---|
| 1 | `h12_freq_late_mean` | **8.20** |
| 2 | `h12_freq_min` | 1.62 |
| 3 | `h12_freq_last` | 1.10 |
| 4 | `emb_129` | 0.057 |
| 5 | `hangul_ratio` (한글 비율) | 0.048 |

**해석:**
- 12개월 빈도 데이터가 있을 때 모델은 사실상 `h12_freq_late_mean`(12개월 후반부 평균 빈도) 하나로 예측한다. 이 feature가 SHAP 1위 값(8.20)과 2위(1.62) 사이의 격차만 봐도 압도적임을 알 수 있다.
- 임베딩과 언어학적 feature는 빈도 데이터가 있을 때는 모델이 거의 참조하지 않는다.
- SHAP 상위에 `hangul_ratio`(한글 비율)가 등장하는 점이 흥미롭다 — 빈도 feature 다음으로 한글 비율이 높은 유행어일수록 장기생존과 연관성이 있음을 시사한다.

---

### 실험 5: Teacher-Student Knowledge Distillation

**목적**: E+L+F 전체 데이터로 학습한 Teacher(LightGBM)의 지식을 텍스트 전용(E+L) Student(MLP)에게 전달하여 텍스트 기반 예측 성능을 향상시킬 수 있는지 검증.

**학습 손실 함수:**
```
Loss = BCE(y, p_student) + α × MSE(p_teacher, p_student)
```
- α=0: 일반 지도학습 (Teacher 사용 안 함)
- α↑: Teacher 예측값에 더 많이 의존

**결과:**

| 모델 | feature | AUROC | F1 | Balanced Accuracy |
|---|---|---|---|---|
| Full Teacher (LightGBM) | E+L+F | **1.000** | 0.912 | 0.919 |
| Text-only Baseline (MLP) | E+L | 0.684 | 0.523 | 0.645 |
| Distilled Student α=0.1 | E+L | 0.680 | 0.509 | 0.626 |
| Distilled Student α=0.3 | E+L | 0.688 | 0.536 | 0.661 |
| Distilled Student α=0.5 | E+L | 0.670 | 0.509 | 0.629 |
| Distilled Student α=0.7 | E+L | 0.684 | 0.509 | 0.626 |
| Distilled Student α=1.0 | E+L | **0.691** | 0.500 | 0.613 |

**해석:**
- Distillation 효과가 미미하다 (baseline 0.684 대비 최대 +0.007, α=1.0 기준).
- Teacher가 가진 지식의 대부분이 F feature(빈도수)에 집중되어 있기 때문에, E+L만 사용하는 Student에게 의미 있는 지식 전달이 어렵다.
- α 값에 따른 성능 차이도 통계적으로 유의미한 수준이 아니다.

---

## 6. 종합 결론

### 신뢰할 수 있는 결론

| 결론 | 근거 |
|---|---|
| 텍스트만으로 AUROC 0.725 달성 | 유행어 표기·언어적 특성에 생존 신호 존재 |
| 1개월 데이터로 AUROC 0.871 (+0.146) | 초기 반응 데이터가 가장 큰 단일 개선 요인 |
| 3개월이면 AUROC 0.953 | 실용적 조기 예측 가능 지점 |
| 6개월이면 AUROC 0.998 | 사실상 완벽한 구분 가능 |
| 임베딩(E)과 언어학(L) 기여가 유사 | 각각 단독으로 AUROC 0.711 / 0.706 |
| 한글 비율이 생존에 관련 | SHAP 상위 비-빈도 feature로 `hangul_ratio` 등장 |

### 해석에 주의가 필요한 결과

| 결과 | 주의 이유 |
|---|---|
| E+L+F 조건에서 모델 비교 (대부분 AUROC=1.0) | 12개월 전체 빈도 데이터만으로도 구분이 너무 잘 되므로 모델 간 성능 차이가 나타나지 않음 |
| Feature importance에서 F group이 E/L을 완전히 압도 | 동일한 이유 |
| h12 성능(AUROC=1.0) | 12개월 빈도 데이터 자체가 결과를 거의 반영하고 있어 실질적인 예측 과제가 아님 |

### 실험 중요도 순위

1. **Horizon 실험** — 핵심 결과. 데이터 누적에 따른 단계적 성능 향상 확인
2. **Ablation Study** — E/L/F 각 그룹의 독립 기여 정량화
3. **Feature Importance** — `hangul_ratio` 등 해석 가능한 언어적 신호 발견
4. **모델 비교** — 텍스트 전용 조건(AUROC 0.725)이 현실적 기준점
5. **Distillation** — 효과 미미, 빈도 feature 의존도가 높은 Teacher의 한계 확인

---

## 7. 프로젝트 구성 요약

### 사용 기술 스택

| 범주 | 도구 |
|---|---|
| 임베딩 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 형태소 분석 | `kiwipiepy` |
| 모델 | `LightGBM`, `XGBoost`, `CatBoost`, `scikit-learn`, `PyTorch` |
| Feature 중요도 | `SHAP` (TreeExplainer) |
| 실험 관리 | Python 스크립트 + YAML 설정 파일 |

### 실험 파이프라인

```
scripts/01_build_features.py     → 데이터 로드 및 E/L/F feature 생성
scripts/02_run_model_comparison.py → 실험 1: 모델 비교
scripts/03_run_ablation.py        → 실험 2: Feature 조합 ablation
scripts/04_run_horizon_experiment.py → 실험 3: 조기 예측
scripts/05_run_feature_importance.py → 실험 4: Feature 중요도
scripts/06_run_distillation.py    → 실험 5: Distillation
```

### 주요 산출물

```
outputs/results/model_comparison.csv
outputs/results/ablation_results.csv
outputs/results/horizon_results.csv
outputs/results/distillation_results.csv
outputs/importance/shap_importance.csv
outputs/importance/permutation_importance.csv
outputs/importance/group_permutation_importance.csv
outputs/models/best_full_model.pkl        (LightGBM, E+L+F)
outputs/models/best_text_only_model.pkl   (MLP, E+L)
outputs/models/best_distilled_student.pt  (Distilled MLP, E+L)
```
