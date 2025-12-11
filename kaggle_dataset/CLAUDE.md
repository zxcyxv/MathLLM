# MathLLM Project - CLAUDE.MD

## Project Overview

**목표**: AIMO3 (AI Mathematical Olympiad Progress Prize 3) Kaggle 대회 참가
- 올림피아드 수준의 수학 문제를 LaTeX 형식으로 푸는 오픈소스 AI 모델 개발
- 상금: 1위 $262,144 / Overall Progress Prize (47/50 달성 시) $1,589,248+

---

## Current Status (2024-12)

### ✅ 완료된 작업

1. **TRM 아키텍처 구현**
   - Qwen-2.5-Math-1.5B + TRM 통합 완료
   - Same-dimension architecture (3584) 적용
   - 3-level recursion (N_sup=16, T=3, n=6)

2. **훈련 파이프라인**
   - ChatML 형식 적용 (`apply_chat_template()`)
   - GSM8K `#### N` → `\boxed{N}` 변환
   - Deep Supervision + EMA 구현
   - Gradient accumulation (Step-wise State Offloading)

3. **추론 파이프라인**
   - KV Cache 구현 (추론 속도 개선)
   - `generate()` 함수 구현
   - EOS 토큰 처리 수정

4. **데이터셋 지원**
   - GSM8K (7.5K samples)
   - NuminaMath-CoT (860K samples) ← **NEW**
   - MATH (7.5K samples)

5. **평가 스크립트**
   - `eval/trm_eval_simple.py` (train/test split 지원)
   - 전체 출력 표시 옵션

6. **문서화**
   - `ARCHITECTURE.md` - 상세 아키텍처 문서
   - `ISSUES.md` - 발견된 버그와 해결 방법

### 🔄 진행 중

- NuminaMath-CoT 데이터셋으로 대규모 훈련 준비

### 📋 TODO

1. **훈련 실행**
   - [ ] NuminaMath-CoT로 본격 훈련
   - [ ] 적절한 epoch/sample 수 결정
   - [ ] 체크포인트 저장 및 평가

2. **Kaggle 제출 준비**
   - [ ] `upload_kaggle_dataset.py`로 코드 업로드
   - [ ] Kaggle 노트북 작성
   - [ ] 추론 시간 최적화 (5시간 제한)

3. **성능 개선**
   - [ ] 하이퍼파라미터 튜닝
   - [ ] 더 큰 모델 (7B) 실험
   - [ ] Self-consistency / majority voting

---

## Competition Constraints

| 항목 | 제한 |
|------|------|
| GPU Notebook | ≤ 5시간 |
| CPU Notebook | ≤ 9시간 |
| 인터넷 | 비활성화 |
| 정답 형식 | 0-99999 정수 |
| 문제 수 | Public 50 / Private 50 |
| 평가 방식 | 2회 실행, 둘 다 맞으면 1점, 하나만 0.5점 |

---

## Quick Start

### 훈련

```bash
# GSM8K로 훈련 (7.5K, 빠른 테스트용)
python train_trm.py --dataset gsm8k --epochs 3

# NuminaMath로 훈련 (860K, 본격 훈련)
python train_trm.py --dataset numina --num_samples 100000 --epochs 1

# MATH로 훈련 (7.5K, 경시대회 수준)
python train_trm.py --dataset math --epochs 3

# 옵션들
python train_trm.py \
    --dataset numina \
    --batch_size 4 \
    --gradient_accumulation 4 \
    --lr 1e-4 \
    --max_length 1024 \
    --freeze_lm_head \        # lm_head 고정 (TRM만 훈련)
    --output_dir ./checkpoints/trm_numina
```

### 평가

```bash
# GSM8K test set 평가
python eval/trm_eval_simple.py --checkpoint ./checkpoints/trm/checkpoint-XXX

# 상세 출력
python eval/trm_eval_simple.py --checkpoint ./checkpoints/trm/checkpoint-XXX -v

# train set으로 검증
python eval/trm_eval_simple.py --checkpoint ./checkpoints/trm/checkpoint-XXX --split train
```

### Kaggle 업로드

```bash
# Kaggle 데이터셋 생성 및 업로드
python upload_kaggle_dataset.py --username YOUR_KAGGLE_USERNAME

# ZIP만 생성 (수동 업로드)
python upload_kaggle_dataset.py --username YOUR_USERNAME --no-upload
```

---

## Training Datasets

| Dataset | Size | Format | 특징 |
|---------|------|--------|------|
| GSM8K | 7,473 | `#### N` → `\boxed{N}` 변환 | 초중등 수준 |
| **NuminaMath-CoT** | **859,494** | `\boxed{}` (변환 불필요) | 다양한 소스, 추천 |
| MATH | 7,500 | `\boxed{}` (변환 불필요) | 경시대회 수준 |

**Column 매핑:**
- GSM8K: `question`, `answer`
- NuminaMath/MATH: `problem`, `solution`

---

## Architecture Summary

### Same-Dimension Architecture
```
Qwen [3584] → Identity → TRM [3584] → Qwen lm_head [3584→vocab]
```

### 3-Level Recursion

| Level | Parameter | Value | 역할 |
|-------|-----------|-------|------|
| Level 1 | N_supervision | 16 | Deep Supervision steps |
| Level 2 | T_recursion | 3 | Deep Recursion (T-1 no_grad + 1 grad) |
| Level 3 | n_latent | 6 | Latent Recursion (z updates) |

### Effective Depth
```
Depth = 2 × (n + 1) × T × N_sup = 2 × 7 × 3 × 16 = 672 layers
```

### Parameter Count

| Component | Parameters | Trainable |
|-----------|------------|-----------|
| Qwen Backbone | ~1.5B | ❌ Frozen |
| Interface (y_init) | ~3.5K | ✅ |
| TRM Block | ~257M | ✅ |
| TRM Heads (lm_head) | ~545M | ✅ (or frozen) |
| **Total Trainable** | **~802M** (or ~257M with `--freeze_lm_head`) |

---

## File Structure

```
MathLLM/
├── CLAUDE.md                    # 프로젝트 문서 (이 파일)
├── ARCHITECTURE.md              # 상세 아키텍처 문서
├── ISSUES.md                    # 버그 및 해결 방법
├── train_trm.py                 # TRM 훈련 스크립트
├── upload_kaggle_dataset.py     # Kaggle 업로드 스크립트
├── src/
│   ├── config.py                # TRMConfig
│   ├── interface.py             # TRMInterface
│   ├── layers.py                # TRMBlock, TRMAttention, RoPE
│   ├── engine.py                # TinyRecursiveTransformer
│   ├── model.py                 # QwenTRM
│   ├── heads.py                 # TRMHeads
│   └── train.py                 # Trainer
├── eval/
│   ├── trm_eval_simple.py       # TRM 평가 (권장)
│   └── gsm8k_eval.py            # GSM8K 평가
└── checkpoints/                 # 모델 저장
```

---

## Known Issues & Solutions

자세한 내용은 `ISSUES.md` 참조. 주요 이슈:

1. **데이터 형식 불일치**: GSM8K `#### N` → `\boxed{N}` 변환 필요
2. **ChatML 형식 미적용**: `apply_chat_template()` 사용 필수
3. **EOS 토큰 미학습**: labels에 EOS 포함 확인
4. **attention_mask 버그**: `generate()` 시 mask 전달하면 반복 출력
5. **Scheduler N_sup 누락**: total_steps에 N_supervision 반영 필수

---

## Critical Notes

### ChatML 형식 필수
```python
messages = [
    {"role": "system", "content": "Please reason step by step..."},
    {"role": "user", "content": question},
    {"role": "assistant", "content": answer}  # training only
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
```

### NuminaMath는 변환 불필요
```python
# GSM8K: convert_format=True (#### → \boxed)
# NuminaMath/MATH: convert_format=False (이미 \boxed)
```

### generate() 시 attention_mask 제거
```python
# 잘못됨 - 반복 출력 발생
model.generate(input_ids, attention_mask=mask, ...)

# 올바름
model.generate(input_ids, eos_token_id=tokenizer.eos_token_id, ...)
```

---

## Related Links

- [AIMO3 Competition](https://kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
- [TRM Paper](https://arxiv.org/abs/2510.04871)
- [Qwen-2.5-Math](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct)
- [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
