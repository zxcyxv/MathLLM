# MathLLM 남은 작업 계획

## 현재 상태 요약

| 항목 | 상태 | 결과 |
|------|------|------|
| Qwen Zero-shot baseline | ✅ 완료 | **93.71%** (GSM8K) |
| TRM Identity Test | ✅ 완료 | Zero Init 검증 통과 |
| TRM 학습 스크립트 | ✅ 완료 | `train_trm.py` |
| Finetune 학습 스크립트 | ✅ 완료 | `train_finetune.py` |
| LoRA 학습 스크립트 | ✅ 완료 | `train_lora.py` |
| TRM 학습 | 🔄 진행중 | Loss 7.08 → 2.2 (Step 100) |

---

## 남은 작업

### Phase 2: Core Experiment (현재 진행중)

#### 2.1 TRM 학습 완료 및 평가
- [ ] TRM 학습 완료 대기 (예상 ~18시간)
- [ ] 학습된 TRM 모델로 GSM8K 평가
- [ ] 결과 기록

#### 2.2 Finetune Baseline 학습
```bash
uv run python train_finetune.py \
  --n_layers 1 \
  --dataset gsm8k \
  --epochs 3 \
  --batch_size 4 \
  --output_dir ./checkpoints/finetune_1layer
```
- [ ] Last 1 layer finetune 학습 (~233M params)
- [ ] GSM8K 평가
- [ ] TRM과 비교

#### 2.3 (Optional) LoRA Baseline
```bash
uv run python train_lora.py \
  --rank 64 \
  --dataset gsm8k \
  --epochs 3 \
  --output_dir ./checkpoints/lora
```
- [ ] LoRA 학습 (~160M params)
- [ ] GSM8K 평가

---

### Phase 3: Extreme Test (Dynamic T)

#### 3.1 Inference-time Depth Scaling
- [ ] `src/model.py`의 `generate()` 메서드에 dynamic T 지원 추가
- [ ] T=3 → T=5 → T=10으로 증가시키며 테스트
- [ ] 어려운 문제(GSM8K hard subset)에서 정답률 변화 측정

#### 3.2 MATH Dataset 평가
- [ ] `eval/math_eval.py` 구현
- [ ] MATH Level 5 문제에서 TRM vs Finetune 비교
- [ ] Test-time compute 효과 검증

---

### Phase 4: Kaggle Submission 준비

#### 4.1 Inference 최적화
- [ ] 5시간 GPU 제한 내 50문제 해결 가능한지 확인
- [ ] 배치 처리 최적화
- [ ] vLLM 또는 TensorRT 적용 검토

#### 4.2 Submission Format
- [ ] Kaggle notebook 형식으로 변환
- [ ] 오프라인 모델 로딩 (인터넷 비활성화 환경)
- [ ] 답변 형식 검증 (0-99999 정수)

#### 4.3 최종 테스트
- [ ] AIMO Public 문제로 테스트
- [ ] 2회 실행 일관성 확인

---

## 빠른 실험을 위한 대안

현재 TRM 학습이 ~18시간 소요 예상. 빠른 검증을 원하면:

### Option A: 샘플 수 제한
```bash
uv run python train_trm.py \
  --num_samples 1000 \
  --epochs 1 \
  --N_supervision 16 \
  --output_dir ./checkpoints/trm_quick
```
예상 시간: ~1시간

### Option B: N_supervision 감소
```bash
uv run python train_trm.py \
  --N_supervision 4 \
  --epochs 1 \
  --output_dir ./checkpoints/trm_n4
```
예상 시간: ~4-5시간

### Option C: T_recursion 감소
```bash
uv run python train_trm.py \
  --T_recursion 1 \
  --epochs 1 \
  --output_dir ./checkpoints/trm_t1
```
예상 시간: ~6시간

---

## 결과 기록 템플릿

| 모델 | 학습 데이터 | Params | GSM8K | MATH Lvl5 | 비고 |
|------|------------|--------|-------|-----------|------|
| Qwen-7B (Zero-shot) | - | 0 | 93.71% | - | 기준점 |
| Qwen + TRM | GSM8K | 176M | % | % | 실험군 |
| Qwen + Finetune (1 layer) | GSM8K | ~233M | % | % | 대조군 |
| Qwen + LoRA (r=64) | GSM8K | ~160M | % | % | 대조군 |

---

## 핵심 질문

> **"TRM이 단순한 파라미터 증량이 아니라, 재귀적 구조 덕분에 성능이 오르는가?"**

성공 조건:
- `Score(TRM) > Score(Finetune)` with 비슷한 param 수
- T 증가 시 어려운 문제 정답률 상승 (Test-time compute 효과)
