# MathLLM Progress Log

## 2024-12-10: Training Optimization + Inference KV Cache

### 1. Training Speed Optimization

#### 완료된 최적화
| 최적화 | 파일 | 효과 |
|--------|------|------|
| Pre-tokenization | train_trm.py | 매 배치 tokenize → 1회 tokenize |
| AMP bfloat16 | train_trm.py, src/train.py | 10-15% 속도 향상 |
| TRM bfloat16 변환 | train_trm.py:145-151 | dtype 불일치 해결 |
| DataLoader 최적화 | train_trm.py | num_workers=0, pin_memory=True |
| Fast path (acc=1) | src/train.py:196-212 | CPU offload 없이 GPU only |

#### 실패한 최적화 (AVOID)
- **Dynamic Padding**: CUDA 메모리 재할당으로 오히려 느려짐
- **persistent_workers**: Multiprocessing deadlock 발생
- **torch.compile mode="reduce-overhead"**: CUDAGraph 충돌

### 2. Inference KV Cache 구현

#### 수정된 파일
- `src/layers.py`: TRMAttention, TRMBlock에 KV cache + attention_mask 지원
- `src/engine.py`: TinyRecursiveTransformer에 past_kvs, attention_mask 파라미터
- `src/model.py`: generate() 완전 재작성 - Prefill/Decode 분리

#### 성능 개선
```
Before: 10분에 0개 샘플 완료
After:  11분에 80개 샘플 완료 (~8.7초/샘플)
```

#### 핵심 아이디어
```python
# Prefill (prompt 처리)
for sup_step in range(16):
    for t in range(T):
        y, z, new_kvs = engine(x, y, z, cos, sin, use_cache=(is_last))
trm_kv_cache = new_kvs  # 마지막 iteration의 KV만 저장

# Decode (토큰 생성)
for token_idx in range(max_new_tokens):
    # 새 토큰만 16 * T번 처리, 과거는 KV cache에서 참조
    new_y, new_z, new_kvs = engine(
        x=new_hidden,      # [B, 1, D] - 현재 토큰만
        past_kvs=trm_kv_cache,  # 과거 KV
        use_cache=True
    )
    trm_kv_cache = merge(trm_kv_cache, new_kvs)
```

### 3. 평가 스크립트 개선

#### eval/trm_identity_eval.py 변경사항
- Checkpoint 로딩 (`--checkpoint` 옵션)
- Batch 처리 (`--batch_size` 옵션, left-padding)
- AMP 지원 (`--amp/--no-amp`)
- PyTorch 2.6 호환 (`weights_only=False`)

### 4. 훈련 결과 (7B 모델)

#### 설정
```
Model: Qwen/Qwen2.5-Math-7B-Instruct
Dataset: GSM8K 7473 samples
Batch size: 8
Gradient accumulation: 4
Epochs: 1
```

#### Loss 수렴
| Step | Loss |
|------|------|
| 0 | 8.45 |
| 10 | 1.15 |
| 20 | 1.09 |
| 30 | 1.00 |
| 40 | 0.87 |
| Final | 1.31 (avg) |

#### 체크포인트
- `./checkpoints/trm/checkpoint-234`
- 234 global steps (934 batches ÷ 4 accumulation)

### 5. GSM8K 평가 결과

#### 조건
- supervision_steps: 8
- batch_size: 8
- AMP: True

#### 결과
- **Accuracy: 3.75%** (3/80 samples)
- 속도: ~8.7초/샘플

#### 분석
- 1 epoch만 학습으로 부족
- Training loss 1.3은 아직 높음
- 더 많은 epoch 필요

### 6. 다음 단계

1. **더 많은 학습**
   ```bash
   uv run python train_trm.py \
       --model Qwen/Qwen2.5-Math-1.5B-Instruct \
       --epochs 5 \
       --batch_size 8 \
       --gradient_accumulation 1
   ```

2. **1.5B 모델로 빠른 iteration**
   - 7B: ~2시간/epoch
   - 1.5B: ~15-20분/epoch 예상

3. **Hyperparameter 튜닝**
   - Learning rate: 1e-4 → 5e-5?
   - N_supervision: 16 → 8?
   - T_recursion: 3 → 2?

---

## File Changes Summary

```
Modified files:
- src/layers.py      (+102 lines) - KV cache, attention_mask
- src/engine.py      (+52 lines)  - past_kvs, attention_mask
- src/model.py       (+165 lines) - generate() rewrite
- src/train.py       (+147 lines) - AMP, fast path
- train_trm.py       (+102 lines) - pre-tokenization, AMP
- eval/trm_identity_eval.py (+219 lines) - batch, checkpoint, AMP
```
