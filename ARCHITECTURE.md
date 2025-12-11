# Qwen-TRM Model Architecture

> TRM (Tiny Recursive Model)을 Qwen-2.5-Math 백본과 결합한 수학 추론 모델

## 1. 전체 구조 개요

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Qwen-TRM Architecture                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   Input IDs ──► Qwen Backbone (Frozen) ──► hidden_states [B, S, 1536]        │
│                        │                                                      │
│                        │ (1회 실행, no_grad)                                  │
│                        ▼                                                      │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │                    TRM Module (Trainable)                             │   │
│   │                                                                       │   │
│   │   hidden_states ──► TRMInterface ──► x, y, z states                  │   │
│   │                           │                                           │   │
│   │                           ▼                                           │   │
│   │              TinyRecursiveTransformer (Engine)                        │   │
│   │                    (3-Level Recursion)                                │   │
│   │                           │                                           │   │
│   │                           ▼                                           │   │
│   │                      TRMHeads ──► logits [B, S, vocab]               │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 모델 컴포넌트 상세

### 2.1 Qwen Backbone (Frozen)

| 항목 | Qwen2.5-Math-1.5B-Instruct |
|------|---------------------------|
| Parameters | ~1.5B |
| Hidden Size | **1536** |
| Layers | 28 |
| Attention Heads | 12 (GQA) |
| Head Dimension | 128 |
| Vocab Size | 151,936 |
| RoPE theta | 1,000,000 |
| 역할 | 입력 토큰을 semantic hidden states로 인코딩 |
| 학습 | ❌ **Frozen** (requires_grad=False) |

```python
# 사용 방식
with torch.no_grad():
    hidden_states = qwen_backbone(input_ids)  # [B, S, 1536]
```

---

### 2.2 TRMInterface (State Initializer)

```
┌─────────────────────────────────────────────────────────────┐
│  TRMInterface                                                │
├─────────────────────────────────────────────────────────────┤
│  입력: hidden_states [B, S, 1536]                           │
│                                                              │
│  extract_context(hidden_states):                             │
│      return hidden_states  # Identity (투영 없음)            │
│                                                              │
│  initialize_states(x):                                       │
│      y = y_init.expand(B, S, -1)  # learnable, 초기값 0     │
│      z = zeros(B, S, 1536)         # 0으로 초기화           │
│      return y, z                                             │
│                                                              │
│  출력: x, y, z  각각 [B, S, 1536]                           │
└─────────────────────────────────────────────────────────────┘
```

| 파라미터 | Shape | 설명 |
|----------|-------|------|
| `y_init` | [1, 1, 1536] | Learnable 초기 상태 (0으로 초기화) |

**총 파라미터: ~1.5K**

---

### 2.3 TinyRecursiveTransformer (Engine)

**Level 3: Latent Recursion** 구현

```
┌─────────────────────────────────────────────────────────────┐
│  TinyRecursiveTransformer                                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  forward(x, y, z, cos, sin):                                │
│                                                              │
│      # n번 z 업데이트 (Reasoning Mode)                       │
│      for i in range(n):  # n=6                              │
│          h = x + y + z           # Additive Fusion          │
│          z = TRMBlock(h)         # Direct Replacement       │
│                                                              │
│      # 1번 y 업데이트 (Prediction Mode)                      │
│      h = y + z                   # x 제외!                   │
│      y = TRMBlock(h)             # Direct Replacement       │
│                                                              │
│      return y, z                                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**핵심 특징:**
- **Direct Replacement**: `z = block(h)` (NOT `z = z + block(h)`)
- **Additive Fusion**: x, y, z를 더해서 입력으로 사용
- **x 제외**: y 업데이트 시 context x를 제외하여 추론과 예측 분리

---

### 2.4 TRMBlock (Shared Transformer Block)

```
┌─────────────────────────────────────────────────────────────┐
│  TRMBlock (1개, 공유됨)                                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input h [B, S, 1536]                                        │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  RMSNorm(1536)                                       │    │
│  │       │                                              │    │
│  │       ▼                                              │    │
│  │  TRMAttention                                        │    │
│  │    • num_heads: 12                                   │    │
│  │    • head_dim: 128                                   │    │
│  │    • Q, K, V projections [1536 → 1536]              │    │
│  │    • RoPE: apply_rotary_pos_emb(q, k, cos, sin)     │    │
│  │    • Scaled Dot-Product Attention (Flash Attention)  │    │
│  │    • O projection [1536 → 1536] (Zero Init)         │    │
│  └─────────────────────────────────────────────────────┘    │
│       │                                                      │
│       ▼ (NO residual!)                                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  RMSNorm(1536)                                       │    │
│  │       │                                              │    │
│  │       ▼                                              │    │
│  │  SwiGLU FFN                                          │    │
│  │    • gate_proj [1536 → 6144]                        │    │
│  │    • up_proj   [1536 → 6144]                        │    │
│  │    • down_proj [6144 → 1536] (Zero Init)            │    │
│  │    • output = down(SiLU(gate(x)) * up(x))           │    │
│  └─────────────────────────────────────────────────────┘    │
│       │                                                      │
│       ▼                                                      │
│  Output [B, S, 1536]                                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Zero Initialization:**
```python
nn.init.zeros_(self.attn.o_proj.weight)   # Attention output
nn.init.zeros_(self.mlp.down_proj.weight) # FFN output
```
→ 초기 출력 = 0, 학습 안정성 확보

**파라미터 계산:**

| 레이어 | Shape | Parameters |
|--------|-------|------------|
| norm1 | [1536] | 1,536 |
| q_proj | [1536, 1536] | 2,359,296 |
| k_proj | [1536, 1536] | 2,359,296 |
| v_proj | [1536, 1536] | 2,359,296 |
| o_proj | [1536, 1536] | 2,359,296 |
| norm2 | [1536] | 1,536 |
| gate_proj | [6144, 1536] | 9,437,184 |
| up_proj | [6144, 1536] | 9,437,184 |
| down_proj | [1536, 6144] | 9,437,184 |
| **Total** | | **~37.8M** |

---

### 2.5 TRMHeads (Output Head)

```
┌─────────────────────────────────────────────────────────────┐
│  TRMHeads                                                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input y [B, S, 1536]                                        │
│       │                                                      │
│       ▼                                                      │
│  RMSNorm(1536)                                               │
│       │                                                      │
│       ▼                                                      │
│  lm_head Linear(1536, 151936, bias=False)                    │
│       │                                                      │
│       ▼                                                      │
│  Output logits [B, S, 151936]                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**lm_head 초기화:**
```python
# Qwen의 lm_head 가중치를 직접 복사 (동일 차원이므로 SVD 불필요)
trm_lm_head.weight.copy_(qwen_lm_head.weight)
```

**파라미터:**

| 레이어 | Shape | Parameters |
|--------|-------|------------|
| norm | [1536] | 1,536 |
| lm_head | [151936, 1536] | 233,373,696 |
| **Total** | | **~233M** |

---

## 3. 3-Level Recursive Training

TRM 논문의 핵심: **3중 재귀 루프**로 깊은 연산 수행

### 3.1 Loop Hierarchy

```
Level 1: Deep Supervision (N_sup = 16)
└── Level 2: Deep Recursion (T = 3)
    └── Level 3: Latent Recursion (n = 6)
        └── TRMBlock 호출
```

### 3.2 Effective Depth

```
Total Block Calls = n_latent × T × N_sup
                  = (6 + 1) × 3 × 16
                  = 336 calls per token
```

### 3.3 Training Algorithm

```python
# ═══════════════════════════════════════════════════════════════
# LEVEL 1: Deep Supervision (N_sup = 16)
# ═══════════════════════════════════════════════════════════════
y, z = None, None

for sup_step in range(N_sup):  # 16번

    # ─────────────────────────────────────────────────────────
    # State Initialization (첫 step만)
    # ─────────────────────────────────────────────────────────
    if y is None:
        y, z = interface.initialize_states(x)

    # ─────────────────────────────────────────────────────────
    # LEVEL 2: Deep Recursion (T = 3)
    # ─────────────────────────────────────────────────────────
    # T-1회: no_grad (메모리 절약)
    with torch.no_grad():
        for t in range(T - 1):  # 2회
            y, z = engine(x, y, z, cos, sin)  # Level 3 포함

    # 마지막 1회: with grad (학습)
    y, z = engine(x, y, z, cos, sin)  # Level 3 포함

    # ─────────────────────────────────────────────────────────
    # Output & Loss
    # ─────────────────────────────────────────────────────────
    logits = heads(y)
    loss = cross_entropy(logits, labels)

    # ─────────────────────────────────────────────────────────
    # Optimization (매 supervision step마다!)
    # ─────────────────────────────────────────────────────────
    loss.backward()
    clip_grad_norm_(params, max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    ema.update()

    # ─────────────────────────────────────────────────────────
    # State Propagation (다음 step으로)
    # ─────────────────────────────────────────────────────────
    y = y.detach()  # 그래프 절단, 값 유지
    z = z.detach()
```

---

## 4. State Variables

| State | Shape | 역할 | 초기화 |
|-------|-------|------|--------|
| **x** | [B, S, 1536] | Context (Anchor) | Qwen hidden_states |
| **y** | [B, S, 1536] | Solution (Answer) | learnable y_init (zeros) |
| **z** | [B, S, 1536] | Reasoning (Thought) | zeros |

**역할 분리:**
- `x`: 입력 문맥, 고정 앵커
- `z`: 중간 추론 과정 (n번 업데이트)
- `y`: 최종 답변 표현 (1번 업데이트)

---

## 5. Parameter Summary

### 5.1 Trainable vs Frozen

| Component | Parameters | Trainable |
|-----------|------------|-----------|
| Qwen Backbone | ~1.5B | ❌ Frozen |
| TRMInterface (y_init) | ~1.5K | ✅ |
| TRMBlock (engine) | ~37.8M | ✅ |
| TRMHeads (norm + lm_head) | ~233M | ✅ |
| **Total Trainable** | **~271M** | |

### 5.2 lm_head Freeze Option

```bash
# lm_head도 훈련 (기본값)
python train_trm.py

# lm_head freeze (TRM block만 훈련)
python train_trm.py --freeze_lm_head
```

lm_head freeze 시 trainable params: **~37.8M**

---

## 6. Inference with KV Cache

### 6.1 문제점

매 토큰 생성 시 16 supervision steps × 전체 시퀀스 처리 → **O(S³)**

### 6.2 해결: KV Cache

```
Prefill Phase:
  - 전체 prompt를 16 × 3 supervision으로 처리
  - 마지막 iteration의 KV를 캐시

Decode Phase (토큰당):
  - 새 토큰 1개만 처리
  - 과거 KV는 캐시에서 참조
  - 새 KV를 캐시에 추가
```

### 6.3 Virtual Layers

TRM engine은 `n+1`번 block을 호출:
- n번: z 업데이트
- 1번: y 업데이트

각 호출을 "virtual layer"로 취급하여 **별도 KV cache** 유지

```python
past_kvs = [
    (k_0, v_0),  # z update 1
    (k_1, v_1),  # z update 2
    ...
    (k_n, v_n),  # z update n
    (k_y, v_y),  # y update
]  # Total: n+1 = 7 entries
```

---

## 7. File Structure

```
src/
├── config.py      # TRMConfig 설정
├── interface.py   # TRMInterface (상태 초기화)
├── layers.py      # RoPE, TRMAttention, SwiGLU, TRMBlock
├── engine.py      # TinyRecursiveTransformer (Level 3)
├── heads.py       # TRMHeads (출력 헤드)
├── model.py       # QwenTRM (Level 2 + 통합)
├── train.py       # Trainer (Level 1 + 최적화)
└── dataset.py     # 데이터셋 처리
```

---

## 8. Key Design Decisions

### 8.1 Same Dimension (No Bottleneck)

```
TRM dim = Qwen hidden dim = 1536
```
- Interface에서 projection 불필요 (Identity)
- lm_head 직접 복사 가능 (SVD 불필요)
- 정보 손실 없음

### 8.2 Direct Replacement (No Residual)

```python
# Standard Transformer
out = h + block(h)

# TRM Block
out = block(h)  # h 안 더함!
```
- Zero Init과 결합하여 안정적 학습
- 초기 출력 = 0 → 값 폭발 방지

### 8.3 RoPE Compatibility

```
head_dim = 128  (Qwen과 동일)
rope_theta = 1e6  (Qwen과 동일)
```
- Qwen의 위치 인코딩 패턴 재활용
- 긴 시퀀스 처리 능력 유지

### 8.4 Deep Supervision

매 supervision step마다 loss 계산 & weight 업데이트
```
배치 1개당 optimizer.step() = 16회
```
- 빠른 수렴
- 안정적 학습

---

## 9. Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| n_latent | 6 | Level 3: z 업데이트 횟수 |
| T_recursion | 3 | Level 2: Deep Recursion (T-1 no_grad) |
| N_supervision | 16 | Level 1: Deep Supervision steps |
| learning_rate | 1e-4 | AdamW LR |
| batch_size | 4 | Per-GPU batch size |
| gradient_accumulation | 1 | Effective batch = batch × acc |
| max_length | 1024 | Max sequence length |
| epochs | 3 | Training epochs |

### Scheduler Note

```python
total_steps = batches × epochs × N_supervision
            = 1868 × 3 × 16
            = 89,664 steps
```
**N_supervision 포함 필수** - 빠뜨리면 LR이 16배 빨리 decay

---

## 10. ChatML Format

Qwen의 공식 대화 형식 사용:

```
<|im_start|>system
Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>
```

**Key Tokens:**
- `<|im_start|>`: 151644
- `<|im_end|>`: 151645 (EOS)
- `<|endoftext|>`: 151643 (PAD)

---

## 11. Known Issues & Notes

### 11.1 attention_mask in generate()

`model.generate()`에 `attention_mask`를 전달하면 반복 붕괴 발생
```python
# 잘못됨
model.generate(input_ids, attention_mask=mask, ...)

# 올바름
model.generate(input_ids, ...)
```

### 11.2 Unicode Sensitivity

데이터셋의 유니코드 문자(예: ' vs ')에 따라 출력 품질 차이 발생 가능

### 11.3 Memory Usage

- Backbone: ~3GB (bfloat16)
- TRM Trainable: ~0.5GB
- Gradients & Optimizer: ~2GB
- **Total**: ~6-8GB per GPU (batch_size=4)
