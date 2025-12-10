# MathLLM Project - CLAUDE.MD

## Project Overview

**목표**: AIMO3 (AI Mathematical Olympiad Progress Prize 3) Kaggle 대회 참가
- 올림피아드 수준의 수학 문제를 LaTeX 형식으로 푸는 오픈소스 AI 모델 개발
- 상금: 1위 $262,144 / Overall Progress Prize (47/50 달성 시) $1,589,248+

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

## Technical Architecture: Qwen-TRM Integrated Model

### Core Philosophy
> "Qwen의 강력한 표현 학습 + TRM의 무한 연산 깊이 결합"

파라미터 증량 없이 **재귀적 추론(Recursive Reasoning)**으로 연산 깊이를 확장

### High-Level Architecture (Simplified)

**핵심 변경: TRM이 Qwen과 동일한 차원(3584)에서 동작**
- Interface projection 제거 (identity)
- lm_head 직접 복사 (SVD 불필요)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input IDs ──► Qwen Backbone (Frozen) ──► hidden_states [B,S,3584]  │
│                        │                                             │
│                        │  (1회 실행, no_grad)                        │
│                        ▼                                             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Deep Supervision Loop (N_sup = 16)               │   │
│  │  ┌────────────────────────────────────────────────────────┐  │   │
│  │  │                                                        │  │   │
│  │  │  hidden_states ──► x [B,S,3584]  (Identity, no proj)  │  │   │
│  │  │                      │                                 │  │   │
│  │  │                      ▼                                 │  │   │
│  │  │  ┌─────────────────────────────────────────────────┐  │  │   │
│  │  │  │         Deep Recursion (T = 3)                  │  │  │   │
│  │  │  │  ┌───────────────────────────────────────────┐ │  │  │   │
│  │  │  │  │      Latent Recursion (n = 6)             │ │  │  │   │
│  │  │  │  │  for i in range(n):                       │ │  │  │   │
│  │  │  │  │      z = TRMBlock(x + y + z)              │ │  │  │   │
│  │  │  │  │  y = TRMBlock(y + z)                      │ │  │  │   │
│  │  │  │  └───────────────────────────────────────────┘ │  │  │   │
│  │  │  │  T-1회: no_grad / 마지막 1회: with grad        │  │  │   │
│  │  │  └─────────────────────────────────────────────────┘  │  │   │
│  │  │                      │                                 │  │   │
│  │  │                      ▼                                 │  │   │
│  │  │  y ──► TRMHeads (Norm + lm_head) ──► logits [B,S,V]   │  │   │
│  │  │                      │                                 │  │   │
│  │  │                      ▼                                 │  │   │
│  │  │  loss = CrossEntropy(logits, labels)                  │  │   │
│  │  │  loss.backward() → optimizer.step() → EMA.update()    │  │   │
│  │  │                      │                                 │  │   │
│  │  │              y, z detach ──► 다음 step으로             │  │   │
│  │  │                                                        │  │   │
│  │  └────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Change Summary

### Before (Bottleneck Architecture)
```
Qwen [3584] → Interface MLP [3584→1024] → TRM [1024] → SVD lm_head [1024→vocab]
```

### After (Same-Dimension Architecture)
```
Qwen [3584] → Identity → TRM [3584] → Qwen lm_head [3584→vocab]
```

### 비교

| 항목 | Before | After |
|------|--------|-------|
| TRM dim | 1024 | **3584** |
| num_heads | 8 | **28** |
| head_dim | 128 | **128** (동일) |
| Interface | MLP (3584→1024) | **Identity** |
| lm_head | SVD 압축 | **Qwen 직접 복사** |
| TRM Block params | ~21M | **~257M** |
| 정보 손실 | Bottleneck 압축 | **없음** |

---

## Training Data Flow (상세)

### Step-by-Step Execution

```python
# ═══════════════════════════════════════════════════════════════════
# 1. BACKBONE ENCODING (1회, frozen, no_grad)
# ═══════════════════════════════════════════════════════════════════
with torch.no_grad():
    hidden_states = qwen_backbone(input_ids)  # [B, S, 3584]

# ═══════════════════════════════════════════════════════════════════
# 2. DEEP SUPERVISION LOOP (N_sup = 16회)
# ═══════════════════════════════════════════════════════════════════
y, z = None, None

for sup_step in range(N_sup):  # 16번 반복

    # ─────────────────────────────────────────────────────────────
    # 2.1 Context (Identity - no projection)
    # ─────────────────────────────────────────────────────────────
    x = hidden_states  # [B, S, 3584] - 그대로 사용

    # 첫 step에서 y, z 초기화
    if y is None:
        y = y_init.expand(B, S, -1)  # learnable, 초기값 0
        z = torch.zeros(B, S, 3584)   # 0으로 초기화

    # ─────────────────────────────────────────────────────────────
    # 2.2 Deep Recursion (T = 3회)
    # ─────────────────────────────────────────────────────────────
    cos, sin = rotary_emb(x, S)  # RoPE 임베딩

    # T-1회: no_grad (메모리 절약)
    with torch.no_grad():
        for t in range(T - 1):  # 2회
            y, z = latent_recursion(x, y, z, cos, sin)

    # 마지막 1회: with grad (학습용)
    y, z = latent_recursion(x, y, z, cos, sin)

    # ─────────────────────────────────────────────────────────────
    # 2.3 Output & Loss
    # ─────────────────────────────────────────────────────────────
    logits = trm_heads(y)  # [B, S, vocab_size]
    loss = cross_entropy(logits, labels)

    # ─────────────────────────────────────────────────────────────
    # 2.4 Optimization
    # ─────────────────────────────────────────────────────────────
    loss.backward()
    clip_grad_norm_(trainable_params, max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    ema.update()

    # ─────────────────────────────────────────────────────────────
    # 2.5 State Propagation
    # ─────────────────────────────────────────────────────────────
    y = y.detach()  # 그래프 절단, 값은 유지
    z = z.detach()
```

### Latent Recursion (Level 3) 상세

```python
def latent_recursion(x, y, z, cos, sin, n=6):
    """
    TRM 논문 Figure 3의 핵심 알고리즘
    - n번 z 업데이트 (reasoning)
    - 1번 y 업데이트 (prediction)
    - Direct Replacement (no residual!)
    """
    # n번 z 업데이트 (Reasoning Mode)
    for i in range(n):  # 6회
        h = x + y + z           # Additive fusion
        z = trm_block(h, cos, sin)  # Direct replacement

    # 1번 y 업데이트 (Prediction Mode)
    h = y + z                   # x 완전 제외!
    y = trm_block(h, cos, sin)  # Direct replacement

    return y, z
```

---

## Module Specifications

### 1. Qwen Backbone (Frozen)

| Hyperparameter | Value | 비고 |
|----------------|-------|------|
| Model | Qwen2.5-Math-7B-Instruct | 수학 특화 |
| Parameters | ~7.61B | Non-embedding: ~6.53B |
| **Hidden Size** | **3584** | TRM도 동일! |
| Layers | 28 | Transformer depth |
| Attention | GQA (28 Q, 4 KV) | 7:1 비율 |
| **head_dim** | **128** | 3584/28 |
| **RoPE theta** | **1,000,000** | 위치 임베딩 주파수 |
| Vocab Size | 151,936 | Byte-level BPE |

### 2. TRM Interface (Simplified)

```
┌─────────────────────────────────────────────────────────────┐
│  Interface (Identity)                                        │
├─────────────────────────────────────────────────────────────┤
│  extract_context(hidden_states) → hidden_states             │
│  (No projection, same dimension)                             │
│                                                              │
│  y_init: learnable [1, 1, 3584], initialized to zeros       │
└─────────────────────────────────────────────────────────────┘
```

- **역할**: State 초기화만 담당
- **파라미터**: ~3.5K (y_init only)

### 3. State Variables

| State | Shape | 역할 | 초기화 |
|-------|-------|------|--------|
| **x** | [B, S, 3584] | Context (Anchor) | hidden_states 그대로 |
| **y** | [B, S, 3584] | Solution (Answer) | learnable y_init (zeros) |
| **z** | [B, S, 3584] | Reasoning (Thought) | zeros |

### 4. TRM Block (Trainable) - Qwen과 동일 구조

```
┌─────────────────────────────────────────────────────────────┐
│  TRMBlock (Qwen-Compatible Architecture)                     │
├─────────────────────────────────────────────────────────────┤
│  Input h [B, S, 3584]                                        │
│       ↓                                                      │
│  RMSNorm(3584)                                               │
│       ↓                                                      │
│  TRMAttention (28 heads, head_dim=128)                       │
│    - Q, K, V projections                                     │
│    - RoPE: apply_rotary_pos_emb(q, k, cos, sin)             │
│    - Scaled Dot-Product Attention (Flash Attention)          │
│    - Output projection (Zero Init)                           │
│       ↓                                                      │
│  RMSNorm(3584)                                               │
│       ↓                                                      │
│  SwiGLU FFN (3584 → 14336 → 3584)                           │
│    - gate_proj, up_proj, down_proj (Zero Init)               │
│       ↓                                                      │
│  Output [B, S, 3584]  ← NO RESIDUAL CONNECTION!             │
└─────────────────────────────────────────────────────────────┘
```

**핵심 설계:**
- **Direct Replacement**: `out = block(h)` (not `out = h + block(h)`)
- **Zero Init**: o_proj, down_proj를 0으로 초기화 → 초기 출력 = 0
- **Qwen 호환**: head_dim=128, num_heads=28, theta=1e6
- **파라미터**: ~257M (3.5x larger than before)

### 5. TRM Heads (Trainable)

```
┌─────────────────────────────────────────────────────────────┐
│  TRMHeads                                                    │
├─────────────────────────────────────────────────────────────┤
│  Input y [B, S, 3584]                                        │
│       ↓                                                      │
│  RMSNorm(3584)        ← 출력 분포 안정화                      │
│       ↓                                                      │
│  lm_head Linear(3584, 151936)  ← Qwen 가중치 직접 복사       │
│       ↓                                                      │
│  Output logits [B, S, 151936]                                │
└─────────────────────────────────────────────────────────────┘
```

**lm_head 초기화 (SVD 불필요!):**
```python
# Same dimension → direct copy
trm_lm_head.weight.copy_(qwen_lm_head.weight)
```
- **파라미터**: ~545M (vocab × 3584)

---

## Training Configuration

### 3-Level Loop Parameters

| Level | Parameter | Value | 역할 |
|-------|-----------|-------|------|
| Level 1 | N_supervision | 16 | Deep Supervision steps |
| Level 2 | T_recursion | 3 | Deep Recursion (T-1 no_grad + 1 grad) |
| Level 3 | n_latent | 6 | Latent Recursion (z updates) |

### Effective Depth
```
Depth = n_layers × (n + 1) × T × N_sup
      = 2 × 7 × 3 × 16
      = 672 effective layers
```

### Optimizer Settings

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| β1, β2 | 0.9, 0.95 |
| Weight Decay | 0.01 |
| Gradient Clipping | 1.0 |
| LR Scheduler | CosineAnnealing |
| EMA Decay | 0.999 |

### Scheduler 계산 (중요!)

```python
# N_sup을 고려한 total_steps 계산
total_optimizer_steps = num_batches × num_epochs × N_supervision
                      = 1868 × 3 × 16
                      = 89,664 steps

scheduler = CosineAnnealingLR(optimizer, T_max=89664)
```

### Gradient Accumulation (Step-wise State Offloading)

TRM의 Deep Supervision 구조에서는 일반적인 gradient accumulation이 불가능합니다.
대신 **Step-wise State Offloading** 방식을 사용합니다:

```python
# 알고리즘 핵심
for sup_step in range(N_sup):  # 16
    optimizer.zero_grad()

    for micro_batch in micro_batches:  # accumulation_steps
        # CPU에서 y, z 상태 로드 → GPU
        y, z = load_states_from_cpu(micro_batch_idx)

        loss = forward(hidden_states, y, z) / acc_steps
        loss.backward()  # gradient 누적

        # GPU에서 y, z 상태 저장 → CPU
        save_states_to_cpu(y, z, micro_batch_idx)

    optimizer.step()  # step 끝에서 1번만 update
```

**핵심 원리:**
- 같은 supervision step 내 모든 micro-batch가 **동일한 가중치**로 forward
- Gradient를 누적한 후 step 끝에서 **1번만 update**
- 다음 step에서는 **업데이트된 가중치** 사용
- y, z 상태는 CPU에 offload하여 GPU 메모리 절약

**사용법:**
```bash
# Effective batch = 4 * 4 = 16
python train_trm.py --batch_size 4 --gradient_accumulation 4
```

---

## Parameter Count

| Component | Parameters | Trainable |
|-----------|------------|-----------|
| Qwen Backbone | ~7.61B | ❌ Frozen |
| Interface | ~3.5K | ✅ (y_init only) |
| TRM Block | ~257M | ✅ |
| TRM Heads | ~545M | ✅ |
| **Total Trainable** | **~802M** | |

**Note**: 이전 구조(~177M) 대비 ~4.5x 증가. 메모리 요구량 증가하지만 정보 손실 없음.

---

## File Structure

```
MathLLM/
├── CLAUDE.md                    # 프로젝트 문서 (이 파일)
├── TRM.pdf                      # TRM 논문 원문
├── train_trm.py                 # TRM 학습 스크립트
├── train_finetune.py            # Finetuning baseline
├── train_lora.py                # LoRA baseline
├── src/
│   ├── config.py                # TRMConfig (d_lat=3584, num_heads=28)
│   ├── interface.py             # TRMInterface (Identity, y_init only)
│   ├── layers.py                # RotaryEmbedding, TRMBlock, TRMAttention
│   ├── engine.py                # TinyRecursiveTransformer (Latent Recursion)
│   ├── model.py                 # QwenTRM (Deep Recursion + encode_backbone)
│   ├── heads.py                 # TRMHeads (Direct copy from Qwen)
│   ├── train.py                 # Trainer (Deep Supervision Loop)
│   └── dataset.py               # GSM8K Dataset
├── eval/
│   └── gsm8k_eval.py            # GSM8K 평가
└── checkpoints/                 # 모델 저장
```

---

## Key Implementation Details

### 1. Same Dimension Architecture (NEW!)

```python
# TRMConfig
d_lat: int = 3584         # Same as Qwen!
num_heads: int = 28       # Same as Qwen!
head_dim: int = 128       # 3584/28 = 128
```

**장점:**
- No information bottleneck
- Qwen lm_head 직접 사용 (사전학습 지식 100% 활용)
- RoPE 완벽 호환

### 2. Direct Replacement (No Residual)

```python
# Standard Transformer (residual)
out = h + attention(norm(h))
out = out + ffn(norm(out))

# TRM Block (direct replacement)
out = attention(norm(h))
out = ffn(norm(out))  # h가 더해지지 않음!
```

**이유**: Zero Init과 결합하여 초기 안정성 확보

### 3. Zero Initialization

```python
# TRMBlock.__init__
nn.init.zeros_(self.attn.o_proj.weight)
nn.init.zeros_(self.mlp.down_proj.weight)
```

**효과**: 학습 시작 시 `block(h) ≈ 0` → 값 폭발 방지

### 4. RoPE Full Compatibility

```python
# TRM과 Qwen 완전 동일
head_dim = 3584 / 28 = 128
rope_theta = 1_000_000
```

**효과**: Qwen의 위치 인코딩 패턴을 그대로 활용

### 5. 속도 최적화

- **RoPE 캐싱**: batch당 1회만 계산, supervision steps간 재사용
- **Gradient clipping 최적화**: trainable params 리스트 캐싱
- **non_blocking transfer**: CPU↔GPU 전송 시 `non_blocking=True`로 비동기 처리
- **torch.compile**: `--compile` 옵션으로 TRM engine 컴파일 가능

---

## Critical Notes

### TRM은 이제 Qwen과 동일 차원
> `d_lat = backbone_dim = 3584`로 설정됨.
> Interface는 identity function이 됨.

### lm_head는 직접 복사
> SVD 압축 불필요. Qwen의 lm_head 가중치를 그대로 복사.
> 사전학습 지식 손실 없음.

### 메모리 증가
> TRM Block이 ~257M (이전 ~21M 대비 12x)
> 전체 trainable params ~802M (이전 ~177M 대비 4.5x)

### Scheduler는 N_sup 고려 필수
> `total_steps = batches × epochs × N_supervision`
> N_sup을 빼먹으면 LR이 16배 빠르게 decay됨.

---

## Related Links

- [AIMO3 Competition](https://kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
- [TRM Paper](https://arxiv.org/abs/2510.04871)
- [Qwen-2.5-Math](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct)
