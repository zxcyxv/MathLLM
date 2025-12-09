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

### System Pipeline
```
Input → Qwen(Encoder) → Projection(Bottleneck) → TRM(Reasoner) → Answer
```

---

## Mini-Qwen 아키텍처 (언어 모델 기능 보존)

### 문제점과 해결책

단순히 TRM을 붙이면 Qwen의 **언어 생성 감각**이 단절됨:
- 위치 정보(RoPE) 없이 순서 파악 불가
- Random 초기화된 lm_head로 사전학습 지식 손실

**해결: "Mini-Qwen" 블록으로 TRM 업그레이드**

```
Before (언어 감각 없음):              After (Qwen 호환):
┌─────────────────────────┐          ┌─────────────────────────┐
│ TRM Block               │          │ TRM Block (Mini-Qwen)   │
│  └─ Attention (RoPE ✗)  │    →     │  └─ RoPE Attention      │ ← 위치 정보
│  └─ Random lm_head      │          │  └─ head_dim=128        │ ← Qwen 동일
└─────────────────────────┘          │  └─ Final RMSNorm       │ ← 분포 안정화
                                     │  └─ SVD lm_head         │ ← 사전학습 활용
                                     └─────────────────────────┘
```

### 핵심 설계 원칙

| 항목 | Qwen-2.5-Math-7B | TRM (Mini-Qwen) | 비고 |
|------|------------------|-----------------|------|
| head_dim | 128 (3584/28) | **128** (1024/8) | 동일하게 맞춤 |
| RoPE theta | 1,000,000 | **1,000,000** | 주파수 호환 |
| Final Norm | RMSNorm | **RMSNorm** | 출력 분포 안정화 |
| lm_head | Pretrained | **SVD 압축** | 사전학습 활용 |

---

## TRM 3중 루프 구조 (Paper Figure 3, Page 5)

**핵심**: 논문의 정확한 3-Level 루프 구조

```
┌─────────────────────────────────────────────────────────────┐
│  Level 1: N_sup = 16 (Deep Supervision)                     │
│  └─ 매 step마다: backward → opt.step → zero_grad            │
│     └─ y, z를 detach해서 다음 step으로 전달                  │
│                                                              │
│     ┌─────────────────────────────────────────────────┐     │
│     │  Level 2: T = 3 (Deep Recursion)                │     │
│     │  └─ T-1번: no_grad로 실행 (메모리 절약)          │     │
│     │  └─ 1번: grad로 실행 (학습용)                    │     │
│     │                                                  │     │
│     │     ┌─────────────────────────────────────┐     │     │
│     │     │  Level 3: n = 6 (Latent Recursion)  │     │     │
│     │     │  └─ n번 z 업데이트: z = net(x+y+z)  │     │     │
│     │     │  └─ 1번 y 업데이트: y = net(y+z)    │     │     │
│     │     └─────────────────────────────────────┘     │     │
│     └─────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Effective Depth
```
n_layers × (n+1) × T × N_sup = 2 × 7 × 3 × 16 = 672 layers
```

### 핵심 알고리즘 (Paper Figure 3 + RoPE)

```python
# Level 3: Latent Recursion (engine.py)
def latent_recursion(x, y, z, cos, sin, n=6):
    for i in range(n):           # n번 z 업데이트
        h = x + y + z            # Additive fusion
        z = block(h, cos, sin)   # RoPE 적용, Direct replacement
    h = y + z                    # x 완전 제외!
    y = block(h, cos, sin)       # RoPE 적용
    return y, z

# Level 2: Deep Recursion (model.py)
def deep_recursion(x, y, z, cos, sin, T=3):
    with torch.no_grad():        # T-1번: 메모리 절약
        for j in range(T - 1):
            y, z = latent_recursion(x, y, z, cos, sin)
    y, z = latent_recursion(x, y, z, cos, sin)  # 마지막: grad
    return y.detach(), z.detach()

# Level 1: Deep Supervision (train.py)
for step in range(N_supervision):  # N_sup = 16
    (y, z), logits = deep_recursion(x, y, z, cos, sin)
    loss = cross_entropy(logits, target)
    loss.backward()
    opt.step(); opt.zero_grad(); ema.update()
```

---

## Module Specifications

### 1. Semantic Encoder (Qwen-2.5-Math-7B)

| Hyperparameter | Value | 비고 |
|----------------|-------|------|
| Parameters | ~7.61B | Non-embedding: ~6.53B |
| **Hidden Size** | **3584** | 일반 7B(4096)와 다름 - 주의! |
| Intermediate | 18,944 | SwiGLU FFN |
| Layers | 28 | Transformer depth |
| Q Heads | 28 | Query heads |
| KV Heads | **4** | GQA 7:1 비율 |
| **head_dim** | **128** | 3584/28 |
| Vocab Size | 151,936 | Byte-level BPE |
| **RoPE theta** | **1,000,000** | 위치 임베딩 주파수 |

### 2. TRM Interface (Projection Layer)

```
Qwen Output [B, S, 3584] → Linear + RMSNorm → TRM Input [B, S, 1024]
```

- **Information Bottleneck**: 3584 → 1024 차원 압축

### 3. State Variables

| State | Shape | 역할 |
|-------|-------|------|
| **x** (Context) | [B, S, 1024] | 문제의 불변 의미 표현 (Anchor, Frozen) |
| **y** (Solution) | [B, S, 1024] | 현재 잠정 정답 임베딩 |
| **z** (Reasoning) | [B, S, 1024] | 추론 궤적 (Hidden Reasoning Path) |

### 4. TRM Block (Mini-Qwen 구조)

**Qwen 호환 설계:**

| 항목 | 설명 |
|------|------|
| **RoPE Attention** | `apply_rotary_pos_emb(q, k, cos, sin)` - 위치 정보 유지 |
| **head_dim=128** | Qwen과 동일 → RoPE 주파수 호환 |
| **SwiGLU FFN** | Qwen과 동일한 활성화 함수 |
| **RMSNorm** | Pre-LN 구조 |

```python
class TRMBlock:
    def forward(self, h, cos, sin):
        # TRM uses DIRECT REPLACEMENT (no residual!)
        # This is different from standard Transformers
        h_attn = attn(norm1(h), cos, sin)  # RoPE 적용
        out = mlp(norm2(h_attn))           # SwiGLU
        return out  # NO residual connection
```

**Zero Init**: Output projections (o_proj, down_proj)를 0으로 초기화하여
초기 출력이 0이 되도록 함. 이로써 학습 시작 시 안정성 확보.

### 5. TRM Heads (Qwen 가중치 활용)

**핵심: SVD 압축으로 사전학습 지식 보존**

```python
class TRMHeads:
    def __init__(self, config, qwen_lm_head):
        self.norm = RMSNorm(1024)           # Final Norm (출력 안정화)
        self.lm_head = Linear(1024, vocab)  # SVD 초기화

    def _init_from_qwen(self, qwen_lm_head):
        # Qwen [vocab, 3584] → SVD → TRM [vocab, 1024]
        U, S, V = svd_lowrank(qwen_lm_head.weight, q=1024)
        self.lm_head.weight = U @ diag(S)
```

### 6. Training Strategy

- **Deep Supervision**: N_sup=16번, 매 step backward
- **Truncated BPTT**: y, z detach로 그래프 절단
- **EMA**: decay=0.999 (학습 안정성)
- **AdamW**: β1=0.9, β2=0.95

---

## Implementation Status

### Phase 1: Foundation ✅
- [x] Qwen-2.5-Math-7B 로딩 및 Freeze
- [x] TRMInterface 구현 (3584 → 1024 Projection)
- [x] State 초기화 로직 (x, y, z)

### Phase 2: TRM Engine ✅
- [x] RotaryEmbedding (theta=1e6, head_dim=128)
- [x] TRMAttention with RoPE (Qwen 호환)
- [x] TRMBlock 구현 (Additive Fusion + RoPE)
- [x] TinyRecursiveTransformer - Latent Recursion

### Phase 3: Training Loop ✅
- [x] TRMHeads (Final Norm + SVD lm_head 초기화)
- [x] QwenTRM - Deep Recursion (T-1 no_grad + 1 grad)
- [x] Trainer - Deep Supervision (N_sup 루프)
- [x] EMA 구현 및 적용
- [x] Zero Init for stable training start
- [x] Direct replacement (no residual) per TRM paper

### Phase 4: Evaluation Infrastructure ✅
- [x] GSM8K evaluation script (`eval/gsm8k_eval.py`)
- [x] Qwen Zero-shot baseline: **93.71%** on GSM8K
- [x] TRM Identity Test passed (Zero Init 검증)

### Phase 5: Training Scripts ✅
- [x] `train_trm.py` - TRM training with Deep Supervision
- [x] `train_finetune.py` - Last N layers finetuning (baseline)
- [x] `train_lora.py` - LoRA training (alternative baseline)

### Phase 6: Experiments (IN PROGRESS)
- [x] TRM training on GSM8K started (Loss 7.08 → 2.57)
- [ ] Complete TRM training and evaluate
- [ ] Train finetune baseline for comparison
- [ ] Dynamic T inference test

### Phase 7: Kaggle Submission (TODO)
- [ ] Dynamic Depth Inference optimization
- [ ] Kaggle Submission Format 호환
- [ ] 5시간 GPU 제한 내 최적화

---

## File Structure

```
MathLLM/
├── CLAUDE.md                    # 프로젝트 컨텍스트 (이 파일)
├── TRM.pdf                      # TRM 논문 원문
├── main.py                      # 메인 실행 스크립트
├── train_trm.py                 # TRM 학습 스크립트
├── train_finetune.py            # Last N layers finetuning
├── train_lora.py                # LoRA 학습 스크립트
├── src/
│   ├── config.py      # TRMConfig (RoPE 설정 포함)
│   ├── interface.py   # TRMInterface (Backbone → TRM projection)
│   ├── layers.py      # RotaryEmbedding, TRMBlock (RoPE Attention, Zero Init)
│   ├── engine.py      # TinyRecursiveTransformer (cos/sin 전달)
│   ├── model.py       # QwenTRM (RoPE 초기화, lm_head 연동)
│   ├── heads.py       # TRMHeads (Final Norm + SVD 초기화)
│   ├── train.py       # Trainer + EMA
│   └── dataset.py     # GSM8K Dataset
├── eval/
│   ├── gsm8k_eval.py  # GSM8K 평가 스크립트
│   └── utils.py       # 평가 유틸리티
└── checkpoints/                 # 학습된 모델 저장
```

---

## Key Hyperparameters (config.py)

```python
@dataclass
class TRMConfig:
    backbone_dim: int = 3584      # Qwen hidden size
    d_lat: int = 1024             # TRM latent dimension
    num_heads: int = 8            # 1024/8 = 128 head_dim (Qwen 동일)
    expansion: int = 4

    # RoPE settings (Qwen 호환)
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768

    # 3-Level Loop Parameters
    n_latent: int = 6             # Level 3: Latent Recursion
    T_recursion: int = 3          # Level 2: Deep Recursion
    N_supervision: int = 16       # Level 1: Deep Supervision

    vocab_size: int = 151936
    use_ema: bool = True
    ema_decay: float = 0.999
```

---

## Critical Notes

### Dimension Mismatch Warning
> Qwen-2.5-Math-7B의 Hidden Size는 **3584**이다 (일반 7B 모델의 4096이 아님).

### head_dim 일치 중요
> TRM의 `num_heads=8` (1024/8=128)은 Qwen의 `head_dim=128`과 동일.
> 이로 인해 Qwen의 RoPE 주파수 설정을 그대로 활용 가능.

### lm_head SVD 초기화
> Qwen의 사전학습된 lm_head [vocab, 3584]를 SVD로 압축하여 [vocab, 1024]로 초기화.
> Random init 대비 학습 초기 수렴 속도 크게 향상.

### 논문 vs Gemini Spec 주의
> `TRM_PAPER_VS_GEMINI_SPEC.md` 참조. Gemini가 생성한 Part1/2/3.txt는 논문과 다름.
> 현재 구현은 **논문 Figure 3 기준** + **Qwen 호환 업그레이드**.

### Memory Considerations
- T-1번은 no_grad로 메모리 절약
- 매 supervision step마다 y, z detach
- Effective depth 672 layers이지만 메모리는 최소화

### Zero Init + Direct Replacement (핵심!)
> TRM의 TRMBlock은 **residual connection이 없다** (standard Transformer와 다름).
> `out = mlp(norm2(attn(norm1(h))))` - 입력 h가 출력에 더해지지 않음.
>
> Zero Init과 함께: 학습 초기에 block output = 0이 되어 안정적인 시작점 제공.
> 재귀 루프에서 값이 폭발하지 않음 (y,z = 0으로 시작 → 점진적 학습).

### State Initialization
- **y**: learnable parameter `y_init` (초기값 0)
- **z**: `torch.zeros(...)` (x가 아닌 0으로 초기화 - 값 폭발 방지)

---

## Related Links

- [AIMO3 Competition](https://kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
- [TRM Paper](https://arxiv.org/abs/2510.04871)
- [Qwen-2.5-Math](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct)
