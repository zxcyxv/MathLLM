# TRM 논문 vs Gemini 명세서 차이점 분석

## 개요

이 문서는 **TRM 논문** (arXiv:2510.04871, "Less is More: Recursive Reasoning with Tiny Networks")과
**Gemini가 작성한 명세서** (Part1.txt, Part2.txt, Part3.txt)의 모든 차이점을 정리합니다.

---

## 1. 입력 융합 방식 (Input Fusion)

### 📄 TRM 논문
```python
# 덧셈 (Addition)
z = net(x + y + z)   # z 업데이트
y = net(y + z)       # y 업데이트
```

**근거**:
- Page 2: `zL ← fL(zL + zH + x)` (HRM 설명)
- Page 6: "since z ← fL(x + y + z) contains x but y ← fH(y + z) does not contains x"
- Figure 1: ⊕ (덧셈) 기호로 x, y, z 연결

### 📋 Gemini 명세서
```python
# Concatenation + Linear Projection
combined = torch.cat([x, y, z], dim=-1)  # [B, S, 3*D]
output = self.net(combined)               # [B, S, D]
```

**위치**: Part2.txt Line 50-52, 104, 205-210

### ⚠️ 차이점
| 항목 | 논문 | Gemini |
|------|------|--------|
| 방식 | `x + y + z` (element-wise addition) | `concat([x,y,z])` → Linear |
| 입력 차원 | D → D | 3D → D |
| 파라미터 | 없음 (같은 차원) | Linear(3D, D) 필요 |

---

## 2. y 업데이트 시 x 처리 방식

### 📄 TRM 논문
```python
y = net(y, z)  # x를 아예 입력하지 않음
```

**근거**:
- Figure 3 (Page 5): `y = net(y, z)  # refine output answer`
- Page 6: "y ← fH(y + z) does **not contains x**"
- Page 6: "the task to achieve... is directly specified by the **inclusion or lack of x** in the inputs"

### 📋 Gemini 명세서
```python
x_dummy = torch.zeros_like(x)
y_new = self._single_step(x_dummy, y, curr_z)  # x를 0으로 마스킹
```

**위치**: Part2.txt Line 93-97, 232-236, 388-390

### ⚠️ 차이점
| 항목 | 논문 | Gemini |
|------|------|--------|
| x 처리 | 입력에서 완전히 제외 | zeros_like(x)로 마스킹 |
| 수식 | `net(y + z)` | `net(0 + y + z)` |
| 의미 | 모델이 x 없이 추론하도록 강제 | 0 벡터가 여전히 concat됨 |

---

## 3. Gradient 처리 (Deep Supervision)

### 📄 TRM 논문
```python
def deep_recursion(x, y, z, n=6, T=3):
    # T-1회는 gradient 없이 실행
    with torch.no_grad():
        for j in range(T-1):
            y, z = latent_recursion(x, y, z, n)

    # 마지막 1회만 gradient로 실행
    y, z = latent_recursion(x, y, z, n)

    return (y.detach(), z.detach()), output_head(y)
```

**근거**: Figure 3 (Page 5)

### 📋 Gemini 명세서
```python
for step in range(T):
    y, z = self.engine.forward_recursion_process(x, y, z)

    # 매 step마다 gradient 계산
    loss = self.ce_loss(logits, labels)
    total_loss += loss

    # step 끝날 때 detach
    y = y.detach()
    z = z.detach()
```

**위치**: Part3.txt Line 70-103

### ⚠️ 차이점
| 항목 | 논문 | Gemini |
|------|------|--------|
| Gradient 범위 | 마지막 1회 full recursion만 | 매 step의 n+1 recursion |
| no_grad 사용 | T-1회 전체를 no_grad로 감쌈 | 사용 안 함 |
| Loss 계산 | 마지막 step만 | 모든 step 누적 |
| detach 시점 | return 시 한 번 | 매 step 끝 |

---

## 4. Deep Supervision Loop 구조

### 📄 TRM 논문
```python
# Figure 3 Deep Supervision Loop
for x_input, y_true in train_dataloader:
    y, z = y_init, z_init
    for step in range(N_supervision):
        x = input_embedding(x_input)
        (y, z), y_hat, q_hat = deep_recursion(x, y, z)

        loss = softmax_cross_entropy(y_hat, y_true)
        loss += binary_cross_entropy(q_hat, (y_hat == y_true))

        # 매 step마다 backward & update
        loss.backward()
        opt.step()
        opt.zero_grad()

        if q_hat > 0:  # early-stopping
            break
```

### 📋 Gemini 명세서
```python
for step in range(steps):
    y, z = self.engine.forward_recursion_process(x, y, z)
    logits, halt_logit = self.heads(y)

    # Loss 누적
    total_loss += step_ce_loss + 0.1 * step_halt_loss

    # detach만 하고 backward는 나중에
    y = y.detach()
    z = z.detach()

# 루프 끝나고 한 번에 backward (또는 매 step backward)
```

### ⚠️ 차이점
| 항목 | 논문 | Gemini |
|------|------|--------|
| backward 시점 | 매 supervision step마다 | 모호함 (누적 후 또는 매번) |
| optimizer step | 매 supervision step마다 | 루프 끝나고 한 번 |
| Early stopping | ACT 기반 early stopping | 없음 (고정 T steps) |

---

## 5. 네트워크 구조 (Architecture)

### 📄 TRM 논문
```python
# 단순 2-layer 구조 (Less is More)
self.net = nn.Sequential(
    SwiGLU(D, 4*D, D),
    RMSNorm(D)
)
# 또는 Self-Attention + MLP (ARC-AGI용)
```

**근거**:
- Page 7: "using 2 layers (instead of 4 layers) maximized generalization"
- Table 1: "TRM (T=3, n=6)" uses 2 layers, 5M params
- Page 7: "Less is More" - 작은 네트워크가 오버피팅 방지

### 📋 Gemini 명세서
```python
# Split Projection Fusion + Transformer Block
class EfficientTRMBlock(nn.Module):
    def __init__(self, d_lat, num_heads):
        self.x_proj = nn.Linear(d_lat, d_lat)      # x용 별도 projection
        self.yz_proj = nn.Linear(2*d_lat, d_lat)   # [y;z]용 별도 projection
        self.attn = nn.MultiheadAttention(...)
        self.ffn = nn.Sequential(...)
```

**위치**: Part2.txt Line 317-356

### ⚠️ 차이점
| 항목 | 논문 | Gemini |
|------|------|--------|
| 구조 | 단순 2-layer (SwiGLU + Norm) | Split Projection + Transformer |
| 파라미터 | 5-7M | 더 많음 (복잡한 구조) |
| 철학 | "Less is More" - 최소화 | KV-Cache 최적화 - 효율화 |
| x 처리 | 덧셈으로 통합 | 별도 projection 후 합산 |

---

## 6. precompute_x 최적화

### 📄 TRM 논문
**없음** - 덧셈 방식이라 x를 미리 계산할 필요 없음

### 📋 Gemini 명세서
```python
def precompute_x(self, x):
    """루프 전에 한 번만 계산"""
    return self.block.x_proj(x)

def forward(self, x_static, y, z):
    # x_static은 미리 계산된 값
    fused_input = x_static + self.yz_proj(torch.cat([y, z], dim=-1))
```

**위치**: Part2.txt Line 367-372

### ⚠️ 차이점
| 항목 | 논문 | Gemini |
|------|------|--------|
| x 캐싱 | 불필요 (덧셈) | x_proj 결과 캐싱 |
| 목적 | - | 연산 효율화 |
| 복잡도 | 단순 | 추가 로직 필요 |

---

## 7. Residual Connection

### 📄 TRM 논문
```python
# Figure 3 pseudocode - 직접 교체
for i in range(n):
    z = net(x, y, z)  # z를 직접 교체
y = net(y, z)         # y를 직접 교체
```

**참고**: Figure 1의 다이어그램에는 "Add & Norm"이 표시되어 있어 residual이 있을 수도 있음

### 📋 Gemini 명세서
```python
for _ in range(self.n):
    delta_z = self._single_step(x, y, curr_z)
    curr_z = curr_z + delta_z  # 명시적 residual

delta_y = self._single_step(x_dummy, y, curr_z)
y_new = y + delta_y  # 명시적 residual
```

**위치**: Part2.txt Line 76-97, 221-236

### ⚠️ 차이점
| 항목 | 논문 | Gemini |
|------|------|--------|
| z 업데이트 | `z = net(...)` (교체) | `z = z + net(...)` (residual) |
| y 업데이트 | `y = net(...)` (교체) | `y = y + net(...)` (residual) |
| 근거 | Pseudocode 기준 | "gradient stability" 언급 |

---

## 8. ACT (Adaptive Computation Time)

### 📄 TRM 논문
```python
# 단순화된 ACT - Binary CE만 사용
loss += binary_cross_entropy(q_hat, (y_hat == y_true))

if q_hat > 0:  # early-stopping
    break
```

**근거**:
- Page 7: "get rid of the continue loss (from the Q-learning)"
- Page 7: "only learn a halting probability through a Binary-Cross-Entropy loss"
- Table 1: "w/ ACT" (86.1%) vs without (87.4%) - ACT 없이도 좋음

### 📋 Gemini 명세서
```python
# Q-learning 기반 ACT (HRM 방식)
class TRMHeads(nn.Module):
    self.halt_head = nn.Sequential(
        nn.Linear(d_lat, d_lat // 2),
        nn.SiLU(),
        nn.Linear(d_lat // 2, 1)
    )

# Halting loss
halt_target = (accuracy > 0.99).float()
step_halt_loss = self.bce_loss(halt_logit.mean(), halt_target)
```

**위치**: Part3.txt Line 1-18, 87-98

### ⚠️ 차이점
| 항목 | 논문 | Gemini |
|------|------|--------|
| 방식 | 단순 BCE | Q-learning 기반 |
| forward pass | 1회 | 2회 (continue loss용) |
| 구현 | 단순 | 복잡 |
| 필요성 | 선택적 (없어도 됨) | 필수처럼 서술 |

---

## 9. EMA (Exponential Moving Average)

### 📄 TRM 논문
```
EMA = 0.999
```

**근거**:
- Page 7: "integrate Exponential Moving Average (EMA) of the weights"
- Page 7: "going from 79.9% to 87.4%; see Table 1"
- Page 11: "TRM uses an Exponential Moving Average (EMA) of 0.999"

### 📋 Gemini 명세서
**언급 없음**

### ⚠️ 차이점
| 항목 | 논문 | Gemini |
|------|------|--------|
| EMA | 0.999 사용 | 언급 없음 |
| 효과 | +7.5% accuracy | - |
| 안정성 | 오버피팅/발산 방지 | 고려 안 됨 |

---

## 10. 상태 변수 해석 (State Variables)

### 📄 TRM 논문
- **x**: Input question (embedded)
- **y**: Current proposed solution (= zH in HRM)
- **z**: Latent reasoning feature (= zL in HRM)

**근거** (Page 6):
> "zH is simply the current (embedded) solution... zL is a latent feature that does not directly correspond to a solution"
> "hierarchy is not needed; there is simply an input x, a proposed solution y, and a latent reasoning feature z"

### 📋 Gemini 명세서
- **x**: Context State (Invariant Semantic Representation)
- **y**: Solution State (잠정적 정답 임베딩)
- **z**: Reasoning State (논리적 사고의 궤적)

**위치**: Part1.txt "B. State Variables Definition"

### ⚠️ 차이점
| 항목 | 논문 | Gemini |
|------|------|--------|
| x 역할 | 입력 질문 | 불변의 문맥 (Anchor) |
| y 역할 | 현재 해답 | 잠정적 정답 |
| z 역할 | 추론 과정 (CoT 유사) | Hidden Reasoning Path |
| 해석 | 단순 (계층 없음) | 복잡 (Information Bottleneck) |

---

## 11. 하이퍼파라미터

### 📄 TRM 논문 (Page 11)
| Parameter | Value |
|-----------|-------|
| n (inner recursion) | 6 |
| T (supervision steps) | 3 |
| Hidden size | 512 |
| Batch size | 768 |
| Optimizer | AdamW (β1=0.9, β2=0.95) |
| Learning rate | 1e-4 |
| Weight decay | 1.0 (Sudoku), 0.1 (ARC) |
| EMA | 0.999 |
| Network layers | 2 |
| Nsup (max) | 16 |

### 📋 Gemini 명세서
| Parameter | Value |
|-----------|-------|
| n_recursion | 6 |
| t_supervision | 3 |
| d_lat | 1024 |
| num_heads | 16 |
| expansion | 4 |
| Network | Transformer Block |

### ⚠️ 차이점
| 항목 | 논문 | Gemini |
|------|------|--------|
| Hidden size | 512 | 1024 |
| Network | 2-layer MLP | Transformer |
| EMA | 0.999 | 없음 |
| Attention | 선택적 | 필수 |

---

## 12. Loss 함수

### 📄 TRM 논문
```python
loss = softmax_cross_entropy(y_hat, y_true)
loss += binary_cross_entropy(q_hat, (y_hat == y_true))
```

### 📋 Gemini 명세서
```python
# CE Loss
step_ce_loss = self.ce_loss(shift_logits, shift_labels)

# Halting Loss (더 복잡)
accuracy = (preds == shift_labels).float().mean()
halt_target = (accuracy > 0.99).float()
step_halt_loss = self.bce_loss(halt_logit.mean(), halt_target)

total_loss += step_ce_loss + 0.1 * step_halt_loss
```

### ⚠️ 차이점
| 항목 | 논문 | Gemini |
|------|------|--------|
| CE Loss | 단순 CE | Shifted CE (Causal LM) |
| Halt target | `y_hat == y_true` (정확 일치) | `accuracy > 0.99` (임계값) |
| Halt weight | 1.0 (동일) | 0.1 |

---

## 13. Self-Attention vs MLP

### 📄 TRM 논문
```python
# Sudoku-Extreme (9x9): MLP가 더 좋음
# Maze-Hard, ARC-AGI (30x30): Self-Attention이 더 좋음
```

**근거** (Page 7):
> "Using an MLP instead of self-attention, we obtain better generalization on Sudoku-Extreme (improving from 74.7% to 87.4%)"
> "we found this architecture to be suboptimal for tasks with large context length"

### 📋 Gemini 명세서
- Self-Attention을 기본으로 사용
- MLP 대안 언급 없음

### ⚠️ 차이점
| 항목 | 논문 | Gemini |
|------|------|--------|
| 접근 | Task별 선택 | Self-Attention 고정 |
| 작은 context | MLP 추천 | - |
| 큰 context | Self-Attention | Self-Attention |

---

## 14. 초기화 전략

### 📄 TRM 논문
```python
y, z = y_init, z_init  # 학습 가능한 초기화
```

**근거**: Figure 3 - `y, z = y_init, z_init`

### 📋 Gemini 명세서
```python
y_0 = self.y_init.expand(batch_size, seq_len, -1)  # 학습 가능

if self.z_init_strategy == "copy_x":
    z_0 = x_latent.clone()  # x 복사
else:
    z_0 = torch.zeros_like(x_latent)
```

**위치**: Part1.txt Line 41-56

### ⚠️ 차이점
| 항목 | 논문 | Gemini |
|------|------|--------|
| y 초기화 | 학습 가능 벡터 | 학습 가능 (0 기반) |
| z 초기화 | 학습 가능 벡터 | x 복사 또는 0 |

---

## 요약: 핵심 차이점 TOP 5

### 1️⃣ 입력 융합
- **논문**: `x + y + z` (덧셈)
- **Gemini**: `concat([x,y,z])` → Linear

### 2️⃣ y 업데이트
- **논문**: `net(y, z)` - x 없음
- **Gemini**: `net(0, y, z)` - x를 0으로 마스킹

### 3️⃣ Gradient 처리
- **논문**: T-1회 no_grad + 1회 grad
- **Gemini**: 매 step gradient 후 detach

### 4️⃣ 네트워크 구조
- **논문**: 단순 2-layer ("Less is More")
- **Gemini**: Split Projection + Transformer

### 5️⃣ EMA
- **논문**: 0.999 (필수적, +7.5% 성능)
- **Gemini**: 언급 없음

---

## 권장 사항

**논문대로 재구현**하는 것이 권장됩니다:

1. 입력 융합을 덧셈 방식으로 변경
2. y 업데이트 시 x를 완전히 제외
3. T-1회 no_grad + 1회 grad 구조 적용
4. 단순 2-layer 네트워크 사용 고려
5. EMA 0.999 추가
6. 매 supervision step마다 backward/step 수행

---

## 참고 자료

- **TRM 논문**: arXiv:2510.04871
- **Gemini 명세서**: Part1.txt, Part2.txt, Part3.txt
- **TRM GitHub**: (clone된 코드 참조)
