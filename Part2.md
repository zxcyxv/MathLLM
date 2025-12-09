# Part 2: The Recursive Reasoning Engine

## 1. Conceptual Design: Dual-Mode Operation

TRM의 핵심은 **단일 네트워크** `f_θ`가 입력 구성에 따라 두 가지 역할을 수행한다는 점이다.

### Reasoning Mode (Update z)
- **입력**: `[x, y, z]` (Context + Solution + Reasoning)
- **수식**: `z_{k+1} = f_θ(x, y_k, z_k)`
- **의미**: 문제 문맥(x)을 참조하며 추론 궤적(z)을 수정

### Prediction Mode (Update y)
- **입력**: `[0, y, z]` (Context Masked)
- **수식**: `y_{k+1} = f_θ(0, y_k, z_new)`
- **의미**: 추론 결과(z)에만 의존하여 해답(y) 생성
- **Information Bottleneck**: x를 마스킹하여 z가 충분한 정보를 압축하도록 강제

---

## 2. Architecture: Transformer Block

MLP 대신 **Transformer Block**을 채택한다. 언어 모델(Qwen)과의 호환성 및 토큰 간 상호작용(Global Dependency) 처리에 유리하다.

### Block Structure (Pre-Norm)
```
Input → RMSNorm → Attention → Residual → RMSNorm → SwiGLU FFN → Residual → Output
```

### Dimensions
| Parameter | Value |
|-----------|-------|
| Model Dim (d_lat) | 1024 |
| Num Heads | 16 |
| Head Dim | 64 |
| FFN Hidden (d_ff) | 4096 (4 × d_lat) |

---

## 3. Split Projection Fusion (KV-Cache Optimization)

### Problem
- x는 재귀 루프 동안 **불변(Static)**
- y, z는 매 루프마다 **가변(Dynamic)**
- 매번 `[x, y, z]`를 융합하면 x에 대한 연산이 중복됨

### Solution
x와 [y, z]의 프로젝션을 분리하여 x 연산을 캐싱한다.

```python
# 루프 시작 전 1회만 계산
x_static = x_proj(x)

# 루프 내에서 매번 계산
fused = x_static + yz_proj(concat([y, z]))
```

### Benefits
- x의 Linear projection 연산을 n회 생략
- State Drift 문제 없이 수학적 정합성 유지

---

## 4. Implementation Specification

### 4.1 SwiGLU Activation
```python
class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)  # Gate
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)  # Value
        self.w3 = nn.Linear(hidden_features, out_features, bias=False) # Output

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
```

### 4.2 TRMAttention
```python
class TRMAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # QKV projection → Split → Scaled Dot-Product Attention → Output projection
        # Uses F.scaled_dot_product_attention with is_causal=True
```

### 4.3 TRMTransformerBlock
```python
class TRMTransformerBlock(nn.Module):
    def __init__(self, d_lat, num_heads):
        # Split Projection for efficiency
        self.x_proj = nn.Linear(d_lat, d_lat, bias=False)
        self.yz_proj = nn.Linear(2 * d_lat, d_lat, bias=False)

        self.norm1 = nn.RMSNorm(d_lat)
        self.attn = TRMAttention(d_lat, num_heads)

        self.norm2 = nn.RMSNorm(d_lat)
        # SwiGLU FFN
        self.w1 = nn.Linear(d_lat, 4 * d_lat, bias=False)
        self.w2 = nn.Linear(d_lat, 4 * d_lat, bias=False)
        self.w3 = nn.Linear(4 * d_lat, d_lat, bias=False)

    def forward(self, x_static, y, z):
        # 1. Efficient Fusion
        h = x_static + self.yz_proj(concat([y, z]))

        # 2. Attention with residual
        h = h + self.attn(self.norm1(h))

        # 3. SwiGLU FFN with residual
        h_norm = self.norm2(h)
        h = h + self.w3(F.silu(self.w1(h_norm)) * self.w2(h_norm))

        return h
```

### 4.4 TinyRecursiveTransformer
```python
class TinyRecursiveTransformer(nn.Module):
    def __init__(self, d_lat=1024, num_heads=16, n_recursion=6):
        self.n = n_recursion
        self.block = TRMTransformerBlock(d_lat, num_heads)

    def forward(self, x, y, z):
        # 1. Precompute static x projection (ONCE)
        x_static = self.block.x_proj(x)

        # 2. Inner Loop: Update z (n times)
        for _ in range(self.n):
            delta_z = self.block(x_static, y, z)
            z = z + delta_z  # Residual update

        # 3. Outer Step: Update y (1 time)
        x_masked = torch.zeros_like(x_static)
        delta_y = self.block(x_masked, y, z)
        y = y + delta_y  # Residual update

        return y, z
```

---

## 5. Computational Depth Analysis

### Effective Depth
- 1회 forward: `n + 1` transformer block 연산
- Deep Supervision loop (T회): `T × (n + 1)` layers
- **Example**: T=3, n=6 → 21 layers of effective depth

### Why Transformer over MLP
1. **Inductive Bias**: 시퀀스 처리 능력 계승, 이전 토큰의 추론 정보 참조 가능
2. **Compatibility**: Qwen과 동일한 구조로 Feature Space 성질 유지
3. **Capacity**: Self-Attention의 높은 표현력을 재귀로 증폭

---

## 6. Integration Pipeline

```
Qwen Backbone (frozen)
    ↓
TRMInterface.extract_context()  →  x [B, S, 1024]
TRMInterface.initialize_states() →  y, z [B, S, 1024]
    ↓
TinyRecursiveTransformer.forward(x, y, z)
    ├─ precompute x_static (1회)
    ├─ inner loop: z update (n회)
    └─ outer step: y update (1회)
    ↓
y_new, z_new  →  TRMHeads (Part 3)
```
