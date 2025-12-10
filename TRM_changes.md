## TRM 아키텍처 변경 사항

### 1. Interface: 선형 투영 → MLP 투영

- **기존 구조 (`TRMInterface`)**
  - Qwen hidden: `3584` → TRM latent: `1024`
  - 단일 선형층 + RMSNorm:
    - `Linear(3584 → 1024, bias=False) -> RMSNorm(1024)`
  - 장점: 구현 단순, 계산비용 낮음  
    단점: 고차원 정보를 한 번에 압축하면서 **표현력/비선형성이 부족하여 정보 손실이 클 수 있음**.

- **변경 후 구조**
  - **MLP 기반 병목**으로 교체:
    - `Linear(3584 → 2*d_lat) -> GELU -> Linear(2*d_lat → d_lat) -> RMSNorm(d_lat)`
    - 현재 설정에서: `3584 → 2048 → 1024`
  - 의도:
    - 중간 hidden 차원을 두고 비선형(GELU)을 넣어, **Qwen hidden space의 유용한 방향들을 더 풍부하게 재조합**하도록 함.
    - 단일 선형 투영 대비, **정보 손실을 줄이고 TRM이 사용할 수 있는 latent 표현력을 키우는 것**이 목적.
  - 구현 위치:
    - `src/interface.py`의 `TRMInterface.__init__` 내 `self.projector` 정의부.

### 2. Engine: Direct Replacement → 아주 작은 Residual 업데이트

- **기존 구조 (`TinyRecursiveTransformer`)**
  - Latent recursion에서 TRMBlock 출력으로 상태를 **완전히 덮어씀**:
    - Reasoning 모드 (z 업데이트):
      - `h = x + y + z`
      - `z = block(h, cos, sin)`  # direct replacement, residual 없음
    - Prediction 모드 (y 업데이트):
      - `h = y + z`
      - `y = block(h, cos, sin)`  # direct replacement
  - 특징:
    - 논문 컨셉(직접 치환, additive fusion)을 잘 반영하지만,
    - residual을 도입하려고 시도할 경우, **같은 블록을 여러 번 재귀로 돌리기 때문에 값이 기하급수적으로 커지는 문제가 발생**할 수 있음.

- **변경 후 구조: 아주 작은 residual 도입**
  - `TRMConfig`에 **residual 스케일 파라미터** 추가:
    - `residual_alpha: float = 0.1`
  - `TinyRecursiveTransformer`에서 상태 업데이트를 다음과 같이 변경:
    - Reasoning 모드 (z):
      - `h = x + y + z`
      - `delta_z = block(h, cos, sin)`
      - `z = z + alpha * delta_z`  (alpha = `config.residual_alpha`)
    - Prediction 모드 (y):
      - `h = y + z`
      - `delta_y = block(h, cos, sin)`
      - `y = y + alpha * delta_y`
  - 의도:
    - 완전 direct replacement 대신, **이전 상태를 보존하면서 작은 보정(update)을 누적**하도록 설계.
    - `alpha << 1`로 두어, 재귀를 여러 번 돌아도 **norm 폭발을 크게 줄이면서 residual의 장점(gradient 흐름, 표현 누적)을 일부 확보**.
  - 구현 위치:
    - `src/config.py`에 `residual_alpha` 필드 추가.
    - `src/engine.py`의 `TinyRecursiveTransformer.__init__` 및 `forward` 내부 업데이트 로직 수정.

### 3. 기대 효과

- **Interface MLP 도입**
  - Qwen의 3584차원 표현에서, 단순 선형 압축보다 **더 비선형적이고 정보 보존적인 투영**을 수행.
  - TRM가 입력으로 받는 `x`의 품질 향상 → 재귀 추론에서 사용할 수 있는 컨텍스트 신호가 풍부해짐.

- **아주 작은 Residual 도입**
  - 동일한 TRMBlock을 n·T·N_sup 번 반복 호출해도, 각 단계의 업데이트 크기를 `alpha`로 명확히 제어.
  - 이전 상태(`y, z`)를 완전히 덮어쓰지 않고, **기존 정보 + 작은 보정** 구조로 학습이 이뤄져,  
    - 값 폭발을 완화하고,
    - 깊은 재귀에서 gradient 흐름과 표현 누적(“덧칠”)의 이점을 일부 가져올 수 있음.


