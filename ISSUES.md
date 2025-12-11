# Qwen-TRM 개발 이슈 정리

> 개발 과정에서 발견된 주요 버그와 해결 방법

---

## 1. 훈련 데이터 형식 불일치

### 문제
- **시스템 프롬프트**: `"put your final answer within \boxed{}"`
- **GSM8K 원본 데이터**: `#### 72` 형식

모델에게 `\boxed{}`로 답하라고 하면서 `####`로 된 데이터로 훈련시킴.

### 증상
- Loss는 낮아지는데 추론 시 이상한 출력
- 모델이 `\boxed{}`와 `####` 중 어떤 형식으로 답해야 할지 혼란

### 해결
```python
def convert_to_boxed(answer: str) -> str:
    """Convert GSM8K '#### 72' format to '\\boxed{72}' format."""
    match = re.search(r'####\s*(-?\d+(?:,\d+)*)', answer)
    if match:
        num = match.group(1)
        old = match.group(0)
        new = '\\\\boxed{' + num + '}'
        answer = answer.replace(old, new)
    return answer
```

**파일**: `train_trm.py`

---

## 2. ChatML 형식 미적용

### 문제
Qwen 모델은 ChatML 형식으로 사전학습됨:
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>
```

하지만 초기 코드는 단순 텍스트 연결 방식 사용:
```
Question: {question}
Solution: {answer}
```

### 증상
- Loss 1.5까지 떨어져도 추론 시 쓰레기 출력
- 무한 반복, 할루시네이션

### 해결
`tokenizer.apply_chat_template()` 사용:
```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": question},
    {"role": "assistant", "content": answer}
]
full_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)
```

**파일**: `train_trm.py`, `eval/trm_eval_simple.py`

---

## 3. EOS 토큰 미학습

### 문제
- EOS 토큰(`<|im_end|>`, ID: 151645)이 labels에서 마스킹되거나 누락
- 모델이 언제 생성을 멈춰야 하는지 학습 못함

### 증상
- 생성이 `max_new_tokens`까지 계속됨
- 무한 반복 출력

### 해결
1. Labels에 EOS 포함 확인
2. `generate()` 호출 시 `eos_token_id` 전달:
```python
model.generate(
    input_ids,
    eos_token_id=tokenizer.eos_token_id,  # 151645
    ...
)
```

**파일**: `src/model.py`, `eval/trm_eval_simple.py`

---

## 4. generate()에서 EOS 토큰 누락

### 문제
EOS 감지 시 break 먼저 하고 토큰 추가 → EOS가 출력에 포함 안 됨

```python
# 잘못된 코드
if (next_token == eos_token_id).all():
    break  # EOS 추가 전에 종료!
generated = torch.cat([generated, next_token], dim=-1)
```

### 해결
순서 변경 - 토큰 추가 후 EOS 체크:
```python
# 올바른 코드
generated = torch.cat([generated, next_token], dim=-1)
if (next_token == eos_token_id).all():
    break  # EOS 포함 후 종료
```

**파일**: `src/model.py`

---

## 5. attention_mask 전달 시 반복 붕괴

### 문제
`model.generate()`에 `attention_mask`를 전달하면 출력이 무한 반복됨

```python
# 문제 발생
model.generate(input_ids, attention_mask=mask, ...)

# 정상 동작
model.generate(input_ids, ...)  # mask 없이
```

### 원인
- TRM의 커스텀 generate()에서 attention_mask 처리 로직 버그
- Left padding + attention_mask 조합에서 위치 계산 오류
- RoPE 위치 인코딩과 mask 불일치

### 증상
```
Model Output: "2 eggs 2 eggs 2 eggs 2 eggs 2 eggs..."
```

### 해결
평가 코드에서 attention_mask 제거:
```python
generated = model.generate(
    input_ids=inputs['input_ids'],
    # attention_mask 제거!
    max_new_tokens=max_new_tokens,
    eos_token_id=eos_token_id
)
```

**파일**: `eval/trm_eval_simple.py`, `eval/trm_identity_eval.py`

### 주의
batch_size > 1에서 padding이 필요한 경우 별도 처리 필요

---

## 6. Response 추출 버그

### 문제
평가 코드에서 response 추출 시 문자열 길이 기반으로 계산:
```python
response = tokenizer.decode(generated, skip_special_tokens=True)
prompt_len = len(item["prompt"])  # special tokens 포함 길이
response_only = response[prompt_len:]  # 잘못된 위치에서 자름!
```

`skip_special_tokens=True`로 디코딩하면 special tokens가 제거되어 길이가 달라짐.

### 증상
- response_only가 이상한 위치에서 시작
- 출력이 잘리거나 prompt 일부가 포함됨

### 해결
토큰 인덱스 기반으로 새 토큰만 디코딩:
```python
input_len = inputs['input_ids'].size(1)
response_only = tokenizer.decode(generated[0, input_len:], skip_special_tokens=True)
```

**파일**: `eval/trm_eval_simple.py`, `eval/trm_identity_eval.py`, `eval/gsm8k_eval.py`

---

## 7. 유니코드 아포스트로피 차이

### 문제
GSM8K 데이터셋의 일부 문자가 유니코드:
- Dataset: `Janet's` (U+2019, RIGHT SINGLE QUOTATION MARK)
- ASCII: `Janet's` (U+0027, APOSTROPHE)

### 증상
- 같은 문제인데 다른 출력
- 하드코딩 시 정답, 데이터셋에서 로드 시 오답

### 영향
토크나이징 결과가 달라져 모델 동작에 영향. 심각한 버그는 아니지만 평가 결과 불일치 가능.

### 해결
데이터 전처리에서 유니코드 정규화 고려 (선택사항)

---

## 8. 체크포인트 저장 경로 혼동

### 문제
`--resume`은 가중치만 로드, 저장은 `--output_dir`에 함.
`--output_dir` 미지정 시 기본값 `./checkpoints/trm` 사용.

### 증상
```bash
# 이전 체크포인트에서 resume
--resume ./checkpoints/trm_chatml/checkpoint-467

# 하지만 저장은 다른 곳에!
# → ./checkpoints/trm/checkpoint-XX
```

### 해결
resume할 때 output_dir도 명시:
```bash
uv run python train_trm.py \
    --resume ./checkpoints/trm_chatml/checkpoint-467 \
    --output_dir ./checkpoints/trm_chatml  # 명시!
```

**파일**: `train_trm.py`

---

## 9. Python 백슬래시 이스케이프 문제

### 문제
`\boxed{}`를 문자열로 만들 때 `\b`가 백스페이스로 해석됨:
```python
f'\\boxed{72}'  # → '\x08oxed{72}' (백스페이스!)
```

### 해결
이중 백슬래시 또는 문자열 연결:
```python
new = '\\\\boxed{' + num + '}'  # "\\boxed{72}"
```

**파일**: `train_trm.py`

---

## 10. Scheduler N_supervision 누락

### 문제
LR scheduler의 total_steps 계산 시 N_supervision(16) 누락:
```python
# 잘못된 계산
total_steps = batches × epochs  # N_sup 빠짐!

# 올바른 계산
total_steps = batches × epochs × N_supervision
```

### 증상
- LR이 16배 빠르게 decay
- 훈련 후반에 학습률이 너무 낮아짐

### 해결
```python
total_optimizer_steps = num_batches × num_epochs × N_supervision
scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps)
```

**파일**: `src/train.py`

---

## 요약 체크리스트

훈련 전 확인사항:

- [ ] ChatML 형식 사용 (`apply_chat_template()`)
- [ ] `#### XX` → `\boxed{XX}` 변환
- [ ] EOS 토큰이 labels에 포함
- [ ] `generate()` 시 `eos_token_id` 전달
- [ ] `generate()` 시 `attention_mask` 제거 (batch_size=1)
- [ ] Response 추출은 토큰 인덱스 기반
- [ ] Scheduler에 N_supervision 반영
- [ ] `--output_dir` 명시

---

## 파일별 수정 위치

| 파일 | 수정 내용 |
|------|-----------|
| `train_trm.py` | ChatML, boxed 변환, output_dir |
| `src/model.py` | EOS 토큰 순서, generate() |
| `src/train.py` | Scheduler N_sup 반영 |
| `eval/trm_eval_simple.py` | attention_mask 제거, response 추출 |
| `eval/trm_identity_eval.py` | attention_mask 제거, response 추출 |
| `eval/gsm8k_eval.py` | response 추출 |
