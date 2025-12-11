# NuminaMath-CoT Dataset Samples

> **Dataset**: [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
> **Size**: 859,494 samples
> **Format**: `\boxed{}` (Qwen-Math 호환)

---

## Sample 1
**Source**: synthetic_math

### Problem
Consider the terms of an arithmetic sequence: $-\frac{1}{3}, y+2, 4y, \ldots$. Solve for $y$.

### Solution
For an arithmetic sequence, the difference between consecutive terms must be equal. Therefore, we can set up the following equations based on the sequence given:
\[ (y + 2) - \left(-\frac{1}{3}\right) = 4y - (y+2) \]

Simplify and solve these equations:
\[ y + 2 + \frac{1}{3} = 4y - y - 2 \]
\[ y + \frac{7}{3} = 3y - 2 \]
\[ \frac{7}{3} + 2 = 3y - y \]
\[ \frac{13}{3} = 2y \]
\[ y = \frac{13}{6} \]

Thus, the value of $y$ that satisfies the given arithmetic sequence is $\boxed{\frac{13}{6}}$.

---

## Sample 2
**Source**: synthetic_math

### Problem
Suppose that $g(x) = 5x - 3$. What is $g^{-1}(g^{-1}(14))$?

### Solution
First, we need to find the inverse function $g^{-1}(x)$. Given $g(x) = 5x - 3$, solve for $x$:
\[ y = 5x - 3 \]
\[ y + 3 = 5x \]
\[ x = \frac{y + 3}{5} \]
Thus, $g^{-1}(x) = \frac{x + 3}{5}$.

Now, apply $g^{-1}$ twice to the given value $14$:
\[ g^{-1}(14) = \frac{14 + 3}{5} = \frac{17}{5} \]
\[ g^{-1}\left(\frac{17}{5}\right) = \frac{\frac{17}{5} + 3}{5} = \frac{\frac{17}{5} + \frac{15}{5}}{5} = \frac{32}{5 \times 5} = \frac{32}{25} \]

Thus, $g^{-1}(g^{-1}(14)) = \boxed{\frac{32}{25}}$.

---

## Sample 3
**Source**: synthetic_math

### Problem
A farmer has a rectangular field with dimensions $3m+8$ and $m-3$ where $m$ is a positive integer. If the field has an area of 76 square meters, find the value of $m$.

### Solution
Using the given dimensions, we set up the area equation:
\[
(3m+8)(m-3) = 76.
\]
Expanding this, we get:
\[
3m^2 - 9m + 8m - 24 = 76,
\]
\[
3m^2 - m - 24 = 76,
\]
\[
3m^2 - m - 100 = 0.
\]
Factoring the quadratic, we find:
\[
(3m+25)(m-4) = 0.
\]
This gives two potential solutions for $m$: $m=-\frac{25}{3}$ and $m=4$. Since $m$ must be a positive integer, the only valid solution is $m = \boxed{4}$.

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total samples | 859,494 |
| Columns | `source`, `problem`, `solution`, `messages` |
| Answer format | `\boxed{}` |
| Sources | synthetic_math, olympiads, cn_k12, orca_math, ... |

### Token Length Distribution (n=10,000 sample)

| Percentile | Tokens |
|------------|--------|
| 50% | 434 |
| 75% | 647 |
| 90% | 919 |
| 95% | 1,100 |
| 99% | 1,497 |

### Coverage by max_length

| max_length | Coverage |
|------------|----------|
| 512 | 61% |
| 768 | 83% |
| **1024** | **93%** (추천) |
| 1536 | 99% |
| 2048 | 99.9% |
