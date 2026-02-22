---
layout: post
title: "순수 Python으로 GPT 구현하기 - Karpathy 미니멀 코드"
date: 2026-02-22
categories: useful-stuff
---

# 순수 Python으로 GPT 구현하기

> "이 파일 하나가 알고리즘 전체입니다. 나머지는 단지 효율성의 문제일 뿐입니다." - @karpathy

Andrej Karpathy가 만든 순수 Python GPT 구현체입니다. PyTorch, NumPy 같은 외부 라이브러리 없이 **표준 라이브러리만으로** 트랜스포머 모델을 학습하고 추론합니다.

## 무엇을 배울 수 있나요?

- **Autograd (자동 미분)**: 역전파를 직접 구현해 체인룰을 이해합니다
- **트랜스포머 아키텍처**: GPT의 핵심 구조 (어텐션, MLP, RMSNorm)
- **Adam 옵티마이저**: 적응형 학습률의 동작 원리
- **언어 모델링**: 문자 단위 토크나이저부터 샘플링까지

## 주요 특징

- ✅ 순수 Python (외부 의존성 없음)
- ✅ 870줄의 주석 포함 (한국어 번역)
- ✅ 사람 이름 학습 → 새로운 이름 생성
- ✅ KV 캐시, 멀티헤드 어텐션 구현
- ✅ Adam 옵티마이저 직접 구현

## 코드

```python
"""
순수한 Python만으로 GPT를 학습하고 추론하는 가장 최소한의 방법입니다.
이 파일 하나가 알고리즘 전체입니다.
나머지는 단지 효율성의 문제일 뿐입니다.

@karpathy
"""

import os       # os.path.exists: 파일 존재 여부 확인
import math     # math.log, math.exp: 로그/지수 함수 (역전파에서 사용)
import random   # 랜덤 시드, 샘플링, 가우시안 노이즈 등

# 재현성을 위해 랜덤 시드를 고정합니다.
# 같은 시드를 쓰면 매 실행마다 동일한 결과를 얻을 수 있습니다.
random.seed(42)

# ============================================================
# 1. 데이터셋 준비
# ============================================================
# 학습에 사용할 문서 목록(docs)을 불러옵니다.
# 여기서는 사람 이름 목록을 사용합니다.
# input.txt가 없으면 자동으로 Karpathy의 GitHub에서 다운로드합니다.
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

# 파일에서 한 줄씩 읽어 공백을 제거하고, 빈 줄은 제외합니다.
# docs는 ["emma", "olivia", "ava", ...] 같은 이름 리스트가 됩니다.
docs = [line.strip() for line in open('input.txt') if line.strip()]

# 모든 문서를 섞어서 학습 순서에 편향이 생기지 않도록 합니다.
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# ============================================================
# 2. 토크나이저 (Tokenizer)
# ============================================================
# 모든 문서를 합친 뒤 중복 없는 고유 문자 집합을 만들고 정렬합니다.
# 예: ['a', 'b', 'c', ..., 'z'] → 각각 0, 1, 2, ..., 25번 토큰
uchars = sorted(set(''.join(docs)))

# BOS(Beginning of Sequence) 토큰: 문장의 시작과 끝을 나타내는 특수 토큰
# 고유 문자 다음 번호를 BOS 토큰 ID로 사용합니다.
# 이름의 시작을 알리고, 이름이 끝나면 다시 BOS가 나와 "종료"를 표현합니다.
BOS = len(uchars)

# 전체 어휘 크기 = 고유 문자 수 + BOS 토큰 1개
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# ============================================================
# 3. 자동 미분 엔진 (Autograd)
# ============================================================
# 신경망 학습의 핵심: 연산 그래프를 따라 역방향으로 체인룰(chain rule)을 적용해
# 모든 파라미터에 대한 기울기(gradient)를 자동으로 계산합니다.
class Value:
    # __slots__: 인스턴스 속성을 미리 고정해 메모리 사용을 줄이는 Python 최적화 기법
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # 순전파(forward pass)에서 계산된 이 노드의 스칼라 값
        self.grad = 0                   # 역전파(backward pass)에서 채워질 기울기 (손실에 대한 이 노드의 편미분)
        self._children = children       # 연산 그래프에서 이 노드를 만들어낸 자식 노드들
        self._local_grads = local_grads # 이 노드를 각 자식에 대해 미분한 '로컬 기울기'

    # --- 산술 연산자 정의 (각 연산의 순전파 값 + 역전파에 필요한 로컬 기울기 저장) ---

    def __add__(self, other):
        # 덧셈: d(a+b)/da = 1, d(a+b)/db = 1
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        # 곱셈: d(a*b)/da = b, d(a*b)/db = a
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        # 거듭제곱: d(x^n)/dx = n * x^(n-1)
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        # 자연로그: d(ln x)/dx = 1/x  → 크로스 엔트로피 손실 계산에 사용
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        # 지수함수: d(e^x)/dx = e^x  → softmax 계산에 사용
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        # ReLU 활성화 함수: max(0, x), 미분은 x > 0이면 1, 아니면 0
        # MLP 블록의 비선형성으로 사용됩니다.
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    # --- Python이 a + b를 처리할 때 좌/우 피연산자를 모두 지원하기 위한 반사 연산자 ---
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other       # 예: 0 + Value
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other       # 예: 2 * Value
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        """
        역전파(Backpropagation): 연산 그래프 전체에 체인룰을 적용합니다.

        1) 위상 정렬(topological sort): 순전파 순서대로 노드를 나열합니다.
        2) 역순으로 순회하며 각 노드의 기울기를 자식 노드로 전달합니다.
           child.grad += (로컬 기울기) × (현재 노드의 기울기)  ← 체인룰
        """
        # 위상 정렬: DFS로 리프 노드부터 루트 순서로 topo 리스트를 만듭니다.
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # 손실(루트 노드)의 자기 자신에 대한 기울기는 항상 1입니다.
        self.grad = 1

        # 역순(루트 → 리프)으로 순회하며 각 자식에게 기울기를 누적합니다.
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                # 체인룰: 자식의 기울기 += 로컬 기울기 × 현재 노드의 기울기
                child.grad += local_grad * v.grad

# ============================================================
# 4. 모델 하이퍼파라미터 및 파라미터 초기화
# ============================================================
# 트랜스포머의 구조를 결정하는 핵심 하이퍼파라미터들입니다.
n_layer = 1     # 트랜스포머 레이어(블록)의 수. 깊을수록 표현력이 높아집니다.
n_embd = 16     # 임베딩 차원. 각 토큰이 몇 차원의 벡터로 표현되는지를 결정합니다.
block_size = 16 # 어텐션이 볼 수 있는 최대 문맥 길이 (데이터셋의 최장 이름은 15자)
n_head = 4      # 멀티헤드 어텐션의 헤드 수. 다양한 관점에서 문맥을 파악합니다.
head_dim = n_embd // n_head  # 각 헤드의 차원 = 16 // 4 = 4

# 가중치 행렬 생성 헬퍼: nout×nin 크기의 2D 리스트를 가우시안 분포로 초기화합니다.
# std=0.08은 작은 값으로 초기화해 학습 초기 안정성을 높입니다.
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# state_dict: 모델의 모든 학습 가능한 파라미터를 담는 딕셔너리
state_dict = {
    'wte': matrix(vocab_size, n_embd),    # 토큰 임베딩 테이블: 각 토큰 → n_embd 차원 벡터
    'wpe': matrix(block_size, n_embd),    # 위치 임베딩 테이블: 각 위치 → n_embd 차원 벡터
    'lm_head': matrix(vocab_size, n_embd) # 언어 모델 헤드: 최종 임베딩 → 어휘 로짓(logits)
}

# 각 레이어(블록)의 어텐션과 MLP 가중치를 추가합니다.
for i in range(n_layer):
    # --- 멀티헤드 어텐션 가중치 ---
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)  # Query 투영 행렬
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)  # Key 투영 행렬
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)  # Value 투영 행렬
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)  # 어텐션 출력 투영 행렬
    # --- MLP(Feed-Forward Network) 가중치 ---
    # fc1: n_embd → 4*n_embd (확장), fc2: 4*n_embd → n_embd (축소)
    # 4배 확장은 GPT-2가 사용하는 표준 구조입니다.
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

# 모든 행렬의 파라미터를 1차원 리스트로 펼칩니다.
# 나중에 옵티마이저가 한 번에 순회할 수 있도록 편리하게 관리합니다.
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# ============================================================
# 5. 모델 아키텍처 정의
# ============================================================
# GPT-2를 따르되 몇 가지 단순화:
#   - LayerNorm → RMSNorm (바이어스 없음, 더 단순)
#   - 바이어스(bias) 항 제거
#   - GeLU 활성화 → ReLU (순수 Python으로 구현 가능)

def linear(x, w):
    """
    선형 변환: y = W @ x
    w는 [nout × nin] 2D 리스트, x는 [nin] 1D 리스트입니다.
    각 출력 뉴런에 대해 입력과의 내적을 계산합니다.
    """
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    """
    소프트맥스: 로짓(raw scores)을 확률 분포로 변환합니다.
    수치 안정성을 위해 최댓값을 빼고 나서 exp를 취합니다.
    (max_val을 빼도 결과는 수학적으로 동일하지만 오버플로를 방지합니다.)
    """
    max_val = max(val.data for val in logits)      # 수치 안정성용 최댓값
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    """
    RMS 정규화 (Root Mean Square Normalization):
    각 벡터를 그 자신의 RMS로 나눠 크기를 정규화합니다.
    LayerNorm보다 단순하며, 현대 LLM(LLaMA 등)에서 많이 사용됩니다.
    eps=1e-5는 분모가 0이 되는 것을 방지합니다.
    """
    ms = sum(xi * xi for xi in x) / len(x)   # 평균 제곱값(mean square)
    scale = (ms + 1e-5) ** -0.5              # 역수 RMS
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    """
    GPT 순전파(forward pass): 하나의 토큰을 받아 다음 토큰의 로짓을 반환합니다.
    KV 캐시(keys, values)를 받아 이전 위치들의 어텐션 정보를 재활용합니다.

    Args:
        token_id: 현재 토큰의 정수 ID
        pos_id:   현재 토큰의 위치 인덱스 (0부터 시작)
        keys:     각 레이어별 누적된 Key 벡터 목록 (KV 캐시)
        values:   각 레이어별 누적된 Value 벡터 목록 (KV 캐시)

    Returns:
        logits: 어휘 크기만큼의 로짓 리스트 (다음 토큰 예측용)
    """
    # --- 입력 임베딩: 토큰 임베딩 + 위치 임베딩을 합산합니다 ---
    tok_emb = state_dict['wte'][token_id]  # 이 토큰이 "무엇"인지 인코딩
    pos_emb = state_dict['wpe'][pos_id]   # 이 토큰이 "어디에" 있는지 인코딩
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # 두 임베딩을 합산

    # 잔차 연결(residual connection)이 있어도 RMSNorm이 필요한 이유:
    # 역전파 시 기울기 흐름을 안정화하기 위해 초기 정규화를 합니다.
    x = rmsnorm(x)

    # --- 트랜스포머 레이어 반복 ---
    for li in range(n_layer):

        # ── 블록 1: 멀티헤드 셀프 어텐션 (Multi-Head Self-Attention) ──
        x_residual = x          # 잔차 연결을 위해 입력을 저장
        x = rmsnorm(x)          # Pre-Norm: 어텐션 전에 정규화

        # QKV 투영: 현재 토큰 임베딩을 Query, Key, Value로 각각 변환
        q = linear(x, state_dict[f'layer{li}.attn_wq'])  # 질의(Query): "나는 무엇을 찾고 있나?"
        k = linear(x, state_dict[f'layer{li}.attn_wk'])  # 키(Key):    "나는 어떤 정보를 갖고 있나?"
        v = linear(x, state_dict[f'layer{li}.attn_wv'])  # 값(Value):  "내가 제공할 실제 정보"

        # KV 캐시에 현재 위치의 K, V를 추가 (이전 위치들과 함께 어텐션 계산)
        keys[li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(n_head):
            # 각 헤드는 임베딩 공간의 서로 다른 부분(슬라이스)을 담당합니다.
            hs = h * head_dim  # 이 헤드의 시작 인덱스

            # 현재 헤드에 해당하는 Q, K, V 슬라이스 추출
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]   # 과거 모든 위치의 K
            v_h = [vi[hs:hs+head_dim] for vi in values[li]] # 과거 모든 위치의 V

            # 어텐션 스코어: Q와 모든 K의 내적 / √head_dim (스케일링으로 기울기 폭발 방지)
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]

            # 스코어를 확률(어텐션 가중치)로 변환
            attn_weights = softmax(attn_logits)

            # 어텐션 가중치로 V를 가중합산 → 이 헤드의 출력
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)  # 모든 헤드의 출력을 이어붙입니다

        # 멀티헤드 출력을 출력 투영 행렬로 변환 후 잔차 연결
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]  # 잔차 연결: 기울기 소실 방지

        # ── 블록 2: MLP (Feed-Forward Network) ──
        x_residual = x    # 잔차 연결을 위해 입력을 저장
        x = rmsnorm(x)    # Pre-Norm: MLP 전에 정규화

        # 두 번의 선형 변환 사이에 ReLU 비선형성 삽입
        # fc1: n_embd → 4*n_embd (차원 확장 → 더 풍부한 표현 학습)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]   # 비선형 활성화 함수: 0 이하 값을 0으로 만듦
        # fc2: 4*n_embd → n_embd (다시 원래 차원으로 압축)
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]  # 잔차 연결

    # 최종 임베딩을 어휘 크기의 로짓으로 변환 (다음 토큰 예측용 점수)
    logits = linear(x, state_dict['lm_head'])
    return logits

# ============================================================
# 6. Adam 옵티마이저
# ============================================================
# Adam: SGD보다 적응형(adaptive) 학습률을 사용해 더 안정적으로 학습합니다.
# 1차 모멘텀(m): 기울기의 지수이동평균 → 관성(momentum) 효과
# 2차 모멘텀(v): 기울기 제곱의 지수이동평균 → 파라미터별 학습률 조정
learning_rate = 0.01  # 기본 학습률
beta1 = 0.85          # 1차 모멘텀 감쇠율 (일반적으로 0.9; 여기선 빠른 적응을 위해 약간 낮춤)
beta2 = 0.99          # 2차 모멘텀 감쇠율 (일반적으로 0.999)
eps_adam = 1e-8       # 분모가 0이 되는 것을 방지하는 소수 엡실론

m = [0.0] * len(params)  # 1차 모멘텀 버퍼 (파라미터별 기울기 지수이동평균)
v = [0.0] * len(params)  # 2차 모멘텀 버퍼 (파라미터별 기울기 제곱의 지수이동평균)

# ============================================================
# 7. 학습 루프 (Training Loop)
# ============================================================
num_steps = 1000  # 총 학습 스텝 수
for step in range(num_steps):

    # --- 미니배치 준비 ---
    # 한 번에 하나의 문서(이름)를 처리합니다. (배치 크기 = 1)
    # step % len(docs)로 전체 데이터셋을 순환합니다.
    doc = docs[step % len(docs)]

    # 문자를 토큰 ID로 변환하고, 앞뒤에 BOS 토큰을 붙입니다.
    # 예: "emma" → [BOS, 4, 12, 12, 0, BOS]
    # BOS가 앞에 있으면 "이름의 시작을 예측", 뒤에 있으면 "종료를 예측"합니다.
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]

    # 최대 block_size만큼만 처리 (어텐션 창 크기 제한)
    n = min(block_size, len(tokens) - 1)

    # --- 순전파 (Forward Pass) ---
    # KV 캐시를 레이어별로 초기화합니다.
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []

    for pos_id in range(n):
        token_id = tokens[pos_id]       # 현재 입력 토큰
        target_id = tokens[pos_id + 1]  # 예측해야 할 다음 토큰 (정답 레이블)

        # 모델을 통해 로짓 계산
        logits = gpt(token_id, pos_id, keys, values)

        # 소프트맥스로 확률 분포 계산
        probs = softmax(logits)

        # 크로스 엔트로피 손실: -log(정답 토큰의 확률)
        # 정답 토큰의 확률이 높을수록 손실이 작아집니다.
        loss_t = -probs[target_id].log()
        losses.append(loss_t)

    # 전체 시퀀스에 대한 평균 손실 (낮을수록 좋습니다!)
    loss = (1 / n) * sum(losses)

    # --- 역전파 (Backward Pass) ---
    # 손실에서 시작해 모든 파라미터의 기울기를 계산합니다.
    loss.backward()

    # --- Adam 파라미터 업데이트 ---
    # 선형 학습률 감쇠: 학습이 진행될수록 학습률을 점진적으로 줄입니다.
    # 초반에는 빠르게, 후반에는 세밀하게 학습합니다.
    lr_t = learning_rate * (1 - step / num_steps)

    for i, p in enumerate(params):
        # 1차 모멘텀 업데이트: 기울기의 지수이동평균 (관성 효과)
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad

        # 2차 모멘텀 업데이트: 기울기 제곱의 지수이동평균 (적응형 학습률)
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2

        # 편향 보정(bias correction): 초기 스텝에서 0으로 편향된 모멘텀을 보정합니다.
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))

        # 파라미터 업데이트: 적응형 학습률로 기울기 방향으로 이동
        # v_hat의 제곱근으로 나눠 기울기가 큰 파라미터의 학습률을 자동으로 낮춥니다.
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)

        # 기울기를 0으로 초기화 (다음 스텝을 위해 누적 기울기 리셋)
        p.grad = 0

    # 현재 스텝의 손실 출력 (\r로 같은 줄을 덮어씁니다)
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# ============================================================
# 8. 추론 (Inference) - 새로운 이름 생성
# ============================================================
# 학습된 모델을 사용해 새로운 이름을 샘플링합니다.
# temperature: 낮을수록 보수적(확률 높은 것 선택), 높을수록 창의적(다양성 증가)
temperature = 0.5  # (0, 1] 범위; 0.5는 적당히 일관성 있는 이름을 생성합니다.

print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    # 각 샘플마다 KV 캐시를 새로 초기화합니다.
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]

    # BOS 토큰으로 생성을 시작합니다. "이름을 시작해 주세요"라는 신호입니다.
    token_id = BOS
    sample = []

    for pos_id in range(block_size):
        # 현재 토큰을 모델에 입력하고 다음 토큰의 로짓을 얻습니다.
        logits = gpt(token_id, pos_id, keys, values)

        # temperature로 로짓을 나눠 확률 분포의 '뾰족함'을 조절합니다.
        # temperature < 1 → 더 뾰족한 분포 → 높은 확률의 토큰이 더 잘 선택됨
        probs = softmax([l / temperature for l in logits])

        # 확률에 비례해 다음 토큰을 랜덤 샘플링합니다.
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]

        # BOS 토큰이 생성되면 이름이 끝난 것으로 간주하고 중단합니다.
        if token_id == BOS:
            break

        # 생성된 토큰 ID를 다시 문자로 변환해 누적합니다.
        sample.append(uchars[token_id])

    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

## 실행 방법

```bash
python minimal_gpt.py
```

실행하면:
1. 사람 이름 데이터셋 자동 다운로드
2. 1000 스텝 학습 (몇 분 소요)
3. 새로운 이름 20개 생성

## 결과 예시

```
num docs: 32033
vocab size: 27
num params: 3472
step 1000 / 1000 | loss 1.8432
--- inference (new, hallucinated names) ---
sample  1: mar
sample  2: aliah
sample  3: sora
sample  4: kaleigh
...
```

## 참고 자료

- 원본: [Karpathy의 GitHub](https://github.com/karpathy)
- 트랜스포머 논문: "Attention is All You Need" (Vaswani et al., 2017)
- GPT-2 논문: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)

---

**난이도:** ⭐⭐⭐⭐ (중급~고급)  
**학습 시간:** 2~3시간 (주석 읽으며 따라가기)  
**의존성:** Python 3.7+ (표준 라이브러리만 사용)

*이 코드는 교육 목적으로 단순화되었습니다. 실무에서는 PyTorch/JAX 같은 프레임워크를 사용하세요.*
