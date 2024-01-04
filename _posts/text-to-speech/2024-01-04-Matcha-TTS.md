---
layout: post
title: "[Text-to-Speech] Matcha-TTS"
description: >
  Matcha-TTS 논문 요약
category: seminar
tags: text-to-speech
author: jh_yun
comments: true
---


# Matcha-TTS

- S. Mehta, “MATCHA-TTS: A Fast TTS Architecture with Conditional Flow Matching”, 2023

# Goal

- Conditional Flow Matching (CFM)을 활용하여 속도가 빠른 TTS 모델을 제안

# Motivation

- Diffusion 기반 모델의 단점
    - 느린 인퍼런스 속도와 많은 연산량으로 인한 낮은 실용성
- Conditional Flow Matching 기반 모델의 장점
    - 속도가 빠르면서 높은 퀄리티를 보여줌
    - CMF 기반의 TTS 모델이 제안된 바 없음

# Contribution

- CMF 기반의 TTS 모델을 최초로 제안
- 빠른 속도와 더불어 높은 음성 합성 성능을 보여줌

# Method

## Optimal-Transport Conditional Flow Matching (OP-CFM)

우리가 원하는 복잡하고 잘 모르는 데이터 분포 $q(x)$ 에서 $d$ 차원의 데이터를 샘플링했다고 생각해봅시다. 

$$
x \in \mathbb{R} ^d
$$

몇 가지 용어를 더 정의하겠습니다. 먼저 *Probability density path* $p_ {t}$ 는 Time-dependent 한 p.d.f 입니다.  $p_ {t}$ 는 $d$ 차원 공간에서 특정 시점 $t$ 에 **데이터 포인트가 존재할 확률밀도**를 나타냅니다.

$$
p_{t} : [0,1] \times \mathbb{R}^{d} \to \mathbb{R}_{>0}, \quad \int p_t(x)dx =1
$$

논문에서 말하기를 데이터 분포 $q$ 에서 샘플을 생성하는 방법 중 하나는 “$t \in [0,1]$에서 **아래와 같은 probability density path $p_{t}$ 를 구축하는 것**” 이라고 말합니다.

$$
p_{0} (x) = N(x; 0,I), \quad t=0
$$

$$
p_{1}(x)\approx q(x), \quad t=1
$$

이제 $p_t$ 구축을 위해서 *Vector field* $v_{t}: [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$   와, $v_{t}$ 가 만드는 *Flow* $\phi_{t}: [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$  를 아래와 같이 **ODE로 정의**합니다. 이 $v_t$는 Probability path 를 생성하는 역할입니다.

$$
\frac{d}{dt} \phi_t 
(x)= v_t (\phi_t(x)); \quad \phi_0(x)=x
$$

[Neural ODE, 2018]에서는 이 **Vector field $v_ {t}$ 를 신경망 $v_ {t} (x;\theta)$ 로 모델링**하는 방법을 제안했고, **Continuous-time Normalizing Flow (CNF)** 라고 칭하겠습니다.

CNF는 아래의 push-forward 방정식을 이용해, 단순한 prior density $p_{0}$ (e.g., pure noise)을 더 복잡한 데이터 분포 $p_{1}$ 으로 변환하는 데 사용됩니다.

$$
p_{t} = [\phi_{t}]*p_{0}
$$

$$
[\phi_{t}]*p_{0} (x) = p_{0} (\phi^{-1} _{t} (x)) \det [\frac{\partial \phi _ {t} ^ {-1} }{\partial x} (x)]
$$

---

이제 Flow Matching (FM) 과 Conditional Flow Matching (CMF) 이야기로 넘어가겠습니다.

Flow matching을 위해 2가지 가정을 짚고 가겠습니다. 

- ***Assumption 1***: $q(x_ 1 )$  에서 얻은 **샘플에 대한 접근은 가능**하지만, density function $q(x_ 1 )$ 에 대한 접근은 불가능 합니다.
- ***Assumption 2***: $p_ 0 = p$ 는 $p(x) = N( x \|0,I)$ 처럼 단순한 분포이고, $p_ 1 \approx q$ 를 만족하는 **Target probability path $p_ t$ 가 존재**합니다. 또한 이런 $p_{t}$ 를 만드는 **Vector field $u_{t} (x)$ 가 존재**합니다.

위의 2가지 가정과 함께 아래와 같은 **Flow Mathcing objective** 를 정의합니다.

$$
\mathcal{L}_ {\text{FM}} ( \theta ) = \mathbb{E}_{t,p_{t} (x)} || u_{t} (x) - v_{t} (x; \theta ) || ^2
$$

하지만, 이러한 접근법은 굉장힌 나이브하면서 몇 가지 문제점이 있습니다.

- ***Promblem 1***: $u_{t}$ 와 $p_t$ 는 Prior에 대한 정보가 없으므로 Intractable 합니다.
- ***Promblem 2***: $p_{1} (x) \approx q(x)$를 만족하는 Probability path 의 가짓수가 너무 많습니다.
- ***Promblem 3***: 원하는 $p_t$ 를 만드는 Vector field $u_{t}$ 에 대해 접근이 불가능 합니다.

---

이러한 문제점을 해결하기 위해서 Conditional Flow Matching이 제안되었습니다.

Conditional Flow Matching은 ‘Per-example’ 접근법 (i.e. conditional)을 사용합니다. 즉, 위의 Assumption 1번에 따라 $q(x_{1} )$에서 샘플 1개를 얻고, 이 샘플에 Conditioned 된 *Conditional probability path* $p_{t} (x \| x_{1})$ 와 이러한 $p_{t} (x \| x_{1})$를 만드는 *Conditional vector field* $u_{t} (x \|x _ {1})$ 를 정의할 수 있습니다. 그리고 여기서 $p_{t} (x \| x_ {1})$는 아래의 두 조건을 만족한다고 가정합니다.

$$
p_{0} (x | x_{1}) = N(x|0,I), \quad t=0
$$

$$
p_{1} (x | x_{1}) = N(x|x_{1},\sigma ^2 I), \quad t=1
$$

이제 Conditional probability path와 Conditional vector field를 marginalizing 하여 우리가 원하는 probability와 vector field에 접근이 가능해 보입니다. 하지만, 아쉽게도 marginalizing 과정에서 아직 intractable 한 항이 존재하여 단순한 marginalizing 으로 이 문제를 해결하기 어렵습니다.

[Flow Mathcing for Generative Modeling, 2022] 에서는 아래와 같은 새로운 objective를 제안합니다.

$$
\mathcal{L}_ {\text{CFM}} ( \theta ) = \mathbb{E}_{t,q(x_{1}),p_{t} (x|x_{1})} || u_{t} (x|x_{1}) - v_{t} (x; \theta ) || ^2
$$

[Flow Mathcing for Generative Modeling, 2022] 에서 증명된 Thm 중 하나는 $\mathcal{L}_ {\text{FM}} ( \theta )$ 와 $\mathcal{L}_ {\text{CFM}} ( \theta )$  가 $\theta$에 대해 동일한 그래디언트를 갖는다는 것입니다. 따라서, $\mathcal{L}_ {\text{CFM}} ( \theta )$ 을 최적화하는 것이 $\mathcal{L}_ {\text{FM}} ( \theta )$ 을 최적화하는 것과 동치라는 결론에 이릅니다.

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/d79164fb-d7b2-4100-af37-b9501485c09a)

---

[Flow Mathcing for Generative Modeling, 2022] 에서는 더욱 일반화된 CFM을 위해 Conditional probability path $p_ {t} (x \| x _ {1})$ 와 Conditional vector field $u_ {t} (x \| x _ {1})$에 대한 논의를 시작합니다.

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/0fea6bc7-01da-49b6-a156-c50a331a5d56)

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/6d0588a8-bf91-4855-843b-6096b36ef8ec)

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/a4fecd11-257a-4551-9e5f-247ed17d341b)

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/639ef708-9a1c-4627-8856-6520189ce302)

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/2971020c-f3ff-4622-850c-cf75d04458a0)

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/52368be0-bc6c-4db0-81b9-f59001fad14a)

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/328a17e8-e60e-44ca-a556-a65fe0c78712)

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/e7f697dc-280b-44ee-b74c-64a30517287f)

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/3e760ca6-cfe5-4959-a42a-f9336943af6f)

Matcha-TTS 에서는[Flow Mathcing for Generative Modeling, 2022]에서 제안한 CMF의 변형인  Optimal-Transport Conditional Flow Matching (OT-CFM)을 이용해서 최종 objective를 정의합니다.

$$
\mathcal{L}_ {\text{OT-CFM}} ( \theta ) = \mathbb{E}_{t,q(x_{1}),p_{0} (x_{0})} || u_{t} ^{\text{OT}} (\phi_ {t} ^{\text{OT}}(x_{0})|x_{1}) - v_{t} (\phi_ {t} ^{\text{OT}}(x_{0})|\mu; \theta ) || ^2
$$

이 때, $\phi_ {t} ^{\text{ OT} } (x_ {0}) = (1-(1-\sigma_ {\min} ) t ) x_ {0} + t x_ {1}$ 로 정의하여 시간이 변함에 따라 $x_ {0}$ 에서 $x_ {1}$ 로 변화하는 Flow로 생각할 수 있습니다.

이렇게 하면 구하고자 하는 Vector field  $u_{t} ^{\text{OT}} (\phi_ {t} ^{\text{OT}}(x_{0}) \|x_{1})$ 는 Linear, Time invariant 하고, 오직 $x_0$와 $x_1$ 값에만 의존하는  $u_{t} ^{\text{OT}} (\phi_ {t} ^{\text{OT}}(x_{0})\|x_{1}) = x_{1} - (1- \sigma _ {\text{min}} ) x_{0}$ 라고 할 수 있습니다. 

# Architecture

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/c043e77c-ba75-4019-80fe-d9dd90b52f4a)

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/dc659157-3502-4821-9989-f52edde5c06d)

- Non autoregressive 인코더-디코더 구조
- Text encoder
    - Glow-TTS, Grad-TTS를 Follow
    - Encoder loss도 Grad-TTS 와 동일하게 아래처럼 정의
        - $\mathcal{L}_ {\text{enc}} = - \sum ^{F} _ {j=1} \log \psi (y_{j}; \tilde{\mu} _{A(j)} , I)$
    - Text-encoder의 아웃풋 벡터들은 Duration predictor에 맞춰 Upsampling(복제)
    - Upsampling 결과를 $\mu$ 라고 표현
    - $\mu$는 최종적으로 주어진 텍스트와  duration에 따라 예측된 Average 어쿠스틱 피쳐
    - $\mu$는 Decoder가 Vector field를 예측할 때에 컨디션으로 사용
        - Decoder:  $v_{t} (\phi_ {t} ^{\text{OT}}(x_{0}) \|\mu; \theta )$
    - 특이점: Grad-TTS와 다르게 맨 처음 노이즈 샘플 $x_0$에는 $\mu$가 사용되지 않음

- Duration predictor
    - Glow-TTS, Grad-TTS를 Follow
    - 소숫점 결과는 자연수로 올림

- Decoder
    - U-Net 기반 (Stable-diffusion에서 영감받음)
        - 1D conv residual block + Transformer block
    - $v_{t} (\phi_ {t} ^{\text{OT}}(x_{0}) \| \mu; \theta )$
    - $\mathcal{L}_ {\text{OT-CFM}} ( \theta ) = \mathbb{E}_ {t,q(x_{1}),p_{0} (x_{0})} \|\| u_{t} ^{\text{OT}} (\phi_ {t} ^{\text{OT}}(x_{0})\|x_{1}) - v_{t} (\phi_ {t} ^{\text{OT}}(x_{0})\|\mu; \theta ) \|\| ^2$
    

# Experiment

### MOS Result

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/290d13db-d9cb-47ca-acd8-219fed253c4a)

### Synthesis speed

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/783e62e3-b584-4da8-8108-022cb6141e45)
