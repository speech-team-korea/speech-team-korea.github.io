---
layout: post
title: "[Generative-Model] Flow matching"
description: >
  Flow matching 논문 요약
category: seminar
tags: generative-model
author: jh_yun
comments: true
---

# Flow matching 수식 정리

<blockquote style="border-left: 2px solid; padding-left: 10px; margin-left: 0;">
Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le <br>
Accepted by ICLR, 2023 <br>
[<a href="https://arxiv.org/pdf/2210.02747.pdf">Paper</a>] <br>
</blockquote>

[참고] 아직 완성본이 아니라 초안입니다. 저도 아직 모르는 부분이 많아 공부중이며, 수시로 정리되는 부분이 있으면 추가하거나 수정하겠습니다.

이 글을 읽기 전에 한 번 읽어보면 좋은 내용

1. 왜 ‘Flow matching’이라고 부르는가?
    
    비슷한 예로 (Score-based) 디퓨전에서 Score matching이라는 말이 자주 나오는데, 이는 실제로 디퓨전 모델이 예측하는 값이 log-확률분포의 기울기를 의미하는 Score $\nabla_ {x} \log p_ {t} (x)$ 이기 때문입니다. 즉, 우리는 Flow matching이라는 방법을 사용하고자, 어떠한 모델 파라미터에 대한 학습을 통해서 flow를 예측하게 됩니다. 
    
2. Flow, Vector field, Probability density path 이 무엇인가?
    
    먼저 아래에서 자주 쓰일 용어들에 대해 간단하게 설명하겠습니다.
    
    - *flow* $\phi_{t}: [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$
        - 우리가 모델을 통해 예측할 값, flow $\phi_ {t}$ 입니다. 후에 우리는 굉장히 심플한 분포 $p_0$와 우리가 원하는 데이터 분포 $p_1$ 사이를 연결하려고 하는데, 이때 사용되는 징검다리(흐름)이 바로 이 flow 입니다.
    - *vector field* $v_{t}: [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$
        - vector field $v_t$ 는 다음과 같은 ODE를 통해서 flow $\phi_t$를 정의합니다. 논문에서는 아래의 ODE를 기반으로 $v_t$와 $\phi_t$ 사이의 관계를 “$v_ t$ 가 $\phi_ t$를 만든다.” 라고 표현하고 있습니다.
        
        $$
        \frac{d}{dt} \phi_t 
        (x)= v_t (\phi_t(x)); \quad \phi_0(x)=x
        $$
        
        - 이 ODE의 의미를 간단하게 이해해보면, flow (두 분포 사이의 흐름) $\phi_t(x)$의 시간에 따른 변화량은 $v_t$ 로 표현이 된다는 것 입니다.
        - 뿐만 아니라 “$v_t$가 밑에서 설명할 Probability density path도 구축한다”고 표현합니다.
    - probability density path $p_{t} : [0,1] \times \mathbb{R}^{d} \to \mathbb{R}_{>0}, \quad \int p_t(x)dx =1$
        - Probability density path $p_t$는 time-dependent probability density function 입니다.즉, 시점 $(0 \leq t \leq 1)$이 딱 정해지면, 그 시점에 대한 P.d.f 를 의미합니다. (위에 적은 정의역을 보시면 이해가 됩니다).
        - 아래의 예시는 GPT가 그려준 예시인데,  $t=0$ 부터 $t=1$까지의 시간을 0.25마다 이산적으로 정해주고, 데이터 차원 $d=1$ 인 p.d.f 입니다. 만약 아래의 p.d.f가 이산적인 시간이 아니라 연속적인 시간에 대응하여 무한 개가 존재한다고 합시다. 그 무한개의 초록색 p.d.f를 입체적으로 앞뒤로 이어붙이면 흐물흐물 거리는 3D 터널이 되고, 그 터널이 지금 설명하고 있는 probability density path $p_t$ 입니다. 즉, 시간이 지남에 따라 터널이 조금씩 굴곡이 생기는 터널로 가볍게 이해할 수 있습니다. 그리고 그 터널을 위에서 가로방향과 평행하게 잘라 단면적을 보면 특정 시점 $t$에서의 p.d.f 라고 할 수 있습니다. (앞으로 편의를 위해서 probability path라고 하겠습니다)
        
        ![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/d5b1ef83-e6b2-49c9-96dd-3bffd9342bab)
        
        - Prabability path 를 터널로 이해하고 보니, Flow와 Probability path 둘 다 심플한 분포에서 원하는 분포로 연결하는 연결다리로 이해될 수도 있습니다. 하지만 이 둘은 엄연히 다릅니다. 이는 후에 더 설명하겠습니다.

---

## Continuous Normalizing Flow (CNF)

이제 기본적인 세팅은 다 끝났습니다. 

먼저 우리가 알아야할 것은 Continuous Normalizing Flow (CNF) 입니다. 

우리가 원하는 복잡하고 잘 모르는 데이터 분포 $q(x)$ 에서 $d$ 차원의 데이터를 샘플링했다고 생각해봅시다. 

$$
x \in \mathbb{R} ^d
$$

생성모델의 본질적인 목적은 데이터 분포를 모델링하여 샘플링하는 것 입니다. 논문에서 말하기를 데이터 분포 $q$ 에서 샘플을 생성하는 방법 중 하나는 “$t \in [0,1]$에서 **아래와 같은 probability density path $p_{t}$ 를 구축하는 것**” 이라고 말합니다.

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

[Neural ODE, 2018]에서는 이 **Vector field $v_ {t}$ 를 신경망 $v_ {t} (x;\theta)$ 로 모델링**하는 방법을 제안했고, **Continuous-time Normalizing Flow (CNF)** 라고 부릅니다.

CNF는 아래의 push-forward 방정식을 이용해, 단순한 prior density $p_{0}$ (e.g., pure noise)을 더 복잡한 데이터 분포 $p_{1}$ 으로 변환하는 데 사용됩니다.

$$
p_{t} = [\phi_{t}]*p_{0}
$$

$$
[\phi_{t}]*p_{0} (x) = p_{0} (\phi^{-1} _{t} (x)) \det [\frac{\partial \phi _ {t} ^ {-1} }{\partial x} (x)]
$$

이렇기 때문에 위에서 터널로 이해한 probability path $p_t$와 flow $\phi_t$ 는 엄연히 다르다고 할 수 있습니다. 즉, $\phi_t$가 위와 같은 연산을 통해 $p_t$를 만들 수 있습니다.

## Flow matching

이제 Flow Matching (FM) 과 Conditional Flow Matching (CMF) 이야기로 넘어가겠습니다.

Flow matching을 위해 2가지 가정을 짚고 가겠습니다. 

- ***Assumption 1***: $q(x_ 1 )$  에서 얻은 **샘플에 대한 접근은 가능**하지만, density function $q(x_ 1 )$ 에 대한 접근은 불가능 합니다.
- ***Assumption 2***: $p_ 0 = p$ 는 $p(x) = N( x \|0,I)$ 처럼 단순한 분포이고, $p_ 1 \approx q$ 를 만족하는 **Target probability path $p_ t$ 가 존재**합니다. 또한 이런 $p_{t}$ 를 만드는 **Vector field $u_{t} (x)$ 가 존재**합니다.

위의 2가지 가정과 함께 아래와 같은 **Flow Mathcing objective** 를 정의합니다.

$$
\mathcal{L}_ {\text{FM}} ( \theta ) = \mathbb{E}_{t\sim U[0,1],x \sim p_{t} (x)} || u_{t} (x) - v_{t} (x; \theta ) || ^2
$$

하지만, 이러한 접근법은 굉장힌 나이브하면서 몇 가지 문제점이 있습니다.

- ***Promblem 1***: $u_{t}$ 와 $p_t$ 는 Prior에 대한 정보가 없으므로 Intractable 합니다.
- ***Promblem 2***: $p_{1} (x) \approx q(x)$를 만족하는 Probability path 의 가짓수가 너무 많습니다.
- ***Promblem 3***: 원하는 $p_t$ 를 만드는 Vector field $u_{t}$ 에 대해 접근이 불가능 합니다.

## Conditional flow matching

이러한 문제점을 해결하기 위해서 Conditional Flow Matching이 제안되었습니다.

Conditional Flow Matching은 ‘Per-example’ 접근법 (i.e. conditional)을 사용합니다. 즉, 위의 Assumption 1번에 따라 $q(x_{1} )$에서 샘플 1개를 얻고, 이 샘플에 Conditioned 된 *Conditional probability path* $p_{t} (x \| x_{1})$ 와 이러한 $p_{t} (x \| x_{1})$를 만드는 *Conditional vector field* $u_{t} (x \|x _ {1})$ 를 정의할 수 있습니다. 그리고 여기서 $p_{t} (x \| x_{1})$는 아래의 두 조건을 만족한다고 가정합니다.

$$
p_{0} (x | x_{1}) = N(x|0,I), \quad t=0
$$

$$
p_{1} (x | x_{1}) = N(x|x_{1},\sigma ^2 I), \quad t=1
$$

이렇게 되면 $t=1$ 일 때, $p_1$ 은 $x_1$을 중심으로 하는 가우시안 분포형 형태를 띄게 됩니다. 이제 이 conditional probability path 를 $q(x_1)$에 대해서 Marginalizing 하면 우리가 원하는 Marginal probability path $p_t (x)$가 됩니다.

$$
p_t(x) = \int p_t(x|x_1) q(x_1) d x_1
$$

그리고 $t=1$일 때, marginal probability path $p_1$은 실제 데이터 분포 $q$를 가우시안 혼합분포로 근사하게 됩니다.

$$
p_1 (x) = \int p_1(x|x_1) q(x_1) d x_1 \approx q(x)
$$

마찬가지로 conditinoal vector field $u_t(\cdot \|x_1)$으로부터 marginal vector field도 정의할 수 있습니다.

$$
u_t(x) = \int u_t (x|x_1) \frac{p_t(x|x_1)q(x_1)}{p_t(x)} dx_1
$$

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/bc492180-b6d0-4791-bb95-c5fc64d57e6b)

이제 Conditional probability path와 Conditional vector field를 marginalizing 하여 우리가 원하는 probability와 vector field에 접근이 가능해 보입니다. 하지만, 아쉽게도 marginalizing 과정에서 아직 intractable 한 항이 존재하여 단순한 marginalizing 으로 이 문제를 해결하기 어렵습니다.

이에 아래와 같은 새로운 objective CFM을 제안합니다.

$$
\mathcal{L}_ {\text{CFM}} ( \theta ) = \mathbb{E}_{t,q(x_{1}),p_{t} (x|x_{1})} || u_{t} (x|x_{1}) - v_{t} (x; \theta ) || ^2
$$

이 때, $\mathcal{L}_ {\text{FM}} ( \theta )$ 와 $\mathcal{L}_ {\text{CFM}} ( \theta )$  가 $\theta$에 대해 동일한 그래디언트를 갖는다는 것이 증명되었습니다. 따라서, $\mathcal{L}_ {\text{CFM}} ( \theta )$ 을 최적화하는 것이 $\mathcal{L}_ {\text{FM}} ( \theta )$ 을 최적화하는 것과 동치라는 결론에 이릅니다.

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/f245336e-904f-409d-9d13-2ea55f66a91e)

즉 위와 같은 Conditional vector field $u_ {t} (x \| x_1 )$ 을 목표로 모델링해도 원래 문제의 최적점에 도달할 수 있게 됩니다. 

이제 더욱 일반화된 CFM을 위해 Conditional probability path $p_ {t} (x \| x _ {1})$ 와 Conditional vector field $u_ {t} (x \| x _ {1})$에 대한 논의를 시작합니다. 다음과 같은 Conditinoal probability path를 생각해봅시다.

$$
p_t (x |x_1) = \mathcal{N} (x | \mu_t (x_1),\sigma_t(x_1)^2 I)
$$

위 path를 생각해보면, 기본적으로 $x_1$에 conditioned 되어 있으면서 시간에 따라 변하는 평균 $\mu : [0,1] \times \mathbb{R} ^ d \to \mathbb{R} ^ d$  와 표준편차 $\sigma : [0,1] \times \mathbb{R} ^ d \to \mathbb{R} _ {>0 }$   에 의해 결정되는 Gaussian probability path라고 이해할 수 잇습니다. 

그리고 위 path에 2가지 조건을 추가적으로 설정해줍니다.

$$
\mu_0 (x_1) = 0, \quad \sigma_ 0 (x_1)= 1 \quad \therefore p_0 (x|x_1) = p(x) = \mathcal{N} (x|0,I), \quad \text {at} \ \ t= 0
$$

$$
\mu_1 (x_1) = x_1, \quad \sigma_ 1 (x_1)= \sigma_{\text{min}} \quad \therefore p_1 (x|x_1) = \mathcal{N} (x|x_1,\sigma_{\text{min}} ^ 2I), \quad \text {at} \ \ t= 1
$$

즉, 위에서 정의한 Conditional probability path는 표준 가우시안분포에서 시작해서 $x_1$을 중심으로한 가우시안분포로 변하는 Conditional probabiltiy path 입니다.

이 때, 한가지 문제점이 있습니다. 논문에서 말하기를 “특정한 Probability path를 만드는 Vector field는 무수히 많다. 만약 Vector field의 어떤 요소가 변한다고 했을 때, 그 요소가 path에 영향을 미치지 않는다면, 같은 path를 만든다고 하더라도 다른 Vector field을 갖기 때문이다.” 라고 합니다. 그래서 가장 단순한 Vector field를 고르기로 했고, 그에 대응하는 Flow를 아래와 같이 정의하겠습니다. 이 Flow는 $x_1$ 에 conditioned 되어 있습니다.

$$
\psi _ {t, x_1} (x) = \psi _ t (x) = \sigma _ t (x_ 1) x + \mu _ t ( x_ 1)
$$

여기서 $x$가 표준 가우시안 분포에 위치한다고 했을 때, 위 수식의 Flow $\psi_t(x)$는 평균이 $\mu_t(x_1)$이고 표준편차가 $\sigma_t (x_1)$인 정규분포로 변환하는 Affine tranformation으로 이해할 수 있습니다. 
즉, $\psi_t (x)$는 노이즈한 분포 $p_0 ( x\| x_ 1 ) = p(x)$를 $p_t (x \|x_1) = \mathcal{N} (x \| \mu_t (x_1),\sigma_t(x_1)^2 I)$ 로 변환시킵니다. (이제는 Flow가 우리가 생각하는 ‘흐름’에 걸맞는 형식으로 쓰여진 것 같습니다.)

$$
[\psi _ t ] * p(x) = p_t(x|x_1)
$$

그리고 이러한 Flow는 위의 Probability path를 만드는 conditional vector field에 의해 정의됩니다.

$$
\frac{d}{dt} \psi _t (x)= u_t (\psi_t(x)|x_1)
$$

$p_t ( x\|x_1 )$  를 $x_0$ 로 Reparameterizing 하고, 위 식을 이 전에 정의했던 CFM loss 식에 대입하면, 아래의 loss 식을 얻을 수 있습니다. 

$$
\mathcal{L}_ {\text{CFM}} ( \theta ) = \mathbb{E}_{t, x_1 \sim q(x_{1}), x_0 \sim p (x_0)} || v_{t} (\psi_{t,x_1} (x_0); \theta) - \frac{d}{dt} \psi _{t, x_1} (x_0) || ^2
$$

즉, 원래는 우리가 원하는 Vector field $u_t$를 찾기위한 식이었지만,  **Flow의 시간에 대한 미분 값을 예측하는 문제로 바뀌었습니다.**

아래의 증명과 Thm을 통해, 지금까지 정한 가우시안 Probability path와 그에 대응하는 Flow $\psi_ {t, x_ 1 }$를 이용해서 $u_ t ( x \| x _ 1 )$ 을 **Closed form으로 정할 수 있습니다.**

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/fc18b6aa-3a58-4c8b-aa32-776296ed2ef8)

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/09b57bcc-e87c-48ad-879c-829fe0b25bfe)

## Optimal transport conditional vector field

논문에서는 Optimal transport를 위해 아래와 같이 path의 평균과 표준편차를 정의합니다.

$$
\mu _ t (x) = t x_1 , \quad \sigma_t (x) = 1- (1 - \sigma_{\text{min}})t
$$

이렇게 정의하더라도, 이전에 위에서 설정해준 $t$에 따른 path의 조건을 만족하게 됩니다.

$$
\mu_0 (x_1) = 0, \quad \sigma_ 0 (x_1)= 1 \quad \ \text {at} \ \ t= 0
$$

$$
\mu_1 (x_1) = x_1, \quad \sigma_ 1 (x_1)= \sigma_{\text{min}} \quad  \text {at} \ \ t= 1
$$

이러한 Conditional probability path는 아래의 Conditional vector field에 의해 생성됩니다. 그리고 이 Conditional vector fields는 Thm 3에 의해, path의 평균과 표준편차로 표현할 수 있습니다. (Thm 3 식에다가 방금 정의한 path의 평균과 표준편차, 그리고 각각의 미분한 값을 대입해주면 아래의 수식을 얻을 수 있습니다.)

$$
u_t(x|x_1 ) = \frac{x_1 - (1- \sigma_{\text{min}})x}{1-(1-\sigma_{\text{min}})t}
$$

그리고  $u_t (x\|x_1)$에 대응하는 Flow는 정의된 path의 평균과 표준편차로 표현 가능합니다.

$$
\psi_{t,x_1}(x) = \psi_t(x) = (1-(1-\sigma_{\text{min}}) t ) x + tx_1
$$

그리고 최종적인 CFM loss는 위의 Flow에 기반하여 아래와 같이 다시 표현할 수 있습니다.

$$
\mathcal{L}_ {\text{CFM}} ( \theta ) = \mathbb{E}_{t, x_1 \sim q(x_{1}), x_0 \sim p (x_0)} || v_{t} (\psi_{t,x_1} (x_0); \theta) - \frac{d}{dt} \psi _{t, x_1} (x_0) || ^2
$$

$$
\to \mathcal{L}_ {\text{CFM}} ( \theta ) = \mathbb{E}_ {t, x_1 \sim q(x_{1}), x_0 \sim p (x_0)} || v_{t} (\psi_{t,x_1} (x_0); \theta) - (x_1 - (1-\sigma_{\text{min}} ) x_0 ) || ^2
$$

위와 같이 평균과 표준편차를 선형적으로 정의하는 방식은 단순할 뿐만 아니라 실제로도 Optimal 한 transport를 만들게 됩니다. 즉,  $\psi_{t,x_1}(x) = \psi_t(x) = (1-(1-\sigma_{\text{min}} )t ) x + tx_1$ 는 실제로 두 개의 가우시안분포 $p_0(x\|x_1)$와 $p_1(x\|x_1)$ 사이의 **Optimal Transport (OT) displacement map** 입니다.
