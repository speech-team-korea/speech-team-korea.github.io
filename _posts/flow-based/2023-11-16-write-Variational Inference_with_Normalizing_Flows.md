# Variational Inference with Normalizing Flows

# Goal

### Proposing a simple approach for learning highly non-Gaussian posterior densities by learning transformations of simple densities to more complex ones through a **normalizing flow**

# Motivations

![Untitled](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled.png)

**Variational Inference**의 목적은 계산이 어려운 사후확률 분포 $p(z|x)$를 계산이 보다 쉬운 approximate posterior distribution인 $q(z|x)$로 근사하는 것입니다.

우리는 *evidence*인 $𝑝(𝑥)$를 최대화하는 모델, 즉 데이터 $𝑥$의 분포를 잘 설명하는 확률모형을 학습시키고자 합니다.

이때 **approximate posterior distribution**의 선택이 중요합니다.

근데 대부분의 경우, approximate posterior을 샘플링과 효율적으로 계산하기 위해서 단순한 distribution을 사용하게 됩니다. (usually it is a multivariate Gaussian)

Variational inference에 simple distribution을 활용하는 경우, inference 성능이 떨어지고 표현력이 부족하다는 단점이 있습니다.

Richer, more faithful posterior approximations는 더 좋은 performance를 보여줄 수 있습니다.

본 논문에서는 simple distribution 대신 **Normalizing flows**를 활용하여 rich posterior approximations을 만들고 이를 **Variational inference**에 활용하겠다는 것입니다.

# Amortized Variational Inference

Variational inference에서는 사후확률에 근사한 $𝑞(𝑧)$를 만들기 위해 Kullback-Leibler divergence을 활용합니다.

## **KL Divergence (KLD)**

<aside>
💡 **KL divergence**

KL divergence는 non-symmetric하게 두 개의 확률 분포 $P$와 $Q$사이의 difference를 측정하는 방법임

여기서 보통 $P$는 the true posterior distribution, $Q$는 the approximate distribution라는 가정을 사용한다

![Untitled](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled%201.png)

discrete와 continuous 확률 분포에 대해 각각 이렇게 정의할 수 있고,

information entropy로 KL divergence를 직관적으로 이해할 수 있음

**KL divergence as information Gain**

Entropy는 간단히 말하면, the average amount of information이다

또한 불확실성(uncertainty)에 대한 척도라고 할 수 있음

- 예시를 통한 **Entropy**에 대한 직관적인 이해
    
    만약 어떤 가방 안에 빨간 공만 들어있다면, 어떤 공을 꺼내도 빨간 공인 것을 이미 알고 있기 때문에 불확실성은 없다고 할 수 있고 우리가 얻을 information이 적다고 할 수 있음 (Entropy는 0)
    
    만약 빨간 공과 초록 공이 반반 들어 있다면, 어떤 공이 더 자주 관찰될 지 모르기 때문에 불확실성이 가장 크다고 할 수 있고 공을 꺼낼 때 우리가 얻을 information이 많다 (Entropy가 가장 큼)
    
    즉, Entropy는 예측하기 쉬운 일에서 보다, 예측하기 힘든 일에서 더 높다
    
    예측하기 쉬운 일에서는 우리가 얻을 정보량은 적고 예측하기 어려운 일에서는 우리가 새롭게 얻을 정보량은 많다고 이해할 수 있다
    
    그래서 Entropy는 불확실성에 대한 척도이자 information의 average라고 하는 것
    

Entropy는 discrete와 continuous 확률 분포에 대해 아래와 같이 정의 가능한데,

![Untitled](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled%202.png)

- **Entropy**를 the minimum number of bits(or symbol) you need to encode an event drawn from your probability distribution으로도 직관적으로 이해 가능
    
    Entropy를 probability distribution의 사건(or symbol)을 encoding하는데 필요한 최소한의 bit 수로도 이해를 할 수 있다 (see [Shannon's source coding theorem](https://en.wikipedia.org/wiki/Shannon%27s_source_coding_theorem))
    
    예를 들어, fair eight-sided 주사위가 있으면 각 결과는 equi-probable
    
    그래서 주사위의 결과를 encoding하는데 필요한 bit의 수는 평균적으로
    
     $`\sum_1^8 -\frac{1}{8}log_2(\frac{1}{8}) = 3`$ 3개가 됨
    
    만약에 주사위가 a weighted eight-sided 주사위라면, (8이 나올 확률이 다른 수보다 40배 높음)
    
    평균적으로, 주사위의 결과를 encoding하는데 1개의 bit가 더 필요함
    
    (to get close, we would assign "8" to a single bit 0, and others to something like 10, 110, 111 ... using a [prefix code](https://en.wikipedia.org/wiki/Prefix_code))
    
    이러한 관점에서 **Entropy**를 이해한다면, 
    
    theoretical average message length에 최대한 가깝도록 symbol을 확률 분포 $P$에서 가져오고 있다는 가정을 사용하고 있는 것이다
    
    다른 분포인 $Q$의 ideal symbol을 사용한다면 average message length(i. e. **Entropy**)는 어떻게 될까? $P$와 $Q$의 **Cross Entropy**가 될 것임
    
    $𝐻(𝑃,𝑄):=𝐸_𝑃[𝐼_𝑄(𝑋)]=𝐸_𝑃[−log(𝑄(𝑋))]$
    
    당연히 ideal encoding보다 클 것이고 average message length가 증가함
    
    즉, $Q$의 code를 사용하면서 분포 $P$의 message(사건)을 transmit할 때, 더 많은 정보(or bits)가 필요하다는 말이다
    
    (앞에서 true posterior distribution을 approximate distribution으로 추정할 때 생기는 차이와 같은 맥락으로 이해할 수 있음)
    

즉, **KL divergence**를 우리가 확률분포 $P$ 대신 확률분포 $Q$를 사용하면서 확률 분포를 잘못 추정할 때 필요한 **average extra-message length**로 볼 수 있다

![Untitled](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled%203.png)

그래서 만약 theoretic minimal distribution인 $P$가 있을때,

approximation distribution인 Q를 찾아서 KL divergence를 최소화하여

P에 가까운 분포를 추정할 수 있는 것이다.

**Forward and Reverse KL Divergence**

KL divergence에서 주목할 점은 not symmetric하다는 것이다

$𝐷_{𝐾𝐿}(𝑃||𝑄)≠𝐷_{𝐾𝐿}(𝑄||𝑃)$

좌변을 **Forward KL divergence**, 우변을 **Reverse KL divergence**라고 함

1. **Forward KL divergence**
식과 그림에서 볼 수 있듯이, $P$ is large and $Q → 0$ 일때, KL divergence는 발산하게 됨
그래서 적절한 approximate distribution $Q$를 고를 때, 최대한 $P$의 non-zero part를 cover하도록 $Q$를 선택하게 된다 (그림에서 $P$는 multimodal이지만 $Q$는 bell shaped임)
이때 문제점은 Forward KL divergence를 최소화하면서, original distribution에서는 low density를 가지지만, approximate distribution에서는 maximum density를 가진다는 점임 (center of $Q$)
    
    ![Untitled](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled%204.png)
    
2. **Reverse KL divergence**
여기서도 마찬가지로 $P$가 theoretic distribution, $Q$가 approximation
식과 그림에서 볼 수 있듯이, $P$ is small and $Q$ is not small이면 발산하게 됨
추가적으로, $P$ is large면 아무 문제없음
그림을 보면, $Q$의 tail쪽에서 훨씬 빠르게 drop off를 하기 때문에 발산하게 되는 문제가 생긴다
적절한 approximate distribution $Q$를 고를때, $P$와 $Q$의 tail이 비슷한 rate로 drop off하도록 Q를 선택하게 된다
그래서 Reverse KL divergence를 최소화하면서, 그림에서도 볼 수 있듯이,
Q는 분포 P의 mode 중 하나에 잘 matching하고 있고 좋은 approximation을 하고 있다
**Reverse KL divergence를 자주 사용하는 이유** 중 하나이고 다른 수학적인 이유도 있다고 한다
    
    ![Untitled](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled%205.png)
    
    ![Untitled](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled%206.png)
    
</aside>

사후확률 분포 p(z|x)와 $𝑞(𝑧)$ 사이의 KLD를 계산하고, KLD가 줄어드는 쪽으로 $𝑞(𝑧)$를 조금씩 업데이트하는 과정을 반복하면 사후확률을 잘 근사하는 $𝑞^*(𝑧)$를 얻게 될 것이라는 게 Variational inference의 핵심 아이디어입니다. 

본 논문에서는 확률 모델의 주변 가능성(marginal likelihood)을 사용하여 inference를 수행하는 것이 핵심입니다.

이를 위해서는 모델의 잠재적인 변수(latent variables)들을 marginalize하는 것이 필요합니다. 

이 과정은 일반적으로 실행 불가능하기 때문에, 대신 주변 가능성(marginal probability)의 하한(lower bound)을 최적화하는 방법을 사용합니다.

![Untitled](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled%207.png)

모델은 관측값(observed data) $*x*$, 잠재 변수(latent variable) $*z$,* 그리고 모델 파라미터 $*θ*$로 구성됩니다.

$*p_θ(x∣z)*$는 likelihood이고 $*p(z)*$는 잠재 변수(latent variable)에 대한 사전 분포(prior distribution)입니다.

젠슨의 부등식(Jensen's inequality)을 통해 최종 하한(Lower Bound)이 도출되는데, 우리는 이 마지막 부등식의 우변을 **Evidence lower bound, 줄여서 ELBO**라고 부릅니다. 

이 ELBO를 maximize함으로써 likelihood 또한 maximize할 수 있는 것입니다. 

이 ELBO를 좀 더 자세히 살펴보면, 

**첫 번째 항은 approximate posterior $q(z|x)$와 prior $p(z)$ 간의 KL divergence**와 같습니다. 

그렇기 때문에 이는 **approximate posterior**와 **prior**가 최대한 비슷하게 만들어주는 **error**라고 할 수 있고, 이는 VAE가 reconstruction task만 잘 하는 것을 방지하기 때문에 **Regularization error**라고 부릅니다.

**두번째 항은 $p(x|z)$와 $q(z|x)$사이의 negative cross entropy와** 같습니다. 그래서 때문에 이는 Encoder와 Decoder가 Auto-encoder처럼 reconstruction을 잘 할 수 있게 만들어주는 error라고 할 수 있기 때문에 **Reconstruction error**라고 부릅니다.

위 과정을 통해, $*log p_{(\theta)}(x)*$에 대해 lower bound를 얻어냈습니다.

이제 $q(z|x)$를 Normalizing Flow를 통해 복잡한 approximate posterior distribution으로 만들고 **ELBO**를 최대화한다면  $*log p_{(\theta)}(x)*$를 잘 근사했다는 결론을 얻을 수 있습니다.

## **Stochastic Backpropagation**

미니배치(mini-batches)와 확률적 그래디언트 하강법(stochastic gradient descent)을 사용하여, 매우 큰 데이터 셋에 Variational Inference를 적용할 수 있게 합니다. 

이를 위해 두 가지 주요 문제를 해결해야 하는데,

1) expected log-likelihood의 기울기를 계산

2) 계산적으로 실행 가능한 가장 풍부한 근사 사후 분포(Approximate posterior)의 선택

입니다.

1) expected log-likelihood를 위해서 stochastic backpropagation을 진행하는데, two step으로 진행됩니다.

1. **Reparameterization**
    
    $*q_ϕ(z)$가 $N(z|μ,σ^2)$*인 가우시안 분포라면, 표준 정규 분포를 기반 분포로 사용하여 *z*를
    
    $*z = μ + σϵ, ϵ∼N(0,1)*$)으로 재매개변수화할 수 있습니다.
    
    즉, 잠재 변수(latent variable)를 알고있는 분포와 미분 가능한 변환을 사용하여 재매개변수화를 통해 미분 가능한 방식으로 기울기를 계산할 수 있게 하는 것입니다.
    
2. **Backpropagation with Monte Carlo**
    
    ![Untitled](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled%208.png)
    

확률적인 샘플링을 기반으로 기울기의 기대치를 추정합니다. 

기본 분포로부터 여러 샘플을 추출하고, 이들을 통해 log-likelihood의 기울기를 추정하는 것입니다.

# Normalizing Flow

**Normalizing Flow**는 invertible mapping의 series를 통해서 단순한 probability density를 transforming하는 방식으로 complex distribution을 만들 수 있는 방법입니다.

Normalizing Flow를 이해하기 전, 알아야 할 개념 먼저 소개하겠습니다.

## Change of Variable Theorem

어떠한 변수 또는 다변수로 나타낸 식을 다른 변수 또는 다변수로 바꿔 나타내는 것을 **[Change of Variable](https://en.wikipedia.org/wiki/Change_of_variables)**이라고 합니다. 

예시를 들면, 어떠한 랜덤 변수 $x$가 있고, $x$에 대한 **확률 밀도 함수(Probability Density Function)**가 있다고 가정하겠습니다.

$x \sim p(x) \\
z\sim \pi(z)$

여기서 **변수 $z$가 변수 $x$를 잘 표현할 수 있는 잠재 변수(latent variable)**이고 $z$의 확률 밀도 함수 $\pi(z)$를 알고 있다고 가정하고, **invertible한** 일대일 대응 함수 $f$를 사용해서, $x = f(z)$로 표현 가능합니다.

그리고 **함수 $f$는 invertible**하다고 가정했기 때문에, $z=f^{-1}{(x)}$로 표현 가능합니다.

여기서 저희가 하고 싶은 것은 $x$의 unknown 확률 분포인  $p(x)$를 구하는 것입니다.

확률 분포의 정의를 통해,  $\int p(x)dx = \int \pi(z)dz = 1$ 로 표현이 가능하고 

**Change of Variable**을 적용하면, $\int p(x)dx = \int \pi(f^{-1}(x))d f^{-1}(x)$ 로 표현할 수 있습니다. 

이것을 $p(x) = \pi (z)\Bigl|\frac{dz}{dx}\Bigl| = \pi(f^{-1}(x))\Bigl|\frac{df^{-1}}{dx}\Bigl| = \pi(f^{-1}(x))\Bigl|(f^{-1})'(x)\Bigl|$ 와 같이 정리할 수 있고 구하고 싶었던 $**p(x)$를 $z$의 확률밀도함수로 표현할 수 있습니다**.

실제로는 고차원의 변수들을 다루기 때문에, 이를 다변수(Multi-variable)관점으로 다시 표현을 하기 위해 행렬을 사용하면,

$\mathbf{z} \sim \pi(\mathbf{z}), \mathbf{x} = f(\mathbf{z}), \mathbf{z} = f^{-1}(\mathbf{x})\\
p(\mathbf{x})=\pi(\mathbf{z})\Bigl|\text{det}\frac{d\mathbf{z}}{d\mathbf{x}}\Bigl| = \pi(f^{-1}(\mathbf{x}))\Bigl|\text{det}\frac{df^{-1}}{d\mathbf{x}}\Bigl|$

와 같이 정리할 수 있습니다.

## Jacobian Matrix and Determinant

![Untitled](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled%209.png)

**[Jacobian matrix](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)**는 위와같이 벡터 $\text{x}$, $\text{y}$에 대한 일차 편미분을 행렬로 나타낸 것입니다.

즉, 우리가 $n$차원 입력 벡터 $\text{x}$를 $m$차원 출력 벡터 $\text{y}$로 mapping하는 ($\text{y}:\mathbb{R}^n \mapsto \mathbb{R}^m$)함수가 주어지면 이 함수의 모든 1차 편미분 함수 행렬을 이렇게 Jacobian matrix로 간단하게 표현할 수 있습니다.

![Untitled](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled%2010.png)

![Untitled](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled%2011.png)

**[Determinant](https://en.wikipedia.org/wiki/Determinant)**는 행렬을 대표하는 값으로, 정방행렬(Square Matrix)에 어떤 특정한 방법으로 하나의 수를 대응시키는 일종의 함수입니다.

Determinant의 성질은 아래와 같습니다.

(1) 행렬 $A$의 임의의 행에 스칼라 곱을 한 뒤 다른 행에 더해 $B$를 만들었을 때 두 행렬의 determinant는 같다.

(2) 행렬 $A$의 임의의 행을 다른 행과 바꾸어 $B$를 만들었을 때 𝑑𝑒𝑡$B$$=−$𝑑𝑒𝑡$A$

(3) 행렬 $A$의 임의의 행에 스칼라 곱을 해 $B$를 만들었을 때 𝑑𝑒𝑡$B$$=$$k$𝑑𝑒𝑡$A$

(4) **삼각행렬(triangular matrix)**의 행렬식은 주 대각원소들의 곱과 같다.

(5) 행렬 $A$가 **가역(invertible)**임과 𝑑𝑒𝑡$A$$≠0$.

(6) 𝑑𝑒𝑡$𝐴^T$$=$𝑑𝑒𝑡$A$

(7) 𝑑𝑒𝑡$AB$$=$(𝑑𝑒𝑡$A$)(𝑑𝑒𝑡$B$)

이제 **Normalizing Flow**의 동작 과정은 간단하게 표현하면 아래와 같습니다.

![Untitled](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled%2012.png)

($x$는 high dimensional data, $z$는 latent variable)

![Untitled](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled%2013.png)

여기서 $**z$의 확률 분포를 알고 있다면 $x$의 확률 분포를 구할 수 있습니다**.

아까 Change of Variable Theorem 설명에서와 마찬가지로, $x = f(z)$, $z=f^{-1}{(x)}$인 상황에서, 

$p(x)$ $=$ $p(z)detJ$로 $z$의 확률 분포에 scalar값인 determinant를 곱해서 $x$의 확률 분포를 표현할 수 있습니다.

그리고 양변에 $log$를 씌우면,  $log($$p(x))$ $=$ $log(p(z))+log(detJ)$로 표현 가능합니다.

그런데 실제 데이터인  $x$는 보통 매우 복잡한 분포를 가지기 때문에 $x$와 $z$를 하나의 함수로 바로 연결하기는 어렵습니다.

![Untitled](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled%2014.png)

$p(x)=p(z_{1})detJ_{1}$ 

$p(z_{1})=p(z_{2})detJ_{2}$ 

$p(z_{2})=p(z_{3})detJ_{3}$ 

….

$p(z_{n-1})=p(z_{n})detJ_{n}$ 

그래서 이렇게 많은 함수를 통해서 mapping을 해주는 것이고 $p(x) = p(z_{n})\Pi_{n}(detJ_{n})$으로 표현할 수 있습니다.

최종적으로 log likelihood는  $log($$p(x))$ $=$ $log(p(z_{n}))+\Sigma_n log(detJ_{n})$로 표현됩니다.

![Untitled](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled%2015.png)

딥러닝에서 Normalizing Flow를 적용하여 $x$의 확률 분포를 알기 위해서는 **2가지 조건이 꼭 충족**되어야 합니다.

1. **함수 $f$의 역함수가 존재해야 함** (**invertible** $f$)
2. **Jacobian의 Determinant도 계산 가능해야 함**

이 2가지 조건을 고려하여 함수  $f$를 선택해야 하고, 이 flow 함수들을 모델마다 어떻게 구현하는 지가 다릅니다.

![Untitled](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled%2016.png)

![Untitled](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled%2017.png)

## **Finite Flows and Infinitesimal Flows**

Finite Flows와 Infinitesimal Flows은 Normalizing Flows에서 사용되는 두 가지 다른 접근 방식입니다.

# Experiments

![Effect of normalizing flow on two distributions](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled%2018.png)

Effect of normalizing flow on two distributions

초기 Unit Gaussian 분포로부터 복잡한 분포 변환 가능한 normalizing flow의 performance를 보여주고 있음

![Approximating four non-Gaussian 2D distributions. The images represent densities for each energy function in table 1 in the range (−4,4)$^2$. (a) True posterior; (b) Approx posterior using the normalizing flow; (c) Approx posterior using NICE; (d) Summary results comparing KL-divergences between the true and approximated densities for the first 3 cases](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled%2019.png)

Approximating four non-Gaussian 2D distributions. The images represent densities for each energy function in table 1 in the range (−4,4)$^2$. (a) True posterior; (b) Approx posterior using the normalizing flow; (c) Approx posterior using NICE; (d) Summary results comparing KL-divergences between the true and approximated densities for the first 3 cases

4개의 Gaussian이 아닌 분포의 근사 과정을 보여줍니다.

- (a) **True Posterior**: 각각의 경우에 대한 실제 사후 분포를 보여줍니다.
- (b) **Approximate Posterior using Normalizing Flow**: Normalizing Flow를 사용하여 각 posterior distribution을 근사한 결과를 보여줍니다.
- (c) **Approximate Posterior using NICE**: NICE 방법을 사용하여 각 posterior distribution을 근사한 결과를 보여줍니다.
- (d) **Comparison of KL-divergences**: **True Posterior**와 **Approximate Posterior 간**의 Kullback–Leibler divergence을 비교합니다. 실제 분포와 근사 분포 사이의 차이를 정량적으로 평가합니다.

![Untitled](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled%2020.png)

![Untitled](assets/img/2023-11-16-write-Variational Inference_with_Normalizing_Flows/Untitled%2021.png)