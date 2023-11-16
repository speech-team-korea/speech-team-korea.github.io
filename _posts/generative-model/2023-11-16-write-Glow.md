---
layout: post
title: "Glow: Generative Flow with Invertible 1x1 Convolutions"
description: >
  아직 미완성입니다..!
category: seminar
tags: generative-model
author: jh_cha
comments: true
---

# Glow: Generative Flow with Invertible 1×1 Convolutions

# Goal

### Proposing Glow, a simple type of generative flow using an invertible 1 x 1 convolution

# Motivations

본 논문에서는 먼저 machine learning에서의 2가지 major problem들에 대해서 언급하고 있습니다.

### Data-efficiency

- 인간처럼 few datapoints만으로 학습하는 능력이 부족함

### Generalization

- Task나 Context의 변화에 robustness를 가지지 못함

이러한 limitation들을 기존의 generative model들은 사람의 supervision이나 labeling없이 input 데이터의 의미있는 feature들을 학습하는 방식으로 개선하고 있습니다.

보통 Generative model은 실제 데이터의 분포를 모델링하고, 모델링한 분포에서 sampling을 통해 새로운 데이터를 생성(Generate)한다고 볼 수 있습니다.

## Types of Generative Models

### GANs [Goodfellow et al., 2014]
![출처 : [https://lilianweng.github.io/posts/2018-10-13-flow-models/](https://lilianweng.github.io/posts/2018-10-13-flow-models/)](assets/img/2023-11-16-write-Glow/fig.png)

출처 : [https://lilianweng.github.io/posts/2018-10-13-flow-models/](https://lilianweng.github.io/posts/2018-10-13-flow-models/)

**장점**

- Synthesize large and realistic images

**단점**

- No encoder to infer the latents
- Datapoints can not be directly represented in a latent space
- Difficulty of optimization and assessing overfitting and generalization

GAN은 암시적(implicitly)으로 실제 학습 데이터의 분포를 학습합니다.

### VAEs [Kingma and Welling, 2013, 2018]

![출처 : [https://lilianweng.github.io/posts/2018-10-13-flow-models/](https://lilianweng.github.io/posts/2018-10-13-flow-models/)](assets/img/2023-11-16-write-Glow/fig1.png)

출처 : [https://lilianweng.github.io/posts/2018-10-13-flow-models/](https://lilianweng.github.io/posts/2018-10-13-flow-models/)

**장점**

- Parallelizability in training and synthesis

**단점**

- Challenging to optimize
- Infer only approximately the value of the latent variables

**VAE(Various AutoEncoder)**는 **likelihood-based method**입니다.

여기서 likelihood는 데이터가 특정 분포로부터 만들어졌을 확률을 말합니다. 

즉, likelihood는 생성 모델이 모델링하고 있는 특정 확률 분포가 학습 데이터를 얼마나 잘 설명하고 있는가를 말한다고 볼 수 있습니다. 모델은 학습을 통해 이 likelihood를 최대화하는 파라미터를 갖게 되는 것이 목표입니다.

VAE에서는 계산이 어려운(intractable) 사후확률 분포(posterior distribution)를 보다 쉬운 분포로 근사하는데, Evidence Lower Bound(ELBO)를 최대화하는 방식으로 데이터의 log-likelihood를 암시적으로(implicitly) 학습합니다.

### Autoregressive models [Van den Oord et al, 2016]

![Visualization of a stack of *dilated* causal convolutional layers in WaveNet [Van den Oord et al, 2016]](assets/img/2023-11-16-write-Glow/fig2.png)

Visualization of a stack of *dilated* causal convolutional layers in WaveNet [Van den Oord et al, 2016]

**장점**

- Simplicity

**단점**

- Limited parallelizability in synthesis
- Troublesome for large images or video

**Autoregressive model**도 **likelihood-based method**입니다.

WaveNet 모델을 예시로 얘기를 하면, $T$개의 오디오 샘플 $x$들로 이루어진 한 Waveform $\text{x}$가 있다고 할 때 ($\text{x} = \{x_{1}, ... ,x_{T}\}$), Waveform $\text{x}$의 joint probability를 아래와 같이 conditional probabilities의 곱으로 표현할 수 있습니다.

$p(\text{x}
) = \Pi^T_{t=1}p(x_{t}|x_{1},...,x_{t-1})$ 

WaveNet은 매 time step $t$마다 모든 이전 time step에서 생성한 샘플들을 조건(conditioned)으로 새로운 오디오 샘플 $x_{t}$를 생성합니다. 

$x_{t}$의 categorical distribution을 output으로 내보내게 되는 것이고 모델의 파라미터에 대하여 데이터의 log-likelihood를 최대화하는 방향으로 최적화(optimization)를 하게 됩니다. 즉, 모델이 주어진 데이터 셋을 최대한 잘 설명하는 파라미터를 가지도록 학습을 한다는 뜻입니다.

위의 예시 그림에서 볼 수 있듯이, Synthesis때 연속적으로 sample을 생성하기 때문에 synthesis의 computational length는 데이터의 dimensionality에 비례하게 됩니다.

따라서, Autoregressive model은 이미지나 비디오와 같이 데이터의 dimensionality가 큰 경우 계산 비용이 상당히 커지는 문제가 생길 수 있습니다.

### Flow-based generative models

![출처 : [https://lilianweng.github.io/posts/2018-10-13-flow-models/](https://lilianweng.github.io/posts/2018-10-13-flow-models/)](assets/img/2023-11-16-write-Glow/fig3.png)

출처 : [https://lilianweng.github.io/posts/2018-10-13-flow-models/](https://lilianweng.github.io/posts/2018-10-13-flow-models/)

**Flow-based model**도 **likelihood-based method**에 속합니다.

Flow-based model은 GAN이나 VAE와 다르게 명시적으로(explicitly) 데이터의 확률 분포를 학습합니다.

이 논문에서 제안하고 있는 방법인 Flow-based generative model은 **NICE**[Dinh et al., 2014]에서 제안되었고, **RealNVP**[Dinh et al., 2016]에서 더 좋은 성능을 보여줬습니다.

본 논문에서는 Flow-based의 장점을 다음과 같이 언급하고 있습니다.

- **Exact latent-variable inference and log-likelihood evaluation**
    
    Flow-based 모델은 VAE와 달리, 데이터의 log-likelihood를 근사적(approximately)으로 추정하지 않고 정확한(exactly) 분포를 얻을 수 있습니다.
    
    그리고 VAE에서 Evidence Lower Bound(ELBO)를 최대화하는 방식으로 데이터의 log-likelihood를 암시적으로(implicitly) 최적화(optimization)하지만, Flow-based 모델은 정확하게 데이터의 log-likelihood의 최적화(optimization)가 가능합니다.
    
- **Efficient to parallelize for both inference and synthesis**
    
    Flow-based 모델은 Autoregressive model과 달리, 병렬화(parallelization)을 하기에 효율적인 모델입니다.
    
- **Useful latent space for downstream tasks**
    
    Flow-based 모델은 Autoregressive model이나 GAN과 달리, datapoint들이 latent space상에서 표현(directly represented)이 가능합니다.
    
    그래서 latent space상에서 데이터 간의 interpolation이나 modification을 통해 다양한 application을 가능하게 해줍니다.
    
- **Significant potential for memory savings**
    
    Reversible neural networks에서 gradient를 계산하는 비용은 depth에 따라 linear하게 증가하는 것이 아니라 일정한 memory(that is constant)를 가진다고 합니다.
    
    [RevNet](https://proceedings.neurips.cc/paper_files/paper/2017/hash/f9be311e65d81a9ad8150a60844bb94c-Abstract.html) paper[Gomez et al., 2017]에 잘 설명되어 있다고 하니 필요하다면 참고하면 될 것 같습니다.
    

Flow-based Generative Model을 설명하기에 앞서, 알고 있어야 할 개념인 the change of variable rule과 Jacobian determinant에 대해서 먼저 소개하겠습니다.

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

![Untitled](assets/img/2023-11-16-write-Glow/fig4.png)

**[Jacobian matrix](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)**는 위와같이 벡터 $\text{x}$, $\text{y}$에 대한 일차 편미분을 행렬로 나타낸 것입니다.

즉, 우리가 $n$차원 입력 벡터 $\text{x}$를 $m$차원 출력 벡터 $\text{y}$로 mapping하는 ($\text{y}:\mathbb{R}^n \mapsto \mathbb{R}^m$)함수가 주어지면 이 함수의 모든 1차 편미분 함수 행렬을 이렇게 Jacobian matrix로 간단하게 표현할 수 있습니다.

![Untitled](assets/img/2023-11-16-write-Glow/fig5.png)

![Untitled](assets/img/2023-11-16-write-Glow/fig6.png)

**[Determinant](https://en.wikipedia.org/wiki/Determinant)**는 행렬을 대표하는 값으로, 정방행렬(Square Matrix)에 어떤 특정한 방법으로 하나의 수를 대응시키는 일종의 함수입니다.

Determinant의 성질은 아래와 같습니다.

(1) 행렬 $A$의 임의의 행에 스칼라 곱을 한 뒤 다른 행에 더해 $B$를 만들었을 때 두 행렬의 determinant는 같다.

(2) 행렬 $A$의 임의의 행을 다른 행과 바꾸어 $B$를 만들었을 때 𝑑𝑒𝑡$B$$=−$𝑑𝑒𝑡$A$

(3) 행렬 $A$의 임의의 행에 스칼라 곱을 해 $B$를 만들었을 때 𝑑𝑒𝑡$B$$=$$k$𝑑𝑒𝑡$A$

(4) **삼각행렬(triangular matrix)**의 행렬식은 주 대각원소들의 곱과 같다.

(5) 행렬 $A$가 **가역(invertible)**임과 𝑑𝑒𝑡$A$$≠0$.

(6) 𝑑𝑒𝑡$𝐴^T$$=$𝑑𝑒𝑡$A$

(7) 𝑑𝑒𝑡$AB$$=$(𝑑𝑒𝑡$A$)(𝑑𝑒𝑡$B$)

# Flow-based Generative Models

## Normalizing Flows

**Normalizing Flow**의 동작 과정은 간단하게 표현하면 아래와 같습니다.

![Untitled](assets/img/2023-11-16-write-Glow/fig7.png)

($x$는 high dimensional data, $z$는 latent variable)

![Untitled](assets/img/2023-11-16-write-Glow/fig8.png)

여기서 $**z$의 확률 분포를 알고 있다면 $x$의 확률 분포를 구할 수 있습니다**.

아까 Change of Variable Theorem 설명에서와 마찬가지로, $x = f(z)$, $z=f^{-1}{(x)}$인 상황에서, 

$p(x)$ $=$ $p(z)detJ$로 $z$의 확률 분포에 scalar값인 determinant를 곱해서 $x$의 확률 분포를 표현할 수 있습니다.

그리고 양변에 $log$를 씌우면,  $log($$p(x))$ $=$ $log(p(z))+log(detJ)$로 표현 가능합니다.

그런데 실제 데이터인  $x$는 보통 매우 복잡한 분포를 가지기 때문에 $x$와 $z$를 하나의 함수로 바로 연결하기는 어렵습니다.

![Untitled](assets/img/2023-11-16-write-Glow/fig9.png)

$p(x)=p(z_{1})detJ_{1}$ 

$p(z_{1})=p(z_{2})detJ_{2}$ 

$p(z_{2})=p(z_{3})detJ_{3}$ 

….

$p(z_{n-1})=p(z_{n})detJ_{n}$ 

그래서 이렇게 많은 함수를 통해서 mapping을 해주는 것이고 $p(x) = p(z_{n})\Pi_{n}(detJ_{n})$으로 표현할 수 있습니다.

최종적으로 log likelihood는  $log($$p(x))$ $=$ $log(p(z_{n}))+\Sigma_n log(detJ_{n})$로 표현됩니다.

![Untitled](assets/img/2023-11-16-write-Glow/fig10.png)

딥러닝에서 Normalizing Flow를 적용하여 $x$의 확률 분포를 알기 위해서는 **2가지 조건이 꼭 충족**되어야 합니다.

1. **함수 $f$의 역함수가 존재해야 함** (**invertible** $f$)
2. **Jacobian의 Determinant도 계산 가능해야 함**

이 2가지 조건을 고려하여 함수  $f$를 선택해야 하고, 이 flow 함수들을 모델마다 어떻게 구현하는 지가 다릅니다.

# Proposed Generative Flow

![One step of our flow in Glow paper](assets/img/2023-11-16-write-Glow/fig11.png)

One step of our flow in Glow paper

본 논문에서는 NICE[Dinh et al., 2014]와 RealNVP[Dinh et al., 2016]의 flow를 기반으로 새로운 generative flow인 Glow 모델을 제안하고 있습니다.

위의 그림에서 볼 수 있듯이, one flow step은 **actnorm**, **invertible 1 x 1 convolution**, **affine coupling layer**로 이루어져 있습니다.

![Multi-scale architecture (Dinh et al., 2016)](assets/img/2023-11-16-write-Glow/fig12.png)

Multi-scale architecture (Dinh et al., 2016)

이렇게 구성된 flow step은 위의 그림에서 볼 수 있듯이, RealNVP에서 제안된 multi-scale architecture 구조에서 활용됩니다

이 구조는 squeezing operation을 통해 구현할 수 있습니다. 

![Untitled](assets/img/2023-11-16-write-Glow/fig13.png)

이미지를 sub-square로 reshape하는데(4X4X1 ->2X2X4), 이 방법은 spatial size를 채널의 수로 효과적으로 trade하는 것입니다.

그래서 전체적인 과정을 보면, x라는 이미지를 입력값으로 넣고 squeeze를 통해 공간해상도를 줄입니다.

그리고 flow step을 K번 반복한 후, split을 진행하고 이를 L-1번 반복하는 multi-scale 구조를 가집니다. 

결론적으로 flow step K번을 L번 반복하는 것이 됩니다.

## Actnorm: scale and bias layer with data dependent initialization

![Untitled](assets/img/2023-11-16-write-Glow/fig14.png)

Actnorm은 Activation Output에 **Affine Transformation**을 적용하는것을 의미합니다. 결과적으로 Batch Normalization과 유사한 역할을 수행하는 것인데 invertible하고 Log Determinant 계산이 쉬워 Normalizing Flow에 적용하기 수월합니다.

**RealNVP**에서는 batch normalization을 활용했지만, 본 논문에서는 **activation normalization** (**actnorm**) layer를 대신 제안하고 있습니다.

기존에 사용했던 Batch normalization에서는 mini-batch size가 작을 때 activation의 noise variation이 커져서 batch normalization의 성능이 떨어진다는 단점이 있었습니다.

큰 이미지를 처리하기 위해서는 GPU 성능 한계때문에 mini-batch size를 줄여야 한다는 문제점이 있었고 이를 해결하기 위해 본 논문에서는 각 channel마다 scale과 bias parameter를 이용해서 **affine transformation of the activation**을 수행합니다.

이러한 과정을 거쳐서 초기의 mini-batch 데이터에 대해 channel 별로 **zero mean**과 **unit variation**을 가지도록 초기화를 시킵니다.

이를 보통 **data dependent initialization[Salimans and Kingma, 2016]**이라고 합니다.

initialization후에 scale과 bias parameter는 데이터와 independent한 regular trainable parameter로서 다루어집니다.

## **Invertible 1 x 1 convolution**

![Untitled](assets/img/2023-11-16-write-Glow/fig15.png)

Invertible 1 x 1 convolution은 Coupling Layer의 Input을 Split 하는 용도로 사용됩니다. 

1×1 Convolution을 사용함으로써 계산 복잡도가 크지 않으면서도 쉽게 Log Determinant를 구할 수 있습니다

RealNVP에서는 channel의 순서를 반대로 바꾸는 permutation이 포함된 flow를 제안했지만,

본 논문에서는 앞서 제안된 fixed permutation 대신에 (learned) invertible 1 x 1 convolution을 제안하고 있습니다.

## Affine Coupling Layers

![NICE의 Coupling Layer](assets/img/2023-11-16-write-Glow/fig16.png)

NICE의 Coupling Layer

![RealNVP의 Affine Transformation](assets/img/2023-11-16-write-Glow/fig17.png)

RealNVP의 Affine Transformation

Coupling Layer란 Input을 둘로 나눠 구성한 Matrix인데, RealNVP에서 제안된 **Affine Coupling layer**를 본 논문에서도 활용하고 있습니다.

![Glow의 affine coupling layer](assets/img/2023-11-16-write-Glow/fig18.png)

Glow의 affine coupling layer

다만, 본 논문에서는 기존의 방식과 세 가지가 포인트가 다릅니다.

### Zero initialization

학습을 시작할 때 Affine Coupling Layer가 Identity Function이 되도록 마지막 convolution layer를 Zero로 초기화합니다. 이는 매우 깊은 모델에서의 학습을 용이하게 합니다.

### Split and concatenation

채널을 나누는 방식으로 NICE와 같이 channel을 따라 절반을 나누는 방법이 있고 RealNVP처럼 checkerboard pattern으로 나누는 방법이 있습니다. Glow는 NICE 방식을 사용합니다.

본 논문에서는 channel dimension에서만 split을 수행하면서 전체 아키텍쳐를 간소화하고 있습니다.

### Permutation

NICE는 Channel의 순서를 반대로(reversely) 바꾸는 방법을 사용했습니다. 

RealNVP는 Fixed Random Permutation 방식을 사용했습니다. 

본 논문의 Glow는 1×1 convolution 방식을 사용하고 있습니다.

![The three main components of our proposed flow, their reverses, and their log-determinants in Glow paper](assets/img/2023-11-16-write-Glow/fig19.png)

The three main components of our proposed flow, their reverses, and their log-determinants in Glow paper

# Experiments

먼저 Quantitative Experiment 결과들을 보면 아래와 같습니다.

## **Gains using invertible** 1 × 1 **Convolution**

![Comparison of the three variants - a reversing operation as described in the RealNVP, a fixed random permutation, and our proposed invertible 1 × 1 convolution, with additive (left) versus affine (right) coupling layers ](assets/img/2023-11-16-write-Glow/fig20.png)

Comparison of the three variants - a reversing operation as described in the RealNVP, a fixed random permutation, and our proposed invertible 1 × 1 convolution, with additive (left) versus affine (right) coupling layers 

Reverse, Shuffle, 1×1 convolution 방식의 Permutation을 실험한 결과입니다. 

그래프를 보면 다양한 Permutation 방법들 중 1×1 convolution 방식의 성능이 가장 좋은 모습입니다.

## **Comparison with RealNVP on standard benchmarks**

![Untitled](assets/img/2023-11-16-write-Glow/fig21.png)

다양한 데이터셋에 대한 RealNVP 모델과의 비교입니다. Bits per Dimension 성능을 측정했습니다.

GLOW는 모든 데이터셋에서 RealNVP 보다 좋은 성능을 보여주고 있습니다.

이제 Qualitative Experiment 결과들을 보면 아래와 같습니다.

Glow 모델이 **고해상도(high resolutions)까지 scaling이 가능한지**, **현실적인 샘플들을 만드는지**, **meaningful latent space를 만들고 있는지**를 중점으로 봐야 합니다.

### Synthesis

![ Random samples from the model, with temperature 0.7](assets/img/2023-11-16-write-Glow/fig22.png)

 Random samples from the model, with temperature 0.7

Glow 모델에서 랜덤하게 샘플링한 이미지들인데, 이미지의 품질이 꽤 좋은 것을 확인할 수 있습니다.

### Interpolation

![Linear interpolation in latent space between real images](assets/img/2023-11-16-write-Glow/fig23.png)

Linear interpolation in latent space between real images

latent space상에서 2개의 실제 데이터의 encoding한 벡터간 linear interpolation을 했고, 그 interpolation latent로부터 샘플링한 이미지들입니다.

꽤 현실적으로 실제 두 이미지 사이의 중간 얼굴 이미지를 보여주고 있는 것을 보아, Glow 모델이 meaningful latent space를 잘 만들고 있는 것을 확인할 수 있습니다.

### Semantic Manipulation of attributes of a face

![Smiling](assets/img/2023-11-16-write-Glow/fig24.png)

Smiling

![Pale Skin](assets/img/2023-11-16-write-Glow/fig25.png)

Pale Skin

![Blond Hair](assets/img/2023-11-16-write-Glow/fig26.png)

Blond Hair

![Young](assets/img/2023-11-16-write-Glow/fig27.png)

Young

![Narrow Eyes](assets/img/2023-11-16-write-Glow/fig28.png)

Narrow Eyes

![Male](assets/img/2023-11-16-write-Glow/fig29.png)

Male

다음은 미소, 피부색 등의 Semantic Maniplation 실험입니다. 

CelebA 데이터셋에는 Smiling, Blond Hair 등의 Label이 존재합니다. 

일단 Labeling 없이 Normalizing Flow를 학습한 다음 True/False Label 데이터들의 Average(z)를 각각 구합니다. 그 다음 두 개의 점에서 Interpolation을 수행하며 Image를 생성해냅니다.

### Effect of model depth

![Samples from shallow model on left vs deep model on right. Shallow model has L = 4
levels, while deep model has L = 6 levels](assets/img/2023-11-16-write-Glow/fig30.png)

Samples from shallow model on left vs deep model on right. Shallow model has L = 4
levels, while deep model has L = 6 levels

모델이 적당히 깊을때 Long Range Dependency를 잘 학습하는 모습을 볼 수 있습니다.

### Effect of temperature

![Effect of change of temperature. From left to right, samples obtained at temperatures
0, 0.25, 0.6, 0.7, 0.8, 0.9, 1.0](assets/img/2023-11-16-write-Glow/fig31.png)

Effect of change of temperature. From left to right, samples obtained at temperatures
0, 0.25, 0.6, 0.7, 0.8, 0.9, 1.0

Temperature가 높으면 noisy한 모습을 확인할 수 있습니다. 논문에서는 0.7일때 가장 적절한 결과를 얻었다고 언급하고 있습니다.

# References

D. P. Kingma and P. Dhariwal ,"[Glow](https://proceedings.neurips.cc/paper_files/paper/2018/hash/d139db6a236200b21cc7f752979132d0-Abstract.html): Generative flow with invertible 1x1 convolutions,” *NeurIPS*, 2018.

A. Oord et al., “[Wavenet](https://arxiv.org/abs/1609.03499): A generative model for raw audio,” *arXiv*, 2016

Dinh, Laurent, David Krueger, and Yoshua Bengio ,"[Nice](https://arxiv.org/abs/1410.8516): Non-linear independent components estimation,” *arXiv*, 2014.

Dinh, Laurent, Jascha Sohl-Dickstein, and Samy Bengio ,“[DENSITY ESTIMATION USING REAL NVP](https://arxiv.org/abs/1605.08803)”, *NeurlPS*, 2016.

Gomez et al., "[The reversible residual network: Backpropagation without storing activations](https://proceedings.neurips.cc/paper_files/paper/2017/hash/f9be311e65d81a9ad8150a60844bb94c-Abstract.html)," *NeurlPS*, 2017.