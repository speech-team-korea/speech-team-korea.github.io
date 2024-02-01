---
layout: post
title: "[Text-to-Speech] GenerSpeech"
description: >
  GenerSpeech 논문 요약
category: seminar
tags: text-to-speech
author: jh_yun
comments: true
---

# GenerSpeech

<blockquote style="border-left: 2px solid; padding-left: 10px; margin-left: 0;">
Rongjie Huang, Yi Ren, Jinglin Liu, Chenye Cui and Zhou Zhao <br>
"GenerSpeech: Towards Style Transfer for Generalizable Out-of-Domain Text-to-Speech"<br>
Accepted by NeurIPS 2022 <br>
[<a href="https://arxiv.org/abs/2205.07211">Paper</a>] [<a href="https://generspeech.github.io/">Demo</a>] [<a href="https://github.com/Rongjiehuang/GenerSpeech">Code</a>] <br>
</blockquote>


# Goal

- Out-of-domain (OOD) custom voice에 대한 Zero-shot style transfer TTS 모델 제안

# Motivation

- Unseen domain에 대해 Adaptation 성능을 높이는 Generalization 관련 연구
    - Domain-agnostic 부분과 Domain-specific 부분으로 나누어 모델링
    - Generalization 성능을 향상
        - [[Deeper, broader and artier domain generalization](https://arxiv.org/abs/1710.03077), 2017]
        - [[Feature-critic networks for heterogeneous domain generalization](https://arxiv.org/abs/1901.11448), 2019]
- TTS 연구에서의 Zero-shot domain generalization
    - 해당 연구를 목표로 진행한 연구들이 많이 없었고, 낮은 성능을 보임

# Contribution

- Multi-level로 구성된 Style adaptor와 Mix-style layer normalization을 사용하여, Style transfer 작업에서 Generalization 성능 향상
- 기존 Domain generalization에서 제안된 agnostic & specific 접근법을 TTS 모델에 적용
---

# Overview

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/ff29b679-33e0-4d68-a11d-17d11f12dc63)

### Problem formulation

- OOD custom voice, 즉 학습 때 마주하지못한 Reference style (speaker identity, emotion, prosody)에 대하여 높은 품질과 유사도를 가지는 Style transfer 작업을 수행

### Backbone model

- [[FastSpeech 2](https://arxiv.org/abs/2006.04558), 2021] 을 사용했습니다.

### Domain generalization

- 이전 Domain generalization 연구의 접근법을 참고하여 모델을 크게 2가지 부분으로 나누었습니다.
    - Style-agnostic part
        - Linguistic content
    - Style-specific part
        - Speaker identity, Emotion, Prosody

### Methods

- Mix-style layer normalization (MSLN)
    - Linguistic content representation에서 Style information을 제거하는 역할입니다.
- Multi-level style adaptor
    - Style attribute를 잘 modeling 하고 transferring 하기 위해 제안했습니다.
    - Global encoder (Speaker 정보와 Emotion 정보 모델링)
    - Local encoder (3단계로 나누어 Prosody 모델링)
- Flow-based post net
    - [[PortaSpeech](https://arxiv.org/abs/2109.15166), 2021] 을 참조했습니다.
    - 전체적으로 Smoothing 되는 현상을 방지하고, Expressive speech의 detail을 구현하기 위해 사용했습니다.

---

## Content adaptor

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/b25c5248-73c8-406e-97a1-8047dbb4b9a6)

Content adaptor의 목표는 다음과 같습니다.

1. Phonetic sequnce 에서 스타일 정보를 제거한다.
2. Style-agnostic Prosodic variation (SAP) 을 예측한다.

### Mix-Style Layer Normalization (MSLN)

Layer normalization은 네트워크의 최종 예측결과에 많은 영향을 끼칩니다. 이와 동시에 학습가능한 스케일 벡터 $\gamma$와 편향 벡터 $\beta$는 적은 파라미터 수를 갖고 있어서 speaker adaptation에서도 매우 중요한 역할을 해왔습니다.

$$
\text{LN}(x) = \gamma \frac{x-\mu}{\sigma} + \beta
$$

특히 [[AdaSpeech](https://arxiv.org/abs/2103.00993), 2020]과 [[Meta-StyleSpeech](https://arxiv.org/abs/2106.03153), 2021]에서 Style conditional layer normalization을 사용해서 Custom voice에 대한 adaptation 능력을 향상 시켰습니다.

$$
\text{CLN} (x,w) = \gamma(w) \frac{x-\mu}{\sigma} + \beta(w); \quad \gamma(w) = E^\gamma * w, \ \beta(x) = E^\delta * w
$$

이러한 방식은 레퍼런스 오디오에 대해서 좋은 Adaptation 성능을 보이기는 하지만, 결국 source domain과 target domain 간의 차이로 인해 모델의 일반화 성능은 향상되지 않습니다. 

스타일 정보를 잘 분리하고 Style-agnostic representation을 잘 학습하기 위한 간단한 방법을 생각해보면,  **Mismatched 스타일 정보로 Condition을 주는 것**입니다. 즉, Style-consistent representation을 생성하는 것을 방지하고, 모델에게 노이즈를 주어 혼란을 야기하는 것으로 이해할 수 있습니다. 

본 논문에서는 최근 Domain generalization 연구에서 제안된 방식을 참조하여 Mix-Style Layer Normalization (MSLN) 을 디자인했습니다. 이는 학습 과정에서 학습 샘플의 스타일 정보를 Perturbing 하는 방식입니다.

- [[Domain generalization with mixstyle](https://arxiv.org/abs/2104.02008), 2020]
- [[Style normalization and restitution for domain generalization and adaptation](https://arxiv.org/abs/2101.00588), 2021]
- [[Unsupervised domain adaptation by backpropagation](https://arxiv.org/abs/1409.7495), 2015]

$$
\gamma_{\text{mix}} (w) = \lambda \gamma(w) + (1-\lambda) \gamma(\tilde{w}), \quad \beta_{\text{mix}} (w) = \lambda \beta(w) + (1-\lambda) \beta(\tilde{w})
$$

$$
\text{Mix-StyleLN} (x,w) = \gamma _{\text{mix} } (w) \frac{x-\mu}{\sigma} + \beta_{\text{mix}} (w)
$$

- $w= \text{style vector},\quad \tilde{w}= \text{Shuffle} (w), \quad \lambda \in \mathbb{R} ^ B \sim \text{Beta}(\alpha, \alpha), \quad \alpha \in (0, \infty)$

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/02686ca7-b982-4db7-9340-e7b2fe95d8dc)

즉 모델은 Perturbed style로 정규화시킨 인풋 피쳐 $x$를 처리하고, **Generalizable style-invariant content representation**을 학습했습니다. 

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/4bde468f-2a90-42c6-8b55-4c2afa4eff10)

최종적으로는 Content adaptor 의 끝단에 pitch predictor(style-agnostic pitch predictor)를 비치하여 style-agnostic prosodic variation을 만듭니다. 또한 Mix-Style Layer Normalization을 이용함으로써 speaker identity, emotion 과 같은 **Global style attributes로부터** **linguistic content-related variation을 분리하여 모델링**할 수 있었습니다.

---


## Multi-level style adaptor

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/5749bb6a-4b93-443e-8095-d08b7199191f)

Multi-level style adaptor는 크게 Global encoder와 Local encoder 로 나눌 수 있습니다.

### Global Representation

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/cf09fd63-92dd-4006-8e2a-d550031fec7d)

Global encoder는 스피커 정보, 감정 정보 등 Global acoustic style 정보를 잡아내야합니다. 따라서 본 논문에서는 Global encoder로 [[wav2vec 2.0](https://arxiv.org/abs/2006.11477), 2020] 를 사용했습니다. 실제로는 스피커 분류와 감정 분류 태스크에서 모델을 파인튜닝 시키기 위해, wav2vec 2.0 모델의 마지막에 Average pooling 레이어와 FC 레이어를 추가하여 Global encoder를 구성했습니다.

파인튜닝 진행 시, 모델의 손실함수는 Additive Margin softmax [[AM-softmax](https://arxiv.org/abs/1801.05599), 2018] 를 사용했습니다. 결과적으로 파인튜닝된 wav2vec 2.0은 스피커 특성과 감정 특성을 모델링한 Discriminatrive gloabl representation $\mathcal{G} _ s$와  $\mathcal{G} _ e$ 을 만들게 됩니다.

### Local Representation

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/5472ba2d-27a5-4886-b6ac-4f79714c9374)

Global encoder만으로는 Fine-grained prosodic detail을 모델링하기 어렵습니다. 본 논문에서는 Fine-grained prosody 정보를 모델링하기 위해 Frame, Phoneme, Word-level로 나누어서 어쿠스틱 컨디션을 모델링하는 방식을 사용했습니다.

각각의 Multi-level local style encoder는 아래와 같은 공통적인 특징을 공유하고 있습니다.

- 인풋 시퀀스는 CNN을 통과합니다.
- Vector quantization bottleneck [[Neural discrete representation learning](https://arxiv.org/abs/1711.00937), 2017]을 통과해서 non-prosodic (non-style)한 정보를 제거합니다.
    - Vector quantization: $e \in \mathbb{R} ^{K\times D}$ ( $D$차원 임베딩 벡터, $K$-way categorical )
    - 손실함수 ( $z_e(x)$는 Vector quantization block의 아웃풋 )
    
    $$
    \mathcal{L}_c = || z_e (x) - \text{sg} [e]||^ 2 _2
    $$
    

**Frame level**

- Optional pooling 레이어를 적용하지 않아, frame-level latent representation $S_ p$ 을 얻습니다.

**Phoeneme level**

- 추가 입력으로 Phoneme boundary를 받고 Pooling 레이어를 적용하여, phoneme-level latent representation $S _ p$ 을 얻습니다.

**Word level**

- 추가 입력으로 Word boundary를 받고 Pooling 레이어를 적용하여, word-level latent representation $S _ w$ 을 얻습니다.

**Style-To-Content Alignment Layer**

다양한 길이의 Local style representation $\mathcal{S}_u$, $\mathcal{S}_p$, $\mathcal{S}_w$와 Phonetic representation $\mathcal{H}_c$를 Align해주기 위해 Style-To-Content Alignment Layer 를 제안했습니다. 이를 통해서 Style과 Content 사이의 Alignment를 학습하게 됩니다.

Scaled dot-product attention 을 활용하였고, Frame-level representation $S_u$ 을 아래 예시로 표현할 수 있습니다. $\mathcal{H}_c$가 Query로 사용되고, $\mathcal{S}_u$가 Key, Value로 사용됩니다.

$$
\text{Attention} (Q,K,V) = 
\text{Attention} (\mathcal{H}_c,\mathcal{S}_u,\mathcal{S}_u) =
\text{Softmax} (\frac{\mathcal{H}_c \mathcal{S}_u ^ T}{\sqrt{d}}) \mathcal{S}_u 
$$

---

### Flow-based post net

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/1372530e-0f09-4321-b2d3-756e0dbc0fe3)

학습과정동안, Flow post net은 합성된 Mel-spectrogram을 Gaussian prior 분포로 변환하고 데이터의 exact log-likelihood 를 계산합니다.

인퍼런스 과정동안, Prior 분포에서 샘플링한 Latent variables 을 Post-net에 역으로 건네주어 Expressive한 Mel-spectrogram 을 만듭니다.

---


### Pre-training

- Global style encoder wav2vec 2.0을 AM softmax 함수를 사용한 다운스트림 태스크에 파인튜닝합니다.

### Training

- GenerSpeech는 학습과정동안 레퍼런스 스피치와 타겟 스피치가 동일합니다.
- 최종 Loss는 다음과 같습니다.
    - $L _ {\text{dur}}$ : Duration prediction loss
        - GT duration과 Predicted duration 사이의 MSE
    - $L _ {\text{Mel}}$ : Mel reconstruction loss
        - GT Mel과 트랜스포머 Decoder가 합성한 Mel 사이의 MAE
    - $L _ {\text{p}}$ : Pitch reconstruction loss
        - GT pitch spectrogram과 SAP와 SSP가 예측한 Pitch spectrogram 사이의 MSE
    - $L _ {\text{pn}}$ : Post-net의 Negative log-likelihood
    - $L _ {\text{c}}$ : Commit loss
        - Vector quantizatioin
        - ${L} _ c = \|\| z_e (x) - \text{sg} [e]\|\|^ 2 _2$

### Inference

1. 텍스트 인코더가 Phoneme sequence를 인코딩하고 Duration에 맞게 복제하여 $\mathcal{H}_c$ 를 생성합니다.
2. 이어서 Style Agnostic Pitch (SAP) 가 Linguistic content 피쳐를 만듭니다. 이는 타겟 목소리 스타일과는 무관한 정보입니다.
3. Reference 음성 샘플이 주어지면, Forced alignment를 통해서 Phoneme-boundary와 Word-boundary를 추출합니다.
4. 음성 샘플과 앞서 구한 boundary를 이용해서 Global style 정보와 local style 정보를 추출합니다.
5. Style Specific Pitch (SSP) 가 Reference 음성 샘플에 잘어울리는 pitch 를 예측합니다.
6. Mel-decoder는 Coarse-grained Mel-spectrogram $\tilde{M}$을 만들고, 이를 Flow post net을 통과시켜 Fine-grained Mel-spectrogram $M$을 형성합니다.

# Experiment

### Dataset

- Pre-training : IEMOCAP (12시간), VoxCeleb1(100,000개 발화, 1,251명)
- Training : LibriTTS (586시간, 2456명), ESD database (13시간, 10명, 화남, 행복, 중립)
- Test : VCTK (108명), ESD database(놀람, 슬픔)

### Training and evaluation

- 100,000 step in 사전학습
- GenerSpeech 200,000 학습
- 보코더 : HiFi-GAN

## Result

### Parallel Style Transfer

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/221f38ef-6738-41ed-a5a8-cc7cc4d5c126)

### Non-parallel style transfer

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/2b26ffe9-caa0-4298-8546-360b34b8b9b1)

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/0c856645-d362-4784-88ab-ea7372c6cffd)
