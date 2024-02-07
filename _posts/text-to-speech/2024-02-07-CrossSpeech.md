---
layout: post
title: "[Text-to-Speech] CrossSpeech"
description: >
  CrossSpeech 논문 요약
category: seminar
tags: text-to-speech
author: jh_yun
comments: true
---

# CrossSpeech

<blockquote style="border-left: 2px solid; padding-left: 10px; margin-left: 0;">
Ji-Hoon Kim, Hong-Sun Yang, Yoon-Cheol Ju, Il-Hwan Kim and Byeong-Yeol Kim <br>
"CrossSpeech: Speaker-Independent Acoustic Representation for Cross-Lingual Speech Synthesis"<br>
Accepted by ICASSP 2023 <br>
[<a href="https://arxiv.org/abs/2302.14370">Paper</a>] [<a href="https://lism13.github.io/demo/CrossSpeech/">Demo</a>] [Code X] <br>
</blockquote>



# Goal

- Acoustic feature space 안에서 Speaker와 Language 정보를 분리시켜 모델링하는 Cross-lingual TTS 모델 제안

# Motivation

- 기존 Cross-lingual TTS 모델들의 단점
    - 타겟 스피커와 언어가 분리가 잘 되지 않아, 자연스러운 발화합성이 어려움
- 기존 Domain generalization 연구들
    - [Generspeech, 2022]
    - [Domain generalization with mixstyle, 2021]

# Contribution

- 기존의 Cross-lingual TTS SOTA 모델과 유사한 성능을 보임
- Acoustic feature space 안에서 Speaker 와 Language 를 따로 모델링하여 성능을 높임

---

# Model architecture

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/e5040499-0f22-425a-a224-2fa2e73f6cf4)

- **Speaker independent generator (SIG)** 모듈과 **Speaker dependent generator (SDG)** 모듈로 구성
    - 음성 생성 파이프라인을 independent-dependent 부분으로 나누면, Disentanglement 를 더욱 잘 수행할 수 있음
    - 각각은 Speaker independent representation $h_{si}$과 dependent representation $h_{sd}$을 생성
        - $h_{si}$ : 특정 스피커 분포에 편향을 가지지 않음
        - $h_{sd}$ : 특정 스피커의 특징 정보를 포함함
    - 마지막에 Residual connection을 사용하여 발음 정확도를 높이며, Speaker와 Language의 Disentanglement가 더욱 잘 됨

- Aligner로 [One TTS alignment to rule them all, 2022] 을 사용
    - 효율적인 학습이 가능하며, 각 언어마다 사전에 Aligner를 준비해야 된다는 의존성을 제거

- Backbone model로 [FastPitch, 2021] 을 사용

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/f7c8c48c-14d2-4204-9e7a-d47a23682c53)

---

## Speaker Independent Generator

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/4ca47403-3854-4988-9af7-670941307889)

Reference 음성 샘플과 무관한 representation을 모델링하기 위해, M-DSLN과 Speaker generalization loss 그리고 Speaker independent pitch (SIP) predictor 를 활용했습니다. 

### Mix-Dynamic Style Layer Normalization (M-DSLN)

[PVAE-TTS, 2022]에서 제안한 Dynamic Style Layer Normalization (DSLN)과 [GenerSpeech, 2022]에서 제안한 Mix Style Layer Normalization (MSLN)을 결합했습니다. 

$$
\text{DSLN} ( h, e_s )  = W (e_s) \odot \text{LN} ( h ) + b ( e_s)
$$

- $\odot$: $1\text{D}$ convolution, $e_s$: 스피커 임베딩, $h$: hidden representation, $W(\cdot), b(\cdot)$: Linear layer

$$
W_{mix} ( e_s) = \gamma W (e_s) + (1- \gamma)W(\tilde{e}_s)
$$

$$
b_{mix} ( e_s) = \gamma b (e_s) + (1- \gamma)b(\tilde{e}_s)
$$

- $\tilde{e}_s$: shuffled $e_ s$ along the batch dimension,$\gamma$ $\gamma \sim \text{Beta} ( \alpha, \alpha)$

$$
\text{M-DSLN} ( h_t, e_s )  = W_{mix} (e_s) \odot \text{LN} ( h_t ) + b_{mix} ( e_s)
$$

### Speaker generalization loss

Speaker generalization loss ($\mathcal{L}_{sgr}$)은 [Style Neophile, 2022]을 참고하여 KL-divergence 를 적용했습니다. 이렇게 설정하게 되면, Mixed speaker information 으로부터 만든 Text encoding과 Original speaker information 으로부터 만든 Text encoding 사이의 일관성을 확보할 수 있습니다.

$$
\mathcal{L}^{o2m}_{sgr} = \text{KL} (\text{DSLN}(h_t, e_s)||\text{M-DSLN}(h_t, e_s))
$$

$$
\mathcal{L}^{m2o}_{sgr} = \text{KL} (\text{M-DSLN}(h_t, e_s)||\text{DSLN}(h_t, e_s))
$$

$$
\mathcal{L}_{sgr} = \mathcal{L}^{o2m}_{sgr} + \mathcal{L}^{m2o}_{sgr} 
$$

M-DSLN 방식과 Speaker generalization loss를 같이 적용함으로써 Lingusitc feature 에서 Speaker-dependent 정보를 분리해낼 수 있었고,  이후에 설명할 SIP와 Duration predictor가 Speaker-independent variation을 예측할 수 있었습니다.

### Speaker independent pitch predictor

실제로 Cross-lingual TTS 시나리오에서는 학습 때 마주하지 못한 Speaker-Language 조합을 보기 때문에, Speech variation을 예측하는 것은 어렵습니다. 이러한 문제를 해결하고자 여러 Speaker들의 발화에서 공통된 특성을 모델링하여 Text-related Pitch variation을 예측할 수 있는 SIP predictor를 제안하였습니다. 

SIP predictor는 M-DSLN 의 아웃풋을 입력으로 받고, Pitch 값의 상승과 하락을 나타내는 Binary pitch contour sequence 을 예측합니다. 

- [Improve cross-lingual text-to-speech synthesis on monolingual corpora with pitch contour information, 2021]

SIP predictor를 학습하기 위해서, [FastPitch, 2021]를 따라 매 프레임마다 Ground Truth Pitch value를 추출합니다. 이 때 추출된 GT Pitch value sequence 는 Speaker dependent 하므로, $p^{(d)}$ 라고 칭하겠습니다. 그리고 SIP predictor는 입력 토큰-level 에서 Pitch 값을 다루므로, GT duration을 이용해 각 입력 토큰에 대해 $p^{(d)}$를 평균내립니다. 마지막으로 평균된 $p^{(d)}$를 아래와 같이 Binary sequence로 변환하여 Speaker-independent target pitch sequence $p^{(i)}$를 얻습니다.

$$
p^{(i)}_n = \begin{cases} 
1,  & \bar{p}^{(d)}_{n-1} < \bar{p}^{(d)}_{n}
\\
0, & \text{otherwise}
\end{cases}
$$

- $\bar{p}^{(d)}_{n}$  : 평균된 $\bar{p}^{(d)}$ 의 $n^{th}$ 값
- ${p}^{(i)} _ n$  : ${p}^{(i)}$의 $n^{th}$ 값
- $N$ : 입력 토큰의 길이
- $n \in \{1,2,3,...,N\}$

SIP predictor의 손실함수는 Binary cross-entropy를 사용했습니다.

$$
\mathcal{L}_{sip}  =  - \sum^N_n [

{p}^{(i)} _ n  \log \hat{p}^{(i)} _ n + (1-{p}^{(i)} _ n ) \log(1-\hat{p}^{(i)} _ n ) ]
$$

- $\hat{p}^{(i)}_{n}$  :  $n^{th}$로 예측된 speaker-independent pitch 값

---

## Speaker Dependent Generator

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/714ef4e9-1c94-4555-bb48-6dcd58fccbdf)

모델의 앞쪽에서 Speaker-independent Linguistic feature 를 잘 모델링하였다면, 뒤쪽에서 Target speaker에 대한 정보를 자연스럽게 결합시켜주어야합니다.

본 논문에서는 DSLN [PVAE-TTS, 2022] 와 SDP [FastPitch, 2021] 를 사용했습니다.  

### Dynamic Style Layer Normalization (DSLN)

- 인풋
    - $e_s$ : Speaker embedding
    - $h_{si}$ : Speaker-independent acoustic representation
- 아웃풋
    - Speaker-adapted hidden feature

### Speaker Dependetn Pitch (SDP) predictor

- 인풋
    - Speaker-adapted hidden feature
- 아웃풋
    - 프레임 단위의 Speaker-dependent pitch embedding
- 손실함수
    - 이전에 구했던 Speaker-dependent target pitch sequence $p^{(d)}$ 와 예측된 Speaker-dependent pitch sequence $\hat{p} ^{(d)}$ 간의 MSE

$$
\mathcal{L} _ {sdp} = || p ^{(d)} - \hat{p} ^{(d)} ||_2
$$

---

## 최종 Loss function

$$
\mathcal{L}_{tot} = \mathcal{L}_{rec} + \mathcal{L}_{align} +\lambda_{dur}\mathcal{L}_{dur} + \lambda_{sgr}\mathcal{L}_{sgr} +\lambda_{sip}\mathcal{L}_{sip} + \lambda_{sdp}\mathcal{L}_{sdp} 
$$

- $\lambda_{dur} = \lambda_{sgr} = \lambda_{sip} = \lambda_{sdp} = 0.1$
- $\mathcal{L}_{rec}$ : Reconstruction loss
- $\mathcal{L}_{align}$ : [One TTS alignment to rule them all, 2022] 에서의 Alignment loss
- $\mathcal{L}_{dur}$ : Duration predictor loss with MSE

---

# Experiments

## Dataset

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/b12efacc-0547-4d08-ab2c-06bd71af68ea)

- 3개 국어 데이터셋 사용
- SR : 22,050 Hz
- Phoneme converting : IPA symbol

 

## Other

- 모델의 전체적인 구성 : [FastPitch, 2021]
- SIP & SDP 의 Pitch 예측 pipeline : [FastPitch, 2021]
- 비교모델
    - [Learning to speak fluently in a foreign language, 2019]
    - [Disentangled speaker and language representations using mutual information minimization and domain adaptation for cross-lingual TTS, 2021]
    - [SANE-TTS, 2022]

## Quality comparison

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/a065b30c-7bc1-40dd-b255-becec42b74c9)

## Ablation study

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/e800e6ae-3ab6-4939-a58a-69789e1ddf7c)

## Acoustic feature space

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/144989499/cb9131bb-e485-4264-9cf1-3cf6ee3e2969)