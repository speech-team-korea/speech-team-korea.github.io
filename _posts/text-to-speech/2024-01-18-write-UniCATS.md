---
layout: post
title: "[Text-to-Speech] UniCATS"
description: >
  UniCATS 논문 요약
category: seminar
tags: text-to-speech
author: jh_cha
comments: true
---

# UniCATS: A Unified Context-Aware Text-to-Speech Framework with Contextual VQ-Diffusion and Vocoding

> Chenpeng Du, Yiwei Guo, Feiyu Shen, Zhijun Liu, Zheng Liang, Xie Chen, Shuai Wang, Hui Zhang, Kai Yu <br>
Accepted by *AAAI*2024<br>
[[Paper](https://arxiv.org/abs/2306.07547)][[Demo](https://cpdu.github.io/unicats/)][[Code](https://github.com/cpdu/unicats?tab=readme-ov-file)]
> 

# Definitions of context-aware TTS task

<img width="632" alt="Untitled 0" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/c854698c-080b-4569-b20e-02433a458ae3">

- **Speech continuation** 
preceding context A가 speech prompt의 역할, zero-shot speak speaker adaptation
- **Speech editing**
    
    주변 context와 매끄럽게 음성이 이어지도록 하면서 원하는 text로 중간 부분을 바꾸어 음성을 생성하는 task
    preceding context A, following context B를 둘 다 고려하는 것이 speech continuation과의 차이점이다
    

# Goal

- **Surrounding semantic context**를 모두 고려한 **semantic token**들을 diffusion을 통해서 만들고 **target** **speech prompt의 acoustic context**를 반영한 waveform 생성

# Motivations

- 크게 두 종류의 **discrete speech token**이 그동안 제안됐음
    - **Semantic tokens** (e.g., vq-wav2vec, wav2vec 2.0, HuBERT)
        - masking prediction, discrimination을 위해 학습됨
        - articulation information을 주로 잡아줄 수 있지만 acoustic details에 대한 정보는 부족
    - **Acoustic tokens** (e.g., Soundstream, Encodec)
        - speech reconstruction을 위해 주로 학습됨
        - acoustic details (특히 speaker identity)을 주로 잡아 줄 수 있지만, residual vector quantization (RVQ)이 필요
- VQTTS처럼 **discrete speech token을 TTS의 중간 representation**으로 사용할 때, **mel을 사용할 때보다 superior naturalness와 robustness**를 보여줬음
    - Textless NLP, AudioLM ⇒ wav2vec 2.0, w2v-BERT
    - InstructTTS ⇒ VQ-diffusion, acoustic token, natural language prompt
    - VALL-E, SPEAR-TTS ⇒ acoustic token, zero-shot speaker adaptation, short speech prompt
    - NaturalSpeech 2 ⇒ discrete acoustic tokens as continuous features
- Discrete speech token을 활용하는 현재 TTS 모델들의 단점
    1. 대부분 autoregressive inference를 하기 때문에 speech editing에는 적절하지 않다
    2. 같은 text로부터 waveform의 frame을 만들더라도 다양한 경우가 생길 수 있기 때문에, Acoustic token을 만들때 residual vector quantization (RVQ)를 활용하게 되고, 이는 모델의 예측을 더 어렵게 만든다
    3. 생성된 음성 품질이 audio codec 모델의 성능에 따라 달라진다
- 위 단점들을 보완하기 위해, 본 논문에서는 **unified context-aware TTS framework, UniCATS를 제안**

# Contributions

- speech continuation task, speech editing task에서 모두 state-of-the-art performance
- surrounding context와 자연스럽게 연결되도록 음성을 생성하기 위해서 contextual VQ-diffusion을 제안 (acoustic model, CTX-txt2vec)
- speech prompt의 acoustic context를 반영하기 위해서 contextual vocoding을 제안 (vocoder, CTX-vec2wav)

# The unified context-aware framework

<img width="1000" alt="Untitled 1" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/bf01cddd-db78-4caa-be7d-26e70bd5fa2b">


- Acoustic model ⇒ CTX-txt2vec
- Vocoder ⇒ CTX-vec2wav

# Methods

## CTX-txt2vec with Contextual VQ-diffusion

<img width="800" alt="Untitled 2" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/eeb8e929-e366-4c1a-9c78-1ac0168d3e23">

- contextual VQ-diffusion을 활용해서 input text로부터 semantic token들을 예측
- semantic token으로 vq-wav2vec tokens을 활용
- Text encoder, duration predictor, length regulator, VQ-diffusion decoder로 구성

### VQ-diffusion

- 데이터 샘플 하나를 아래와 같이 가정
    - $x_0=[x^{(1)}_0, x^{(2)}_0, ..., x^{(l)}_0]$, where $x^{(i)}_0 \in \{{1,2,…,K}\}$
- Forward diffusion step $t$동안 **making, substitution, remain unchanged**를 적용해서 $x_t$를 만들 수 있고 forward process는 아래 식과 같다
    
    <img width="500" alt="Untitled 3" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/8f70b81c-8540-4e63-a092-2a5d0cf76f16">
    
    - $v(x_t)\in \mathbb{R}^{(K+1)}:$ $x_t=k$일 때, $k$번째 value만 1인 one-hot vector
    - $K+1$번째 value는 [$\text {mask}$] 토큰을 말한다
    - $Q_{t}\in\mathbb{R}^{(K+1)\times(K+1)}:$ step $t$의 transition matrix를 말한다
- step $t$까지의 전체 Forward process을 아래와 같이 나타낼 수 있다
    
    <img width="399" alt="Untitled 4" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/1b078c3e-a41f-49c0-804f-9455057cc4cc">
    
    - $\bar Q_t = Q_t\cdot\cdot\cdot Q_1$
- 여기서 Bayesian rule을 적용하면,
    
    <img width="477" alt="Untitled 5" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/0be6c454-2c45-4c28-8a48-40c616a2ca19">
    
- VQ-diffusion 모델은 $y$를 condition으로 $x_t$로부터 $x_0$의 분포를 추정하기 위해 학습 ⇒ $p_{\theta}(\tilde x_{0} \mid x_t, y)$
- 결과적으로 Backward process에서 $x_t$와 $y$를 가지고 $x_{t-1}$을 샘플링 할 수 있다
    
    <img width="495" alt="Untitled 6" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/d383cc80-b452-4224-a939-c20636175912">
    

### Contextual VQ-diffusion

<img width="600" alt="Untitled 7" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/86e3051e-c51e-4227-bd05-357680f60654">

- 본 논문은 speech editing과 continuation task에 VQ-diffusion을 적용하기 위해서 input text를 condition $y$로, 생성할 semantic token들을 data $x_0$으로 다룬다
- 기존의 VQ-diffusion이랑 다른 점은 backward process에서 추가적인 context token들인 $c^A$와 $c^B$를 고려한다는 점이다
    
    <img width="362" alt="Untitled 8" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/5ac51541-5e2a-49e7-83eb-67013eb94d78">

    
- 본 논문에서는 Diffusion step $t$에서 corrupted semantic token들인 $x_t$와 clean context token인 $c^A$와 $c^B$를 결합하는 것을 제안한다 ⇒ $[c^A, x_t, c^B]$
- VQ-diffusion model은 transformer 기반 모델이기 때문에, self-attention layer를 통해서 contextual information을 효과적으로 다룰 수 있다
- contextual VQ-diffusion의 posterior는 식 (4)와 비슷하게 아래와 같이 계산할 수 있다
    
    <img width="442" alt="Untitled 9" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/97c465f9-e68a-4abb-a36e-025c791fba60">
    
- 기존의 VQ-diffusion과 또 다른 점은 text encoding $h$를 다르게 넣어준다는 것이다
    - self-attention layer뒤에 text encoding $h$를 넣어줘서 semantic token들과 $h$사이의 strict alignment를 더 쉽게 만들어준다

### Training scheme

- 각 문장을 3가지 경우로 구성할 수 있다
1. **Both context $A$ and $B$**
    
    ⇒  context $A$, $x_0$, context $B$로 랜덤하게 나눈다 ($x_0$의 길이가 전체 문장 길이보다는 짧지만 100개 frame보다는 길게)
    
2. **only context $A$**
    
    ⇒ context $A$의 길이를 2~3초로 랜덤하게 결정
    
3. **no context**
    
    ⇒ 전체 문장을 $x_0$으로 context 없이 다룬다
    
- 이 3가지 경우를 전체에서 0.6 : 0.3 : 0.1의 비율로 구성해서 학습
- $x_0$가 결정되면 forward step을 통해서 $x_t$를 만들고 VQ-diffusion decoder의 input으로 사용한다
- Training loss는 아래와 같다
    
    <img width="464" alt="Untitled 10" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/ba1e0de2-9bad-4ebe-9e79-4dc23c01b7d7">
    
    - $L_{\text {CTX-txt2vec}}$: CTX-txt2vec의 training loss
    - $L_{duration}$: duration prediction의 Mean Square Error(MSE) 가중합
    - $L_{\text{VQ-diffusion}}$: VQ-diffusion loss
    - $\gamma$: hyper-parameter

### Inference algorithm

<img width="546" alt="Untitled 11" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/b4c6c95e-7573-48bb-949d-55c53dc4c6c0">

- $\alpha$: context와 비슷한 speech speed를 유지하기 위한 scale factor

## CTX-vec2wav with Contextual Vocoding

<img width="800" alt="Untitled 12" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/e0adc430-ae62-4390-9f8c-d24dd0dd7403">

- acoustic context(특히 speaker identity)를 고려해서 semantic token들로 waveform을 생성
- speaker embedding이나 acoustic token을 사용하지 않고 mel spectrogram $m$을 활용해서 acoustic context를 prompt
- 2개의 semantic encoder, feature adaptor, HifiGAN generator로 구성된다
    - Feature adaptor는 FASTSpeech2의 variance adaptor와 유사한 역할
    - Training때는 GT feature를 condition으로 사용
    - 첫번째 semantic encoder의 output으로부터 feature들을 예측하도록 학습
    - Inference에서는 예측된 auxiliary feature를 condition으로 활용

### Semantic encoder block based on Conformer

<img width="600" alt="Untitled 13" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/45f535c4-665d-43c1-87e4-7733eb2ecf38">

- $M$개의 Conformer-based block으로 구성
- 기존의 Conformer-based block에 cross-attention layer가 추가해서 mel을 acoustic context로 활용
- mel encoder로 mel spectrogram $m$의 encoding $m'$을 생성해서 사용
    - mel encoder는 간단한 1D convolution layer로 구성
    - no position encoding ⇒ 학습때는 짧은 mel을 사용하더라도 inference때 길이 상관없이 acoustic context를 뽑을 수 있게 해준다
    - 대신 speech prompt인 mel의 길이가 길수록 생성된 음성의 speaker similarity는 증가해서 future work로 남겨둔다고 언급

### Training scheme

- speaker identity가 각 문장마다 일관성있게 유지된다고 가정
- 각 문장을 두 segment로 나눈다
    - 첫번째 segment ⇒ 2~3초의 랜덤 길이, mel을 추출하고 acoustic context를 prompting하는 역할
    - 두번째 segment ⇒ semantic token을 추출하고 vocoding할 부분
- 나머지 학습 과정은 HifiGAN과 같고 auxiliary feature prediction을 위해 $\text L1$ loss가 추가된다
- VQ-TTS에서 제안된 multi-task warmup도 같이 활용

# Experiment Setup

## Datasets

- **LibriTTS** (2306 speakers)
    - Training 데이터로 약 580시간 음성 데이터 사용
    - test 데이터 셋으로 총 500문장 사용
        - test set A ⇒ 369 speakers from ‘test-clean’, 약 3초의 speech prompt
        - test set B ⇒ unseen 37 speakers, zero-shot adaptation 평가용, 약 3초의 speech prompt
        - test C ⇒ RetrieverTTS와 같은 test 데이터 셋 구성, speech editing 평가용

## Training setup

- CTX-txt2vec
    - 50 epochs, AdamW, diffusion step $\text T= 100$
- CTX-vec2wav
    - 800k steps, Adam

# Results

## Speech resynthesis from semantic tokens

<img width="1037" alt="Untitled 14" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/79a214e5-33ed-48fa-90d1-5f939cad337c">


## Speech continuation for zero-shot speaker adaptation

<img width="1000" alt="Untitled 15" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/3051e89c-89ca-4ba3-bc0d-010c6787d661">

## Speech editing

<img width="454" alt="Untitled 16" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/906c53df-55d7-4388-86f8-50aa53eaa8d8">

# Conclusion

- speech continuation과 editing task에서 활용할 수 있는 unified context-aware TTS framework인 UniCATS를 제안
- speaker embedding이나 acoustic token 없이 2~3초의 짧은 speech prompt를 사용해서 음성 생성
- 두 task에서 SOTA 성능을 보여줌

# References
* 추가 예정입니다