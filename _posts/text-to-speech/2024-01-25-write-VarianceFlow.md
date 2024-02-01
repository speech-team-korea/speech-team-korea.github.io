---
layout: post
title: "[Text-to-Speech] VarianceFlow"
description: >
  VarianceFlow 논문 요약
category: seminar
tags: text-to-speech
author: jh_cha
comments: true
---

# VARIANCEFLOW: HIGH-QUALITY AND CONTROLLABLE TEXT-TO-SPEECH USING VARIANCE INFORMATION VIA NORMALIZING FLOW

> Yoonhyung Lee, Jinhyeok Yang, and Kyomin Jung.<br>
Accepted by *ICASSP*2022<br>
[[Paper](https://arxiv.org/abs/2302.13458)][[Demo](https://leeyoonhyung.github.io/VarianceFlow-demo/)][Code x]
****
> 

# Goal

- Normalizing flow를 통해서 pitch와 energy와 같은 variance 정보를 더 정확하게 예측하고 조절

# Motivations

<img width="625" alt="Untitled 0" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/efdef4a5-eb66-4d7a-91ea-4d9043fac899">

- FastSpeech 2와 같이 MSE loss를 통해서 text conditioned variance를 모델링하는 방식과 다르게, normalizing flow를 통해서 variance 예측하는 것이 더 좋다
    1. Normalizing flow를 활용하면 일대다 관계(text-variance)를 학습할때 더 robust
    2. text와 latent variance representation을 disentangling할 수 있어서 variance 조절이 더 정확

# Methods

<img width="700" alt="Untitled 1" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/6553725b-89bf-4135-86e5-7e596e067ec3">

- FastSpeech 2를 baseline으로 사용하고 있고 normalizing flow를 활용해서 variance distribution과 prior distribution을 matching
- FastSpeech 2의 Energy predictor와 Pitch predictor를 NF 모듈로 교체한 것
    
    <img width="500" alt="Untitled 2" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/1ba81cd3-8d52-469f-98bf-7b4cd1949dbd">
    
    - NF 모듈은 VITS의 stochastic duration predictor를 참고했다고 언급
    - Training: complex latent variance distribution을 simple prior distribution으로 matching
    - Inference: prior로부터 latent representation을 바로 sampling
    
    <img width="487" alt="Untitled 3" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/1fcd2def-e890-4625-8a07-24c01a8cbb72">
    
- variance information $x$의 probability density를 위와 같이 계산 가능 (by the change of variables)
- latent variance distribution을 simple prior distribution (unit Gaussian)으로 가정
    - $x$: a variance factor, $h$: hidden representation (여기서는 text), $z$: a latent representation, $f_i$: bijective transform(Neural spline flows)
- variance information의 log-likelihood를 최대화하도록 학습

<img width="490" alt="Untitled 4" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/5987d559-ac56-46b7-9c38-ed26359e76e1">

- $L_{melspec}$, $L_{duration}$: FastSpeech 2의 loss와 같음
- $L_{pitch}$, $L_{energy}$: pitch와 energy의 NF loss (negative log-likelihood)
- $\alpha$: 논문에서 0.1로 설정해서 학습했다고 언급
    
    <img width="440" alt="Untitled 5" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/007558a2-f723-40c4-84f6-d411f1049ab6">

    - [Network-to-Network Translation with Conditional Invertible Neural Networks](https://proceedings.neurips.cc/paper/2020/hash/1cfa81af29c6f2d8cacb44921722e753-Abstract.html) Appendix B 참고
    
- 각 NF 모듈에서 variance 정보를 학습하면서 동시에 variance와 text를 disentangle
    - 위의 식과 같이 $L_{NF}$을 decomposition할 수 있는데, 두번째 entropy term인 $H(x \mid h)$가 상수라서 conditional latent variance distribution인 $q_\theta(z \mid h)$와 prior distribution인 $p(z)$사이의 KL-divergence를 최소화하도록 학습 ⇒ prior $p(z)$를 $h$(text)를 고려하지 않고 independent하게 gaussian 분포로 가정했기 때문에 h와 z를 disentangle하도록 학습되는 것
- 이렇게 학습한 latent representation인 $z$로 더 정확하게 pitch나 energy의 control이 가능하다

<img width="494" alt="Untitled 6" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/124e0141-f67e-4243-a8bc-9490a877d85b">

- 가정한 Gaussian 분포에서 sampling한 $z$를 inverse transform을 통해서 raw variance space로 보내고, 여기서 scalar값을 곱해서 원하는 pitch나 energy로 조절하고 다시 NF모듈을 통해 latent representation $z'$로 변환한다
- Control을 raw variance space에서 하고, 조절한 variance factor를 NF모듈을 통해 변환한 $z'$를 FFT decoder의 input에 더해주는 것임

# Experiment Setup

### Datasets

- LJSpeech
- log-mel spectrograms (1024 fft window, 256 hop lengths)
- log scale pitch (Parsel-mouth)
- $\alpha$ =  0.1
- zero-mean Gaussian prior distribution ($\sigma$ = 0.333)
- HiFi-GAN vocoder, baseline으로는 FastsSpeech 2
- NF module
    - 4-layer rational-quadratic coupling transform
    - VITS의 stochastic duration predictor 구조를 따라했다고 언급

### Training setup

- batch size: 16, AdamW ($\beta_1$=0.9, $\beta_2$=0.98) with Noam learning rate scheduling

# Results

## Speech quality

<img width="495" alt="Untitled 7" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/67811c19-d546-4129-b237-85d3377014c0">

## Controllability

<img width="158" alt="Untitled 8" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/2b158135-5ee0-4b99-a7af-5dafff2777ef">

- f0 frame error rate (FFE) 측정
- 위의 식과 같이 pitch value를 semitone 단위로 조절
- VarianceFlow-reversed
    - raw variance information을 바로 FFT decoder의 Input으로 활용한 모델
    - variance factor를 text와 같이 다른 factor와 disentangle하는 것을 보여주기 위해 비교

<img width="1010" alt="Untitled 9" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/7f7c4e35-a2e1-41ac-9636-1b468005238c">

<img width="900" alt="Untitled 10" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/b3a98763-22e2-4592-8c0a-86f65d69cd63">


- VarianceFlow-reversed와 VarianceFlow의 결과 비교가 중요

## Diversity

<img width="700" alt="Untitled 11" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/bbb97488-43b6-483c-b44b-b8a235afb601">

- $\sigma$ = {0.0, 0.667} 의 범위에서는 speech quality가 유지된다고 언급
- 근데 실험에서 latent representation을 사용하지 않아도 자연스러운 음성을 생성한다고 함
    - 하지만 다른 샘플들끼리 비슷한 prosody로 들린다고함

# Conclusion

- Normalizing flow를 기반으로 variance information을 제공해서 variance controllability를 향상시키면서 speech quality도 향상시킴

# References
- Yi Ren et al. "[Fastspeech 2](https://arxiv.org/abs/2006.04558): Fast and high-quality end-to-end text to speech.", *ICLR*, 2021.
- D. P. Kingma and P. Dhariwal ,"[Glow](https://proceedings.neurips.cc/paper_files/paper/2018/hash/d139db6a236200b21cc7f752979132d0-Abstract.html): Generative flow with invertible 1x1 convolutions,” *NeurIPS*, 2018.
- Kim, Jaehyeon, Jungil Kong, and Juhee Son. "[Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech](https://proceedings.mlr.press/v139/kim21f.html).", *PMLR*, 2021.
- Robin Rombach, Patrick Esser, and Bjorn Ommer. "[Network-to-network translation with conditional invertible neural networks.](https://proceedings.neurips.cc/paper_files/paper/2020/file/1cfa81af29c6f2d8cacb44921722e753-Paper.pdf)", *NeurIPS*, 2020.