---
layout: post
title: "[Emotional Speech Synthesis] Fine-Grained Emotional Control of Text-to-Speech: Learning to Rank Inter-And Intra-Class Emotion Intensities"
description: >
  Fine-Grained Emotional Control of Text-to-Speech: Learning to Rank Inter-And Intra-Class Emotion Intensities 논문 리뷰
category: seminar
tags: emotional-speech-synthesis
author: jh_cha
comments: true
---

# Fine-Grained Emotional Control of Text-to-Speech: Learning to Rank Inter-And Intra-Class Emotion Intensities

- Wang Shijun, Jón Guðnason, and Damian Borth, "[Fine-Grained Emotional Control of Text-to-Speech: Learning to Rank Inter-and Intra-Class Emotion Intensities](https://arxiv.org/abs/2303.01508).”, *ICASSP,* 2023

# Goal

- Proposing a **fine-grained controllable emotional TTS**, that considers both **inter-** and **intra-class distances** and be able to synthesize speech with **recognizable intensity difference**
    
    inter-class 거리와 intra-class 거리 모두를 고려한 fine-grained controllable emotional TTS를 제안
    

# Motivation

- Global emotion representation을 condition으로 사용하여 다양한 감정을 표현하는 방법은 단조로운 감정 표현을 가지게 됨
- 다양한 감정 표현을 위해서 GST(a single vector), RFTacotron(a sequence of vectors) 등이 있었지만, reference 음성의 nuance를 잘 가져오지 못함
    - sad reference 음성과 depressed reference 음성으로 같은 음성 샘플이 만들어짐
- 기존 Fine-grained controllable emotional TTS는 phoneme 또는 word단위의 Intensity label과 Rank 알고리즘을 사용함
    - Rank 알고리즘에서는 동일한 감정 class의 음성 샘플은 비슷한 rank를 가지고, 중립 감정의 intensity는 가장 약하다라는고 가정
    - 여기서 **동일한 클래스에 속하는 샘플들은 intensity가 달라도 동일하게 간주되는 문제점**이 있었음(**클래스 내의 거리는 무시**)
    - 그래서 median-level intensity의 음성과 strong-level intensity(또는 weak-level)의 음성을 비교할 때 혼란이 있었음

# Contribution

- Proposing **a fine-grained controllable emotional TTS** model based on a **novel Rank model**
    - **inter-class distance**(emotion class 결정)와 **intra-class distance**(intensity of non-neutral emotion 결정)둘 다 고려함
    - neutral emotion과 non-neutral emotion을 비교하는 것이 아닌, **Mixup**을 통해 augmentation한 샘플들을 비교
- Proposed **Rank model** is **simple and efficient to extract intensity information**
- **Outperforming** two state-of-the-art fine-grained controllable emotional TTS models (Controllability, emotion expressiveness, naturalness)

# Rank Model

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/c4c3cecf-b860-40c9-b411-b455f4b6947e)

- **Rank Model**은 emotion intensity representations를 추출하는 것이 목표
    - Input $\text X$ : **1. Mel-Spectrogram, 2. Pitch contour, 3. Energy** 의 concatenation
    - output : **A rank score** (the emotion intensity)
    - $\text X_{new}$: An input for **neutral** class
    - $\text X_{emo}$: An input from other **non-neutral** class

![Untitled 1](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/6a62f2d1-99ce-4a00-a4dd-0a3ad8f7c1fd)


- pair ($\text X_{new}$, $\text X_{emo}$)에 Mixup augmentation을 수행함
    - $\lambda_i$, $\lambda_j \thicksim Beta(1,1)$

## Mixup augmentation

- Z. Hongyi et al, "[mixup](https://arxiv.org/abs/1710.09412): Beyond empirical risk minimization.", *ICLR*, 2018

![Untitled 2](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/8f482031-d9ea-47a6-9f3c-8d84f32618f7)

- 보통 모델을 학습할 때, ERM(Empirical Risk Minimization)의 문제점이 있음
    - 실제 데이터의 일부인 학습 데이터(Empirical data)로만 학습을 하기 때문에, 일반화 능력이 떨어짐
    - Overfitting이 생길 수 있고 각 data point간의 간격에 대한 가정이 없음
    - class간의 경계가 복잡해질수록 취약함

![Untitled 3](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/dedf00a6-fff1-463b-b17d-8a53de58be7b)

- ERM과 Mixup으로 각각 학습한 모델의 output 확률 분포
- 모델의 일반화 성능이 높아짐

![Untitled](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/c4c3cecf-b860-40c9-b411-b455f4b6947e)

- **The Intensity Extractor**는 intensity representations $I$ 를 추출
    - **FastSpeech 2**의 Feed-Forward Transformer(FFT)를 활용해서 input을 처리
    - output에 emotion embedding vector($\text X_{emo}$의 emotion class)가 더해지면서 intensity representations $I^i_{mix}$, $I^j_{mix}$을 생성
- Intensity representations $I^i_{mix}$, $I^j_{mix}$ 각각을 평균으로 $h^i_{mix},  h^j_{mix}$을 생성

![Untitled 4](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/da71dd81-f8b1-4973-a77f-56d782bcdea5)

- **Original Mixup loss**가 여기서 적용이 됨
    - $y_{emo}$: labels for **non-neutral** emotion
    - $y_{neu}$: labels for **neutral**
- Mixup 방법이 효과적인 regularization이지만, **클래스 내 간격(intra-class distance)에 대한 sensitivity에 대해서는 충분한 증거가 없음**
    - 그래서 $L_{mixup}$(inter-class)외에 **intra-class information을 위해서 추가적으로 $L_{rank}$를 제안함**
- **Projector**는 $h^i_{mix}, h^j_{mix}$를 scalar pair인 $r^i_{mix}, r^j_{mix}$로 mapping함
    - $r_{mix}$: 음성에 존재하는 non-neutral emotion의 양을 수치적으로 나타내는 점수임(즉, i**ntensity에 대한 score**)
    

![Untitled 5](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/03e4eec5-3273-4fc1-b449-0dc99f1f517a)

- 모델이 intensity에 대한 score를 정확하게 만들게 하기 위해서, 먼저 score차이에 Sigmoid 함수를 적용

![Untitled 6](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/278cfc47-b3f9-4623-823d-910ae851663a)
- $p^{ij}$를 통해 $L_{rank}$를 계산
    - $\lambda_{diff}$: normalized $( \lambda_i - \lambda_j)$
    - $\lambda_i > \lambda_j ⇒ \lambda_{diff} \in (0.5, 1)$
    - $\lambda_j < \lambda_j ⇒ \lambda_{diff} \in (0, 0.5)$
    - 예를 들어, 만약 $\lambda_i > \lambda_j$인 상황(비중립 감정이 $\text X^i_{mix}$에  $\text X^j_{mix}$보다 더 많이 있는 상황), $\lambda_{diff}$는 0.5보다 큼
    - 이 경우, $L_{rank}$를 감소시키기 위해, 모델은 $\text X^i_{mix}$에 대해 더 큰 score  $r^i_{mix}$를 할당해서 sigmoid의 output을 0.5보다 크게 만들 것임
    - **비중립 감정을 포함하는 두 샘플**인 $\text X^i_{mix}$와 $\text X^j_{mix}$를 **모델이 올바르게 ranking하도록** 하는 것
    

![Untitled 7](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/10acbed2-a693-43bc-8715-cd4d81270203)
- **Rank model의 total loss가 됨**

# TTS Model

![Untitled 8](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/935f4f6b-27e7-43f7-a3be-733ee184ac87)

- TTS Model은 당연히 음성(mel-spectrogram)을 생성하는 것이 목표
    - **FastSpeech 2**를 Intensity 정보와 함께 활용함
    - **Variance Adaptor**는 pitch, energy, 그리고 duration을 예측(각 phoneme의 frame 길이 단위로)
    - Intensity 정보를 포함시키기 위해, **pre-trained Intensity Extractor**는 고정되어 활용
    - $I$의 길이가 phoneme length와 같지 않기 때문에, Montreal Forced Aligner (MFA)를 사용하여 각 phoneme에 해당하는 intensity segment를 얻음
    - 그리고 intensity segment는 intensity representations의 길이와 phoneme의 길이를 같게 만들기 위해 평균화

## Training

- 먼저 Rank 모델을 학습하고 FastSpeech 2를 학습함
- Rank model, 20k iterations, 1e-6 learning rate, Adam
- FastSpeech2, 250k iterations, 1e-4 learning rate, Adam
- Total loss의 하이퍼 파라미터 $\alpha$: 0.1, $\beta$: 1.0

## Inference

- **phonemes**와 **manual intensity labels**을 사용하여 emotion intensity를 조절
    - **The Intensity Extractor**는 음성에서 intensity representation만 생성하기 때문에, **manual intensity labels**로 intensity를 조절하기 위해서, 아래와 같은 전략을 사용
        1. **pre-trained Intensity Extractor**로 모든 intensity representations과 그 intensity score를 얻음
        2. 모든 **score를 여러 구간으로 나누는데**, 이때 **각 구간은 하나의 intensity 수준**을 나타냄
        3. 각 intensity 수준에 해당하는 intensity representations을 **하나의 vector로 average**
        4. 마지막으로, inference할때, **manual intensity labels**을 **intensity representations**과 mapping가능
    - emotion class마다 각각 적용하기 때문에, individual intensity representation을 찾을 수 있음

# Experiments

## Dataset

- EmoV-DB 데이터 셋
    - 4 speakers, 5 emotions(Amused, Angry, Disgusted, Sleepy and Neutral)
    - 7 hours, 16KHz sampling rate
- Mel-Spectrogram, pitch, energy 전처리
    - 50-millisecond window
    - 50 percent overlap ratio to extract energy
    - Mel-Spectrogram with 80 Mel Coefficients
    - pitch 뽑기 위해, PyWorld 사용

## Baselines

- [FEC](https://ieeexplore.ieee.org/abstract/document/9383524?casa_token=-IppEeySY7cAAAAA:JIvP7TsuOzQ0tuLMaSj4FhKZSk_DekFckxeAumwYqvoNzMheWNxVvMZO4Fiy78o4Zv4Y2PM)
- [RFTacotron](https://ieeexplore.ieee.org/abstract/document/8683501?casa_token=I7z8TLll6t8AAAAA:nO3cAQA0rmG8SLR_hFDFMWizArHU_Z30Xx3R4jBcdpT_erQ8W4HzDIu5XMERlAGSeVJNy8Q)
- vocoder로는 Parallel wavegan([PWGAN](https://ieeexplore.ieee.org/abstract/document/9053795?casa_token=mJk-9rBgM48AAAAA:i4ywXVI1cWfJpmOZBVtXV3VStKoWHDklp4IqSe86imUXVyjYHzdr9r_DJUKElXlU_LoImZI)) 사용

## **Emotion Intensity Controllability**

![Untitled 9](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/f3e35f0b-d6b9-402e-aac2-7d05e18007dc)


- 세가지 intensity 수준으로 Min, Median, Max intensity로 생성된 음성 샘플을 쉽게 식별할 수 있는지 평가
- Min-Max, Min-Median, Median-Max와 같이 pair를 만들고 실험 참여자들은 한 pair에서 더 강한 intensity를 가진 샘플을 선택하도록 요청
- **FEC에서는 클래스 내 거리 정보가 부분적으로 손실**될 수 있지만, 본 논문에서 제시하고 있는 모델은 **클래스 내 거리 정보를 잘 반영**하고 있다는 것을 확인할 수 있음
- 본 논문의 모델은 Max 및 Min 수준의 샘플을 생성할 수 있을 뿐 아니라, Median 수준의 음성 샘플도 인식 가능할 정도로 합성 가능하다는 것을 알 수 있음

## **Emotion Expressiveness**

![Untitled 10](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/bc6ff60a-4bd8-4aef-8e83-04c40418a0e8)

- 모델들이 명확한 감정을 표현할 수 있는지 평가
- intensity를 고려하지 않기 때문에, Median 수준으로 생성한 음성 샘플만 사용
- 본 논문이 제안하는 Rank 모델의 emotion 표현이 FEC보다 좋은 것을 확인할 수 있음

## **Quality and Naturalness Evaluation**

![Untitled 11](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/ba953d19-716d-4499-81c3-66f6778effec)

- 생성한 샘플의 음성 품질과 자연스러움을 추가로 평가
- 객관적 측정지표인 **Mean Cepstral Distortion (MCD)**와 주관적 측정 지표인 **Mean Opinion Score(MOS)** 사용
- Reference에서 감정을 전달하는 것과 달리, 직접 **intensity representations**을 할당하는 것이 모델에게 좋은 품질의 음성을 생성하기 쉬움
- FastSpeech 2는 Tacotron 2보다 적은 데이터로도 고품질 음성을 생성
- 본 논문에서 제시하는 Rank 모델이 음성의 품질에도 영향을 주는 것을 확인할 수 있음

# Conclusion

- **새로운 Rank 모델**을 기반으로 하는 **fine-grained controllable emotional TTS**를 제안
- **Inter-class** and **Intra-class** distance information 모두를 고려해서 의미있는 intensity representations를 생성
- 기존의 two state-of-the-art baselines(**FEC**, **RFTacotron**)보다 좋은 performance를 보여줌

# References

- Wang Shijun, Jón Guðnason, and Damian Borth, "[Fine-Grained Emotional Control of Text-to-Speech: Learning to Rank Inter-and Intra-Class Emotion Intensities](https://arxiv.org/abs/2303.01508).”, *ICASSP,* 2023
- Zhang Hongyi et al.,"[mixup](https://arxiv.org/abs/1710.09412): Beyond empirical risk minimization.", *ICLR*, 2018.
- Ren Yi, et al., “[Fastspeech 2](https://arxiv.org/abs/2006.04558): Fast and high-quality end-to-end text to speech.”, *ICLR*, 2021.
- Lei Yi, Shan Yang, and Lei Xie, "[Fine-grained emotion strength transfer, control and prediction for emotional speech synthesis.](https://ieeexplore.ieee.org/abstract/document/9383524)", *SLT*, 2021.
- Younggun Lee and Taesu Kim,"[Robust and fine-grained prosody control of end-to-end speech synthesis](https://ieeexplore.ieee.org/abstract/document/8683501).", *ICASSP*, 2019.
- R. Yamamoto, Eunwoo Song, and Jae-Min Kim, "[Parallel WaveGAN](https://ieeexplore.ieee.org/abstract/document/9053795): A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram." *ICASSP*, 2020.