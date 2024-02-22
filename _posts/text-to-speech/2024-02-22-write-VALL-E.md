---
layout: post
title: "[Text-to-Speech] VALL-E"
description: >
  VALL-E 논문 요약
category: seminar
tags: text-to-speech
author: jh_cha
comments: true
---

# VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers

# Goal

- TTS에 Language model을 도입하여 zero-shot 성능 향상

# Motivations

- 현재 TTS는 일반화 능력이 부족하다
    - NLP에서처럼 가능한 크고 다양한 데이터를 학습해야 일반화 능력 향상 가능
    - Large, diverse, multi-speaker speech 데이터를 학습하기 위해서, language 모델을 TTS에 도입

## In-context learning

- 추가적인 파라미터의 업데이트 없이 unseen input에 대한 label을 예측 가능한 능력

# Contributions

- GPT-3처럼 강한 in-context learning 능력을 가진 TTS 프레임워크인 VALL-E를 제안
- 많은 양의 semi-supervised 데이터로 학습해서, speaker dimension에서 일반화된 TTS 시스템을 제안
- Acoustic prompt의 emotion과 acoustic environment를 유지하면서 같은 input text에 대해 다양한 output을 생성
- zero-shot 시나리오에서 natural speech with high speaker similarity 생성

# The overview of VALL-E

<img width="678" alt="Untitled 0" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/1c6cfc41-a4a8-4b63-9303-4fd5f98dab8b">

- VALL-E는 Text prompt(Phoneme)과 Acoustic code prompt를 기반으로 discrete audio codec code를 생성한다
- Phoneme ⇒ Discrete code ⇒ Waveform
    - Text prompt : target content
    - Acoustic code prompt : target speaker’s voice
- Zero-shot TTS, Editing 또는 GPT-3와 결합해서 활용 가능

<img width="757" alt="Untitled 1" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/f90afa1d-5f04-4762-9e36-d75bfd6e4077">

- pre-trained 모델로 EnCodec을 tokenizer로 사용하여, 각 audio sample을 discrete acoustic code  $\text C$로 encoding한다
- 24 kHz audio, Encoder는 input Waveform에서 75 Hz마다 embedding을 생성 (320배 reduction)
- 각 embedding은 residual vector quantization (RVQ)로 모델링
- 논문에서는 8개 quantizer, 1024 entries
- Dataset $\text D = \{ {\text x}_i, {\text y}_i \}$
    - $\text y$: a audio sample
    - $\text x$ $= \{ {x_0, x_1, …, x_L} \}$: phoneme transcription
- $\text {Encodec}(y) = \text C^{T \times 8}$
    - $\text C$: acoustic code matrix
        - $c_{t,:}$: code matrix의 row vector, t번째 frame에서 8개의 code
        - $c_{:,j}$: code matrix의 column vector, j번째 codebook의 code sequence, $j \in \{1, …, 8\}$
    - $T$: the downsampled utterance length
- $\text {Decodec}(\text C) \approx {\tilde y}$
    - quantization후에 codec decoder를 통해 waveform을 reconstruct

# Methods

## 1. Regarding TTS as Conditional Codec Language Modeling

- 본 논문에서는 zero-shot TTS를 conditional codec language modeling task로 설명한다
    - Condition: phoneme sequence $\text x$와 acoustic prompt matrix $\tilde {\text C}^{T' \times 8}$
    - Acoustic code matrix인 $\text C$를 생성하도록 language model을 학습
    - Optimization objective : $\text {max}$ $p(\text C \mid \text x, \tilde {\text C})$
- Language model이  $\text x$와  $\tilde {\text C}^{T' \times 8}$로부터 semantic context와 speaker information을 추출하도록 학습한다

## 2. Training: Conditional Codec Language Modeling

- Codec의 Residual quantization으로, 각 code는 hierarchical 구조를 가진다
    - 각 quantizer는 이전 quantizer들로부터 residual을 모델링하도록 학습된다
    - 즉, 한 토큰은 이전 토큰들로부터 speaker identity 같은 acoustic property를 가져오고, 이어지는 quantizer에서 more fine acoustic detail을 학습하게 된다
- 본 논문에서 이 hierarchical 구조를 반영하여 두 종류의 conditional language model로 설계한다
    1. Autoregressive (AR) Codec Language Modeling
    2. Non-Autoregressive (NAR) Codec Language Modeling
    
    <img width="919" alt="Untitled 2" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/b0775b3f-4888-4b85-a469-b98cd6f3ae58">)
    

### Autoregressive Codec Language Modeling

<img width="806" alt="Untitled 3" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/d45d5d99-4529-44e0-b3b1-2f4c84f5cd51">

- 첫번째 quantizer의 discrete tokens $\text c_{:,1}$을 뽑기 위해서, autoregressive (AR) decoder-only language model을 학습
- AR language 모델은  a phoneme embedding $W_x$, an acoustic embedding $W_a$, a transformer decoder, a prediction layer로 구성
- 각 token $c_{t,1}$은  이전의 토큰 $(\text x, \text c_{\leq t,1})$에 대해서만 attention
- 즉, 첫번째 코드북에서 다음 토큰의 확률을 최대화하도록 학습한다
    
    <img width="570" alt="Untitled 4" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/e6b758d6-c613-4de0-8384-9ed509e95921">
    
    - condition: phoneme sequence $\text x$, the acoustic prompt  $\tilde {\text C}_{:,1}$
    - acoustic token은 AR decoding에서 고정

### Non-Autoregressive Codec Language Modeling

<img width="829" alt="Untitled 5" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/a826f390-2203-4626-ae57-e46970148560">

- 두번째 quantizer부터는 마지막 quantizer까지의 discrete tokens  $\text c_{:,j \in [2,8]}$을 뽑기 위해서, non-autoregressive (NAR) language model을 학습
- AR 모델과 비슷한 구조이고 8개의 acoustic embedding layer가 있다는 점이 다르다
- Training step마다 랜덤하게 training stage $i$를 샘플링, $i \in [2,8]$
- 샘플링한 $i$번째 quantizer의 codebook의 acoustic token들을 최대화하도록 학습한다
    
    <img width="571" alt="Untitled 6" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/932bdf0e-0bfb-43be-850b-e0432dead06c">
    
    - condition: phoneme sequence $\text x$, the acoustic prompt $\tilde {\text C}$, predicted acoustic tokens in previous codebooks ${\text C}_{:,<j}$
- stage $1$부터 stage $i-1$까지의 acoustic token들을 모델의 input으로 합쳐서 사용
    
    <img width="442" alt="Untitled 7" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/702caadf-fb37-4380-86e4-f427b7eb79e7">
    
    - $\odot$: index selection
- Acoustic prompt인 $\tilde {\text C}^{T \times 8}$의 8개 codebook의 embedded representation들도 $e_{\tilde {\text c}_{t}}$로 합쳐서 input으로 사용
    - 같은 문장에서 3초의 Waveform을 random하게 잘라서 사용한다
    - $e_{\tilde {\text c}_{t}} = \sum^{i-1}_{j=1} e_{\tilde {\text c}_{t,j}}$
- 최종적으로 $i$번째 codebook의 acoustic token들을 예측하기 위한 transformer의 input 
⇒ the concatenation of $(e_{\text x}, e_{\tilde {\text c}}, \mathbb{e_{\text c_{:,<i}}})$
    - positional embedding은 prompt와 acoustic sequence에 따로 적용
    - 현재 stage인 $i$에는 Adaptive Layer Normalization을 적용
    - $\text {AdaLN}(h,i)=a_i \text {LayerNorm(} h \text {)} +b_i$
        - $h$: the intermediate activations
        - $a_i$,  $b_i$: stage embedding의 linear projection으로 도출

- 이렇게 AR 모델과 NAR 모델의 조합은 speech quality와 inference speed간의 good trade-off를 가져올 수 있다
    - 음성의 사람마다 다양하게 달라지는 길이를 예측하는데 AR 모델이 더 유연하게 작용 가능
    - NAR 모델을 사용함으로써, time complexity를 줄일 수 있다
    
- 최종적으로 $\text C$의 예측은 아래와 같이 모델링이 가능한 것이다

<img width="761" alt="Untitled 8" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/2e5edce9-3cf7-4e00-9f80-f906f4dd86ff">

## 3. Inference: In-Context Learning via Prompting

- TTS에서 추가적인 파라미터의 업데이트 없이 unseen speaker의 고품질 음성을 생성할 수 있다면, 모델의 in-context learning 능력이 있다고 간주할 수 있다
- Language model에서처럼 zero-shot scenario에서 in-context learning이 가능하게 하려면 prompting이 필수적이다
- 본 논문에서는 아래와 같이 prompt와 inference를 정의한다
    - **VALL-E**
        - Given a text sentence, a segment of enrolled speech and its transcription
        - **Phoneme prompt**: transcription phoneme + target text phoneme sequence
        - **Acoustic prompt**: the enrolled speech $\tilde c_{:,1}$
        - 모델이 Acoustic prompt의 speaker 목소리로 target text의 acoustic token들을 생성
    - **VALL-E-continual**
        - Given the whole transcription, the first 3 seconds of the utterance
        - **Phoneme prompt**: the whole transcription
        - **Acoustic prompt**: the first 3 seconds of the utterance
        - 모델이 utterance에서 첫번째 3초 이후의 acoustic token들을 생성 (continuations)

# Experiment set up

- 16개의 NVIDA TESLA V100 32GB Gpu로 학습
- batch size : 6k acoustic tokens per GPU for 800k steps, AdamW optimizer
- 비교 모델로는 YourTTS, TTS-Portuguese

## Dataset

- LibriLight
    - 60k hours of unlabelled speech form audiobooks in English
    - 약 7000명의 speaker
    - ASR 모델을 활용하여 960 hours transcription을 만듬
    - EnCodec 모델로 60k hours의 acoustic code matrix를 생성

# Results

## Zero-shot TTS Evaluation

- LibriSpeech의 unseen speaker에 대한 결과

<img width="921" alt="Untitled 9" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/9de6205c-c0fe-4e2c-aa1d-3c48b286f3f5">

⇒ WER과 speaker similarity score를 비교했을때 VALL-E가 robustness와 speaker similarity에 대해서 성능이 더 좋은 것을 확인할 수 있다

⇒ VALL-E-continual에서 WER이 더 낮은 이유는 처음 3초의 acoustic token은 GT를 사용했기 때문

<img width="906" alt="Untitled 10" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/fce89585-a1a1-422b-88d8-5cc6dc61bdae">

⇒ VALL-E의 음성이 더 자연스럽고 현실적인 음성을 생성

- Ablation study
    - NAR-no prompt: prompt 없이 학습
    - NAR-phn prompt: phoneme sequence prompt로만 학습
    - NAR-2 prompts: phoneme prompt와 acoustic prompt 둘 다 사용

<img width="912" alt="Untitled 11" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/b4e11ee8-4260-4621-8f3c-271d7f88e873">

⇒ prompt가 없는 상황에서는 GT token들을 사용해도 성능이 좋지 못함

⇒ phoneme prompt를 추가했을때 WER이 크게 감소하는 것으로 보아 phoneme prompt가 주로 생성하는 content에 영향을 주는 것을 확인할 수 있다

<img width="409" alt="Untitled 12" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/006a3bbd-bc25-493e-8406-5129d0f0112b">

⇒ AR 모델에 대한 ablation study

⇒ NAR 모델에 대해서는 NAR-2 prompt 세팅을 그대로 사용

⇒ Acoustic prompt가 speaker similarity에 영향을 많이 주는 것을 확인할 수 있다

- VCTK에 대한 결과

<img width="921" alt="Untitled 13" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/acaefaf9-f750-4430-a576-a024bdca9a99">

⇒ YourTTS는 학습때 VCTK에서 97명의 speaker에 대한 데이터는 본 적 있음

⇒ 그래도 VALL-E의 speaker similarity가 더 높다

<img width="898" alt="Untitled 14" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/6f74e536-5462-4300-b1a4-05b0c3809375">

⇒ SMOS, CMOS도 마찬가지로 VALL-E가 더 높다

⇒ CMOS가 GT보다 높은 이유는 VCTK의 GT utterance에 조금 noise가 있기 때문이라고 언급

## Diversity

- Speaker similarity를 비교할때, LibriSpeech보다 VCTK가 더 다양한 accent를 가진 speaker가 있어서 어렵다

<img width="819" alt="Untitled 15" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/b83e120a-b20c-432f-abd6-3b6af7d10681">

<img width="906" alt="Untitled 16" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/16947b5f-1e31-413d-b294-936117237cd2">


⇒ VALL-E는 discrete token을 생성할 때 sampling 기반의 방식을 사용하기 때문에, 같은 input text를 사용하더라도 다양한 output을 생성할 수 있다

# References