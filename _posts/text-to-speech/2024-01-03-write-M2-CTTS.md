---
layout: post
title: "[Text-to-Speech] M2-CTTS"
description: >
  M2-CTTS 논문 요약
category: seminar
tags: text-to-speech
author: jh_cha
comments: true
---


# M2-CTTS: End-to-End Multi-Scale Multi-Modal Conversational Text-to-Speech Synthesis

> Jinlong Xue, Yayue Deng, Fengping Wang, Ya Li, Yingming Gao, Jianhua Tao, Jianqing Sun, Jiaen Liang
Accepted by ICASSP2023
[[Paper](https://arxiv.org/abs/2305.02269)][[Demo](https://happylittlecat2333.github.io/icassp2023/)]
> 

# Goal

- Speech의 prosody와 naturalness를 향상시키기 위해, **Multi-scale, Multi-modal CSS**인 $\text M^2$$-\text {CTTS}$를 제안

# Motivations

- 기존의 conversational TTS 연구에서는 대화에서 global information을 추출하는데만 관심
- 본 논문에서는 fine-grained information(keyword나 emphasis)을 포함하고 있는 local prosody features도 중요하다라고 주장 (multi-grained context information를 고려)
- 또한, text feature뿐만 아니라 acoustic feature도 함께 고려해야 한다고 언급, **M2(Multi-scale, Multi-Modal)**

# Contributions

- Global features, local features 모두 활용해서 speech의 expressiveness를 높임
- 대화의 textual information과 acoustic information을 둘 다 활용하는 방법을 제안
- Style-Adaptive Layer Normalization (SALN)과 prosody predictor module을 학습때만 사용해서 더 좋은 prosody embedding을 만들어서 활용

# Model Architecture

<img width="1414" alt="Untitled 0" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/7d2d0c1f-78a3-4f61-8dc7-78d90bc40fc3">

- **FastSpeech 2** 기반
- Conversation history로부터 textual, acoustic information을 가져오는 구조
- **Textual Context Module** (TCM)
    - **Text Utterance-Level Module** (TUM)
    - **Text Phoneme-Level Module** (TPM)
- **Acoustic Context Module** (ACM)
    - **Wave Utterance-Level Module** (WUM)
    - **Wave Phoneme-Level Module** (WPM)
- **Conditional Decoder**, **Prosody Predictor Module** (PPM)

# Methods

## Multi-scale & Multi-modal Dialogue Modeling

- 대화를 이해하기 위해서, Global information(Emotion, Intention 등등)뿐만 아니라 detailed information(keywords, emphasis)도 활용하기 위한 구조
- 고정된 길이의 대화 전체 embedding은 평균적인 prosody 정보를 modeling하는 단점
- 그래서 **Coarse-grained Context Modeling**, **Fine-grained Context Modeling** 2가지 scale로 context를 modeling
- **Multi-head attention**
- $Conversation = \{ A_{t-c}, B_{t-c+1},…,A_{t-1},B_t\}$
    - $c$ : memory capacity parameter (speaker $A$와 $B$의 대화 turn 수)

### Coarse-grained Context Modeling

문장 level로 대화 정보를 모델링

전체 문장으로부터 정보를 얻거나 문장 간 연결에서 Context 정보를 추출

- **Text Utterance-Level Module** (TUM)
    - **pre-trained Sentence BERT**를 사용해서 semantic information을 잡음
    - 하나의 GRU layer를  통해서 history embedding을 encoding
        - $E^T_{t-c:t-1}$: history sentence embeddings, $E^T_t$: current utterance embedding
    - **Coarse-grained Textual Encoder**
        - linear layer 하나와 Attention을 통해 weighted global textual context embedding 추출
        - $H^T_t$: weighted global textual context embedding
- **Wave Utterance-Level Module** (WUM)
    - Speech Emotion recognition (SER) task의 **Fine-tuned Wav2vec 2.0(on IEMOCAP)**을 사용해서 prosody 정보 추출
        - $E^A_i$: speaker A의 i번째 utterance에 대한 prosody embedding (**speaker A만 고려**)
        - spectral-based feature보다 더 comprehensive acoustic features를 얻을 수 있음 (spectral-based feature에서는 phase 정보, prosodic data가 raw audio에 비해 부족)
    - **Coarse-grained Acoustic Encoder**
        - TUM의 coarse-grained textual encoder와 같은 구조

### Fine-grained Context Modeling

Phoneme level로 대화 정보를 모델링

실제 대화 상황에서 사람은 특정 word나 phrase에 주목하는 것과 유사

- **Text Phoneme-Level Module** (TPM)
    - **pre-trained BERT**로 hidden feature sequence인 $P_i$(i번째 문장)를 얻음
    - 하나의 긴 sequence로 모든 지난 dialogue($c$ 개의 문장)를 aggregate
    - Speaker embedding, position embedding을 더해줌
        - position embedding을 더해주는 이유는 대화에서 문장 순서를 나타내기 위해
    - **Fine-grained Textual Encoder**에서 ****1D convolution layer를 활용하여 contextualization
    - TTS encoder의 output을 **query**, fine-grained representation sequence를 **key와 value**로 **multi-head cross-attention**을 적용
    - Residual connection처럼 attention의 결과인 wighted context representation이 TTS encoder의 output에 다시 더해짐
- **Wave Phoneme-Level Module** (WPM)
    - acoustic 관점에서 local한 표현이나 강조에 집중할 수 있음
    - **pre-trained Wav2vec 2.0**로 hidden feature sequence인 $P_i$(i번째 발화)를 얻음
    - **Fine-grained Acoustic Encoder**
        - 나머지는 TPM에서와 같음

## Conditional Decoder

- [Meta-Style-Speech](https://proceedings.mlr.press/v139/min21b.html)의 ****decoder에서 **Style-Adaptive Layer Normalization (SALN)**를 가져와서 conditional decoder로 활용

## Prosody Predictor Module (PPM)

- multi-modal global context embedding인 $H^T_t$와 $H^A_t$로부터 current utterance의 prosody embedding인 $\hat E^A_t$을 예측
- **[10](https://arxiv.org/abs/2106.10828)의 next predictor를 활용(2 feed-foward layers)**

![Untitled 1](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/8825d3e3-c399-42df-8a91-fb6524537c5f)

- 위와 같이 MSE loss를 적용해서, **대화 history를 기반으로 current utterance의 prosody expression을 예측하도록** **coarse-grained context module을 강제**함 (Fine-tuned Wav2vec으로 뽑은 $E^A_t$를 Ground-true prosody embedding으로 사용)
- 이 PPM 모듈과 loss는 **학습때만 사용**

## Experiments

- **TTS Backbone으로 FastSpeech 2 구조를 사용**
- Montreal Forced Aligner와 같이 외부 aligner를 가진 supervised duration model을 사용하지 않고 **[unsupervised duration modeling](https://arxiv.org/abs/2108.10447)**을 사용
    - MFA는 out-of-distribution 문제 발생 가능
    - expressive 성능에는 soft alignment가 더 좋음
- **Vocoder로는 pre-trained HiFi-GAN을 활용**
- **mel-spectrogram dimension**: 80 , **sampling rate**: 22050 Hz
- **Training step: 400K step, batch size: 16, GeForceRTX 3090 ~~1개(불확실)~~**

### Dataset

- **DailyTalk** (English corpus)
- 1 male speaker, 1 female speaker, 2541 dialogues, 23773 audio clips
- 모든 dialogue는 5개의 turn이 넘음, 총 20시간

# Results

<img width="500" alt="Untitled 2" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/b8805e45-d491-4a1a-a197-d87c1eedcd47">

- Text와 Audio를 함께 고려했을때, 평균적인 prosody를 만드는 문제점을 완화할 수 있음

<img width="500" alt="Untitled 3" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/7e504f16-098a-47ee-9f33-688c8ef0bc54">

- **대화에서 어떤 음성의 prosody가 더 expressive하고 적합한지를 비교**하는 CMOS 실험
- M4(text 모듈만 활용) vs M5(Wave 모듈만 활용) 비교 결과를 보면, **acoustic features를 사용하는게 더 효과적**임을 알 수 있음
- **Attention**을 시각화했을때, **TUM이나 WUM** 대부분의 경우 **모두 같은 문장에 초점을 둠(특히 가장 최근 turn의 문장)**
- M2(TUM만 사용) vs M7(M2-CTTS), M6(TUM, WUM만 사용) vs M7(M2-CTTS) 비교 결과를 보면, **기존의 CSS연구에 있던 방법들([9](https://arxiv.org/abs/2005.10438),[10](https://arxiv.org/abs/2106.10828))보다 본 논문의 M2-CTTS가 더 좋은 성능을 보여줌**

# Conclusion

- **Multi-scale**, **Multi-modal** CSS인 $\text{M}^2\text{-CTTS}$를 제안
- audio와 text를 함께 활용하면 생성된 음성의 prosody와 naturalness를 향상시킬 수 있음
- coarse-grained feature와 fine-grained feature를 함께 활용하면 expressiveness 증가

# References

- J. Xue et al, "[M2-CTTS: End-to-End Multi-Scale Multi-Modal Conversational Text-to-Speech Synthesis.](https://arxiv.org/abs/2305.02269)", *ICASSP,* 2023.
- G. Haohan et al, "[Conversational end-to-end tts for voice agents.](https://arxiv.org/abs/2005.10438)", *SLT*, 2021.
- C. Jian et al, "[Controllable context-aware conversational speech synthesis.](https://arxiv.org/abs/2106.10828)", *INTERSPEECH*, 2021
- B. Rohan et al, "[One TTS alignment to rule them all.](https://arxiv.org/abs/2108.10447)", *ICASSP, 2022.*
- M. Dongchan et al, "[Meta-stylespeech](https://proceedings.mlr.press/v139/min21b.html): Multi-speaker adaptive text-to-speech generation.", *PMLR*, 2021.
- Ren Yi et al., “[Fastspeech 2](https://arxiv.org/abs/2006.04558): Fast and high-quality end-to-end text to speech.”, *ICLR*, 2021
- • W. Yuxuan et al., “[Style tokens](https://proceedings.mlr.press/v80/wang18h.html?ref=https://githubhelp.com): Unsupervised style modeling, control and transfer in end-to-end speech synthesis.”, PMLR, 2018.
- [Sentence BERT](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v1)
- [Wav2vec 2.0 fine-tuned on IEMOCAP](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP)