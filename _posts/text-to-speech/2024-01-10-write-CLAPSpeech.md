---
layout: post
title: "[Text-to-Speech] CLAPSpeech"
description: >
  CLAPSpeech 논문 요약
category: seminar
tags: text-to-speech
author: jh_cha
comments: true
---

# CLAPSpeech: Learning Prosody from Text Context with Contrastive Language-Audio Pre-training

> Zhenhui Ye, Rongjie Huang, Yi Ren, Ziyue Jiang, Jinglin Liu, Jinzheng He, Xiang Yin, Zhou Zhao<br>
Accepted by *ACL*2023 (Main Conference)<br>
[[Paper](https://arxiv.org/abs/2305.10763)][[Demo](https://clapspeech.github.io/)]
> 

# Goal

- **Expressive TTS**를 위해서, **Prosody prediction**에 **더 좋은 text representation을 만드는 것**

# Motivations

- **Expressive TTS를 위한 기존 연구들**
    - TTS에 prosody variance를 반영하기 위해서, **External variation predictor**(**prediction-based**, **PB** - e.g., FastSpeech 2)를 사용하거나 **variational generative model**(**variational-based**, **VB** - e.g., Glow-TTS, Diffsinger)을 활용
    - pre-trained large **masked language model** task(e.g., [BERT](https://arxiv.org/abs/1810.04805), [Png BERT](https://arxiv.org/abs/2103.15060), [Speech BERT](https://ieeexplore.ieee.org/abstract/document/9413864))를 활용해서 text representation을 만듬
    - input text를 기반으로 **masked mel-spectrogram**을 reconstruct하기도 함(e.g., [MAM](https://arxiv.org/abs/2010.11445), [A3t](https://proceedings.mlr.press/v162/bai22d.html))
- **Reconstruction loss로 prosody를 implicitly 학습하기 때문에,** prosody modeling을 향상시키기 어려움
- **Pronunciation space와 prosody space를 분리하지 않기 때문에**, training efficiency가 낮고 model capacity를 낭비
- 기존의 text representation은 **다른 text context에서의 prosody variance를 잡지 못함**
    - 본 논문에서는 **prosody**를 **다른 조건(e.g., text context와 speaker)에서 동일한 token의 pitch와 duration의 variance**로 생각할 수 있다고 언급
    - *"higher"*를 예시로 한다면*, "higher up"* or *"slightly higher"*에서 **같은 단어이지만 다른 prosody**
    - 다른 문맥에서 똑같은 text token의 prosody variance에 대한 모델링이 필요
    - 그래서 **prosody correlated to the text context**를 연구하는 것을 본 논문의 주 목적

<img width="900" alt="Untitled 0" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/45c484d7-74b6-40bc-bbff-9dc36bf8b0a3">

- 본 논문은 **Cross-modal contrastive learning**을 활용한 **[CLIP](https://proceedings.mlr.press/v139/radford21a)**, **[CLAP](https://ieeexplore.ieee.org/abstract/document/10095889)처럼 contrastive learning method를 제안**
    - **text-speech joint multi-modal space**에서 **text context와 high-level prosody pattern을 연결**
    - **C**ontrastive **L**anguage-**A**udio **P**re-training for Text-to-**Speech** ⇒ **CLAPSpeech**

# Contributions

- 기존의 prosody representation methods보다 **더 작은 model scale로도 더 의미있는 prosody representation**을 제공
    - CLAPSpeech는 cross-modal contrastive loss를 통해서 context-correlated prosody를 explicitly<br> 학습
- **TTS 시스템 앞단에서 CLAPSpeech의 text representation을 간단하게 추가**함으로써, 효과적인 prosody modeling 가능
- **Fine-grained prosody transfer**도 가능하다는 것을 보여줌

# The contrastive pre-training process

<img width="913" alt="Untitled 1" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/94cadc52-5a83-4374-abeb-dc0b23f9a8ec">

- **Joint prosody space에서 speech segment와 text token을 연결하는 것이 training objective**
    - **Text encoder**는 text context를 효율적으로 process하도록
    - **Prosody encoder**는 speech segment로부터 high-level prosody만 뽑도록 (timbre 같은 다른 feature는 제외)
    - **Text-Speech multi-modal prosody embedding space를 만드는 것이 목표**
- **Multi-scale**로 contrastive pre-training framework를 제안
    - Phoneme levels
    - Word levels (위의 figure에 해당)
- $N$개의 real pairs(text encoding, speech encoding)간의 cosine similarity를 최대화
- $N^2-N$개의 incorrect pairs간의 cosine similarity는 최소화

# Methods

## Text Encoder

<img width="300" alt="Untitled 2" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/e04e186b-0b95-41bb-b99e-ea64e1e5a9a2">

- 똑같은 발음을 가진(same pronounceable) token의 prosody는 text context에 따라서 다양하게 변한다
- Input : **phoneme and BPE sequence**
    - **Phonological habit** (e.g., 연음)**과 관련된 prosody pattern**과 **semantic information** (다른 emotional overtones과 관련)을 뽑는데 도움
- **3 Feed Forward Transformers (FFT)**
    - **Phoneme FFT block**: phonetic space에서 phonological habits를 모델링
    - **BPE FFT block**: semantic information을 추출
    - **Additional FFT block**: **final phoneme-level text encoding**을 만듬
- **Token Encoding**
    - **Pre-training에서 하나의 token만** 다루기 때문에, **phoneme-level text로부터 indexing**을 해서 **선택된 token의 encoding**을 얻고 **multi-modal embedding space에 linearly projection**
- **Word pooling and Expanding**
    
    <img width="600" alt="Untitled 3" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/2ee70e4d-3cc6-4e1c-b5cd-e2ca5c5593b0">
    
    - Phoneme sequence와 BPE sequence를 결합하는데 길이가 맞지 않음
    - **Word-level pooling** (WP from PortaSpeech)을 사용해서 BPE encoding을 word level로 average pooling을 하고 phoneme sequence에 복사해서 길이를 맞춰 더해줌

## Prosody Encoder

<img width="300" alt="Untitled 4" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/5251897f-10e7-407d-ad69-83c76340cad9">

- **선택된 token의 GT speech segment로부터 prosody pattern을 추출**하는 것이 목표
- Input : **GT speech의 mel의 segment (선택된 token에 해당)**
    - mel-spectrogram을 word 또는 phoneme 단위로 잘라서 사용
    - contextual information없이 local prosody information만을 가짐
- ResNet-50 구조를 backbone으로 활용
    - original version과 다른 점은 **Layer normalization + 1D convolution으로 mel-spectrogram을 더 잘 처리**하도록 구성
    - **speech segment의 길이가 유동적**이기 때문에 **CLIP**의 **attentive pooling layer를 활용**해서 output feature map으로 합침
- 이렇게 만들어진 **prosody encoding**은 contrastive learning setting으로 **disentangled from phonetic and speaker space**
    - positive sample과 negative sample이 모두 똑같은 발음을 가진(same pronounceable) 토큰이기 때문에 **phonetic information을 배제**
    - 학습때 speaker information을 text encoder에 주지 않기 때문에, prosody encoder는 **최종 encoding에서 prosody information을 최대화하기 위해 speaker information을 배제**

### Context-aware text encoding과 Context-unaware mel encoding을 연결
- **Prosody encoder**는 speech segment로부터 high-level prosody information을 추출하도록 학습
- **Text encoder**는 prosody encoder로부터 추출된 prosody를 잘 예측하도록 text context를 활용

## The total loss

<img width="600" alt="Untitled 5" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/61a59aaa-c5a4-488d-a815-62427ca1e597">

- $T_{ph}$: phoneme token representation, $S$: speech representation
    - $T_{ph}\in \reals^{N \times C}$, $S \in \reals^{N \times C}$, $C$: channel size
- $X_{text}$: 선택된 phoneme token이 포함된 text context
- $X_{speech}$: phoneme token의 speech segment
    - $X_{speech}\in \reals^{F \times T}$, $F$: Mel bins의 수, $T$: Time bins의 수
- $f_{text}(\cdot)$: text encoder, $f_{speech}(\cdot)$: speech encoder, $i_{ph}$: phoneme token의 index
- $LN$: layer normalization, $L_{text}$, $L_{speech}$: linear projections


<img width="600" alt="Untitled 6" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/a773859c-2d38-4c29-9ef1-de2073b76073">

- $L_{ph}$: **Phoneme level CLAPSpeech의 training loss**
- $C_{ph}$: $T_{ph}$와 $S$간의 cosine similarity matrix
    - $C_{ph}=T_{ph}\cdot S^T$, $C_{ph} \in \reals^{N \times N}$
- $\tau$: 학습가능한 temperature 파라미터 (logits의 scale을 조절)
- $l_k$: Cross entropy function along the text and speech axis in $C$
    - $l_k=\frac 1 N \Sigma^N_{i=0} \log \text {diag}(\text {softmax}(C))$
    - Real pairs에 대한 cosine similarity는 크게, incorrect pairs에 대한 cosine similarity 작게


<img width="600" alt="Untitled 7" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/5f2958df-85ab-4845-afd4-62bdaab114dc">

- $L_{word}$: **Word level CLAPSpeech의 training loss**
- $C_{word}$: $T_{word}$와 $S$간의 cosine similarity matrix
    - 나머지는 phoneme level loss와 같음

# CLAPSpeech plugged in TTS

- **CLAPSpeech의 text encoder**는 풍부한 prosody information을 가진 text representation 제공 가능
- 그래서 기존 TTS 시스템에서 prosody prediction을 더 잘하도록 plugin 방식으로 활용 가능

<img width="447" alt="Untitled 8" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/4148c5d1-6894-462b-a86f-7ce8ea913403">


<img width="384" alt="Untitled 9" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/90e972ff-600a-418a-aad3-c6f1212f792b">

- variation-based TTS 중 PortaSpeech를 예시로 보여줌
    - CLAPSpeech의 pre-trained text encoder가 auxiliary encoder로 추가
    - TTS 학습때는 CLAPSpeech의 text encoder는 freeze
    - 추가적으로 multi-length adversarial training으로 TTS 모델의 음성 품질을 향상시킴

# Experiment Setup

## Datasets and Baselines

- **CLAPSpeech pre-training (two ASR datasets 활용)**
    1. LibriSpeech (English, 982 hours, 2484 speakers)
    2. WenetSpeech (Chinese, 10,000 hours 중에서 correctness confidence level이 0.95이상인 1000 hours만 사용)
- **3개의 TTS datasets**으로 평가
    1. LJSpeech (English, single-speaker, 13,100 clips, 24 hours)
    2. Biaobei(Chinese, single-speaker, 10,000 sentences, 12 hours)
    3. LibriTTS(English, 1151 speakers, 149,736 clips, 245 hours 중에서 train-clean-360, train-clean-100 사용)
- **Pre-training baseline**으로 2가지 비교
    1. $\text {BERT}$
    2. $\text{A}^3\text{T}$
- TTS 모델은 **FastSpeech 2 (Prediction-based, PB)**, **PortaSpeech (Variation-based, VB)** 사용

## Training and Evaluation

### Pre-training CLAPSpeech

- **Nvidia 3090Ti GPU 4개** 사용, **batch size : 1024 text-speech pairs**, **640,000 iterations** <br>⇒ 논문에서는 약 1주일 걸렸다고 함
- **CLIP**의 cosine learning rate schedule과 동일한 설정

### Training TTS

- Nvidia 2080Ti GPU 1개 사용, batch size : 64 sentences, vocoder로는 HiFi-GAN 사용

### Evaluation

- MOS, CMOS
- Average dynamic time warping (**DTW**) distances
    - between the pitch contours of GT speech and synthesized speech
    - 생성된 음성의 pitch accuracy를 측정하기 위한 objective metrics
- Average duration error (**DE**) in micro-seconds
    - 생성된 음성의 duration accuracy를 측정하기 위한 objective metrics

# Results

## Performance

<img width="911" alt="Untitled 10" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/23775d73-e6ba-47a7-bc65-fd526545fd68">

- PB, VB TTS에서 모두 CLAPSpeech의 representation이 prosody modeling에 더 의미있다는 것을 보여줌
    - 더 적은 파라미터 수로 더 좋은 성능: CLAPSpeech는 **phonetic space를 배제**하고 **prosody space에 대해서만 집중**하기 때문에 **parameter-efficient**

![Untitled 11](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/5bfe04e2-8d90-42a8-ace5-64136ad47a1d)

- mel을 시각화한 것에서 볼 수 있듯이, CLAPSpeech의 text representation을 사용한 것이 더 GT와 가까운 pitch contours를 보여줌
    - CLAPSpeech를 TTS에서 활용하면 더 현실적인 pitch contour를 가진 음성을 생성 가능<br>⇒ **more expressive and prosodic audio**

## Token Representation Self-similarity

- 기존의 representation 학습 방식과 비교하기 위한 실험
- 똑같은 token에 대한 Averaged similarity를 아래와 같이 정의

<img width="600" alt="Untitled 12" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/57f7f706-34d0-41e8-8d5e-88de3a229754">

- $T_i, T_j$: 동일한 단어이지만 다른 text context의 text encoding
- $s(T)$ 값이 낮을수록 선택된 token 자체가 representation을 만들때 하는 역할이 작다고 직관적으로 이해 가능<br>(text context가 달라질 때마다 동일한 단어 token의 representation이 많이 다르다는 뜻이니까)<br>⇒ 선택된 토큰이 아닌 input text sequence에서 context-related 정보를 가지고 더 좋은 prosody를 예측

### Quantitative Evaluation

<img width="600" alt="Untitled 13" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/8aa5ff94-8f8b-4ed8-82b4-2409eb1085fe">

- CLAPSpeech가 동일한 단어의 representation이지만 context에 따라 다른 representation을 만드는 것을 알 수 있음

### Qualitative Evaluation

![Untitled 14](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/e6b5dcc4-8714-48f0-a22b-ccb73b587670)

<img width="550" alt="Untitled 15" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/71a89237-5719-41e8-83d9-1eb5682165e1">

- self-similarity matrix $M$ ( $M_{i,j}=cosine(T_i,T_j)$ )을 시각화한 figure
- 어두운 색 ⇒ higher self-similarity score, 밝은 색 ⇒ lower self-similarity score
- off-diagonal entries에서 CLAPSpeech의 cosine similarity가 제일 낮은 것을 바로 확인 가능

## Fine-grained Prosody Transfer

<img width="750" alt="Untitled 16" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/c70f7952-6e00-4c1f-bdeb-5792c57b20db">

- CLAPSpeech의 **text-speech joint multi-modal space가 high-level prosody pattern을 잘 represent하는 space**임을 보여주기 위한 실험
- “higher”에 대한 text encoding(source)을 다른 문장(reference)에 있는 “higher”의 text encoding으로 교체해서 사용
    - 생성된 음성(transferred)에서 성공적으로 reference의 “higher”의 prosody를 가져옴

## Ablation Study

<img width="600" alt="Untitled 17" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/7782093e-0925-496c-8ca7-cd310f62fa75">

# Conclusion

- **CLAPSpeech**는 TTS 시스템에서 **풍부한 prosody 정보를 가진 text representation을 제공** 가능

# Limitations

1. **현재 문장의 text context와 관련된 prosody만 고려**<br>(future work로 long-form text의 expressive TTS를 위해 inter-sentence prosody를 개선하는 것에 집중하겠다고 언급)
2. **prosody외에 speaker나 감정과 같은 다른 조건들을 고려하지 않음**

# References

- Y. Zhenhui, et al. "[CLAPSpeech: Learning Prosody from Text Context with Contrastive Language-Audio Pre-training](https://arxiv.org/abs/2305.10763).", *ACL*(Main Conference)*,* 2023.
- Ren Yi et al., “[Fastspeech 2](https://arxiv.org/abs/2006.04558): Fast and high-quality end-to-end text to speech.”, *ICLR*, 2021.
- K. Jaehyeon et al.,"[Glow-tts: A generative flow for text-to-speech via monotonic alignment search.](https://proceedings.neurips.cc/paper/2020/hash/5c3b99e8f92532e5ad1556e53ceea00c-Abstract.html)”, *NeurlPS*, 2020.
- Yi Ren, Jinglin Liu, and Zhou Zhao., "[Portaspeech: Portable and high-quality generative text-to-speech.](https://proceedings.neurips.cc/paper/2021/hash/748d6b6ed8e13f857ceaa6cfbdca14b8-Abstract.html)", *NeurlPS*, 2021.
- L. Jinglin et al.,"[Diffsinger: Singing voice synthesis via shallow diffusion mechanism.](https://ojs.aaai.org/index.php/AAAI/article/view/21350)", *AAAI, 2022.*
- D. Jacob et al.,"[Bert: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805).", *NAACL-HLT*, 2018.
- J. Ye et al.,"[PnG BERT: Augmented BERT on phonemes and graphemes for neural TTS.](https://arxiv.org/abs/2103.15060)", *INTERSPEECH*, 2021.
- C. Liping et al.,"[Speech bert embedding for improving prosody in neural tts.](https://ieeexplore.ieee.org/abstract/document/9413864)", *ICASSP,* 2021.
- B. He et al. "[A3T: Alignment-Aware Acoustic and Text Pretraining for Speech Synthesis and Editing.](https://proceedings.mlr.press/v162/bai22d.html)", *ICML*, 2022.
- C. Junkun et al.,"[Mam: Masked acoustic modeling for end-to-end speech-to-text translation.](https://arxiv.org/abs/2010.11445)", *arXiv preprint,* 2020.
- R. Alec et al., "[Learning transferable visual models from natural language supervision.](https://proceedings.mlr.press/v139/radford21a)", *ICML*, 2021.
- E. Benjamin et al., "[Clap learning audio concepts from natural language supervision.](https://ieeexplore.ieee.org/abstract/document/10095889)" *ICASSP,* 2023.