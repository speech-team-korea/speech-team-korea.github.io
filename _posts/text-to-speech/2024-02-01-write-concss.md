---
layout: post
title: "[Text-to-Speech] CONCSS"
description: >
  CONCSS 논문 요약
category: seminar
tags: text-to-speech
author: jh_cha
comments: true
---

# CONCSS: Contrastive-based Context Comprehension for Dialogue-appropriate Prosody in Conversational Speech Synthesis

> Yayue Deng, Jinlong Xue, Yukang Jia, Qifei Li, Yichen Han, Fengping Wang, Yingming Gao, Dengfeng Ke, Ya Li <br>
Accepted by *ICASSP*2024 <br>
[[Paper](https://arxiv.org/abs/2312.10358)][[Demo](https://anonymous.4open.science/w/DEMO-ICASSP2024-5A69/)]
> 

# Goal

- Contrastive learning 기반의 CSS 프레임워크를 제안하여, context-sensitive representation을 만들고 dialogue-appropriate prosody를 가진 음성 생성

# Motivations

- 기존 CSS 연구의 context encoder는 여전히 context representation를 잘 만들지 못한다
    - Interpretability, representative capacity, context-sensitive discriminability가 부족
- 그래서 효과적이고 context-sensitive representation을 학습하기 위해서 contrastive learning을 활용
    - unlabeled 대화 데이터 셋을 학습하고 context vector의 context sensitivity와 discriminability를 향상시킴

# Contributions

- 논문의 언급으로는 CSS task에 처음 contrastive learning을 적용해서 self-supervised 방식으로 context understanding을 다뤘다고 함
- 다양한 distinct context representation을 잘 만들기 위해서 새로운 pretext task와 sampling strategy를 제안

# Model Architecture

<img width="994" alt="Untitled 0" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/f2e92b26-5e61-426f-a73a-158328c13da6">

- Context encoder의 input으로 대화의 N개 utterance [($u_1$, $p_1$), ($u_2$, $p_2$), …, ($u_N$, $p_N$)] 사용하고 context vector $h_i$를 만듬
    - $u_i$: utterance, $p_i$: speaker
- context vector $h_i$를 VITS에 활용하여 dialogue-appropriate prosody를 가진 음성 생성
- VITS 모델에 아래의 4가지를 추가한 구조
    1. pretext task와 context-dependent pseudo-label을 활용
    2. [M2-CTTS](https://arxiv.org/abs/2305.02269)의 textual and acoustic context encoder 구조를 그대로 활용 (아래 그림에 해당)
    3. Hard negative sampling을 활용해서 triplet loss를 적용
    4. Prosody BERT를 활용한 autoregressive prosodic modeling (APM) 모듈 사용

![Untitled 1](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/c746bb8b-e89a-422c-8f72-48fe7b172883)

# Methods

## Context-aware Contrastive Learning

<img width="600" alt="Untitled 2" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/edc3a8b6-f12c-4e1a-85ce-c5646eb607fd">

### Pretask Definition

- 좋은 context understanding 능력을 가지고 있는 context encoder
    
    ⇒ 이전 문장들의 variation을 잘 감지하고 context마다 구별되는 적절한 representation을 만든다고 가정
    
- contrastive learning을 위해 context-based pretext task를 아래와 같이 정의한다
    - Ground truth sample ($h_i$)
        - anchor sample
    - Positive sample ($h^p_i$)
        - context encoder의 input이 same dialogue(길이만 달라짐)인 경우
        - output ⇒ $h^p_i$
    - Negative sample ($h^n_i$)
        - context encoder의 input이 different(non-overlapping) dialogue인 경우
        - output ⇒ $h^n_i$
    - context encoder가 생성한 context vector는 아래의 기준을 만족해야 한다
        
        <img width="400" alt="Untitled 3" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/25984a6d-9f1a-42c6-ae6e-a96b4f31ef8e">
        
        - $D(\cdot)$: squared Euclidean distance in the context representation space
        - GT sample이 Negative sample보다 Positive sample이랑 더 가깝게

### Sampling Strategy

- hard negative sample을 고려하기 위해 두 가지 방법으로 negative sample을 만든다
    1. Intra-speaker classes
        
        같은 speakers, 완전히 다른 context  ⇒ hard negative samples
        
        context variation이 매우 달라도 상관없이 같은 speaker들로 이루어진 대화는 비슷한 speaker-related prosody를 생성하는 경향이 있음
        
    2. Inter-speaker classes
        
        다른 speakers, 완전히 다른 context ⇒ negative samples
        

### Multi-modal Context Comprehension with Triplet Loss

<img width="588" alt="Untitled 4" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/8e57966b-206a-4ebd-b987-a7c0979deb41">


- context vector의 context-sensitive discriminability를 향상시키기 위해서, triplet loss를 통해서 positive pairs의 유사도는 최대화하고 negative pairs의 유사도는 최소화한다
    
    <img width="600" alt="Untitled 5" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/c8c546c3-eee4-4b2a-a73f-39ce5e75e050">
    
    - $m$: margin parameter
    
- 각각 Textual and acoustic modality별로 batch단위로 각 loss의 평균으로 계산
    
    <img width="500" alt="Untitled 6" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/671d1171-7a2a-4c4e-8a7e-b5e5759fe297">
    
    <img width="500" alt="Untitled 7" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/867661f6-5cbf-4b5f-8b1a-c23f159c2a0a">
    
    - $H_{audio}$: acoustic context vector, $H_{text}$: textual context vector
    
- $L_{contra}$를 최소화하는 것이 아래 (5)식을 만족하도록 context encoder를 학습하는 것이다
    
    <img width="600" alt="Untitled 8" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/f2f97421-7abf-4fa3-bfdc-7eeace05042b">
    
    - $\tau$: 모든 가능한 triplet을 포함하는 training set

- Contrastive learning을 통해 context encoder는 context variation을 잘 반영할 수 있는 representation을 만들 수 있고, 이 representation을 CSS task에 transferring knowledge로 사용한다

## Autoregressive Prosodic Modeling (APM)

<img width="649" alt="Untitled 9" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/8619ad36-cac4-4e1f-b48f-0b974fa5bc47">

- Fluent and natural prosody를 위해 APM모듈을 사용
- Current latent prosody vector인 $LPV_{i+1}$를 만들때 word-level의 prosody information과 이전의 latent prosody vectors인 $LPVs$를 모두 고려한다
- Attention mechanism, Prosody BERT 활용
- ProsoSpeech의 LPV predictor를 참고했다고 언급 (아래 그림은 [ProsoSpeech](https://arxiv.org/abs/2202.07816)의 architecture)

<img width="841" alt="Untitled 10" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/3de6488c-6fa3-4a10-8bad-6614ce561c95">

# Experiment Setup

- [open-source Chinese conversational speech corpus](https://magichub.com/datasets/mandarin-chinese-conversational-speech-corpus-mutiple-devices/)
    - 10 hours transcribed Mandarin conversational speech
    - 30 speakers on certain topics
    - ffmpeg toolkit을 사용해서 dialouge들을 distinct audio clip으로 다 자르고 non-lexical noise들을 제거
    - 전처리후 데이터는 9.2 hours
- CSS의 backbone으로는 [VITS](https://proceedings.mlr.press/v139/kim21f.html)를 활용
    - [Biaobei Chinese TTS dataset](https://www.data-baker.com/open_source.html)으로 pretrain, 5k steps
    - pretraining 후에 Chinese conversation 데이터 셋으로 다시 학습, 20k steps, batch size: 16
- APM 모듈에서 사용한 Prosody BERT는 Biaobei data의 word-level prosodic annotation으로 fine-tuning
- 비교 모델 및 평가 metric은 아래와 같다

<img width="500" alt="Untitled 11" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/cf1fa77d-1922-4910-a09c-9ee5bc57c023">

[[12](https://arxiv.org/abs/2005.10438)], [[11](https://arxiv.org/abs/2305.02269)], [[22](https://ieeexplore.ieee.org/abstract/document/1467314)], [[17](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html)], [[23](https://link.springer.com/chapter/10.1007/978-3-540-74048-3_4)]

# Results

## Comparison of Dialogue-Appropriate Prosody

<img width="876" alt="Untitled 12" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/3c6f1a32-e897-4202-b5d6-228681f4e304">

<img width="600" alt="Untitled 13" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/cb778a26-05c7-48d3-b28a-df09e1a43bca">

<img width="425" alt="Untitled 14" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/16438a1d-b899-4eb5-90aa-b1407633290b">

- multi-modal setting의 MOS가 uni-modal의 MOS보다 좋음 
⇒ acoustic context, textual context 둘 다 고려하는 것이 효과적
- 제안하는 모델(S4)의 MOS가 가장 높고 Mel loss, Log F0 RMSE, MCD 지표에서도 제일 좋은 성능
- S1과 S2 비교 ⇒ triplet loss를 적용하는 것이 prosody performance 향상에 효과적
- S1과 S3 비교 ⇒ hard negative sampling strategy가 prosody performance 향상에 효과적
- S3과 S4 비교 ⇒ APM 모듈을 활용하는 것이 prosody 향상에 효과적
- Real과 Fake 차이가 클수록 context-sensitive
- Contrastive-based approach가 다른 context에 더 sensitive한 것을 추론할 수 있다

## Comparison of Context-sensitive Distinctiveness

<img width="700" alt="Untitled 15" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/e175c5ce-2ab9-4345-96f2-0f6408fabb9d">

- 다른 context modeling 방법을 사용했을때 context vector의 discriminability를 비교한 실험
context vector들의 distance를 시각화함
- (a): M2CTTS는 context vector를 history context보다 current text를 기반으로 context vector를 생성하는 것을 확인할 수 있음
- (c), (d): context-sensitive discriminative capability를 보여줌 ( (a), (b)와 비교 )

# Conclusion

- contrastive-based CSS framework인 CONCSS를 제안함
- Contrastive learning을 통해 context variation을 잘 반영하는 context latent representation을 만들고, 이를  speech synthesis에 활용해서 speech의 prosodic expression을 향상시킬 수 있다
- 논문에서 제안하는 방법들이 context comprehension을 향상시키고 well-representative context vector를 생성해서 more appropriate and context-sensitivity prosody를 가진 음성 생성을 가능하게 함

# References

- Y. Deng et al. "[CONCSS: Contrastive-based Context Comprehension for Dialogue-appropriate Prosody in Conversational Speech Synthesis.](https://arxiv.org/abs/2312.10358)", *ICASSP*, 2024.
- Kim, Jaehyeon, Jungil Kong, and Juhee Son. “[Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech](https://proceedings.mlr.press/v139/kim21f.html).”, *PMLR*, 2021.
- D. Jacob et al.,”[Bert: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805).”, *NAACL-HLT*, 2018.
- J. Xue et al, “[M2-CTTS: End-to-End Multi-Scale Multi-Modal Conversational Text-to-Speech Synthesis.](https://arxiv.org/abs/2305.02269)”, *ICASSP,* 2023.
- Y. Ren et al, "[Prosospeech: Enhancing prosody with quantized vector pre-training in text-to-speech](https://arxiv.org/abs/2202.07816)." *ICASSP, 2022.*
- G. Haohan et al, “[Conversational end-to-end tts for voice agents.](https://arxiv.org/abs/2005.10438)”, *SLT*, 2021.
