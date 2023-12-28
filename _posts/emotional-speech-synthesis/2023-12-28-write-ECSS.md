---
layout: post
title: "[Emotional Speech Synthesis] ECSS"
description: >
  Emotion Rendering for Conversational Speech Synthesis with Heterogeneous Graph-Based Context Modeling 논문 요약
category: seminar
tags: emotional-speech-synthesis
author: jh_cha
comments: true
---


# Emotion Rendering for Conversational Speech Synthesis with Heterogeneous Graph-Based Context Modeling

- Liu Rui et al., "[Emotion Rendering for Conversational Speech Synthesis with Heterogeneous Graph-Based Context Modeling](https://arxiv.org/abs/2312.11947).”, *AAAI*, 2024

# Task Definition

## Conversational Speech Synthesis (CSS)

- **CSS**는 **conversational context에** **적절한** **prosody와 emotional inflection**을 가진 utterance를 정확하게 표현하는 것이 목표
- 여기서, **conversational context에 적절한 prosody와 emotional inflection**이란, **dialogue history를 고려**한 prosody와 emotion inflection을 말한다
- 각 dialogue history의 different emotion cue를 고려했을때, 적절한 emotional inflection을 가진 utterance 예시

![Untitled 0](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/a779eb80-c287-4ac6-9c7d-303951231199)

$u_1$(Happy), $u_2$(Happy), $u_3$(Surprise), $u_4$(Happy) → $u_5$(Surprise)

![Untitled 1](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/b66d8567-7753-4e18-9408-73f44288ca35)

$u_1$(Angry), $u_2$(Happy), $u_3$(Angry), $u_4$(Angry) → $u_5$(Disgust)

- 사람들은 자신의 생각과 기분을 표현하기 위해, 다양한 intensity의 감정을 가지고 표현한다
- 기존의 single utterance를 위한 Speech Synthesis와는 달리, **Conversational Speech Synthesis(CSS)**는 **dialogue interaction history**에 따라 target utterance의 speaking style을 추론
- **Human-machine conversations**(such as virtual assistants, voice agents, etc.)에서 conversational context에 적절한 prosody와 emotion style를 추론하는 것은 중요하다

# Goal

- **multi-modal dialogue history**(text, speaker id, audio, emotion id, emotion intensity)**를 고려한 target utterance의 적절한 emotion style 추론**

# Motivations

- 기존의 CSS 연구들은 conversational context의 이해를 통해, 생성하고자 하는 음성의 적절한 speaking style을 결정
- 본 논문에서는 conversational context에서도 emotion에 대한 이해가 중요하다고 주장
    - Dialogue context에서 emotional expression은 target utterance의 speaking style을 직접적으로 결정
    - Contextual information을 통해, 적절한 emotional rendering(emotion id, emotion intensity)이 중요
- 기존의 CSS task에서 emotional expressiveness problem에 대한 연구가 부족
    - Emotional conversational datasets의 부족
    - Stateful emotion modeling의 어려움

# Contributions

- **Emotional expressiveness**를 모델링한 emotional CSS 모델인 **ECSS**를 제안
- **Heterogeneous graph 기반의 emotional context modeling**을 통해, 정확한 emotional conversational speech 생성

# Methods

<img width="1083" alt="Untitled 2" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/2687a709-2532-47bd-8586-b123d4f7b3c1">

## 1. Multi-source knowledge

- **Dialogue history** **(A historical conversation, Multi-Modal)**
 A sequence of utterances이며, 각 utterance는 multi-modal data의 tuple로 구성
(1. text, 2. speaker id, 3. audio, 4. emotion id, 5. emotion intensity)
Multi-turn conversation의 text, speaker, audio emotion, emotion intensity 정보를 포함
    - $(utt_1, utt_2, …, utt_{\jmath}, utt_C)$ : a conversation, a sequence of utterances
    - $\{utt_1, utt_2, …, utt_{\jmath}\}$:  $\jmath$ 턴까지의 대화 history
    - $utt_j(j \in [1, \jmath])$: multi-modal context를 고려하기 위해, 대화 History에서 각 utterance는 tuple인 <$text_j,$ $speaker_j$, $audio_j$, $emotion_j$, $emotion\ intensity_j$>, 즉 $<u_j, s_j, a_j, e_j, i_j>$로 표현
- **Target utterance (reply, current utterance)**
모델이 생성하고자 하는 utterance이며 주어진 speaker id와 text로 모델은 적절한 speaking style을 가진 utterance를 생성
    - $utt_C$: 모델이 생성하고자 하는 current(target) utterance이며, tuple인 <$text_j,$ $speaker_j$>, 즉 $<u_j, s_j>$로 표현되고 emotion id와 emotion intensity 정보는 모델에 의해 추론
- **Appropriate** **Speaking style**
    - Conversation ****history, Dialogue context에 적절해야 한다
    ⇒ emotion id, emotion intensity, prosody(duration, energy, pitch)

## 2. Emotion Understanding (Heterogeneous Graph-based Emotional Context Encoder)

Heterogeneous Graph-based emotional context encoder가 Emotional Conversational Graph (ECG)를 만들고, 이 ECG를 통해서, multi-source knowledge 사이의 복잡한 관계를 모델링

이 과정으로 context에서 Emotion cue를 이해할 수 있다

### Homogeneous Graph vs Heterogeneous Graph

Homogeneous Graph :  모든 노드가 같은 성질을 가지고 있는 그래프

Heterogeneous Graph : 반대로 그래프의 노드가 다른 여러 종류의 성질을 가지고 있는 그래프

- 대화의 구조(Multi-Source Knowledge)가 Heterogeneous structure를 가지고 있기 때문에, 본 논문에서는 Heterogeneous Graph를 활용

### ECG Construction

- 앞서 정의한 multi-source knowledge를 5가지의 노드로 표현해서 ECG를 만든다
    - Emotional Conversational Graph (ECG) ⇒ $G(N, \varepsilon)$
    - $N$: the set of nodes
        - text $f_u$, audio $f_a$, speaker $f_s$, emotion $f_e$, intensity $f_i$
    - $\varepsilon$: the set of edges
        - 두 노드간의 관계를 보여줌, 14가지 종류
        - past-to-future, future-to-past 양방향 관계 모두를 모델링

### ECG Initialization

- multi-turn dialogue를 다양한 encoder를 활용해서 각 utterance의  $f_u, f_a, f_s, f_e, f_i$ 노드를 초기화
    - **Text Node $f_u$** : [pre-trained BERT model](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v1)로 linguistic feature 추출,  $f_{u_j} = \text {BERT}(u_j)$
    - **Audio Node $f_a$** : [GST](https://proceedings.mlr.press/v80/wang18h.html?ref=https://githubhelp.com) module(reference encoder, style token layer)로 acoustic feature 추출,  $f_{a_j} = \text {GST}(a_j)$
    - **Speaker, Emotion, Emotion intensity Node ($f_{s_j}, f_{e_j}, f_{i_j}$)** : Randomly initialized trainable parameter matrix로 2 speaker identity features, 7 emotion label features, 3 emotion intensity label features를 학습
    - 7 emotion : happiness, sadness, anger, disgust, fear, surprise, neutral
    - 3 emotion intensity : weak, medium, strong
    - 여기서 $f_{u_c}$, $f_{i_c}$는 current(target) utterance, $f_{u_j}$, $f_{i_j}$는 dialogue history의 utterance

### ECG Encoding

- 초기화된 constructed ECG를 활용해서, dialogue context의 emotion cue 정보를 encoding
- Heterogeneous Graph에 Transformer의 방법들을 적용
    - Heterogeneous Mutual Attention (HMA), Heterogeneous Message Passing (HMP
    ), Emotional knowledge Aggregation (EKA) network를 통해 emotional conversation에서의dependency를 모델링
    - Heterogeneous Graph Transformer ([HGT](https://arxiv.org/abs/2003.01332))의 architecture 참고
    
    <img width="1031" alt="Untitled 3" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/19b01002-645b-4d4c-aa1b-56e00ca118fc">
    
    - Target Node ⇒ Query vector, Source Node ⇒ Key vector, dot production
- Final emotion-aware graph-enhanced feature representation ⇒ $f'_u, f'_s, f'_a, f'_e, f'_i$

## 3. Emotion Rendering (Emotional Conversational Speech Synthesizer)

앞의 ECG encoding에 맞추어 current(target) utterance의 정확한 emotion state를 만드는 것이 목표

- **Text Encoder**
    - 생성하고자 하는 current utterance의 text ⇒ content features $H^c_C$
- **Speaker Encoder**
    - 생성하고자 하는 current utterance의 speaker ⇒ speaker identity features $H^s_C$
- **Emotion Renderer**
    - Emotion predictor, Intensity predictor, Prosody predictor로 구성
    - Emotion predictor
        - $f'_e$로부터 current utterance의 emotion representation인 $H^e_C$를 추론
        - ECG encoding후의 $f'_e$는 dialogue history의 모든 emotion 노드들의 universal representation
        - 2 convolution layers, a bidirectional LSTM layer, 2 fully connected layers
        - $H^e_C=\text {FC}(\text{BiLSTM}(\text{CNN}(f'_e)))$
    - Intensity predictor
        - $f'_i$로부터 current utterance의 emotion intensity representation인 $H^i_C$를 추론
        - ECG encoding후의 $f'_i$는 dialogue history의 모든 emotion intensity 노드들의 universal representation
        - 2 convolution layers, a bidirectional LSTM layer, 2 fully connected layers, a mean pooling layer
        - $H^i_C=\text{AvgPooling}(\text {FC}_2(\text{BiLSTM}(\text{CNN}_2(f'_i))))$
    - Prosody predictor
        - text 노드들의 feature representation으로부터 current utterance의 speaking prosody information을 추론
        - ECG encoding과정에서 이미 text 노드에서 audio information 정보를 가지고 있기 때문에, audio 노드들은 사용하지 않음
        - MSE loss (target으로 GST-based prosody extractor와 비교)
        - Multi-head attention layer
    - 결국 Emotion Renderer에서는 graph-enhanced node features ($f'_e, f'_i, f'_p$) ⇒ current utterance의 emotion, intensity, prosody features를 예측 ($H^e_C, H^i_C, H^p_C$)
- **Feature aggregator module**
    - 앞의 five features ($H^c_C, H^s_C, H^e_C, H^i_C, H^p_C$) ⇒ the final mixup feature $H_C$
    - current utterance의 좀 더 robust한 feature representation($H_C$)로 만든다
- **Acoustic Decoder**
    - [FastSpeech2](https://arxiv.org/abs/2006.04558)를 backbone으로 사용
    - Variance adapter에서 $H_C$를 input으로 duration, energy, pitch를 예측
    - vocoder로 pre-trained HiFi-GAN를 활용하고 바람직한 emotion style을 가진 음성을 생성

## Contrastive Learning Training Criterion

- **Emotion-supervised contrastive learning** loss인 $L^{cl}$을 제안
- **Self-Supervised Contrastive Learning** vs **Supervised Contrastive Learning**
    
    <img width="853" alt="Untitled 4" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/49a21128-e6b0-42fc-b047-3956b809c9ed">
    
    - **Self-Supervised Contrastive Learning**
        - label이 없는 데이터로도 의미있는 표현(representation) 학습 가능
        - 하나의 기준 이미지(anchor)를 두고 다양한 augmentation(flip, crop 등등)을 통해 유사한 이미지(positive class)를 생성
        - 다른 이미지들을 negative class로 설정해서 positive와 negative가 latent space에서 분리되도록 학습
        - 단점 : negative의 기준이 다른 이미지이기 때문에, 실제로 같은 class의 이미지도 negative로 분류가 되버림
    - **Supervised Contrastive Learning**
        - 모든 데이터의 label이 있다면, data augmentation을 그대로 사용하면서 같은 class끼리는 positive로 유사한 표현(representation)을 얻도록 학습
        - 결국 latent space상에서 같은 클래스끼리 군집(cluster)를 만들게 학습하는 것
        - 좋은 representation을 얻을 수 있는 방법
- 본 논문에서는 Supervised Contrastive Learning을 사용
- $L^{cl}_{emo}$ ⇒ Contrastive loss for Emotion category
    
    <img width="487" alt="Untitled 5" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/2c26d7e1-edde-4458-a148-760b671fabad">
    
    - $H^K=[H^e_{C1},H^e_{C2}, ...,H^e_{CK}  ]$
        - a batch of K개의 emotion representations
    - $H^e_C$
        - emotion feature representation
    - $sim(\cdot, \cdot)$
        - 코사인 유사도 계산
    - $B(k)=H^K\setminus\{ H^e_{Ck}\}$
        - $H^K$에서  $H^e_{Ck}$를 제외한 모든 representation
    - $P(k)$
        - $H^e_{Ck}$와 같은 emotion label을 가진 positive samples 집합
- $L^{cl}_{int}$ ⇒ Contrastive loss for Emotion intensity
    
    <img width="493" alt="Untitled 6" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/085f0dfe-7462-4bef-a738-e319ebd77600">
    
- 최종 Total loss $L$은 아래와 같다 ($L_{fs2}$: FastSpeech2의 acoustic feature loss, $L^{mse}_{pro}$: prosody predictor의 MSE loss)

    $L = L^{cl}_{emo} + L^{cl}_{int} + L^{mse}_{pro} + L_{fs2}$

# Experiments

## Dataset

- [DailyTalk](https://github.com/keonlee9420/DailyTalk) 데이터 셋 활용
    
    ![Untitled 7](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/b735eced-197b-4ff3-9ae1-c4de31118e66)
    
    - public dataset for conversational speech synthesis
    - 23733 audio clips, 20 hours, 2541 conversations
    - 남자 speaker 1명, 여성 speaker 1명으로 구성
    - 44.10 kHz sampling rate
- Emotion category와 intensity label에 대해서 fine-grained labeling을 추가적으로 진행함
- 최종적으로 7 emotion category labels (happy : 3871, sad : 722, angry : 226, disgust : 186, fear : 74, surprise : 497, neutral : 18197), 3 emotion intensity labels (weak : 19973, medium : 3646, strong : 154)으로 구성
- 22.05 kHz으로 모두 Resampling
- window length : 25ms
- shift length : 10ms
- batch size : 16, 600K steps, a Tesla V100
- 학습할때 dialogue history 길이는 10으로 설정

# Results

<img width="846" alt="Untitled 8" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/8ceaf229-a84d-405d-b892-48b78471de54">

- Dialogue-level Mean Opinion Score (DMOS)
    - naturalness DMOS (N-DMOS) : current utterance의 speaking prosody에 초점
    - emotional DMOS (E-DMOS) : current utterance의 emotional expression의 richness, context의 emotional expression과 일치하는지에 초점
- Acoustic features (mel-spectrum, pitch, energy, duration)간의 Mean Absolute Error (MAE)
- 비교 모델
    - No emotional context modeling : vanilla FastSpeech2
    - GRU-based context modeling : Text modality만 사용해서 contextual dependency를 모델링
    - Homogeneous Graph-based modeling : 노드 하나가 utterance 하나를 뜻함

<img width="386" alt="Untitled 9" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/f3160139-9b1f-4a27-9e8c-4ee25fcea6f4">

- dialogue history의 길이에 따라 결과 비교
- DailyTalk의 평균 대화 turn수는 9.3
- context information이 불충분하거나 너무 많아도 context에서 emotion cue를 찾는 것에 도움이 안됨

<img width="543" alt="Untitled 10" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/7f90569a-a286-4b05-8fa3-ddf11da54391">


<img width="538" alt="Untitled 11" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/bbb11d45-f173-47b6-ba88-47ae06ebd823">

- 생성한 speech의 emotional expressiveness를 시각화한 결과
- pre-Trained Speech Emotion Recognition([SER](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition)) model 활용

# References

- Liu Rui et al., "[Emotion Rendering for Conversational Speech Synthesis with Heterogeneous Graph-Based Context Modeling](https://arxiv.org/abs/2312.11947).”, *AAAI*, 2024.
- Ren Yi, et al., “[Fastspeech 2](https://arxiv.org/abs/2006.04558): Fast and high-quality end-to-end text to speech.”, *ICLR*, 2021.
- Wang, Yuxuan, et al., "[Style tokens](https://proceedings.mlr.press/v80/wang18h.html?ref=https://githubhelp.com): Unsupervised style modeling, control and transfer in end-to-end speech synthesis.", PMLR, 2018.
- [Speech Emotion Recognition](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition) By Fine-Tuning Wav2Vec 2.0
- Hu Ziniu, et al. "[Heterogeneous graph transformer.](https://arxiv.org/abs/2003.01332)", *Proceedings of the web conference, 2020*.
- Khosla, Prannay, et al. "[Supervised contrastive learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html).", *NeurIPS*, 2020.
- DailyTalk Dataset official GitHub repo, [https://github.com/keonlee9420/DailyTalk](https://github.com/keonlee9420/DailyTalk)
- [pre-trained Bert](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v1) Model