---
layout: post
title: "[Emotional Speech Synthesis] EMOQ"
description: >
  EMOQ 논문 요약
category: seminar
tags: emotional-speech-synthesis
author: dh_cho
comments: true
---

# Emo-Q

EMOQ-TTS 논문 요약 (test)

**EMOQ-TTS: Emotion Intensity Quantization for Fine-Grained Controllable Emotional Text-to-Speech**
ICASSP 2022
- GOAL: 세분화된 감정 강도로 음소 단위의 감정 정보를 예측하는 제어 가능한 감성 텍스트 음성 변환 모델
- Motivation: 기존에는 전체 문장이 하나의 전역 정보에 의해 조절되기 때문에 합성된 음성이 단조로운 표현을 갖는 단점이 존재
- Contributions:
    - 사람의 자연스러운 말과 유사한 감성 음성을 생성하기 위해서는 음소 수준에서 감정 강도에 따른 세분화된 감정 표현을 고려
        - 감정 특징 추출 분류기를 사용하여 감정별로 음소 단위의 특징 벡터를 군집화
        - 거리 기반 감정 강도 정량화 크기 이용 → LDA
    - 두 개의 보조 예측자를 추가하여 감정과 관련된 피치 및 에너지 예측
        
![](/assets/img/2023-11-14-write-EMOQ/fig.png)
