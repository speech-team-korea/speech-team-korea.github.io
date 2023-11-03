---
layout: post
title: "[Pronunciation Editing] FluentSpeech"
description: >
  FluentSpeech 논문 요약
category: pronunciation-editing
tags: pronunciation-editing
author: hw_bae
comments: true
---


# FluentSpeech

ACL 2023

FluentSpeech: Stutter-Oriented Automatic Speech Editing with Context-Aware Diffusion Models

Code, Audio Sample URL : [https://github.com/Zain-Jiang/Speech-Editing-Toolkit](https://github.com/Zain-Jiang/Speech-Editing-Toolkit)

- 발음 교정 모델 Baseline으로 쓰고 있는 TTS 모델
- Goal: Utterance에서 Stutter(말 더듬는) 부분에 대해서 제거하는 교정 모델

기존 Text 기반의 Speech editing 모델들의 한계

1) 교정된 음성의 Over-smoothing 문제

2) Stutter에 의한 noise 때문에 robustness가 부족하다는 문제

3) Stutter을 제거하기 위해 교정 부분(edited region)을 정해주어야 하는 문제

![](/assets/img/2023-11-03-write-FluentSpeech/fig1.png)
