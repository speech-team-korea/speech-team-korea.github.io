---
layout: post
title: "F.cross_entropy 팁"
category: coding
tags: pytorch
author: hs_oh
comments: true
---

F.cross_entropy 팁

```
loss = (F.cross_entropy(pred, lbl, reduction='none') * mask.squeeze(1)).sum() / mask.sum()
```

#### 각 변수
- pred: 예측 값 (B, N, L)
- lbl: 타겟 라벨 (B, L)
- mask: 마스크 (B, 1, L)
- B: Batch 크기, N: 클래스 수, L: 길이 

```
torch.Size([16, 5, 144]) torch.Size([16, 144]) torch.Size([16, 1, 144])
```