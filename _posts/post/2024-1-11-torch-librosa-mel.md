---
layout: post
category: post
tags: post
author: hs_oh
comments: true
---

# Torch로 librosa Mel-spectrogram이랑 똑같이 만들기

### HiFiGAN에서 사용하는 Mel-spectrogram 전처리
```
import librosa

sr = 22050
n_fft = 1024
hop_length = 256
win_length = 1024
n_mels = 80
fmin = 0
fmax = 8000
eps = 1e-6

stft = librosa.stft(
    y.squeeze(0).numpy(),
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window="hann",
    pad_mode="constant",
)
mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
linear_spc = np.abs(stft)
mel = mel_basis @ linear_spc
mel_spec = torch.from_numpy(np.log(np.maximum(eps, mel)))
```

### Torch 버전 
```
import torch

sr = 22050
n_fft = 1024
hop_length = 256
win_length = 1024
n_mels = 80
fmin = 0
fmax = 8000
eps = 1e-6
stft = torch.stft(
    y,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window=torch.hann_window(win_length),
    return_complex=True,
)
melscale = MelScale(
    n_mels=n_mels,
    f_min=fmin,
    f_max=fmax,
    norm="slaney",      # 중요
    mel_scale="slaney", # 중요
    sample_rate=sr,
    n_stft=n_fft // 2 + 1,
)
linear_spc = torch.abs(y_stft).squeeze()
mel = melscale(linear_spc)
mel_spec = torch.log(torch.clamp(mel, min=eps))
```
