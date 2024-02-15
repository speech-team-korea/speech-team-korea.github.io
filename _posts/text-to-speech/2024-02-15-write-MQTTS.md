# MQTTS: A Vector Quantized Approach for Text to Speech Synthesis on Real-World Spontaneous Speech

> Li-Wei Chen, Shinji Watanabe, Alexander Rudnicky<br>
Accepted by *AAAI* 2023 <br>
[[Paper](https://arxiv.org/abs/2302.04215)][[Demo](https://b04901014.github.io/MQTTS/)][[Code](https://github.com/b04901014/MQTTS)]
> 

# Goal

- multiple codebook의 discrete code를 활용하여 풍부한 real-world data로 spontaneous speech 생성

# Motivations

- 사람은 비언어적 정보 (paralinguistic information)를 전달하는 다양한 prosody로 자연스럽게(spontaneously) 음성을 생성
- TTS 시스템이 사람처럼 자연스럽게 (spontaneously) 음성을 생성할 수 있으려면 real-world speech 데이터로 학습해야 함
- 하지만 real-word speech를 학습에 사용하면 mel-spectrogram 기반의 autoregressive TTS 시스템은 inference때 적절한 alignment를 잡아주지 못한다
- 그래서 본 논문에서는 mel 대신 multiple codebook을 학습하여 이 문제를 해결하려고 함
    - discrete representation은 input noise에 유연 (higher resiliency to input noise)
    - single codebook은 codebook 사이즈를 크게 하더라도 다양한 prosody pattern을 학습하지 못한다

# Contribution

- Multiple discrete code를 mel 대신 사용해서 spontaneous speech를 생성하는 MQ-TTS (multi-codebook vector quantized TTS)를 제안

# Overview of MQTTS

<img width="856" alt="Untitled 0" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/24f1f0f1-45a8-44a4-b73a-7201e8c472db">

MQTTS는 학습을 아래와 같이 2 stage로 진행

1. **Quantization of Raw Speech**
input인 raw waveform $\text x$ 를 discrete codes $\text c$로 mapping하도록 $L_{VQ}$와 input $\text x$와 output $\text y$사이의 reconstruction loss로 quantizer를 학습
    - $\text c= \{c_t\}, t \in [1,...,T]$
    - $Q_E$: encoder in quantizer
    - $Q_D$: decoder in quantizer
    - $D$: discriminator
    - $G_i$: learnable codebook embeddings where $i \in [1, …, N]$, $N$: codebook의 개수
2. **Conditional Synthesis with Transformer**
Quantizer는 fix된 상태에서 transformer를 speaker embedding $\text s$, phoneme sequence $\text h$를 condition으로 discrete code인 $\text c$를 autoregressive하게 생성하도록 학습

# Methods

## 1. Quantization of Raw Speech

<img width="856" alt="Untitled 0" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/24f1f0f1-45a8-44a4-b73a-7201e8c472db">

### Quantizer architecture

- Quantizer decoder $Q_D$, discriminator $D$의 backbone architecture로는 HiFi-GAN을 활용
    - mel-spectrogram 대신 discrete code의 학습된 embedding을 input으로 사용
- Quantizer encoder $Q_E$는 HiFi-GAN generator의 구조를 reverse하고 deconvolution layer를 convolution layer로 대체
- 학습의 안정성을 위해 residual block의 output에 group normalization을 적용하고 aggregation
- Decoder $Q_D$의 input에 speaker embedding s를 broadcast-add

### Multiple Codebook Learning

- single codebook이 아닌 multiple codebook을 사용
- 최종 embedding은 N개의 코드북에서 선택된 코드 임베딩을 concat해서 만들어진다

<img width="449" alt="Untitled 1" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/1fe8f46e-53c8-4d95-aca6-3cbc20f513f9">

- $c_{t,i}$: codebook $G_i$에 해당하는 discrete code
- $z^c_t$: encoder의 output
- $z^c_{t,i}$: $z^c_t$를 N개 embedding으로 자름

<img width="554" alt="Untitled 2" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/6c765be2-98f6-47c5-871b-b395637ea604">

- $z^q_t$: decoder의 input

<img width="578" alt="Untitled 3" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/cd2650bd-5cfd-4690-a65f-77fc8022456a">

- $L_F$: 최종 total loss
- $L_{VQ}$: quantization loss, $L_{GAN}$: HiFi-GAN loss, $\lambda$: hyper-parameter (실험에서 10으로 설정)

<img width="578" alt="Untitled 4" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/e8400870-555a-47f1-be5e-132001b9320c">

- $\gamma$: hyper-parameter (실험에서 0.25로 설정)
- quantization loss에서 reconstruction loss가 없는건 HiFi-GAN loss에 mel loss가 이미 포함돼있기 때문 (아래는 VQ-VAE의 loss)
    
    <img width="602" alt="Untitled 5" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/a596cc66-e873-47c5-bddd-f8c242cfade9">
    

## 2. Conditional Synthesis with Transformer

<img width="468" alt="Untitled 6" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/3ef4cb01-e0ad-41b1-bc0f-b82d8c07c5d6">

- phoneme sequence $\text h$와 global speaker embedding $\text s$를 condition으로 $c_t$를 생성
    - speaker embedding은 [pre-trained speaker embedding extractor](https://huggingface.co/pyannote/embedding) 사용
- unique alignment를 위해서 마지막 decoder layer에서 single cross-attention을 적용
- Transformer가 multiple output을 생성하기 위해서, 추가적인 sub-decoder module을 사용
    - codebook group의 고정된 size인 N번으로 sequentially 작용
- $[\text R]$은 repetition token으로 speech signal의 high temporal similarity를 반영한 것

<img width="490" alt="Untitled 7" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/89e6d8ce-df8f-4fd0-8719-22ef64bbaf37">

- 위의 log-likelihood를 최대화하도록 transformer를 학습

## 3. Inference

### Monotonic Alignment

<img width="308" alt="Untitled 8" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/a5c29b5a-303c-4cf7-a9c9-e1a3cb025ff8">

- Monotonic Chunk-wise Attention과 비슷한 개념을 사용한다고 언급

<img width="478" alt="Untitled 9" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/bf454f34-5140-450f-884f-d347a7f82913">

- $A(i, j)$: cross attention value
    - $i$: encoder state at step , $j$: decoder state at step
- $N_w$: encoder states context window

### Audio Prompt

- silence clip을 encoding한 discrete code의 3개 frame을 사용해서 (prepend), Noisy corpus로 학습을 해도 background noise를 제거한 clean speech를 생성할 수 있다
- Clean speech는 보통 clean silence가 선행된다는 점을 고려한 방법

# Experiment Setup

- GigaSpeech
    - transcribed audio from audiobooks, Podcasts, Youtube
    - 16 kHz sampling rate
    - 여기서는 Podcast와 youtube만 사용
- VoxCeleb
    - speaker reference audio로 VoxCeleb의 test set 사용
    - 각 speaker마다 40개의 utterances
- Batch size : 200 input frames per GPU
- 4개의 RTX A6000 GPU 사용
- quantizer는 600k step, transformer는 800k step 학습
- 비교 모델은 VITS, Tacotron 2, Transformer TTS
- code group의 개수 $N$ = 4, 160 code size으로 설정

# Results

## Comparing autoregressive models on their alignments

<img width="1017" alt="Untitled 10" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/056972e2-d9f5-4f80-b53a-e9f6ffc84979">

## Comparison of TTS models

<img width="961" alt="Untitled 11" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/a4634ca9-037b-4206-aec5-ea3b51ce1474">


## Pitch contour for the utterance

- “How much variation is there?” from two models within the same speaker

<img width="496" alt="Untitled 12" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/ae6df376-380b-44d9-abc4-aa4606b0a098">

## Audio Prompt and SNR

<img width="475" alt="Untitled 13" src="https://github.com/speech-team-korea/speech-team-korea.github.io/assets/87218795/70448715-c942-4ac1-aaff-6b91f9e39979">

- standard deviation $\sigma$를 다르게 하면서 audio prompt white noise를 만듬
- $\sigma$가 증가할수록 signal-to-noise ratio (SNR)이 감소
    - $\sigma$가 커질수록 모델이 prompt의 영향을 많이 받음