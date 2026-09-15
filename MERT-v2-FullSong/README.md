---
license: cc-by-nc-4.0
library_name: transformers
pipeline_tag: feature-extraction
tags:
- audio
- music
- music-understanding
- representation-learning
- mert2
- custom_code
---
<h1 align="center">🤗 MERT-v2-FullSong</h1>
<p align="center"><strong>Music representations with full-song context</strong></p>
<p align="center">24 kHz mono · 632M parameters · 24 layers · 1,024 dimensions · 25 Hz</p>

<p align="center">
  <a href="https://map-yue2.github.io/">🎵&nbsp;YuE2&nbsp;project</a>
  ·
  <a href="#quick-start">🚀&nbsp;Quick&nbsp;start</a>
  ·
  <a href="#marble">📊&nbsp;MARBLE</a>
  ·
  <a href="#layer-guide">🎛️&nbsp;Layer&nbsp;guide</a>
  ·
  <a href="#citation">📚&nbsp;Citation</a>
</p>
<p align="center">
  <a href="https://huggingface.co/m-a-p/MERT-v2-30s"><img alt="MERT-v2-30s" src="https://img.shields.io/badge/MERT--v2--30s-374151?logo=huggingface&amp;logoColor=FFD21E" height="20" /></a>
  &nbsp;
  <a href="https://huggingface.co/m-a-p/MERT-v2-FullSong"><img alt="MERT-v2-FullSong" src="https://img.shields.io/badge/MERT--v2--FullSong-374151?logo=huggingface&amp;logoColor=FFD21E" height="20" /></a>
  &nbsp;
  <a href="https://huggingface.co/m-a-p/YuE2-3B"><img alt="YuE2-3B" src="https://img.shields.io/badge/YuE2--3B-374151?logo=huggingface&amp;logoColor=FFD21E" height="20" /></a>
  &nbsp;
  <a href="https://huggingface.co/m-a-p/YuE2-Vae"><img alt="🤗 YuE2-Vae" src="https://img.shields.io/badge/YuE2--Vae-374151?logo=huggingface&amp;logoColor=FFD21E" height="20" /></a>
  &nbsp;
  <a href="https://huggingface.co/m-a-p/YuE2-Vae-legacy"><img alt="🤗 YuE2-Vae-legacy" src="https://img.shields.io/badge/YuE2--Vae--legacy-374151?logo=huggingface&amp;logoColor=FFD21E" height="20" /></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/m-a-p/WildSongBench"><img alt="🤗 WildSongBench" src="https://img.shields.io/badge/WildSongBench-374151?logo=huggingface&amp;logoColor=FFD21E" height="20" /></a>
  &nbsp;
  <a href="https://huggingface.co/m-a-p/SheetSage2"><img alt="SheetSage2" src="https://img.shields.io/badge/SheetSage2-374151?logo=huggingface&amp;logoColor=FFD21E" height="20" /></a>
</p>

**MERT-v2-FullSong is a bidirectional music encoder adapted to complete songs lasting 30–360 seconds.** Extract general-purpose music representations at the frame or recording level. Load with standard Hugging Face Transformers, using the familiar [MERT](https://huggingface.co/m-a-p/MERT-v1-330M) workflow.

It continues pretraining from [MERT-v2-30s](https://huggingface.co/m-a-p/MERT-v2-30s) and preserves the same feature interface.

![MERT2 architecture and training](assets/mert2-architecture.png)

*MERT-v2-30s uses the bidirectional backbone in (B); MERT-v2-FullSong continues through the full-song branch in (C). The causal branch is used for YuE2 tokenization.*

<a id="quick-start"></a>

## 🚀 Quick start

Install the matching PyTorch packages:

```bash
python -m pip install torch==2.6.0 torchaudio==2.6.0 transformers==4.53.2 huggingface-hub safetensors soundfile
```

```python
import soundfile as sf
import torch
import torchaudio.functional as AF
from transformers import AutoFeatureExtractor, AutoModel

repo = "m-a-p/MERT-v2-FullSong"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoFeatureExtractor.from_pretrained(repo, trust_remote_code=True)
model = AutoModel.from_pretrained(repo, trust_remote_code=True).eval().to(device)

audio, sr = sf.read("music.wav", dtype="float32", always_2d=True)
audio = audio[:30 * sr].mean(axis=1)  # First 30 seconds, mixed to mono.
waveform = AF.resample(torch.from_numpy(audio), sr, processor.sampling_rate)
inputs = processor(
    waveform.numpy(), sampling_rate=processor.sampling_rate, return_tensors="pt",
).to(device)

with torch.inference_mode():
    output = model(**inputs, output_hidden_states=True)

frames = output.last_hidden_state         # [batch, frames, 1024], 25 Hz
layers = output.hidden_states             # 24 tensors, one per block
mask = output.feature_attention_mask[..., None]
embedding = (frames * mask).sum(1) / mask.sum(1).clamp_min(1)  # [batch, 1024]
```

`hidden_states[0]` is block 1; `hidden_states[23]` is block 24. Remove the audio slice to process a complete song.

<a id="marble"></a>

## 📊 MARBLE

Reported frozen-encoder results on [MARBLE](https://github.com/a43992899/MARBLE). All scores are multiplied by 100; higher is better. **Bold** marks the best displayed value in each column, including ties.

Baseline scores and parameter counts are reproduced from Tables III and V of [PupuJEPA](https://arxiv.org/html/2606.25713v2); the baselines were not rerun for this release. Evaluation protocols may differ across sources.

**General music understanding.** MTT = MagnaTagATune; key = GiantSteps refined key accuracy; genre and beat = GTZAN; valence and arousal = EmoMusic. ROC = ROC-AUC; AP = average precision.

| Model | Params | MTT ROC | MTT AP | Key acc. | Genre acc. | Beat F1 | Valence R² | Arousal R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MERT-Large | 330M | 90.6 | 37.9 | 64.1 | 77.6 | 86.8 | 56.7 | 76.1 |
| Dasheng-1.2B | 1.2B | 91.5 | 40.4 | 58.0 | 81.4 | 87.7 | 57.4 | 75.0 |
| MuQ | 310M | 90.5 | 38.5 | 63.2 | 83.8 | 90.1 | 58.3 | 76.4 |
| MusicFM | 330M | 90.9 | 38.3 | 63.0 | 84.1 | 90.2 | 57.2 | 74.4 |
| AudioMAE++ | 307M | 91.2 | 39.5 | 61.7 | 80.3 | 90.0 | 59.0 | 75.7 |
| MATPAC++ | 307M | 90.6 | 38.2 | 63.7 | 81.4 | 90.1 | 57.8 | 74.7 |
| A-JEPA | 307M | 91.0 | 39.2 | 65.0 | 83.8 | 90.0 | 57.4 | 74.8 |
| PupuJEPA-Large | 307M | 91.7 | 40.8 | 66.1 | 86.9 | **91.0** | 62.5 | 76.8 |
| PupuJEPA-Huge | 632M | 91.3 | 39.7 | 64.8 | 85.9 | 90.5 | 62.0 | 78.5 |
| **MERT-v2-30s** | 632M | **91.91** | **41.29** | 66.97 | **91.72** | 90.59 | 63.23 | **80.01** |
| **MERT-v2-FullSong** | 632M | 91.74 | 41.20 | **67.05** | 90.69 | 90.57 | **63.52** | 78.14 |

**MTG-Jamendo tagging.** Mood denotes mood/theme tags.

| Model | Params | Instrument ROC | Instrument AP | Mood ROC | Mood AP | Genre ROC | Genre AP | Top-50 ROC | Top-50 AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MERT-Large | 330M | 75.5 | 18.8 | 75.3 | 13.5 | 86.1 | 18.0 | 82.6 | 29.1 |
| Dasheng-1.2B | 1.2B | 75.0 | 19.0 | 76.1 | 15.5 | 85.5 | 18.8 | 82.4 | 29.6 |
| MuQ | 310M | 74.8 | 19.1 | 73.7 | 13.2 | 85.4 | 19.1 | 83.0 | 30.2 |
| MusicFM | 330M | 74.6 | 18.5 | 74.9 | 14.1 | 85.3 | 19.4 | 81.9 | 29.7 |
| AudioMAE++ | 307M | 77.1 | 19.9 | 75.6 | 14.0 | 86.3 | 18.9 | 83.1 | 31.1 |
| MATPAC++ | 307M | 77.2 | 19.7 | 75.1 | 14.1 | 85.7 | 19.6 | 82.5 | 30.2 |
| A-JEPA | 307M | 76.6 | 19.3 | 74.6 | 14.3 | 85.5 | 19.2 | 82.5 | 29.6 |
| PupuJEPA-Large | 307M | 78.4 | 21.2 | 76.2 | 15.3 | 86.1 | 20.1 | 82.8 | 30.5 |
| PupuJEPA-Huge | 632M | 77.6 | 20.5 | 75.9 | 14.7 | 85.9 | 20.1 | 83.1 | 30.7 |
| **MERT-v2-30s** | 632M | **80.27** | 22.89 | **79.44** | **16.68** | **88.01** | **21.22** | **84.18** | **32.17** |
| **MERT-v2-FullSong** | 632M | **80.27** | **23.51** | 78.74 | 15.74 | 87.98 | 20.66 | 84.13 | 31.62 |

**Supplemental:** Chords1217 frame accuracy is **78.48** for MERT-v2-30s and **77.79** for MERT-v2-FullSong.

<!-- mert2-hf-validation:start -->
**HF reproduction verified:** Both models reproduce all ten MARBLE tasks with the fixed evaluation settings. [Scores and best settings](marble_results.json).
<!-- mert2-hf-validation:end -->

<a id="layer-guide"></a>

## 🎛️ Layer guide

Recommended probe settings with the encoder frozen. L1 = `hidden_states[0]`; All-layer MLP uses all 24 layers.

| Task / dataset | MERT-v2-30s layer | Probe LR | MERT-v2-FullSong layer | Probe LR |
|---|---|---:|---|---:|
| Genre · GTZAN | L23 | 5e-3 | L24 | 5e-4 |
| Beat · GTZAN | L21 | 1e-3 | L23 | 1e-3 |
| Key · GiantSteps | L4 | 1e-3 | L23 | 1e-3 |
| Emotion · EmoMusic | All-layer MLP | 5e-5 | L24 | 5e-4 |
| Chords · Chords1217 | All-layer MLP | 1e-4 | All-layer MLP | 5e-4 |
| Tagging · MagnaTagATune | L22 | 1e-3 | L23 | 1e-3 |
| Instrument · MTG-Jamendo | L14 | 1e-3 | L12 | 1e-3 |
| Mood/theme · MTG-Jamendo | L16 | 1e-3 | L13 | 1e-3 |
| Genre · MTG-Jamendo | L19 | 1e-3 | L16 | 1e-3 |
| Top-50 · MTG-Jamendo | L13 | 1e-3 | L22 | 1e-3 |

<a id="citation"></a>

## 📚 Citation

**Technical report coming soon.** For now, please cite [MERT (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/33dffa2e3d2ab74a783d1a8c292f66d9-Abstract-Conference.html) and [MARBLE (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7cbeec46f979618beafb4f46d8f39f36-Abstract-Datasets_and_Benchmarks.html) when using MERT-v2 or its benchmark results in your research.

```bibtex
@inproceedings{li2024mert,
  title = {MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training},
  author = {Li, Yizhi and Yuan, Ruibin and Zhang, Ge and Ma, Yinghao and Chen, Xingran and Yin, Hanzhi and Xiao, Chenghao and Lin, Chenghua and Ragni, Anton and Benetos, Emmanouil and Gyenge, Norbert and Dannenberg, Roger and Liu, Ruibo and Chen, Wenhu and Xia, Gus and Shi, Yemin and Huang, Wenhao and Wang, Zili and Guo, Yike and Fu, Jie},
  booktitle = {International Conference on Learning Representations},
  year = {2024},
  url = {https://proceedings.iclr.cc/paper_files/paper/2024/hash/33dffa2e3d2ab74a783d1a8c292f66d9-Abstract-Conference.html}
}

@inproceedings{yuan2023marble,
  title = {MARBLE: Music Audio Representation Benchmark for Universal Evaluation},
  author = {Yuan, Ruibin and Ma, Yinghao and Li, Yizhi and Zhang, Ge and Chen, Xingran and Yin, Hanzhi and Zhuo, Le and Liu, Yiqi and Huang, Jiawen and Tian, Zeyue and Deng, Binyue and Wang, Ningzhi and Lin, Chenghua and Benetos, Emmanouil and Ragni, Anton and Gyenge, Norbert and Dannenberg, Roger and Chen, Wenhu and Xia, Gus and Xue, Wei and Liu, Si and Wang, Shi and Liu, Ruibo and Guo, Yike and Fu, Jie},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {36},
  pages = {39626--39647},
  year = {2023},
  doi = {10.52202/075280-1722},
  url = {https://proceedings.neurips.cc/paper_files/paper/2023/hash/7cbeec46f979618beafb4f46d8f39f36-Abstract-Datasets_and_Benchmarks.html}
}
```

Weights: [CC BY-NC 4.0](LICENSE). [Dependency notices](THIRD_PARTY_NOTICES.md).

**YuE2 family:** [Song generation](https://huggingface.co/m-a-p/YuE2-3B) · [Audio decoder](https://huggingface.co/m-a-p/YuE2-Vae) · [Benchmark decoder](https://huggingface.co/m-a-p/YuE2-Vae-legacy).
