---
license: cc-by-nc-4.0
library_name: transformers
base_model: m-a-p/MERT-v2-FullSong
base_model_relation: adapter
tags:
- audio
- music
- music-transcription
- midi
- abc-notation
- custom_code
---
<h1 align="center">🤗 SheetSage2</h1>
<p align="center"><strong>Music audio to editable scores</strong></p>
<p align="center">Melody · Chords · Beats · Key · Structure</p>

<p align="center">
  <a href="https://map-yue2.github.io/">🎵&nbsp;YuE2&nbsp;project</a>
  ·
  <a href="#quick-start">🚀&nbsp;Quick&nbsp;start</a>
  ·
  <a href="#benchmarks">📊&nbsp;Benchmarks</a>
  ·
  <a href="#citation">📚&nbsp;Citation</a>
</p>
<p align="center">
  <a href="https://huggingface.co/m-a-p/YuE2-3B"><img alt="🤗 YuE2-3B" src="https://img.shields.io/badge/YuE2--3B-374151?logo=huggingface&amp;logoColor=FFD21E" height="20" /></a>
  &nbsp;
  <a href="https://huggingface.co/m-a-p/YuE2-Vae"><img alt="🤗 YuE2-Vae" src="https://img.shields.io/badge/YuE2--Vae-374151?logo=huggingface&amp;logoColor=FFD21E" height="20" /></a>
  &nbsp;
  <a href="https://huggingface.co/m-a-p/YuE2-Vae-legacy"><img alt="🤗 YuE2-Vae-legacy" src="https://img.shields.io/badge/YuE2--Vae--legacy-374151?logo=huggingface&amp;logoColor=FFD21E" height="20" /></a>
  &nbsp;
  <a href="https://huggingface.co/m-a-p/MERT-v2-30s"><img alt="🤗 MERT-v2-30s" src="https://img.shields.io/badge/MERT--v2--30s-374151?logo=huggingface&amp;logoColor=FFD21E" height="20" /></a>
  &nbsp;
  <a href="https://huggingface.co/m-a-p/MERT-v2-FullSong"><img alt="🤗 MERT-v2-FullSong" src="https://img.shields.io/badge/MERT--v2--FullSong-374151?logo=huggingface&amp;logoColor=FFD21E" height="20" /></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/m-a-p/WildSongBench"><img alt="🤗 WildSongBench" src="https://img.shields.io/badge/WildSongBench-374151?logo=huggingface&amp;logoColor=FFD21E" height="20" /></a>
  &nbsp;
  <a href="https://huggingface.co/m-a-p/SheetSage2"><img alt="SheetSage2" src="https://img.shields.io/badge/SheetSage2-374151?logo=huggingface&amp;logoColor=FFD21E" height="20" /></a>
</p>

**SheetSage2 turns music recordings into lead sheets and timed musical annotations.** Transcribe a complete song, edit its ABC or MIDI, render a piano preview, or extract embeddings and token predictions for your own tools.

Built on [MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong), with adapters that merge automatically when you load the model.

![SheetSage2 architecture](assets/architecture.png)

<a id="quick-start"></a>

## 🚀 Quick start

Use Python 3.10 or 3.11 and FFmpeg 6.1 with its shared libraries. Sign in with access to this repository and its MERT-v2 parent:

```bash
python -m pip install huggingface-hub==0.36.0
huggingface-cli download m-a-p/SheetSage2 --local-dir SheetSage2
cd SheetSage2
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "m-a-p/SheetSage2", trust_remote_code=True,
).eval().to("cuda" if torch.cuda.is_available() else "cpu")

result = model.transcribe("song.mp3", output_dir="output")
```

Files are decoded, mixed to mono and resampled automatically. Long songs use overlapping windows.

| Output | File |
|---|---|
| Editable score | `score.abc` |
| Melody and chord accompaniment | `transcription.mid` |
| Separate melodies and chords | `melody_vocal.mid`, `melody_instrumental.mid`, `chords.mid` |
| Timed annotations | `events.json`, `*.lab` |

Omit `output_dir` to keep results in memory. Pass a Tensor or NumPy waveform with its sample rate:

```python
# waveform: [samples] or [channels, samples]
result = model.transcribe(waveform, sampling_rate=24000)
abc = result["abc"]        # str
midi = result["midi"]      # bytes
events = result["events"]  # list of timed events
```

Paths, encoded audio bytes and binary streams are also accepted. `result["midis"]` contains the separate MIDI parts; `result["labs"]` contains annotation text. Request `export_logits=True`, `export_scores=True` or `export_embeddings=True` for CPU tensors in `result["tensors"]`, grouped by window; add `output_hidden_states=True` for all 24 MERT layers. With `output_dir`, these tensors are saved as safetensors instead. Low-level `forward()` returns logits and optional hidden states; `generate()` returns symbolic tokens.

Save a self-contained model for offline use:

```python
model.save_pretrained("sheetsage2-local")
model = AutoModel.from_pretrained(
    "sheetsage2-local", trust_remote_code=True, local_files_only=True,
)
```

### Melody-only ABC for covers

Set `melody_only=True` to retain both the `Vocal` and `Ins` melodies while omitting chord symbols from the ABC and chord accompaniment from playback/combined MIDI. Raw predicted annotations remain available; the default full transcription is unchanged.

```python
result = model.transcribe("song.mp3", output_dir="cover-score", melody_only=True)
abc = result["abc"]  # Also saved as cover-score/score.abc.
```

```bash
python infer.py song.mp3 --output cover-score --melody-only
```

Review the transcription, then pass `result["abc"]` or the saved `cover-score/score.abc` to [YuE2](https://huggingface.co/m-a-p/YuE2-3B) as `abc`, with `cot="melody"` and your target `style` and `lyrics`. If the requested ABC cannot be produced, Python raises an error (partial results are available as `error.result`) and the CLI exits with a nonzero status.

### Command line and rendering

```bash
python infer.py song.mp3 --output output

# Optional piano audio and printable sheet music:
python setup_render.py
python infer.py song.mp3 --output output --render-audio --render-score

# Render existing results without loading the model:
python render.py --input output --output rendered --audio --score pdf,svg,png
```

On minimal Linux servers, use `python setup_render.py --with-deps`. Audio follows the original MIDI timing. Scores preserve the vocal and instrumental staves. For separate piano previews, use `--render-parts vocal,instrumental,chords` with `infer.py`, or `--parts vocal,instrumental,chords` with `render.py`.

Python also accepts `render_audio=True, render_score="pdf,svg,png"`. In memory mode, `result["rendered"]` contains `audio` (part → WAV bytes) and `score` (format → pages as PDF/PNG bytes or SVG text).

<a id="benchmarks"></a>

## 📊 Benchmarks

Scores (%) on H800 with BF16 inference; higher is better. Use `preset="paper"` or `--preset paper` with the dataset prompts in [scores and settings](benchmark_results.json).

| Task | Dataset | Metric | SheetSage1 | madmom | Specialist | SheetSage2 |
|---|---|---|---:|---:|---:|---:|
| Beat | GTZAN | F1 | 85.79 | 85.79 | **88.75** [Beat This!][beat] | 85.65 |
| Beat | osu2017 | F1 | 91.55 | 91.55 | 88.18 [Beat This!][beat] | **92.29** |
| Downbeat | GTZAN | F1 | 64.33 | 64.33 | 78.28 [Beat This!][beat] | **79.51** |
| Downbeat | osu2017 | F1 | 83.22 | 83.22 | 84.99 [Beat This!][beat] | **91.97** |
| Key | GiantSteps | Weighted | 43.89 | 74.62 | 72.09 [S-KEY][key] | **77.73** |
| Key | GTZAN | Weighted | 54.56 | 72.05 | 74.43 [S-KEY][key] | **75.77** |
| Chord | osu2017 | Maj/min | 79.82 | 77.42 | 84.59 [Jiang et al.][chord] | **90.08** |
| Chord | Chords1217 | Maj/min | 72.98 | 83.52* | **84.09**† [ChordFormer][chordformer] | 83.81 |
| Structure | HarmonixSet | Accuracy | — | — | 80.03 [SongFormer][structure] | **80.51** |
| Structure | HarmonixSet | F1 @ 0.5 s | — | — | **70.63** [SongFormer][structure] | 67.96 |
| Structure | HarmonixSet | F1 @ 3 s | — | — | 79.50 [SongFormer][structure] | **82.86** |
| Melody | RWC-Pop | Vocal F1 | 62.71 | — | 62.71 [SheetSage1][sheetsage] | **82.51** |
| Melody | RWC-Pop | Full F1 | 64.02 | — | 64.02 [SheetSage1][sheetsage] | **75.29** |

Melody F1 uses pitch classes. SheetSage1 beat/downbeat results use madmom. *The madmom chord model includes Chords1217 in training. †ChordFormer uses five-fold cross-validation; SheetSage2 uses one checkpoint across all tracks.

[beat]: https://doi.org/10.5281/zenodo.14877491
[key]: https://doi.org/10.1109/ICASSP49660.2025.10890222
[chord]: https://archives.ismir.net/ismir2019/paper/000078.pdf
[chordformer]: https://arxiv.org/abs/2502.11840
[structure]: https://arxiv.org/abs/2510.02797
[sheetsage]: https://arxiv.org/abs/2212.01884

<a id="citation"></a>

## 📚 Citation

**Technical report coming soon.** For now, please cite [YuE](https://arxiv.org/abs/2503.08638) and [MERT](https://proceedings.iclr.cc/paper_files/paper/2024/hash/33dffa2e3d2ab74a783d1a8c292f66d9-Abstract-Conference.html) when using SheetSage2 in your research.

```bibtex
@article{yuan2025yue,
  title = {{YuE}: Scaling Open Foundation Models for Long-Form Music Generation},
  author = {Yuan, Ruibin and Lin, Hanfeng and Guo, Shuyue and Zhang, Ge and Pan, Jiahao and Zang, Yongyi and Liu, Haohe and Liang, Yiming and Ma, Wenye and Du, Xingjian and Du, Xinrun and Ye, Zhen and Zheng, Tianyu and Jiang, Zhengxuan and Ma, Yinghao and Liu, Minghao and Tian, Zeyue and Zhou, Ziya and Xue, Liumeng and Qu, Xingwei and Li, Yizhi and Wu, Shangda and Shen, Tianhao and Ma, Ziyang and Zhan, Jun and Wang, Chunhui and Wang, Yatian and Chi, Xiaowei and Zhang, Xinyue and Yang, Zhenzhu and Wang, Xiangzhou and Liu, Shansong and Mei, Lingrui and Li, Peng and Wang, Junjie and Yu, Jianwei and Pang, Guojian and Li, Xu and Wang, Zihao and Zhou, Xiaohuan and Yu, Lijun and Benetos, Emmanouil and Chen, Yong and Lin, Chenghua and Chen, Xie and Xia, Gus and Zhang, Zhaoxiang and Zhang, Chao and Chen, Wenhu and Zhou, Xinyu and Qiu, Xipeng and Dannenberg, Roger and Liu, Jiaheng and Yang, Jian and Huang, Wenhao and Xue, Wei and Tan, Xu and Guo, Yike},
  journal = {arXiv preprint arXiv:2503.08638},
  year = {2025},
  eprint = {2503.08638},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2503.08638}
}

@inproceedings{li2024mert,
  title = {MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training},
  author = {Li, Yizhi and Yuan, Ruibin and Zhang, Ge and Ma, Yinghao and Chen, Xingran and Yin, Hanzhi and Xiao, Chenghao and Lin, Chenghua and Ragni, Anton and Benetos, Emmanouil and Gyenge, Norbert and Dannenberg, Roger and Liu, Ruibo and Chen, Wenhu and Xia, Gus and Shi, Yemin and Huang, Wenhao and Wang, Zili and Guo, Yike and Fu, Jie},
  booktitle = {International Conference on Learning Representations},
  year = {2024},
  url = {https://proceedings.iclr.cc/paper_files/paper/2024/hash/33dffa2e3d2ab74a783d1a8c292f66d9-Abstract-Conference.html}
}
```

Weights: [CC BY-NC 4.0](LICENSE). [Third-party notices](THIRD_PARTY_NOTICES.md).

**YuE2 family:** [Song generation](https://huggingface.co/m-a-p/YuE2-3B) · [Music representations](https://huggingface.co/m-a-p/MERT-v2-FullSong) · [Audio decoder](https://huggingface.co/m-a-p/YuE2-Vae).
