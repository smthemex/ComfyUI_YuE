# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
import comfy.model_management as mm
from PIL import Image
import numpy as np
from comfy.utils import common_upscale
from datetime import datetime
import folder_paths
import soundfile as sf
import uuid
import hashlib
import json
from pathlib import Path
from transformers import AutoConfig
from safetensors.torch import load_file
from .SheetSage2.modeling_sheetsage2 import SheetSage2Model
cur_path = os.path.dirname(os.path.abspath(__file__))

def build_mert2_encoder(config, files):
    from .SheetSage2.configuration_mert2 import MERT2Config
    from .SheetSage2.modeling_mert2 import MERT2Model
    from .SheetSage2.modeling_sheetsage2 import BASE_CODE_HASHES
    encoder_config = MERT2Config(**config.backbone_config)
    encoder_config._attn_implementation = config.encoder_attn_implementation
    encoder = MERT2Model(encoder_config)
    expected_hash = getattr(config, "base_model_sha256", None)
    state = load_file(files)
    keys = set(encoder.state_dict())
    if not keys & set(state):                     
        for prefix in ("model.", "encoder."):
            stripped = {key[len(prefix):] if key.startswith(prefix) else key: value for key, value in state.items()}
            if keys & set(stripped):
                state = stripped
                break
    report = encoder.load_state_dict(state, strict=False)
    del state
    if report.missing_keys:
        raise ValueError(f"MERT-v2 parent 权重不全，缺 {len(report.missing_keys)} 个键: {report.missing_keys[:8]}")
    return encoder.to(dtype=torch.float32)


def loadsheetsage(model_file_path, mert2_file, dtype=torch.float32, device=torch.device("cpu")):
    """序列化加载 SheetSage2，不调用 from_pretrained。

    - 权重含 encoder.*（merged 快照）: 直接按 merged 配置构建后灌权重，无需父模型。
    - 权重只有 adapter.*（原始发布版）: 建 encoder 骨架 + 灌父权重 + merge_lora（必须在 CPU/float32 上合并）。
    """
    config = AutoConfig.from_pretrained(os.path.join(cur_path, "SheetSage2"), trust_remote_code=True, local_files_only=True)
    state = load_file(model_file_path)
    if any(key.startswith("encoder.") for key in state):
        config.weights_format = "merged"          # 必须在构造之前改：__init__ 靠它决定建不建 encoder
        model = SheetSage2Model(config)
        model.load_state_dict(state, strict=False)
    else:
        model = SheetSage2Model(config)           # adapter 分支: encoder=None, adapter=EncoderAdapters, lora_merged=False
        model.load_state_dict(state, strict=False)
        missing = [key for key in model.state_dict() if key.startswith("adapter.") and key not in state]
        if missing:
            raise ValueError(f"Adapter 权重不全，缺 {len(missing)} 个键: {missing[:8]}")
        model.encoder = build_mert2_encoder(config, mert2_file)
        model.merge_lora()                        # 把 adapter 的 LoRA 加进主干，并置 lora_merged=True
    del state
    model.to(dtype=torch.float32)                 # 合并/加载统一在 float32
    if dtype is not None:
        model.to(dtype=dtype)
    if device is not None:
        model.to(device)
    return model.eval().requires_grad_(False)


def load_yue2_pipeline(repo="m-a-p/YuE2-3B",yue_ckpt=None,vae=None,device="cuda"):
    from .src.yue2.pipeline import YuE2Pipeline
    return YuE2Pipeline.from_pretrained(repo, yue_ckpt=yue_ckpt, vae=vae, device=device)

def yue2_generate(pipe,request,vae_model=None):
    song = pipe(**request,vae_model=vae_model)
    return song
    

def audio2path(audio,):
    prefix=datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    audio_file = os.path.join(folder_paths.get_temp_directory(), f"audio_refer_temp_{prefix}.wav")
    waveform = audio["waveform"].squeeze(0)
    waveform_np = waveform.cpu().numpy() if hasattr(waveform, 'cpu') else waveform.numpy()
    
    # 3. 格式转换：torchaudio 格式为 (channels, samples)，soundfile 需要 (samples, channels)
    # 如果是单声道音频 (1, samples)，转置后变成 (samples, 1)，符合 soundfile 的单声道要求
    if waveform_np.ndim == 2:
        waveform_np = waveform_np.T
        
    sf.write(audio_file, waveform_np, audio["sample_rate"])
    
    return audio_file


def clear_comfyui_cache():
    cf_models=mm.loaded_models()
    try:
        for pipe in cf_models:
            pipe.unpatch_model(device_to=torch.device("cpu"))
    except: pass
    mm.soft_empty_cache()
    torch.cuda.empty_cache()
    max_gpu_memory = torch.cuda.max_memory_allocated()
    print(f"After Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")

def gc_cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def tensor2image(tensor):
    tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def tensor2pillist(tensor_in):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [tensor2image(tensor_in)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[tensor2image(i) for i in tensor_list]
    return img_list

def tensor2pillist_upscale(tensor_in,width,height):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [nomarl_upscale(tensor_in,width,height)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[nomarl_upscale(i,width,height) for i in tensor_list]
    return img_list

def tensor2list(tensor_in,width,height):
    if tensor_in is None:
        return None
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        tensor_list = [tensor_upscale(tensor_in,width,height)]
    else:
        tensor_list_ = torch.chunk(tensor_in, chunks=d1)
        tensor_list=[tensor_upscale(i,width,height) for i in tensor_list_]
    return tensor_list

def tensor_upscale(tensor, width, height):
    samples = tensor.movedim(-1, 1)
    samples = common_upscale(samples, width, height, "bilinear", "center")
    samples = samples.movedim(1, -1)
    return samples

def nomarl_upscale(img, width, height):
    samples = img.movedim(-1, 1)
    img = common_upscale(samples, width, height, "bilinear", "center")
    samples = img.movedim(1, -1)
    img = tensor2image(samples)
    return img


def map_0_1_to_neg1_1(t):

    if not torch.is_tensor(t):
        t = torch.tensor(t)
    t = t.float()

    try:
        vmax = float(t.max())
    except Exception:
        vmax = 1.0
    if vmax > 2.0:
        t = t / 255.0
    try:
        vmin = float(t.min())
        vmax = float(t.max())
    except Exception:
        vmin, vmax = -1.0, 1.0
    if vmin >= 0.0 and vmax <= 1.1:
        t = t * 2.0 - 1.0
    return t

def map_neg1_1_to_0_1(t):
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    t = t.float()
    t = (t + 1.0) * 0.5
    t = t.clamp(0.0, 1.0)
    return t
