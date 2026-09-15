# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from comfy_api.latest import  io
import folder_paths
from pathlib import Path
from datetime import datetime
from .node_utils import clear_comfyui_cache,audio2path,load_yue2_pipeline,yue2_generate,loadsheetsage

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

MAX_SEED = np.iinfo(np.int32).max
node_yue_cr_path = os.path.dirname(os.path.abspath(__file__))
weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)
folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir

class YUE_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YUE_SM_Model",
            display_name="YUE_SM_Model",
            category="YUE_SM",
            inputs=[
                io.Combo.Input("diffusion_models",["none"] +folder_paths.get_filename_list("diffusion_models")),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, diffusion_models) -> io.NodeOutput:
        clear_comfyui_cache()
        dit_path=folder_paths.get_full_path("diffusion_models",diffusion_models) if diffusion_models != "none" else None
        return io.NodeOutput(load_yue2_pipeline(os.path.join(node_yue_cr_path, "YuE2-3B"),yue_ckpt=dit_path, vae=None, device=device))
    

class YUE_SM_Vae(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YUE_SM_Vae",
            display_name="YUE_SM_Vae",
            category="YUE_SM",
            inputs=[
                io.Combo.Input("vae",options=folder_paths.get_filename_list("vae")),
            ],
            outputs=[
                io.Vae.Output(display_name="vae"),
                ],
            )
    @classmethod
    def execute(cls,vae) -> io.NodeOutput:
        clear_comfyui_cache()   
        from .src.yue2.modeling_vae import YuE2VAE
        return io.NodeOutput(YuE2VAE.from_pretrained(os.path.join(node_yue_cr_path, "vae"), decoder_only=True, device="cpu",local_files_only=True,ckpt_path=folder_paths.get_full_path("vae",vae)))
    

class YUE_SM_Clip(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YUE_SM_Clip",
            display_name="YUE_SM_Clip",
            category="YUE_SM",
            inputs=[
                io.Combo.Input("clip",options= folder_paths.get_filename_list("clip")),
                io.Combo.Input("mert2",options= folder_paths.get_filename_list("clip")),
            ],
            outputs=[
                io.Clip.Output(display_name="clip"),
                ],
            )
    @classmethod
    def execute(cls,clip,mert2) -> io.NodeOutput:
        clear_comfyui_cache()
        return io.NodeOutput(loadsheetsage(folder_paths.get_full_path("clip",clip),folder_paths.get_full_path("clip",mert2)))


class YUE_SM_Cond(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YUE_SM_Cond",
            display_name="YUE_SM_Cond",
            category="YUE_SM",
            inputs=[
                io.Clip.Input("clip"),
                io.Audio.Input("audio"),
            ],
            outputs=[
                io.String.Output(display_name="abc_file"),
                ],
            )
    @classmethod
    def execute(cls,clip,audio,) -> io.NodeOutput:
        clear_comfyui_cache()
        clip.to(device)
        prefix=datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = os.path.join(folder_paths.get_output_directory(),"audio", f"cover_score_{prefix}")
        result = clip.transcribe(audio2path(audio), output_dir=output_dir, melody_only=True,)
        clip.to("cpu")
        if not result.get("abc") or result.get("abc_error"):
            raise RuntimeError("Transcription did not produce a usable melody score")
        abc_file= os.path.join(output_dir,"score.abc")
        return io.NodeOutput(abc_file)

class YUE_SM_Sampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YUE_SM_Sampler",
            display_name="YUE_SM_Sampler",
            category="YUE_SM",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.String.Input("style", default="English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM",multiline=True),
                io.String.Input("lyrics", default="[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\nFold the night and leave it here\nMorning has a sky to clear\n\n[Chorus]\nLet the day come into view\nEvery road begins with you\nHold a little room for light\nWe will sing beyond the night",multiline=True),
                io.Combo.Input("cot", options=["full", "melody", "off"], default="full"),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Boolean.Input("only_plan", default=False,tooltip="If True, just return the plan abc files, not the audio."),
                io.String.Input("abc_file", default="E:\\ComfyUI313\\ComfyUI\\custom_nodes\\ComfyUI_YuE\\examples\\melody.abc",multiline=False),
            ],
            outputs=[
                io.Audio.Output(display_name="audio"),
            ],
        )

    @classmethod
    def execute(cls, model,vae,style,lyrics,cot,seed,only_plan,abc_file) -> io.NodeOutput:
        clear_comfyui_cache()
        request = {"lyrics": lyrics,"style": style,"seed": seed,"cot": cot,"id": "comfyui_node",}
        print(abc_file)
        if abc_file:
            request["abc"] = Path(abc_file).read_text(encoding="utf-8")
        if request.get("abc") is not None and request.get("cot", "full") == "off":
            raise ValueError("A supplied score requires full or melody mode. 使用乐谱时，cot参数必须为full或melody。")
        prefix=datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination=os.path.join(folder_paths.get_output_directory(),"audio", f"destination_{prefix}")
        
        if not only_plan:
            result=yue2_generate(model, request,vae_model=vae)
            result.save_artifacts(destination)
            waveform = torch.from_numpy(result.audio).permute(1, 0).to("cpu").float()
            print(f"Generated audio shape: {waveform.shape}, sample rate: {result.sample_rate}") # torch.Size([2,3836096]), sample rate: 48000
            return io.NodeOutput({"waveform": waveform.unsqueeze(0), "sample_rate": result.sample_rate})
        else:
            print("only_plan,no audio generated")
            from  .src.yue2.protocol import SongRequest
            request=SongRequest(**request)
            plan = model.plan(request=request, vae_model=vae)
            plan.save(destination)

            result = {"mode": request.cot, "truncated": {"abc": plan.truncated}}
            return io.NodeOutput({"waveform": torch.zeros(1, 1, 1), "sample_rate": 48000})  
        
from aiohttp import web
from server import PromptServer
import base64


@PromptServer.instance.routes.post("/yue_sm/get_file_path")
async def get_file_path(request):
    try:
        data = await request.json()
        filename = data.get('filename', 'temp.abc')
        content_base64 = data.get('content', '')
        if not content_base64:
            return web.json_response({"error": "No file content provided"}, status=400)
        file_content = base64.b64decode(content_base64)
        temp_dir = os.path.join(folder_paths.get_output_directory(), "yue_sm_temp")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
        print(f"File saved to: {file_path}")

        return web.json_response({"path": file_path})
    except Exception as e:
        print(f"Error in get_file_path: {str(e)}")
        return web.json_response({"error": str(e)}, status=500)