from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .yue_node import YUE_SM_Model,YUE_SM_Vae, YUE_SM_Sampler,YUE_SM_Clip,YUE_SM_Cond

WEB_DIRECTORY = "./js"

class YUE_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            YUE_SM_Model,
            YUE_SM_Vae,
            YUE_SM_Sampler,
            YUE_SM_Clip,
            YUE_SM_Cond,
        ]   

async def comfy_entrypoint() -> YUE_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return YUE_SM_Extension()


