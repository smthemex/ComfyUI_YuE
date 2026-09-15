# ComfyUI_YuE
[YuE](https://github.com/multimodal-art-projection/YuE) is a groundbreaking series of open-source foundation models designed for music generation, specifically for transforming lyrics into full songs (lyrics2song). you can use it in comfyUI

# Update
* update to yue2 / 简单更新到yue2版本； 

# 1. Installation

In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_YuE.git
```
---

# 2. Requirements  
```
pip install -r requirements.txt
```

# 3.models
* 3.1 base infer [3B dit](https://huggingface.co/m-a-p/YuE2-3B) and [vae](https://huggingface.co/m-a-p/YuE2-Vae) ,only model.safetensors and vae.safetensors is needed /只需要模型文件；  
* 3.1 cover mode  [SheetSage2](https://huggingface.co/m-a-p/SheetSage2) and [MERT2](https://huggingface.co/m-a-p/MERT-v2-FullSong) ,only model.safetensors and vae.safetensors is needed /只需要模型文件; 

* base infer:   
```
--   ComfyUI/models/diffusion_models
    ├── yue2_model.safetensors # rename from model.safetensor or not 
--   ComfyUI/models/vae
    ├── yue2_vae.safetensors # rename from model.safetensor or not 
```
* if use cover:  
```
--   ComfyUI/models/clip
    ├── mert2_model.safetensors # rename from model.safetensor or not 
    ├── sheetsage2.safetensors # rename from model.safetensor or not 
```

# 4.Example
![](https://github.com/smthemex/ComfyUI_YuE/blob/main/example_workflows/example.png)

  
# 5.Citation
```
@article{li2023mert,
  title = {{MERT}: Acoustic Music Understanding Model with Large-Scale Self-supervised Training},
  author = {Li, Yizhi and Yuan, Ruibin and Zhang, Ge and Ma, Yinghao and Chen, Xingran and Yin, Hanzhi and Xiao, Chenghao and Lin, Chenghua and Ragni, Anton and Benetos, Emmanouil and Gyenge, Norbert and Dannenberg, Roger and Liu, Ruibo and Chen, Wenhu and Xia, Gus and Shi, Yemin and Huang, Wenhao and Wang, Zili and Guo, Yike and Fu, Jie},
  journal = {arXiv preprint arXiv:2306.00107},
  year = {2023},
  eprint = {2306.00107},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2306.00107}
}

@article{yuan2025yue,
  title = {{YuE}: Scaling Open Foundation Models for Long-Form Music Generation},
  author = {Yuan, Ruibin and Lin, Hanfeng and Guo, Shuyue and Zhang, Ge and Pan, Jiahao and Zang, Yongyi and Liu, Haohe and Liang, Yiming and Ma, Wenye and Du, Xingjian and Du, Xinrun and Ye, Zhen and Zheng, Tianyu and Jiang, Zhengxuan and Ma, Yinghao and Liu, Minghao and Tian, Zeyue and Zhou, Ziya and Xue, Liumeng and Qu, Xingwei and Li, Yizhi and Wu, Shangda and Shen, Tianhao and Ma, Ziyang and Zhan, Jun and Wang, Chunhui and Wang, Yatian and Chi, Xiaowei and Zhang, Xinyue and Yang, Zhenzhu and Wang, Xiangzhou and Liu, Shansong and Mei, Lingrui and Li, Peng and Wang, Junjie and Yu, Jianwei and Pang, Guojian and Li, Xu and Wang, Zihao and Zhou, Xiaohuan and Yu, Lijun and Benetos, Emmanouil and Chen, Yong and Lin, Chenghua and Chen, Xie and Xia, Gus and Zhang, Zhaoxiang and Zhang, Chao and Chen, Wenhu and Zhou, Xinyu and Qiu, Xipeng and Dannenberg, Roger and Liu, Jiaheng and Yang, Jian and Huang, Wenhao and Xue, Wei and Tan, Xu and Guo, Yike},
  journal = {arXiv preprint arXiv:2503.08638},
  year = {2025},
  eprint = {2503.08638},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2503.08638}
}

```

