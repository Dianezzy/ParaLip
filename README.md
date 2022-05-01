# Parallel and High-Fidelity Text-to-Lip Generation
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2107.06831)
[![GitHub Stars](https://img.shields.io/github/stars/Dianezzy/ParaLip?style=social)](https://github.com/Dianezzy/ParaLip)
[![downloads](https://img.shields.io/github/downloads/Dianezzy/ParaLip/total.svg)](https://github.com/Dianezzy/ParaLip/releases)


This repository is the official PyTorch implementation of our AAAI-2022 [paper](https://arxiv.org/abs/2107.06831), in which we propose ParaLip (for text-based talking face synthesis) .

## Video Demos
<img src=https://user-images.githubusercontent.com/48660888/166135987-d27e4ec7-d740-46ce-a12b-bbaaa151a613.gif width="300"/>

Video samples can be found in our [demo page](https://paralip.github.io/).
 
:rocket: **News**: 
 - Feb.24, 2022: Our new work, NeuralSVB was accepted by ACL-2022 [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2202.13277). [Project Page](https://neuralsvb.github.io).
 - Dec.01, 2021: ParaLip was accepted by AAAI-2022.
 - July.14, 2021: We submitted ParaLip to Arxiv [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2107.06831).

## Environments
```sh
conda create -n your_env_name python=3.7
source activate your_env_name 
pip install -r requirements.txt   
```

## ParaLip 
### 1. Preparation

#### Data Preparation
We provide the first frame of each test example for inference. Besides, we include the audio pieces of 5 test examples to generate talking lip videos with human voice.

a) Download and decompress the [TCD-TIMIT dataset](https://github.com/Dianezzy/ParaLip/releases/download/v0.1.0-alpha/timit.tar), then put them in the `data` directory
 
 ```sh
tar -xvf timit.tar
mv timit data/
```

b) Run the following scripts to pack the dataset for inference.

```sh
export PYTHONPATH=.
python datasets/lipgen/timit/gen_timit.py --config configs/lipgen/timit/lipgen_timit.yaml
```

We don't provide the full datasets of TCD-TIMIT because of the licence issue. You can download it by yourself if necessary.

### 2. Inference Example

```sh
CUDA_VISIBLE_DEVICES=0 python tasks/timit_lipgen_task.py --config configs/lipgen/timit/lipgen_timit.yaml --exp_name timit_2 --infer --reset        

```

We also provide:
 - the pre-trained model of [ParaLip on TCD-TIMIT](https://github.com/Dianezzy/ParaLip/releases/download/v0.1.0-alpha/model_ckpt_steps_32000.ckpt). 
Remember to put the pre-trained models in `checkpoints/timit_2` directory respectively.

                                                              
## Citation
```bib
@misc{https://doi.org/10.48550/arxiv.2107.06831,
  doi = {10.48550/ARXIV.2107.06831},
  
  url = {https://arxiv.org/abs/2107.06831},
  
  author = {Liu, Jinglin and Zhu, Zhiying and Ren, Yi and Huang, Wencan and Huai, Baoxing and Yuan, Nicholas and Zhao, Zhou},
  
  keywords = {Multimedia (cs.MM), Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Parallel and High-Fidelity Text-to-Lip Generation},
  
  publisher = {arXiv},
  
  year = {2021},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
    

