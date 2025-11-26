# DiCo: Learning Dynamic Collaborative Network for Semi-supervised 3D Vessel Segmentation (CVPR 2025)
## Introduction
> [**Learning Dynamic Collaborative Network for Semi-supervised 3D Vessel Segmentation**](),   <br/>
> **Jiao Xu**, Xin Chen, Lihe Zhang. <br/>
> In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025*  <br/>
> [[bibtex](https://github.com/xujiaommcome/DiCo)][[supp](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Learning_Dynamic_Collaborative_Network_for_Semi-supervised_3D_Vessel_Segmentation_CVPR_2025_paper.pdf)]

<div align="center" border=> <img src=framework.png width="700" > </div>

## News
- [2025.10.22] Our codes are released!
- [2025.06.16] Repo created. Paper and code will come soon.

## Installation
- PyTorch 2.1.2
- CUDA 11.8 
- Python 3.8.18

## Usage
### Dataset and Pre-processing

The datasets used in our paper are [ImageCAS dataset](https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas), 
[Parse2022 dataset](https://parse2022.grand-challenge.org/Dataset/), 
and [CAS2023 dataset](https://codalab.lisn.upsaclay.fr/competitions/9804).

### Training Steps

**Note:** Please replace the original `unetr.py` in the MONAI library with `code/networks/unetr.py` from this repository.  
For more details, please refer to issue #4.

1. Clone the repo and create data path:
```
git clone https://github.com/xujiaommcome/DiCo
cd DiCo
mkdir data # create data path
```
2. Put the preprocessed data and then
```cd code```
3. We train our model on one single NVIDIA 3090 GPU for each dataset.

To produce the claimed results for  Parse2022 dataset:
```
# For 5% labeled data,
CUDA_VISIBLE_DEVICES=0 python DiCo_main.py --labelnum=5

```

## Citation
If this code is useful for your research, please consider giving star to our repository and citing our work:
```
@inproceedings{xu2025learning,
  title={Learning Dynamic Collaborative Network for Semi-supervised 3D Vessel Segmentation},
  author={Xu, Jiao and Chen, Xin and Zhang, Lihe},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={10445--10454},
  year={2025}
}
```
## Questions
If you have any questions, welcome contact me at 'xjmmcome@mail.dlut.edu.cn'
