# To The Point: Correspondence-driven monocular 3D category reconstruction (TTP)
[Filippos Kokkinos](https://fkokkinos.github.io/)  and [Iasonas Kokkinos](http://www0.cs.ucl.ac.uk/staff/I.Kokkinos/)

[Paper](https://arxiv.org/abs/2106.05662)
[Project Page](https://fkokkinos.github.io/to_the_point/)

<img src="https://fkokkinos.github.io/to_the_point/resources/images/teaser.jpg" width="80%">

## Requirements
* Python 3.6+
* PyTorch 1.7.1
* PyTorch3D 0.4.0
* cuda 11.0



## Installation Instructions

Clone code from repo:
```
git clone https://github.com/fkokkinos/to_the_point_3d_reconstruction
cd to_the_point_3d_reconstruction/
```

Setup Conda Env:
* Create Conda environment
```
conda create -n ttp
conda activate ttp
conda env update --file conda_env.yml
```

* Install other packages and dependencies

  Refer [here](https://pytorch.org/get-started/previous-versions/) for pytorch installation
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -c pytorch
```

* Setup Pytorch3D

  To setup Pytorch3D follow the instructions located [here](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)


* Install torch parallel svd 
  
  We are using the torch-batch-svd package to compute in parallel multiple SVDs. To install follow instructions from https://github.com/KinglittleQ/torch-batch-svd

## Training and Evaluation Instructions

#### Setup annotations and data directories


#### Download dataset pre-trained models
Download pre-trained models from [here](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabfko_ucl_ac_uk/EbRRWms5c4pEuqJObPIq0uYBZXBeJojT0BgvFKbE3exHTg?e=TJDrWI) and unzip in misc folder

```
cd misc/
7z t cachedir.7z

```
CUB dataset can be downloaded from [here](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). Unzip the file in misc/, else set the --cub_dir argument appropriately.

#### Training and Testing on CUB dataset
#####  Training:

Training on CUB is done with 
```
python main.py --name EXP_NAME --kp_loss_wt 0. --mask_loss_wt 2. --rigid_wt 3. --bdt_reg_wt 3. --print_freq 12 --display_freq 100 --tex_loss_wt 1. --tex_dt_loss_wt 1. --equiv_loss_wt 1. --weighted_camera --split train --vis_loss_wt 1. --batch_size 8 --flip_train --learnable_kp=False --mesh_dir meshes/bird_v2.obj --kp_dict meshes/bird_kp_dictionary_v3.pkl --tex_subdivision 1 --triangle_reg_wt 1. --tri_basis_wt 1. --basis_k 16 --arap_reg_wt 3. --betas_loss_wt 0. --sil_loss_wt 0. --normal_loss_wt 0. --def_loss_wt 5. --basis_cycle_loss_wt 0. --arap_basis_loss_wt 0.1 --save_epoch_freq 20 --def_steps 4
```

Exact hyper-parameters used for training of pre-trained models can be found in misc/cachedir/snapshots folder. 



##### Evaluation:


CUB experiment trained with keypoints:
```
python3 -m benchmark.evaluate --name bird_kp  --num_train_epoch 130 --split test --weighted_camera  --basis_k 16 --tex_subdivision 1 --mesh_dir meshes/bird_v2.obj  --kp_dict meshes/bird_kp_dictionary_v3.pkl --def_steps 4  --split test
```

CUB experiment trained without keypoints:
```
python3 -m benchmark.evaluate --name bird_nokp  --num_train_epoch 130 --split test --weighted_camera  --basis_k 16 --tex_subdivision 1 --mesh_dir meshes/bird_v2.obj  --kp_dict meshes/bird_kp_dictionary_v3.pkl --def_steps 4  --split test
```


## BibTex
If you find the code useful for your research, please consider citing:-
```
@inproceedings{
               kokkinos2021to,
               title={To The Point: Correspondence-driven monocular 3D category reconstruction},
               author={Filippos Kokkinos and Iasonas Kokkinos},
               booktitle={Advances in Neural Information Processing Systems},
               editor={A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
               year={2021},
               url={https://openreview.net/forum?id=AWMU04iXQ08}
               }
```

## Acknowledgements

This code repository uses code from [ACFM](https://github.com/fkokkinos/acfm_video_3d_reconstruction), [CMR](https://github.com/akanazawa/cmr/), [CSM](https://github.com/nileshkulkarni/csm/), and [BPnPNet](https://github.com/dylan-campbell/bpnpnet) repos.

## Contact
For questions feel free to contact me at filippos.kokkinos[at]ucl.ac.uk .