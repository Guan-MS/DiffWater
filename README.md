# DiffWater: Underwater Image Enhancement Based on Conditional Denoising Diffusion Probabilistic Model

[[paper](https://ieeexplore.ieee.org/document/10365517)]

```
This Repo includes the training and testing codes of our DiffWater. (Pytorch Version)
If you use our code, please cite our paper and hit the star at the top-right corner. Thanks!

@ARTICLE{10365517,
  author={Guan, Meisheng and Xu, Haiyong and Jiang, Gangyi and Yu, Mei and Chen, Yeyao and Luo, Ting and Zhang, Xuebo},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={DiffWater: Underwater Image Enhancement Based on Conditional Denoising Diffusion Probabilistic Model}, 
  year={2024},
  volume={17},
  number={},
  pages={2319-2335},
  keywords={Image color analysis;Colored noise;Noise reduction;Image restoration;Image enhancement;Visualization;Lighting;Color compensation;conditional denoising diffusion probabilistic model (DDPM);underwater image enhancement (UIE)},
  doi={10.1109/JSTARS.2023.3344453}}

```

## Environment

```
pip install -r requirement.txt
```

### Train and Test sets:


To make use of the [train.py](train.py) and [test.py](test.py)  the dataset folder names should be lower-case and structured as follows.

```
└──── <data directory>/
    ├──── UIEB_R90/
    |   ├──── input_256/
    |   |   ├──── 01.png/
    |   |   ├──── ...
    |   |   └──── 90.png/
    |   ├──── target_256/
		├──── 01.png/
		├──── ...
		└──── 90.png/
```

## Dataset

(1) LSUI : [Data](https://lintaopeng.github.io/_pages/UIE%20Project%20Page.html) 

(2) UIEB : [Data](https://li-chongyi.github.io/proj_benchmark.html) 

(3) SQUID : [Data](https://csms.haifa.ac.il/profiles/tTreibitz/datasets/ambient_forwardlooking/index.html) 

(4) U45 : [Data](https://github.com/IPNUISTlegal/underwater-test-dataset-U45-/tree/master) 

## The test results of the paper
[U90,C60,U45,S16,L504][Google](https://drive.google.com/drive/folders/19QJ1xAxZ4CPaEHjs96HL2dwPzH49LDdV) and [BaiduYun](https://pan.baidu.com/s/1uU_C2B6skEOEj7Nqp5ce1A)) password:gms1 
## Evaluation

To resume from a checkpoint file, simply use the `--resume` argument in [test.py](test.py) to specify the checkpoint.

For your convenience, we provide the pre-trained model in our paper. [BaiduYun](https://pan.baidu.com/s/1_woeIfvT6zpUxn-3rItPCg ) password:gms1  and [Google](https://drive.google.com/drive/folders/11T9ao0pmNFv9lVZLcVliNbZ3iJ5F2t7s)

## Training

To resume from a checkpoint file, simply use the `--resume` argument in  [train.py](train.py)  to specify the checkpoint.

## Acknowledgement

Our code is adapted from the original [SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement) repository. We thank the authors for sharing their code.

## Contact
If you have any questions, please contact Meisheng Guan at 1971306417@qq.com.
