# DiffWater: Underwater Image Enhancement Based on Conditional Denoising Diffusion Probabilistic Model

```
This Repo includes the training and testing codes of our DiffWater. (Pytorch Version)
If you use our code, please cite our paper and hit the star at the top-right corner. Thanks!
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

## Evaluation

To resume from a checkpoint file, simply use the `--resume` argument in [test.py](test.py) to specify the checkpoint.

For your convenience, we provide all paired model used in our paper. [BaiduYun](https://pan.baidu.com/s/1_woeIfvT6zpUxn-3rItPCg ) password:gms1

## Training

To resume from a checkpoint file, simply use the `--resume` argument in  [train.py](train.py)  to specify the checkpoint.

## Acknowledgement

Our code is adapted from the original [SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement) repository. We thank the authors for sharing their code.
