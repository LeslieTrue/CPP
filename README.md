# CPP(Clustering via Principle of rate reduction and Pretrained model)

by [Tianzhe Chu*](https://tianzhechu.com), [Shengbang Tong*](https://tsb0601.github.io/petertongsb/), [Tianjiao Ding*](https://tianjiaoding.com), [Xili Dai](https://delay-xili.github.io/), [Benjamin David Haeffele](https://www.cis.jhu.edu/~haeffele/), [Rene Vidal](http://vision.jhu.edu/rvidal.html), [Yi Ma](http://people.eecs.berkeley.edu/~yima/) (* means equal contribution)

## Introduction
This repo is the official implementation for the paper "Image Clustering via the Principle of Rate Reduction in the Age of Pretrained Models".
This paper proposes a novel image clustering pipeline that integrates pre-trained models and rate reduction, enhancing clustering accuracy and introducing an effective self-labeling algorithm for unlabeled datasets at scale.

## Version

(2023.6 Version 0) Not yet ready, still updating!

## Install Dependencies

We adopt the pretrained CLIP model from OpenAI's official repository https://github.com/openai/CLIP. To install all the dependencies, run the following command:
```python
pip install -r requirements.txt
```
## Preparing Data
Since we use CLIP's image encoder as a frozen backbone, there are two ways to define the networks: i. with backbone inside; ii. without the backbone; which correspond to the following ways of dataset initialization respectively.
### i. As RGB Images
It's a regular procedure to train a CPP model using datasets with RGB images when defining the network with backbone inside.
### ii. As CLIP Features
To reduce the inference time of frozen pretrained networks, we suggest to preprocess the dataset using CLIP's image encoder and train a CPPNet using a network without the backbone.

The following command will help to preprocess datasets into CLIP features without shuffling. 

```python
python ./data/preprocess.py --data imagenet --path ./data --feature_dir ./imagenet-feature.pt
```

## Training
Example training command for CIFAR-10

```python
python main.py --data_dir ./data --bs 1024 --desc train_CPP_CIFAR10\
 --lr 1e-4 --lr_c 1e-4 --pieta 0.175 --epo 15 --hidden_dim 4096 --z_dim 128 --warmup 50
```

## Optimal Number of Clusters Measurement
```python
python optimalcluster.py
```

## Self-Labeling
```python
python labeling.py
```

## Cite

If you find the repo useful or interesting, plz give a star~~ :D

```
@article{chu2023image,
  title={Image Clustering via the Principle of Rate Reduction in the Age of Pretrained Models},
  author={Chu, Tianzhe and Tong, Shengbang and Ding, Tianjiao and Dai, Xili and Haeffele, Benjamin David and Vidal, Rene and Ma, Yi},
  journal={arXiv preprint arXiv:2306.05272},
  year={2023}
}
```

