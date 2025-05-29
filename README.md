## _SynCo_-v2: Unsupervised Training of Vision Transformers with Synthetic Negatives

This is a PyTorch implementation of the [SynCo-v2 paper](https://arxiv.org/abs/XXXX.XXXXX):
```
@inproceedings{giakoumoglou2025syncov2,
title={Unsupervised Training of Vision Transformers with Synthetic Negatives},
author={Nikolaos Giakoumoglou and Andreas Floros and Kleanthis Marios Papadopoulos and Tania Stathaki},
booktitle={Proceedings of the CVPR 2025 Workshop on Visual Concepts},
year={2025},
url={https://openreview.net/forum?id=dg8FuaOKnC}
}

```
It also contains the implementation of [BYOL paper](https://arxiv.org/abs/2006.07733):

```
@misc{grill2020byol,
      title={Bootstrap your own latent: A new approach to self-supervised Learning}, 
      author={Jean-Bastien Grill and Florian Strub and Florent Altché and Corentin Tallec and Pierre H. Richemond and Elena Buchatskaya and Carl Doersch and Bernardo Avila Pires and Zhaohan Daniel Guo and Mohammad Gheshlaghi Azar and Bilal Piot and Koray Kavukcuoglu and Rémi Munos and Michal Valko},
      year={2020},
      eprint={2006.07733},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2006.07733}, 
}
```
and [MoBY paper](https://arxiv.org/abs/2105.04553):
```
@misc{xie2021moby,
      title={Self-Supervised Learning with Swin Transformers}, 
      author={Zhenda Xie and Yutong Lin and Zhuliang Yao and Zheng Zhang and Qi Dai and Yue Cao and Han Hu},
      year={2021},
      eprint={2105.04553},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2105.04553}, 
}
```

### Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

This repo is based on [MoBY](https://github.com/SwinTransformer/TransformerSSL) and [SynCo](https://github.com/giakoumoglou/synco) code:
```
diff main_pretrain.py <(curl https://raw.githubusercontent.com/SwinTransformer/TransformerSSL/moby_main.py)
diff main_linear.py <(curl https://raw.githubusercontent.com/SwinTransformer/TransformerSSL/moby_linear.py)
```

The scripts expect the following dataset structures:

```
[your imagenet-folder with train and val folders]/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

### Environment Setup

To set up the environment correctly, follow these steps:

```bash
conda create -n moby -c conda-forge cudatoolkit=11.8 python=3.10.11
conda activate moby
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install timm==0.4.9 --user
pip install diffdist --user
pip install Pillow --user
pip install pyyaml
pip install yacs
pip install termcolor
pip install scipy
pip install numpy==1.21.5
```

**Note:** These commands set up a complete environment compatible with CUDA 11.7/11.8 and the specific PyTorch version required for this implementation.

### Unsupervised Training

To do unsupervised pre-training with MoBY framework using ViT-Small backbone on ImageNet on a 4-gpu machine, run:

```
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=12345 \
    main_pretrain.py \
    --cfg configs/moby_vit_tiny.yaml \
    --data-path [your imagenet-folder with train and val folders] \
    --batch-size 128 \
    --output [output folder] \
    --tag [tag folder] \
```

To run unsupervised pre-training with SynCo or BYOL instead of MoBY, you would use the same command structure as shown in the MoBY example, but you'd need to change the configuration file to point to the appropriate framework's config. For different model architectures such as Swin-Tiny, Swin-Small, Swin-Base, ViT-Small, or ViT-Base, simply select the corresponding configuration file (see [./configs](configs)) while keeping all other command parameters the same.

### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 4-gpu machine, run:

```
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=12345 \
    main_linear.py \
    --cfg configs/moby_swin_tiny.yaml \
    --data-path [your imagenet-folder with train and val folders] \
    --output [output folder] \
    --tag [tag folder] \
```

Make sure the ```config``` file, ```output``` director and ```tag``` are the same as in the pre-training stage.

### Transfer Learning

With a pre-trained model, to evaluate on downstream datasets (CIFAR-10, CIFAR-100, STL-10, Oxford Flowers102, Oxford Pets, Food101, Stanford Cars, Caltech101, DTD, FGVC Aircraft, SUN397, VOC2007, Places365) using linear probing on a 4-gpu machine, run:

```
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=12345 \
    main_linear.py \
    --cfg configs/moby_swin_tiny.yaml \
    --data-path [your imagenet-folder with train and val folders] \
    --output [output folder] \
    --tag [tag folder] \
    --opts DATA.DATASET cifar10
```

For full finetuning instead of linear probing (by default ```LINEAR_EVAL.WEIGHTS frozen```), add: ```--opts DATA.DATASET cifar10 LINEAR_EVAL.WEIGHTS finetune```.

Replace `cifar10` with the desired dataset: `cifar100`, `stl10`, `flowers`, `pets`, `food101`, `cars`, `caltech101`, `dtd`, `aircraft`, `sun397`, `voc2007`, or `places365`. Datasets will be automatically downloaded to `./data/` directory.
### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
