## _SynCo_-v2: An Empirical Study of Training Self-Supervised \\Vision Transformers with Synthetic Hard Negatives

<img width="1173" height="376" alt="image" src="https://github.com/user-attachments/assets/67e49f69-51e4-438b-a284-77e970dab956" />

This is a PyTorch implementation of the [SynCo-v2 paper](https://giakoumoglou.com/src/syncov2/syncov2-main.pdf), currently available at [giakoumoglou.com](giakoumoglou.com):
```
@misc{giakoumoglou2025syncov2,
      title={{SynCo-v2: An Empirical Study of Training Self-Supervised Vision Transformers with Synthetic Hard Negatives}}, 
      author={Nikolaos Giakoumoglou and Andreas Floros and Kleanthis Marios Papadopoulos and Tania Stathaki},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings Workshop},
      year={2026}
}
```

It also contains the implementation of [BYOL](https://arxiv.org/abs/2006.07733) and [MoBY](https://arxiv.org/abs/2105.04553).


## Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

This repo is based on [MoBY](https://github.com/SwinTransformer/TransformerSSL) and [SynCo](https://github.com/giakoumoglou/synco) code:

```bash
diff main_pretrain.py <(curl https://raw.githubusercontent.com/SwinTransformer/TransformerSSL/moby_main.py)
diff main_linear.py <(curl https://raw.githubusercontent.com/SwinTransformer/TransformerSSL/moby_linear.py)
```

The scripts expect the following dataset structure:

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


## Environment Setup

To set up a compatible environment (CUDA 11.7/11.8) on [PBS](https://en.wikipedia.org/wiki/Portable_Batch_System), follow these steps:

```bash
conda create -n syncov2 -c conda-forge cudatoolkit=11.8 python=3.10.11
conda activate syncov2
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install timm==0.4.9 diffdist Pillow pyyaml yacs termcolor scipy numpy==1.21.5 opencv-python tqdm
```

## Unsupervised Training

To do unsupervised pre-training of a ViT-Base model on ImageNet in an 8-gpu machine, run:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=12345 \
    main_pretrain.py \
    --cfg configs/synco_vit_base.yaml \
    --data-path [your imagenet-folder with train and val folders] \
    --batch-size 64 \
    --output [output folder] \
    --tag [tag folder]
```

To use MoBY or BYOL instead, swap the config file accordingly. For different architectures (Swin-Tiny, Swin-Small, Swin-Base, ViT-Small, ViT-Base), select the corresponding config from [`./configs`](configs).

## Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=12345 \
    eval_linear.py \
    --cfg configs/synco_vit_base.yaml \
    --data-path [your imagenet-folder with train and val folders] \
    --output [output folder] \
    --tag [tag folder]
```

Use the same `--cfg`, `--output`, and `--tag` as in pre-training. By default, this performs linear probing with frozen features (`LINEAR_EVAL.WEIGHTS freeze`). For full fine-tuning, set `LINEAR_EVAL.WEIGHTS finetune` in the config.


## Transfer Learning

To evaluate on downstream datasets (CIFAR-10, CIFAR-100, STL-10, Oxford Flowers102, Oxford Pets, Food101, Stanford Cars, Caltech101, DTD, FGVC Aircraft, SUN397, VOC2007, Places365), append `--opts DATA.DATASET <dataset>` to the linear evaluation command. For full fine-tuning, add `--opts LINEAR_EVAL.WEIGHTS finetune`.

## *k*-NN Classification, Image Retrieval, Video Object Segmentation

Use the evaluation scripts from the official [DINO repository](https://github.com/facebookresearch/dino).

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

### Acknowledgments

This codebase is built upon [MoBY](https://github.com/SwinTransformer/TransformerSSL), [SynCo](https://github.com/giakoumoglou/synco), [DINO](https://github.com/facebookresearch/dino), [Swin Transformer](https://github.com/microsoft/Swin-Transformer). We thank the authors for their excellent work and for making their code publicly available.
