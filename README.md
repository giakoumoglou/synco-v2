## _ViTAMINS_: An Empirical Study of Training Self-Supervised Vision Transformers with Synthetic Hard Negatives

<div align="center">
  <img width="1000" height="376" alt="image" src="https://i.postimg.cc/xj6Dvq0j/teaser-vitamins.png" />
</div>

This is a PyTorch implementation of the [ViTAMINS paper](https://arxiv.org/abs/TO.DO), accepted at **WACV 2027**:
```
@misc{giakoumoglou2026vitamins,
      title={{ViTAMINS}: An Empirical Study of Training Self-Supervised Vision Transformers with Synthetic Hard Negatives}, 
      author={Nikos Giakoumoglou and Andreas Floros and Kleanthis-Marios Papadopoulos and Tania Stathaki},
      year={2026},
      eprint={2609.01041},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.01041}, 
}
```

It also contains the implementation of [BYOL](https://arxiv.org/abs/2006.07733), [MoBY](https://arxiv.org/abs/2105.04553) and [DINO](https://arxiv.org/abs/2104.14294).


## Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

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


## Code Structure

```
.
├── main_pretrain.py            # entrypoint: unsupervised pre-training
├── main_linear.py              # entrypoint: linear probing / fine-tuning
├── configs/                    # one folder per method, one file per architecture
│   ├── byol/                   # swin_tiny, swin_small, swin_base, vit_small, vit_base
│   ├── moby/                   # swin_tiny, swin_small, swin_base, vit_small, vit_base
│   ├── dino/                   # swin_tiny, swin_small, swin_base, vit_small, vit_base
│   └── vitamins/               # swin_tiny, swin_small, swin_base, vit_small, vit_base
├── scripts/                    # PBS job scripts, one folder per method
│   ├── byol/ moby/ dino/ vitamins/    …/{swin_tiny,swin_small,swin_base,vit_small,vit_base}.pbs
│   └── extract_ILSVRC2012.sh
└── src/
    ├── config.py               # default config and yaml merging
    ├── logger.py
    ├── lr_scheduler.py
    ├── optimizer.py
    ├── utils.py
    ├── datasets/               # data loading, ssl augmentations, transfer datasets
    └── models/
        ├── build.py            # model factory
        ├── byol.py
        ├── moby.py
        ├── dino.py
        ├── vitamins.py
        └── backbones/          # vision transformer and swin transformer
```

## Environment Setup

To set up a compatible environment (CUDA 11.7/11.8) on [PBS](https://en.wikipedia.org/wiki/Portable_Batch_System), follow these steps:

```bash
conda create -n vitamins -c conda-forge cudatoolkit=11.8 python=3.10.11
conda activate vitamins
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
    --cfg configs/vitamins/vit_base.yaml \
    --data-path [your imagenet-folder with train and val folders] \
    --batch-size 64 \
    --output [output folder] \
    --tag [tag folder]
```

To use BYOL, MoBY or DINO instead, swap the config file accordingly, e.g. `--cfg configs/dino/vit_base.yaml`.

### Submitting on PBS

Ready-made job scripts are provided for every method and architecture. Submit them **from the repository root**, since each script does `cd $PBS_O_WORKDIR`:

```bash
qsub scripts/vitamins/vit_base.pbs
```

Each script runs pre-training followed by linear evaluation, writes to `./logs/`, and requests the GPU count matching its architecture (4 GPUs at batch 128, or 8 GPUs at batch 64, for a total batch of 512 either way).

## Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=12345 \
    main_linear.py \
    --cfg configs/vitamins/vit_base.yaml \
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

This project is under the MIT license. See [LICENSE](LICENSE) for details.

### Acknowledgments

This codebase is built upon [MoBY](https://github.com/SwinTransformer/TransformerSSL), [SynCo](https://github.com/giakoumoglou/synco), [DINO](https://github.com/facebookresearch/dino), [Swin Transformer](https://github.com/microsoft/Swin-Transformer). We thank the authors for their excellent work and for making their code publicly available.
