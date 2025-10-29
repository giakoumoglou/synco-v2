## _SynCo_-v2: Unsupervised Training of Vision Transformers with Synthetic Negatives

This is a PyTorch implementation of the [SynCo-v2 paper](https://arxiv.org/abs/XXXX.YYYYY):
```
@inproceedings{giakoumoglou2025unsupervised,
title={Unsupervised Training of Vision Transformers with Synthetic Negatives},
author={Nikolaos Giakoumoglou and Andreas Floros and Kleanthis Marios Papadopoulos and Tania Stathaki},
booktitle={Proceedings of the CVPR 2025 Workshop on Visual Concepts},
year={2025},
url={https://openreview.net/forum?id=dg8FuaOKnC}
}
```

```
@inproceedings{giakoumoglou2025syncov2,
title={SynCo-v2: An Empirical Study of Training Self-Supervised Vision Transformers with Synthetic Hard Negatives},
author={Nikolaos Giakoumoglou and Andreas Floros and Kleanthis Marios Papadopoulos and Tania Stathaki},
year={2025},
eprint={XXXX.YYYYY},
archivePrefix={arXiv},
primaryClass={cs.CV},
url={https://arxiv.org/abs/XXXX.YYYYY}, 
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

To set up the environment correctly in [PBS](https://en.wikipedia.org/wiki/Portable_Batch_System), follow these steps:

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
pip install opencv-python
pip install tqdm
```

**Note:** These commands set up a complete environment compatible with CUDA 11.7/11.8 and the specific PyTorch version required for this implementation.

## Unsupervised Training

To do unsupervised pre-training with MoBY framework using ViT-Small backbone on ImageNet on a 4-gpu machine, run:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=12345 \
    main_pretrain.py \
    --cfg configs/moby_vit_tiny.yaml \
    --data-path [your imagenet-folder with train and val folders] \
    --batch-size 128 \
    --output [output folder] \
    --tag [tag folder]
```

To run unsupervised pre-training with **SynCo** or **BYOL** instead of MoBY, you would use the same command structure as shown in the MoBY example, but you'd need to change the configuration file to point to the appropriate framework's config. For different model architectures such as **Swin-Tiny**, **Swin-Small**, **Swin-Base**, **ViT-Small**, or **ViT-Base**, simply select the corresponding configuration file (see [./configs](configs)) while keeping all other command parameters the same.

## Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights on a 4-gpu machine, run:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=12345 \
    eval_linear.py \
    --cfg configs/moby_swin_tiny.yaml \
    --data-path [your imagenet-folder with train and val folders] \
    --output [output folder] \
    --tag [tag folder]
```

Make sure the `config` file, `output` directory and `tag` are the same as in the pre-training stage.

**Note:** By default, this performs linear evaluation with frozen features (`LINEAR_EVAL.WEIGHTS freeze`). For full fine-tuning instead, the configuration would set `LINEAR_EVAL.WEIGHTS finetune`.


## _k_-NN Classification

To evaluate a model with k-NN classification on ImageNet validation set, run:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=12345 \
    eval_knn.py \
    --cfg configs/moby_swin_tiny.yaml \
    --data-path [your imagenet-folder with train and val folders] \
    --output [output folder] \
    --nb-knn 10 20 100 200 \
    --temperature 0.07
```

**Optional arguments:**
- `--nb-knn`: List of k values to test (default: 10, 20, 100, 200)
- `--temperature`: Temperature for the weighted k-NN (default: 0.07)
- `--dump-features [path]`: Save extracted features to avoid recomputation
- `--load-features [path]`: Load precomputed features


## Image Retrieval

We evaluate our models on the revisited Oxford and Paris datasets following the [DINO evaluation protocol](https://github.com/facebookresearch/dino).

Download the revisited Oxford and Paris datasets:

```bash
# Download Oxford5k and Paris6k
mkdir -p data/revisited_paris_oxford
cd data/revisited_paris_oxford

# Oxford
wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz
mkdir -p roxford5k/jpg && tar -xzf oxbuild_images.tgz -C roxford5k/jpg

# Paris
wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1.tgz
wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2.tgz
mkdir -p rparis6k/jpg
tar -xzf paris_1.tgz -C rparis6k/jpg
tar -xzf paris_2.tgz -C rparis6k/jpg

# Download ground truth
wget https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz
tar -xzf gt_files_170407.tgz -C roxford5k/

wget https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_120310.tgz
tar -xzf paris_120310.tgz -C rparis6k/

# Download revisited annotations
git clone https://github.com/filipradenovic/revisitop.git
python revisitop/python/dataset.py --dataset roxford5k --data_root roxford5k
python revisitop/python/dataset.py --dataset rparis6k --data_root rparis6k

cd ../..
```

Evaluate image retrieval on Oxford5k:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=12345 \
    eval_retrieval.py \
    --cfg configs/moby_swin_tiny.yaml \
    --data-path ./data/revisited_paris_oxford \
    --dataset roxford5k \
    --output [output folder]
```

For Paris6k, simply change `--dataset roxford5k` to `--dataset rparis6k`.

## Video Object Segmentation

We evaluate video object segmentation on the DAVIS 2017 validation set following the [DINO evaluation protocol](https://github.com/facebookresearch/dino).

Download DAVIS 2017 dataset:

```bash
# Create directory
mkdir -p data/davis
cd data/davis

# Download DAVIS 2017 (TrainVal 480p)
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip

# Unzip
unzip DAVIS-2017-trainval-480p.zip

cd ../..
```

Evaluate video object segmentation on DAVIS 2017:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=12345 \
    eval_video_seg.py \
    --cfg configs/moby_swin_tiny.yaml \
    --data-path ./data/davis \
    --output [output folder]
```

After running the evaluation, use the official [DAVIS evaluation toolkit](https://github.com/davisvideochallenge/davis2017-evaluation) to compute J&F metrics:

```bash
# Install DAVIS evaluation toolkit
pip install git+https://github.com/davisvideochallenge/davis2017-evaluation

# Evaluate
python -m davis2017.evaluation --results_path [output folder] --davis_path ./data/davis
```


## Transfer Learning

With a pre-trained model, to evaluate on downstream datasets (**CIFAR-10**, **CIFAR-100**, **STL-10**, **Oxford Flowers102**, **Oxford Pets**, **Food101**, **Stanford Cars**, **Caltech101**, **DTD**, **FGVC Aircraft**, **SUN397**, **VOC2007**, **Places365**) using linear probing on a 4-gpu machine, run:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=12345 \
    eval_linear.py \
    --cfg configs/moby_swin_tiny.yaml \
    --data-path [your imagenet-folder with train and val folders] \
    --output [output folder] \
    --tag [tag folder] \
```

For **full fine-tuning** instead of linear probing (by default `LINEAR_EVAL.WEIGHTS freeze`), add: 
```bash
--opts LINEAR_EVAL.WEIGHTS finetune
```

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

### Acknowledgments

This codebase is built upon [MoBY](https://github.com/SwinTransformer/TransformerSSL),[SynCo](https://github.com/giakoumoglou/synco), [DINO](https://github.com/facebookresearch/dino), [Swin Transformer](https://github.com/microsoft/Swin-Transformer). We thank the authors for their excellent work and for making their code publicly available.
