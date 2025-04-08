## _SynCo_-v2: Unsupervised Training of Vision Transformers with Synthetic Negatives

This is a PyTorch implementation of the [SynCo-v2 paper](https://arxiv.org/abs/XXXX.XXXXX):
```
@misc{giakoumoglou2025syncov2,
      title={Unsupervised Training of Vision Transformers with Synthetic Negatives}, 
      author={Nikolaos Giakoumoglou and Andreas Floros and Kleanthis Marios Papadopoulos and Tania Stathaki},
      year={2025},
      eprint={0000.00000},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/0000.00000}, 
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

### Unsupervised Training

To do unsupervised pre-training with MoBY framework using DeiT-Small backbone on ImageNet on a 4-gpu machine, run:

```
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=12345 \
    main_pretrain.py \
    --cfg configs/moby_swin_tiny.yaml \
    --data-path [your imagenet-folder with train and val folders] \
    --batch-size 128 \
    --output [output folder] \
    --tag [tag folder] \
```

Similarly for SynCo or BYOL.

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

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
