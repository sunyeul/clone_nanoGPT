# clone_nanoGPT

This repository contains the code for the repository [nanoGPT](https://github.com/karpathy/nanoGPT)

## quick start

We download it as a single (1MB) file and turn it from raw text into one large stream of integers:

```python
$ poetry run python data/shakespeare_char/prepare.py
```

This creates a train.bin and val.bin in that data directory. Now it is time to train your GPT. The size of it very much depends on the computational resources of your system:

I have a GPU. Great, we can quickly train a baby GPT with the settings provided in the config/train_shakespeare_char.py config file:

```python
$ poetry run python src/train.py configs/train_shakespeare_char.py
```

I only have a macbook (or other cheap computer). No worries, we can still train a GPT but we want to dial things down a notch. A simple train run could look as follows:

```python
$ poetry run python src/train.py configs/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

```python
$ poetry run python src/sample.py --out_dir=out-shakespeare-char
```
