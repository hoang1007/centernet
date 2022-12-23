# Centernet: Object as Points

This repo contains implementations of the [CenterNet: Object as Points](https://arxiv.org/abs/1904.07850)

# Usage

To train the model, run the following command:

```bash
python3 src/train.py \
  experiment=centernet_resnet18 \
  trainer=gpu
```

For Model evaluation, view [this notebook](./notebooks/test.ipynb)