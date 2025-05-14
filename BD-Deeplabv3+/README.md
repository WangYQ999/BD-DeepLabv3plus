This project is an enhanced and modified version of [DeepLabv3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch), originally developed by [VainF].

# BD-DeepLabv3Plus-Pytorch

## Quick Start 

```python
python main.py --model deeplabv3plus_resnet50 --gpu_id 0 --lr 0.01 --crop_size 1024 --batch_size 4 --output_stride 16
```
Backbone:  deeplabv3plus_resnet50, deeplabv3plus_resnet101, deeplabv3plus_mobilenet.

## Acknowledgement
Many thanks to the original author for their excellent work. 

## Reference

[1] [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

[2] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)
