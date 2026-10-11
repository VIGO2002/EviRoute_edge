# Upstream model and library notices

**Project and algorithm adaptation:** Chenwei He (何晨炜).

EviRoute-Edge implements research adaptation, paired evaluation and export
engineering using third-party pretrained models and libraries, including:

- **DINOv3** (pretrained vision representation), whose weights and any
  applicable distribution terms remain with their original authors.
- **timm (PyTorch Image Models)**: https://github.com/huggingface/pytorch-image-models
- **PyTorch and torchvision**: https://github.com/pytorch/pytorch
- **RKNN-Toolkit2** and RKNN runtime components from Rockchip, distributed
  under their respective SDK licenses: https://github.com/airockchip/rknn-toolkit2
- Other scientific Python libraries (NumPy, SciPy, pandas, scikit-learn,
  Pillow, OpenCV, ONNX, ONNX Runtime).

No third-party pretrained weights, RKNN runtime binaries or SDK distributions
are redistributed here. Original attribution and license requirements apply;
this file does not relicense upstream code or checkpoints.
