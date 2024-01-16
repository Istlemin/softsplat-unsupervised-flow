# Forward Warping for Unsupervised Optical Flow

This repository explores the usage of forward warping for unsupervised optical flow. The implementation is based on the repository [Unsupervised-Optical-Flow](https://github.com/ily-R/Unsupervised-Optical-Flow), with significant modifications made to the loss calculation in `utils.py`, the models and warping operations in `models.py` and the training loop in `unsup_train.py`. `softsplat.py` contains two implementations of the splatting operation, an original one written by me in pure PyTorch, and one taken from [softmax-splatting](https://github.com/sniklaus/softmax-splatting). The PWC-NET implementation in `models.py` is based on [pytorch-pwc](https://github.com/sniklaus/pytorch-pwc).


To run training with softmax-splatting, use the following command (after placing the Flying Chairs dataset in ../datasets):
```
python unsup_train.py --root ../datasets/FlyingChairs_release/data/ --model pwc_net --path softsplat --lr 1e-4 --forward_splat
```