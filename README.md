# Forward Warping for Unsupervised Optical Flow

This repository explores the usage of forward warping for unsupervised optical flow. The implementation is based on the repository [Unsupervised-Optical-Flow](https://github.com/ily-R/Unsupervised-Optical-Flow), with significant modifications made to the loss calculation in `utils.py`, the models and warping operations in `models.py` and the training loop in `unsup_train.py`. `softsplat.py` contains an implementation of the for


To run, 

```
python unsup_train.py --root ../datasets/FlyingChairs_release/data/ --model pwc_net --path softsplat --lr 1e-4 --forward_splat
```