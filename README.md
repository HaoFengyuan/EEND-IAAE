# End-to-End Neural Speaker Diarization with an Iterative Adaptive Attractor Estimation

This project is the implementation of EEND-IAAE, which has been has been submitted to Neural Networks. There are two main parts in the proposed IAAE network: an attention-based pooling was designed to obtain a rough estimation of the attractors based on the diarization results of the previous iteration, and an adaptive attractor was then calculated by using transformer decoder blocks.

In this project, the primary basis was the original Chainer implementation of [EEND](https://github.com/hitachi-speech/EEND) and the PyTorch implementation [EEND-Pytorch](https://github.com/Xflick/EEND_PyTorch).

Notably, the project only encompassed the inferring phase. For specifics on data preparation, please refer to [there](https://github.com/hitachi-speech/EEND/blob/master/egs/callhome/v1/run_prepare_shared.sh). For details regarding the training phase, please refer to the [there](https://github.com/Xflick/EEND_PyTorch/blob/master/run.sh).

## Pretrained Models
We provide the pretrained SA-EEND trained on simulated data and real datasets respectively.

`exp/simu_EEND.th` was trained on Sim2spk with $\beta = 2$, and `exp/real_EEND.th` was adapted on the CALLHOME adaptation set. In the training phase, we basically followed the training protocol described in [the original paper](https://arxiv.org/abs/2003.02966).

Building upon these pretrained models, we can proceed to train the proposed EEND-IAAE.

## Citation
If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry.
```

```
