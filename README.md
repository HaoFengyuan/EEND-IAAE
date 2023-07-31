# End-to-End Neural Speaker Diarization with an Iterative Adaptive Attractor Estimation

This project is the implementation of EEND-IAAE, which has been accepted by Neural Networks. There were two main parts in the proposed IAAE network: an attention-based pooling was designed to obtain a rough estimation of the attractors based on the diarization results of the previous iteration, and an adaptive attractor was then calculated by using transformer decoder blocks.

In this project, the primary basis was the original Chainer implementation of [EEND](https://github.com/hitachi-speech/EEND) and the PyTorch implementation [EEND-Pytorch](https://github.com/Xflick/EEND_PyTorch).

Notably, the project only encompassed the inference phase. For specifics on data preparation, please refer to [there](https://github.com/hitachi-speech/EEND/blob/master/egs/callhome/v1/run_prepare_shared.sh). For details regarding the training phase, please refer to the [there](https://github.com/Xflick/EEND_PyTorch/blob/master/run.sh).

## Pretrained Models
We provided the pretrained SA-EEND trained on simulated data and real datasets respectively.

`exp/simu_EEND.th` was trained on Sim2spk with $\beta = 2$, and `exp/real_EEND.th` was adapted on the CALLHOME adaptation set. In the training phase, we basically followed the training protocol described in [the original paper](https://arxiv.org/abs/2003.02966).

Building upon these pretrained models, we can proceed to train the proposed EEND-IAAE.

## Results

We chose the DER results at the 2nd iteration step for further comparison with other different variants of the EEND.

| Method |  <br> $\beta = 2$ | Sim2spk <br> $\beta = 3$ |  <br> $\beta = 5$ | CALLHOME |
|:-|:-:|:-:|:-:|:-:|
| __Clustering-based__ |
| &emsp; i-vector + AHC | 33.74 | 30.93 | 25.96 | 12.10 |
| &emsp; x-vector (TDNN) + AHC | 28.77 | 24.46 | 19.78 | 11.53 |
| __EEND-based__ |
| &emsp; BLSTM-EEND | 12.28 | 14.36 | 19.69 | 26.03 |
| &emsp; SA-EEND | 4.56 | 4.50 | 3.85 | 9.54 |
| &emsp; CB-EEND | 2.85 | N/A | N/A | 9.70 |
| &emsp; RX-EEND | 4.18 | 3.93 | 4.01 | 9.17 |
| &emsp; EEND-NAA | 2.97 | 2.77 | 3.39 | 7.83 |
| &emsp; AL-EEND | 4.29 | 4.11 | 4.15 | 8.67 |
| &emsp; EEND-IAAE, $(i=2)$ | __2.83__ | __2.57__ | __3.23__ | __7.58__ |

(\* More details can be found in the paper.)


## Citation
If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry.
```
@article{Hao2023NN,
         title = {{End-to-End} Neural Speaker Diarization with an Iterative Adaptive Attractor Estimation},
         journal = {Neural Networks},
         year = {2023},
         doi = {Accepted},
         author = {Fengyuan Hao and Xiaodong Li and Chengshi Zheng}
}
```
