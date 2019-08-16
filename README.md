## Few-Shot Learning with Global Class Representations
Created by <a href="https://tiangeluo.github.io/" target="_blank">Tiange Luo*</a>, <a href="" target="_black">Aoxue Li*</a>, <a href="http://personal.ee.surrey.ac.uk/Personal/T.Xiang/index.html" target="_blank">Tao Xiang</a>, <a href="https://www.weiranhuang.com/" target="_blank">Weiran Huang</a> and <a href="https://scholar.google.com/citations?user=VZHxoh8AAAAJ&hl=zh-CN" target="_blank">Liwei Wang</a>

![Overview](https://github.com/tiangeluo/fsl-global/blob/master/material/overview.png)


## Introduction
This is the repository for our ICCV 2019 paper (arXiv report [here](https://arxiv.org/abs/1908.05257)).

In this paper, we propose to tackle the challenging few-shot learning (FSL) problem by learning global class representations using both base and novel class training samples. In each training episode, an episodic class mean computed from a support set is registered with the global representation via a registration module. This produces a registered global class representation for computing the classification loss using a query set. Though following a similar episodic training pipeline as existing meta learning based approaches, our method differs significantly in that novel class training samples are involved in the training from the beginning. To compensate for the lack of novel class training samples, an effective sample synthesis strategy is developed to avoid overfitting. Importantly, by joint base-novel class training, our approach can be easily extended to a more practical yet challenging FSL setting, i.e., generalized FSL, where the label space of test data is extended to both base and novel classes. Extensive experiments show that our approach is effective for both of the two FSL settings.

For more details of our architecture, please refer to our paper or <a href="https://tiangeluo.github.io/GlobalRepresentation.html" target="_blank">project website</a>.

## Citation
If you find our work useful in your research, please consider citing:

        @article{luo2019few,
          title={Few-Shot Learning with Global Class Representations},
          author={Luo, Tiange and Li, Aoxue and Xiang, Tao and Huang, Weiran and Wang, Liwei},
          journal={arXiv preprint arXiv:1908.05257},
          year={2019}
        }

## About this repository
Due to company and patent issues, the authors are striving for releasing source codes. We will try our best and release the core code of the proposed module at least.
