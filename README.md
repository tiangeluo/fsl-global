## Few-Shot Learning with Global Class Representations
Created by <a href="https://tiangeluo.github.io/" target="_blank">Tiange Luo*</a>, <a href="" target="_black">Aoxue Li*</a>, <a href="http://personal.ee.surrey.ac.uk/Personal/T.Xiang/index.html" target="_blank">Tao Xiang</a>, <a href="" target="_blank">Weiran Huang</a> and <a href="http://www.liweiwang-pku.com" target="_blank">Liwei Wang</a>

![Overview](https://github.com/tiangeluo/fsl-global/blob/master/material/overview.png)

## About this repository
This is the repository for our ICCV 2019 [paper](https://arxiv.org/abs/1908.05257). In this paper, we propose to tackle the challenging few-shot learning (FSL) problem by learning global class representations for each class via involving both base and novel classes training samples from the beginning. For more details of our framework, please refer to our [paper](https://arxiv.org/abs/1908.05257) or <a href="https://tiangeluo.github.io/GlobalRepresentation.html" target="_blank">project website</a>.

Due to company and patent issues (patent #[201910672533.7](https://www.vipzhuanli.com/patent/201910672533.7/)), the author only release the codes of the proposed module. If you want to run those codes, you have to implement and train the hallucinator proposed in [Low-shot learning from imaginary data](https://arxiv.org/abs/1801.05401).

## Generalized FSL in the paper
The generalized FSL test way proposed in the paper is highly similar to the typical 5-way-5-shot FSL. The only difference is that we would do a **all-way** classification (100-way in section 4.2.1, 120-way classification in section 4.3.2). We set three types of evaluation. Please refer to section 4.2.1, in each episode, we will sample 5 classes from base classes (acc_b), novel classes (acc_n), or all classes (acc_a). Then, sampling 5 shots from each class. Finally, for all three types, we will do a all-way classification.

In section 4.2.1, follow the class splits as the original miniImageNet, we use the validation set to tune hyper-parameters, use both the training and validation sets as the base classes (80), and use the test set as the novel classes (20). In section 4.3.2, we still use the original training and validation sets as the base classes (80), but use the test set and the new added 20 classes as the novel classes (40).

## Usage

- convnet.py: the network for extracting feature and registrating.
- train_100way.py: train the model on the 100-way setting.
- test_100way.py: test in generalized FSL setting descrebed in section 4.2.1.
- samplers.py: code for sampling data in a 5-way-5-shot manner.
- csv: dataset splits.


## Citation
If you find our work useful in your research, please consider citing:

        @article{luo2019few,
          title={Few-Shot Learning with Global Class Representations},
          author={Luo, Tiange and Li, Aoxue and Xiang, Tao and Huang, Weiran and Wang, Liwei},
          journal={arXiv preprint arXiv:1908.05257},
          year={2019}
        }
