![logo](https://camo.githubusercontent.com/796826bb3b2df53e9e47da85833adc751f0a415b/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f372f37372f54726964656e745f6c6f676f2e737667)
# HiGradPy

This is the Python package for HiGrad (Hierarchical Incremental Gradient Descent), an algorithm for statistical inference for online learning and stochastic approximation.

#### Description
Stochastic gradient descent (SGD) is an immensely popular approach for online learning in settings where data arrives in a stream or data sizes are very large. However, despite an ever-increasing volume of work on SGD, much less is known about the statistical inferential properties of SGD-based predictions.

Taking a fully inferential viewpoint, this paper introduces a novel procedure termed HiGrad to conduct statistical inference for online learning, without incurring additional computational cost compared with the vanilla SGD. The HiGrad procedure begins by performing SGD iterations for a while and then split the single thread into a few, and this procedure hierarchically operates in this fashion along each thread.

With predictions provided by multiple threads in place, a t-based confidence interval is constructed by decorrelating predictions using covariance structures given by the Ruppert–Polyak averaging scheme. Under certain regularity conditions, the HiGrad confidence interval is shown to attain asymptotically exact coverage probability.

#### Installation
```python
pip install higradpy
```

#### Reference
Weijie Su and Yuancheng Zhu. (2018) Statistical Inference for Online Learning and Stochastic Approximation via Hierarchical Incremental Gradient Descent.
