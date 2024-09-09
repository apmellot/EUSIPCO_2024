# EUSIPCO-2024

## Physics-informed and Unsupervised Riemannian Domain Adaptation for Machine Learning on Heterogeneous EEG Datasets
This repository contains the code and tools to reproduce the results obtained in the paper: [Physics-informed and Unsupervised Riemannian Domain Adaptation for Machine Learning on Heterogeneous EEG Datasets](https://arxiv.org/pdf/2403.15415)

## Abstract

Combining electroencephalogram (EEG) datasets for supervised machine learning (ML) is challenging due to session, subject, and device variability. ML algorithms typically require identical features at train and test time, complicating analysis due to varying sensor numbers and positions across datasets. Simple channel selection discards valuable data, leading to poorer performance, especially with datasets sharing few channels. To address this, we propose an unsupervised approach leveraging EEG signal physics. We map EEG channels to fixed positions using field interpolation, facilitating source-free domain adaptation. Leveraging Riemannian geometry classification pipelines and transfer learning steps, our method demonstrates robust performance in brain-computer interface (BCI) tasks and potential biomarker applications. Comparative analysis against a statistical-based approach known as Dimensionality Transcending, a signal-based imputation called ComImp, source-dependent methods, as well as common channel selection and spherical spline interpolation, was conducted with leave-one-dataset-out validation on six public BCI datasets for a right-hand/left-hand classification task. Numerical experiments show that in the presence of few shared channels in train and test, the field interpolation consistently outperforms other methods, demonstrating enhanced classification performance across all datasets. When more channels are shared, field interpolation was found to be competitive with other methods and faster to compute than source-dependent methods.

## Citation
```
@article{mellot2024physics,
  title={Physics-informed and Unsupervised Riemannian Domain Adaptation for Machine Learning on Heterogeneous EEG Datasets},
  author={Mellot, Apolline and Collas, Antoine and Chevallier, Sylvain and Engemann, Denis and Gramfort, Alexandre},
  journal={arXiv preprint arXiv:2403.15415},
  year={2024}
}
```
