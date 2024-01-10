# ALT-PINN

An official source code for paper [Auxiliary-Tasks Learning for Physics-Informed Neural Network-Based Partial Differential Equations Solving](https://arxiv.org/abs/2307.06167). Any communications or issues are welcomed. Please contact shuaicaijunjun@126.com. If you find this repository useful to your research or work, it is really appreciate to star this repository. :heart:

-------------

### Overview

<p align = "justify"> 
Physics-Informed Neural Networks (PINNs) have emerged as promising surrogate models for solving Partial Differential Equations (PDEs) due to their capacity to capture solution-related features using neural networks. However, the accuracy of original PINNs is often hindered by the uncertainty of the nonlinear PDE system and the instability of neural networks, particularly in elucidating complex physical phenomena. To address these limitations, we have undertaken a comprehensive study into the training processes and convergence mechanisms of physics-informed learning, leading to the development of ATL-PINN, an auxiliary-task learning framework tailored for PINNs. Our ATL-PINN framework concurrently solves multiple PDE tasks to narrow the latent solution space, enhancing the inductive bias and robustness of the underlying PDE's prediction accuracy. Subsequently, ATL-PINN introduces a network architecture integrated with customized feature extractors and gating networks to ensure resilient feature fusion. We also incorporate a gradient cosine similarity algorithm to steer the gradient descent process, mitigating the detrimental impacts of shared parameter updates. Our experimental results on three diverse PDE problems demonstrate that ATL-PINN substantially improves solution accuracy, with a peak performance enhancement of 96.62% (averaging 32.27%) compared to original PINNs.
</p>


### Requirements

1. Torch == 1.12.x
2. Numpy == 1.21.x
3. h5py

### Dataset

All the PDEs case studies we used in our benchmark are download from [PDEBench Datasets](https://github.com/pdebench/PDEBench), and their files are publicly available on [PDEBench Datasets](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986). We randomly selected 100 tasks for each PDE problem to build a sub-dataset for our experiments. The sub-dataset was published in [Google Drive](https://drive.google.com/drive/folders/1n1lHasFJGIEEg_Nm792at2rJeB1QQh0W?usp=sharing).

### Citation

If you use code or datasets in this repository for your research, please cite our paper.

```
@misc{yan2023auxiliarytasks,
      title={Auxiliary-Tasks Learning for Physics-Informed Neural Network-Based Partial Differential Equations Solving}, 
      author={Junjun Yan and Xinhai Chen and Zhichao Wang and Enqiang Zhou and Jie Liu},
      year={2023},
      eprint={2307.06167},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

