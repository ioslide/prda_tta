# Prerequisites
```bash
conda create -n tta python=3.8.1
conda activate tta
conda install -y ipython pip

# install the required packages
pip install -r requirements.txt 
```
# Preparation

## Datasets
To run one of the following benchmark tests, you need to download the corresponding dataset.
  - `CIFAR100 → CIFAR100-C`: CIFAR100-C dataset is automatically downloaded when running the experiments or manually download from [here 🔗](https://zenodo.org/record/3555552#.ZDES-XZBxhE).
  - `ImageNet → ImageNet-C`: Download [ImageNet-C 🔗](https://github.com/hendrycks/robustness) dataset from [here 🔗](https://zenodo.org/record/2235448#.Yj2RO_co_mF).
  - `ImageNet → ImageNet-3DCC`: Download [ImageNet-3DCC 🔗](https://github.com/hendrycks/robustness) dataset from [here 🔗](https://github.com/EPFL-VILAB/3DCommonCorruptions?tab=readme-ov-file#3dcc-data).

For non-source-free methods (like RMT, etc.), you need to download the [ImageNet 🔗](https://www.image-net.org/download.php) dataset.

## Models

For the TTA benchmarks, pre-trained models from [RobustBench](https://github.com/RobustBench/robustbench), [Torchvision](https://pytorch.org/vision/0.14/models.html), and [Timm](https://huggingface.co/timm) are used.

# Run Experiments

Python scripts are provided to run the experiments. For example, to run the IMAGNET → IMAGNET-C with `OURS`, run the following command:
```bash
python CTTA.py -acfg configs/adapter/cifar100/OURS.yaml -dcfg configs/dataset/cifar100.yaml -ocfg configs/order/cifar100/0.yaml SEED 0
```

Bash scripts are provided to run the experiments. For example, run the following command:
```bash
nohup bash run.sh > run.log 2>&1 &
```

# Competitors
The repository currently supports the following methods: [TEA](https://openaccess.thecvf.com/content/CVPR2024/papers/Yuan_TEA_Test-time_Energy_Adaptation_CVPR_2024_paper.pdf), [RMT](https://arxiv.org/abs/2211.13081), [BN](https://arxiv.org/pdf/1603.04779.pdf), [Tent](https://openreview.net/pdf?id=uXl3bZLkr3c), [CoTTA](https://arxiv.org/abs/2203.13591), [SAR](https://openreview.net/pdf?id=g2YraF75Tj), [RoTTA](https://openaccess.thecvf.com/content/CVPR2023/papers/Yuan_Robust_Test-Time_Adaptation_in_Dynamic_Scenarios_CVPR_2023_paper.pdf), [TRIBE](https://ojs.aaai.org/index.php/AAAI/article/view/29435)

# Acknowledgements
This project is based on the following projects:
+ Robustbench [official](https://github.com/RobustBench/robustbench)
+ CoTTA [official](https://github.com/qinenergy/cotta)
+ Tent [official](https://github.com/DequanWang/tent)
+ SAR [official](https://github.com/mr-eggplant/SAR)
+ RoTTA [official](https://github.com/BIT-DA/RoTTA)
+ RMT [official](https://github.com/mariodoebler/test-time-adaptation)
+ TRIBE [official](https://github.com/Gorilla-Lab-SCUT/TRIBE/)
+ TEA [official](https://github.com/yuanyige/tea)

# Contact
If you have any questions about our work, please contact <a href="mailto:im@xhy.im">im@xhy.im</a>
