# DualAug Combined with DA-Fusion


Effective Data Augmentation With Diffusion Models

## Installation

To install the package, first create a `conda` environment.

```bash
conda create -n da-fusion python=3.7 pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6 -c pytorch
conda activate da-fusion
pip install diffusers["torch"] transformers pycocotools pandas matplotlib seaborn scipy
```

Then download and install the source code.

```bash
git clone git@github.com:brandontrabucco/da-fusion.git
pip install -e da-fusion
```


## Setting Up PASCAL VOC

Data for the PASCAL VOC task is adapted from the [2012 PASCAL VOC Challenge](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar). Once this dataset has been downloaded and extracted, the PASCAL dataset class `semantic_aug/datasets/pascal.py` should be pointed to the downloaded dataset via the `PASCAL_DIR` config variable located [here](https://github.com/brandontrabucco/da-fusion/blob/main/semantic_aug/datasets/pascal.py#L14).

Ensure that `PASCAL_DIR` points to a folder containing `ImageSets`, `JPEGImages`, `SegmentationClass`, and `SegmentationObject` subfolders.


## Fine-Tuning Tokens

We perform textual inversion (https://arxiv.org/abs/2208.01618) to adapt Stable Diffusion to the classes present in our few-shot datasets. The implementation in `fine_tune.py` is adapted from the [Diffusers](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py) example. 

We wrap this script for distributing experiments on a slurm cluster in a set of `sbatch` scripts located at `scripts/fine_tuning`. These scripts will perform multiple runs of Textual Inversion in parallel, subject to the number of available nodes on your slurm cluster.

If `sbatch` is not available in your system, you can run these scripts with `bash` and manually set `SLURM_ARRAY_TASK_ID` and `SLURM_ARRAY_TASK_COUNT` for each parallel job (these are normally set automatically by slurm to control the job index, and the number of jobs respectively, and can be set to 0, 1).

## Few-Shot Classification

Code for training image classification models using augmented images from DA-Fusion is located in `train_classifier.py`. 
This script accepts a number of arguments that control how the classifier is trained:

Run for DA-Fusion and DA-Fusion+DualAug
```bash
sh basicaug.sh 
sh dualaug.sh
```


## Citation

If you find our method helpful, consider citing our preprint!

```
@misc{https://doi.org/10.48550/arxiv.2302.07944,
  doi = {10.48550/ARXIV.2302.07944},
  url = {https://arxiv.org/abs/2302.07944},
  author = {Trabucco, Brandon and Doherty, Kyle and Gurinas, Max and Salakhutdinov, Ruslan},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Effective Data Augmentation With Diffusion Models},
  publisher = {arXiv},
  year = {2023},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
