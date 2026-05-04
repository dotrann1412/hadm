"""
Simplified HADM: Detecting Human Artifacts from Diffusion Models.

Paper: https://arxiv.org/abs/2411.13842

Pure PyTorch + torchvision re-implementation.
Original uses EVA-02 ViT + Cascade R-CNN via Detectron2/xformers/mmcv.
This version uses ResNet-50 + FPN + Faster R-CNN via torchvision.
"""

from .model import HADM, load_hadm_weights
from .backbone import Backbone
from .dataset import HADDataset
from .train import train_one_epoch, evaluate, run_inference, get_class_names

__all__ = ["HADM", "Backbone", "HADDataset", "train_one_epoch", "evaluate", "run_inference", "get_class_names", "load_hadm_weights"]