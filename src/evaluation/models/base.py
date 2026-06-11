import torch.nn as nn
import nibabel as nib
from torch import Tensor


class Model(nn.Module):
    """
    Wrap an sMRI encoder model. Takes an input batch and returns an embedding tensor,
    shape (batch_size, embed_dim).
    """

    def forward(self, batch: dict[str, Tensor]) -> Tensor: ...


class Transform:
    """
    Model specific data transform. Takes an input nibabel nifti image and returns a sample
    dict with all model-specific preprocessing applied.

    The sample must be batch collatable.
    """

    def __call__(self, img: nib.Nifti1Image) -> dict[str, Tensor]: ...


ModelTransformPair = tuple[Model, Transform]
