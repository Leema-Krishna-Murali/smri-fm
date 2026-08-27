import nibabel as nib
from nibabel.processing import resample_to_output, smooth_image


def canonical(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Repacked, because the hf Nifti feature hands back a wrapper `as_reoriented` cannot rebuild."""
    return nib.as_closest_canonical(nib.Nifti1Image(img.dataobj, img.affine, img.header))


def acquired_at(img: nib.Nifti1Image, mm: float) -> nib.Nifti1Image:
    """An isotropic acquisition at `mm`."""
    img = canonical(img)
    return resample_to_output(smooth_image(img, mm), mm, order=1)


def thick_slice(img: nib.Nifti1Image, mm: float) -> nib.Nifti1Image:
    """An anisotropic axial acquisition: `mm` slices, in-plane resolution untouched."""
    img = canonical(img)
    in_plane = nib.affines.voxel_sizes(img.affine)[:2]
    return resample_to_output(smooth_image(img, (0, 0, mm)), (*in_plane, mm), order=1)


PERTURBATIONS = {
    "thick_slice_5mm": lambda img: thick_slice(img, 5.0),
    "acquired_at_2mm": lambda img: acquired_at(img, 2.0),
}
