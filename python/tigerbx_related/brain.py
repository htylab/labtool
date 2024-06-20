import numpy as np
def get_conform(nib_ima, nib_mask, nib_betmask, voxel_size=(1.0, 1.0, 1.0)):
    from nibabel.processing import conform
    ima = conform(nib_ima, order=3, cval=-10, voxel_size=voxel_size).get_fdata(caching='unchanged')
    mask = conform(nib_ima, order=0, cval=-10, voxel_size=voxel_size).get_fdata(caching='unchanged')
    betmask = conform(nib_ima, order=0, cval=-10, voxel_size=voxel_size).get_fdata(caching='unchanged')
    xx, yy, zz = np.nonzero(ima>0)
    x1, x2 = np.min(xx), np.max(xx)
    y1, y2 = np.min(yy), np.max(yy)
    z1, z2 = np.min(zz), np.max(zz)

    ima = ima[x1:(x2+1), y1:(y2+1), z1:(z2+1)]
    mask = mask[x1:(x2+1), y1:(y2+1), z1:(z2+1)]
    betmask = betmask[x1:(x2+1), y1:(y2+1), z1:(z2+1)]
    return ima, mask, betmask