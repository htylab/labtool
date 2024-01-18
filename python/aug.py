import random
import numpy as np
import torchio as tio

def permute48(vol0, mask0):
    axis3 = [0, 1, 2]
    random.shuffle(axis3)
    vol = np.transpose(vol0, axis3)
    mask = np.transpose(mask0, axis3)

    xx = random.choice([1, -1])
    yy = random.choice([1, -1])
    zz = random.choice([1, -1])
    vol = vol[::xx, ::yy, ::zz].copy()
    mask = mask[::xx, ::yy, ::zz].copy()
    return vol, mask

import torch
import torchio as tio
import numpy as np

def aug(image, mask):
    input_is_numpy = False
    
    # Check if the input is a numpy array, and if so, convert to PyTorch tensor
    if isinstance(image, np.ndarray):
        input_is_numpy = True
        image = torch.from_numpy(image).unsqueeze(0)  # Convert from (D, H, W) to (1, D, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)    # Convert from (D, H, W) to (1, D, H, W)
    else:
        # Add a channel dimension to the 3D tensors if input is a tensor
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

    # Define the probabilities for each augmentation
    prob_affine = 0.2
    prob_noise = 0.1
    prob_bias = 0.1
    prob_motion = 0.1
    prob_ghosting = 0.05
    prob_spike = 0.05
    prob_blur = 0.05

    # Define the augmentation pipeline
    transforms = tio.Compose([
        tio.Resample((1, 1, 1)),  # Resample to (1,1,1) voxel spacing
        tio.RandomAffine(
            scales=(0.9, 1.1), 
            degrees=(10, 10, 10), 
            translation=(10, 10, 10),
            isotropic=False,
            image_interpolation='linear',
            p=prob_affine
        ),
        tio.RandomNoise(std=(0, 0.25), p=prob_noise),
        tio.RandomBiasField(coefficients=0.5, p=prob_bias),
        tio.RandomMotion(p=prob_motion),
        tio.RandomGhosting(p=prob_ghosting),
        tio.RandomSpike(p=prob_spike),
        tio.RandomBlur(p=prob_blur),
        tio.RandomAnisotropy(p=0.05),
        # tio.CropOrPad(target_shape=[96, 96, 96])
    ])

    # Create a subject with the image and mask
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=image),
        mask=tio.LabelMap(tensor=mask)
    )

    # Apply the transforms
    transformed_subject = transforms(subject)
    transformed_image = transformed_subject['image']['data'].squeeze(0)  # Remove channel dimension
    transformed_mask = transformed_subject['mask']['data'].squeeze(0)    # Remove channel dimension

    # Convert back to numpy array if the input was numpy
    if input_is_numpy:
        transformed_image = transformed_image.cpu().detach().numpy()
        transformed_mask = transformed_mask.cpu().detach().numpy()

    return transformed_image, transformed_mask
