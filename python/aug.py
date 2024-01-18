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

def aug(image, mask):
    # Add a channel dimension to the 3D tensors
    image = image.unsqueeze(0)  # Convert from (D, H, W) to (1, D, H, W)
    mask = mask.unsqueeze(0)    # Convert from (D, H, W) to (1, D, H, W)
    
    prob_affine = 0.2
    prob_noise = 0.1
    prob_bias = 0.1
    prob_motion = 0.1
    prob_ghosting = 0.05
    prob_spike = 0.05
    prob_blur = 0.05
        

    # Define the augmentation pipeline
    transforms = tio.Compose([
        tio.Resample(target_spacing=(1, 1, 1)),
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
        tio.CropOrPad(target_shape=[96, 96, 96])
    ])

    # Create a subject with the image and mask
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=image),
        mask=tio.LabelMap(tensor=mask)
    )

    # Apply the transforms
    transformed_subject = transforms(subject)
    transformed_image = transformed_subject['image'][tio.DATA].squeeze(0)  # Remove channel dimension
    transformed_mask = transformed_subject['mask'][tio.DATA].squeeze(0)    # Remove channel dimension

    return transformed_image, transformed_mask
