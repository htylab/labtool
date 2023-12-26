import numpy as np
import nibabel as nib

def get_dsc(nii_file1, nii_file2):
    # Load the image data
    img1 = nib.load(nii_file1).get_fdata()
    img2 = nib.load(nii_file2).get_fdata()

    # Get a list of unique labels in the images
    labels = np.unique(np.concatenate([np.unique(img1), np.unique(img2)]))
    
    # Remove the background label if it exists (commonly labeled as 0)
    labels = labels[labels != 0]

    # Calculate Dice coefficient for each label
    dice_scores = []
    
    if len(labels) == 0:
        return 1
    for label in labels:
        img1_binary = img1 == label
        img2_binary = img2 == label

        intersection = np.logical_and(img1_binary, img2_binary)
        
        union = np.logical_or(img1_binary, img2_binary)
        print(intersection.sum(), union.sum(), img1_binary.sum(), img2_binary.sum())
        total = img1_binary.sum() +  img2_binary.sum()
        dice = (2. * intersection.sum() + 1e-6) / (total + 1e-6)
        dice_scores.append(dice)

    # Return the average Dice coefficient across all labels
    return np.round(np.mean(dice_scores),6)