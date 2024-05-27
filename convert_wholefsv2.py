import glob
import nibabel as nib
import numpy as np

ASEG_pairs = [
    (2, 41), (3, 42), (4, 43), (5, 44), (7, 46), (8, 47),
    (10, 49), (11, 50), (12, 51), (13, 52), (17, 53), (18, 54),
    (26, 58), (28, 60), (30, 62), (31, 63)
]

whole_fsv2 = [
      0,    4,    5,    7,    8,   10,   11,   12,   13,   14,   15,   16,   17,   18,
     24,   26,   28,   30,   31,   85,  251,  252,  253,  254,  255, 1001, 1002, 1003,
   1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018,
   1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032,
   1033, 1034, 1035, 3001, 3002, 3003, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012,
   3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026,
   3027, 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3035, 5001, 5002]

ffs = glob.glob('/work/tyhuang0908/Dataset/tigerdatav4/*/*/wmparc.nii.gz')

cc = 0
for f in ffs:
    cc += 1
    try:
        nib_orig = nib.load(f)
        data = nib_orig.get_fdata().astype(int)
        print('before', np.unique(data).size)

        temp = data.copy()
        temp[(data >2000) & (data <3000)] = temp[(data >2000)  & (data <3000)] - 1000
        temp[(data >4000) & (data <5000)] = temp[(data >4000)  & (data <5000)] - 1000
        for left, right in ASEG_pairs:
            temp[data==right] = left
        temp2 = temp * 0
        count = 0
        for kk in whole_fsv2:
            count += 1
            temp2[temp == kk] = count

        output_nib = nib.Nifti1Image(temp2.astype(np.uint8), nib_orig.affine, nib_orig.header)
        output_nib.header.set_data_dtype(np.uint8)
        output_file = f.replace('wmparc.nii.gz', 'wholefsv2_relabel.nii.gz')
        nib.save(output_nib, output_file)
        print(cc, len(ffs), 'writing outputfile....', output_file)
    except Exception as e:
        print('error', f, e)