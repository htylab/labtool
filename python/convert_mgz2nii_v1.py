import sys
import glob
import os
import time
import subprocess
from os.path import join, basename, dirname
import shutil
from multiprocessing import Process, Pool
import numpy as np
import nibabel as nib
import glob
from tqdm import tqdm
#跑之前先執行下面這行
# export FREESURFER_HOME=/NFS/show/tools/freesurfer/freesurfer; source $FREESURFER_HOME/SetUpFreeSurfer.sh;

FREESURFER_DIR = '/NFS/show/tools/freesurfer/freesurfer'
DATA_DIR = '/NFS/show/tools/freesurfer/tmp_out'
SAVE_DIR = '/NFS/show/tools/freesurfer/nii_out'

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'aseg'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'wmparc'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'dkt'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'ct-N'), exist_ok=True)

num_pools = int(sys.argv[1])

temp_dir = 'temp'
os.makedirs(temp_dir, exist_ok=True)
    
def systemx(cmd, display=True):
    #print('Executing :' + cmd)
    subprocess.call(cmd, shell=display)

    
ffs = sorted(glob.glob(os.path.join(DATA_DIR, '*')))[:2]
def convert(ff):
    try:
        name = os.path.basename(ff)
        
        output_name = name
        temp_path = temp_dir + r'/temp_' + '_' + name
        
        output_path = SAVE_DIR
        
        if os.path.exists(temp_path) or os.path.exists(f'{output_path}/ct-N/{output_name}'):
            return
        else:
            os.makedirs(temp_path, exist_ok=True)

        # cmd = f'mri_label2vol ' 
        # cmd += f' --seg {ff}/mri/aseg.mgz --temp {ff}/mri/orig/001.mgz --o {output_path}/aseg/{output_name}_aseg.nii.gz --regheader '
        # systemx(cmd)
        
        # cmd = f'mri_label2vol ' 
        # cmd += f' --seg {ff}/mri/wmparc.mgz --temp {ff}/mri/orig/001.mgz --o {output_path}/wmparc/{output_name}_wmparc.nii.gz --regheader '
        # systemx(cmd)
        
        # cmd = f'mri_label2vol ' 
        # cmd += f' --seg {ff}/mri/aparc.DKTatlas+aseg.mgz --temp {ff}/mri/orig/001.mgz --o {output_path}/dkt/{output_name}_dkt.nii.gz --regheader '
        # systemx(cmd)

        cmd = f'mri_surf2vol --o {temp_path}/thickness-in-volume.nii.gz --ribbon {ff}/mri/aseg.mgz ' 
        cmd += f' --so {ff}/surf/lh.white {ff}/surf/lh.thickness --so {ff}/surf/rh.white {ff}/surf/rh.thickness'
        systemx(cmd)
        
        cmd = f'mri_vol2vol ' 
        cmd += f' --mov {temp_path}/thickness-in-volume.nii.gz --targ {ff}/mri/orig/001.mgz --o {output_path}/ct-N/{output_name}_ct_n.nii.gz --regheader --no-save-reg --nearest '
        systemx(cmd)
        
        shutil.rmtree(f'{temp_path}')

    except KeyboardInterrupt:
        shutil.rmtree(f'{temp_path}')
        raise
        
    except:
        shutil.rmtree(f'{temp_path}')
        pass

with Pool(num_pools) as p:
    list(tqdm(p.imap(convert, ffs), total=len(ffs)))
    
if len(glob.glob(f'{temp_dir}/*'))==0:
    shutil.rmtree(temp_dir)