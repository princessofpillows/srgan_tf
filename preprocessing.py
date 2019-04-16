import os, cv2, glob, h5py
from pathlib import Path
from tqdm import trange


def resize(path, res):
    img = cv2.imread(str(path), 1)
    return cv2.resize(img, res)

def process_lr(cfg, paths, f):
    lr = []
    # Process low resolution images
    for i in trange(100):
        lr.append(resize(paths[i], cfg.hr_resolution))

    f.create_dataset('lr', data=lr, dtype='uint8')
    return f

def process_hr(cfg, paths, f):
    hr = []
    # Process high resolution images
    for i in trange(100):
        hr.append(resize(paths[i], cfg.hr_resolution))

    f.create_dataset('hr', data=hr, dtype='uint8')
    return f

def package_data(cfg):
    cwd = os.getcwd()
    data_dir = Path(cwd + cfg.data_dir)

    # ensure we are searching a valid directory
    if not data_dir.is_dir():
        print('Error: path to', data_dir, 'does not exist, change data_dir to a valid directory')
        exit(1)

    # ensure subdirectories are valid
    if not (Path(data_dir / 'HR').is_dir() and Path(data_dir / 'LR').is_dir()):
        print('Error: data directory', data_dir, 'does not contain the two required subdirectories HR and LR')
        exit(1)

    # Get image paths
    paths_lr = list((data_dir / 'LR').glob('*.JPG'))
    paths_hr = list((data_dir / 'HR').glob('*.png'))
    if not (paths_hr and paths_lr):
        print('Error: HR or LR is empty or not in correct format')
        exit(1)

    # Write to h5 file
    f = h5py.File(data_dir / 'data.h5', 'w')
    # Get our real world noise low resolution images
    f = process_lr(cfg, paths_lr, f)
    # Get our high resolution images
    f = process_hr(cfg, paths_hr, f)
    f.close()

