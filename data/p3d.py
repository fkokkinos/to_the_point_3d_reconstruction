"""
Data loader for pascal VOC categories.
Should output:
    - img: B X 3 X H X W
    - kp: B X nKp X 2
    - mask: B X H X W
    - sfm_pose: B X 7 (s, tr, q)
    (kp, sfm_pose) correspond to image coordinates in [-1, 1]
"""

from __future__ import absolute_import, division, print_function

import os.path as osp
from collections import Counter, OrderedDict

import numpy as np
import scipy.io as sio
from absl import flags
from torch.utils.data import DataLoader

from . import base as base_data

# -------------- flags ------------- #
# ---------------------------------- #
curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'misc', 'cachedir')

if osp.exists('/data1/shubhtuls'):
    kData = '/data1/shubhtuls/cachedir/PASCAL3D+_release1.1'
elif osp.exists('/scratch1/storage'):
    kData = '/scratch1/storage/PASCAL3D+_release1.1'
elif osp.exists('/home/shubham/data/'):
    kData = '/home/shubham/data/PASCAL3D+_release1.1'
else:  # Savio
    kData = '/home/filippos/data/PASCAL3D+_release1.1'

flags.DEFINE_string('p3d_dir', kData, 'PASCAL Data Directory')
flags.DEFINE_string('p3d_anno_path', osp.join(cache_path, 'p3d'), 'Directory where pascal annotations are saved')
flags.DEFINE_string('p3d_class', 'aeroplane', 'PASCAL VOC category name')
flags.DEFINE_string('p3d_names_list', None, 'PASCAL VOC category name')

opts = flags.FLAGS


# -------------- Dataset ------------- #
# ------------------------------------ #
class P3dDataset(base_data.BaseDataset):
    '''
    VOC Data loader
    '''

    def __init__(self, opts, filter_key=None):
        super().__init__(opts)
        keys = []
        if filter_key is not None:
            with open(filter_key) as fin:
                for f in fin:
                    keys.append(f.strip())
            keys.sort()
        d_keys = OrderedDict(Counter(keys))
        self.img_dir = osp.join(opts.p3d_dir, 'Images')
        self.annotations = osp.join(opts.p3d_dir, 'Annotations', opts.p3d_class + '_pascal')
        self.kp_path = osp.join(
            opts.p3d_anno_path, 'data', '{}_kps.mat'.format(opts.p3d_class))
        self.anno_path = osp.join(
            opts.p3d_anno_path, 'data', '{}_{}.mat'.format(opts.p3d_class, opts.split))
        self.anno_sfm_path = osp.join(
            opts.p3d_anno_path, 'sfm', '{}_{}.mat'.format(opts.p3d_class, opts.split))

        # Load the annotation file.
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']
        self.kp_perm = sio.loadmat(
            self.kp_path, struct_as_record=False, squeeze_me=True)['kp_perm_inds'] - 1

        if len(keys) > 0:
            # According to Kar et al. Category Specific reconstruction
            # remove non existent
            anno_new = []
            anno_sfm_new = []
            annotations_new = []
            for idx_anno, [anno, anno_sfm] in enumerate(zip(self.anno, self.anno_sfm)):
                anno_strp = anno.rel_path.split('/')[1].split('.')[0]
                if anno_strp in d_keys:
                    for _ in range(d_keys[anno_strp]):
                        anno_new.append(anno)
                        anno_sfm_new.append(anno_sfm)
                        annotations_new.append(
                            osp.join(opts.p3d_dir, 'Annotations', opts.p3d_class + '_pascal', anno_strp + '.mat'))
                    del d_keys[anno_strp]
            self.anno = np.array(anno_new)
            self.anno_sfm = np.array(anno_sfm_new)
            self.annotations = annotations_new
        opts.num_kps = len(self.kp_perm)
        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)


# ----------- Data Loader ----------#
# ----------------------------------#

def data_loader(opts, filter_key=None, shuffle=True, drop_last=True):
    dset = P3dDataset(opts, filter_key)
    # dset = d_set_func(opts, filter_key=filter_key)
    return DataLoader(
        dset,
        batch_size=opts.batch_size,
        shuffle=shuffle,
        num_workers=opts.n_data_workers,
        pin_memory=True, drop_last=drop_last)
