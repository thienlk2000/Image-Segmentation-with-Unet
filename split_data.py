import os
import shutil
import numpy as np
img_dir = 'train'
mask_dir = 'train_masks'
test_dir = 'test'
mask_test_dir = 'test_masks'

img_file = os.listdir(img_dir)

shuffle_idx = np.random.permutation(len(img_file))
test_size  = int(len(img_file)*0.2)
test_idx = shuffle_idx[:test_size]
test_file = [img_file[f] for f in test_idx]

for f in test_file:
    src_img = os.path.join(img_dir, f)
    src_mask = os.path.join(mask_dir, f.replace('.jpg', '_mask.gif'))
    des_img = os.path.join(test_dir, f)
    des_mask = os.path.join(mask_test_dir, f.replace('.jpg', '_mask.gif'))
    shutil.move(src_img, des_img)
    shutil.move(src_mask, des_mask)