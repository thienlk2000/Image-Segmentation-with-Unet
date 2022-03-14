import os
test_img = 'test'
mask = 'test_masks'
for f in os.listdir(test_img):
    if  not os.path.exists(os.path.join(mask, f.replace('.jpg', '_mask.gif'))):
        print(f)