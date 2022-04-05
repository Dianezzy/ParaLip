
import numpy as np
import json
import os
import matplotlib.image as mpimg
from utils.text_encoder import TokenTextEncoder

def _process_mp4(vid_path, img_shape):
    vid_imgs_dir = vid_path
    if os.path.exists(vid_imgs_dir):
        img_files = os.listdir(vid_imgs_dir)
    else:
        img_files = []
    img_files.sort()
    vid_array = []
    for img_file in img_files:
        img_path = os.path.join(vid_imgs_dir, img_file)  # path/vid_id/img_id.png
        img = mpimg.imread(img_path)
        if img.shape == img_shape:
            vid_array.append(img)
    vid_array = np.asarray(vid_array)  # (~75, 60, 100)
    return vid_array

def build_phone_encoder(data_dir):
    phone_list_file = os.path.join(data_dir, 'phone_set.json')
    phone_list = json.load(open(phone_list_file))
    return TokenTextEncoder(None, vocab_list=phone_list)
