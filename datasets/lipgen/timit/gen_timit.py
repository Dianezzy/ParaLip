import os
os.environ["OMP_NUM_THREADS"] = "1"
from utils.hparams import set_hparams, hparams
from datasets.lipgen.utils import build_phone_encoder

import glob
import json
from tqdm import tqdm

from multiprocessing.pool import Pool
from utils.indexed_datasets import IndexedDatasetBuilder


set_hparams()

class TimitProcessor:
    def __init__(self):
        self.save_wav = False
        # below is my code
        self.item2ph_and_dur = {os.path.splitext(os.path.basename(v))[0].replace('_PHN',''): v
                         for v in glob.glob(f"{hparams['ph_dur_dir']}/*/*/*.txt")}  # v: spk_id/sent_id/spkid_sentid_PHN.txt
        self.item_names = sorted(list(self.item2ph_and_dur.keys()))
        self.set_usr = []

    def _phone_encoder(self):
        ph_set = [x.strip() for x in open(f"{hparams['data_dir']}/dict.txt").readlines()]
        json.dump(ph_set, open(f"{hparams['data_dir']}/phone_set.json", 'w'))
        print("| phone set: ", ph_set)
        return build_phone_encoder(hparams['data_dir'])

    @property
    def test_item_names(self):
        valid_set = [x.strip() for x in open(f"{hparams['data_dir']}/Test.txt").readlines()]
        valid_set.remove('56M_sx67')
        return valid_set

    def meta_data(self, prefix):
        frame_dir = hparams['frame_dir']

        if prefix == 'test':
            item_names = self.test_item_names
        else:
            raise NotImplementedError

        print('len : ', len(item_names))
        print(item_names[:3])
        for item_name in item_names:
            if item_name not in self.item_names:
                print(f"| Label not found. Skip {item_name}.")
                continue
            spk_id, sent_id = item_name.split('_')  # eg. 01M
            ph_and_dur_fn = self.item2ph_and_dur[item_name]
            frames_fn = f"{frame_dir}/{spk_id}/{sent_id}/lips"
            yield item_name, ph_and_dur_fn, frames_fn

    def process(self):
        os.makedirs(hparams['data_dir'], exist_ok=True)
        self.phone_encoder = self._phone_encoder()
        self.process_data('test')

    def process_data(self, prefix):
        data_dir = hparams['data_dir']
        futures = []
        p = Pool(int(os.getenv('N_PROC', os.cpu_count())))
        for m in self.meta_data(prefix):
            item_name, ph_and_dur_fn, frames_fn = m
            futures.append([
                m, p.apply_async(self.process_item, args=(
                    item_name, ph_and_dur_fn, frames_fn, self.phone_encoder, hparams))])
        p.close()


        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        all_keys = []
        lengths = []

        for f_id, future in enumerate(tqdm(futures)):
            inp = future[0]
            res = future[1].get()
            if res is None:
                continue

            item_name= inp[0]

            item = {
                'item_name': item_name,
                'txt': res['txt'],
                'phone': res['phone_encoded'],
                'guide_face': res['vid'][0],
            }
            builder.add_item(item)
            lengths.append(res['vid'].shape[0])
            all_keys.append(item_name)
            futures[f_id] = None
        p.join()
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_all_keys.npy', all_keys)
        np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)

    @staticmethod
    def process_item(item_name, ph_and_dur_fn, frames_fn, encoder, hparams):
        spk_id, sent_id = item_name.split('_')

        ph_and_dur = open(ph_and_dur_fn).readlines()
        ph_lst = [ph.split()[1] for ph in ph_and_dur]
        ph = ' '.join(ph_lst)
        phone_encoded = encoder.encode(ph)

        vid = _process_lips_from_faces(frames_fn, img_shape=(hparams['img_h'], hparams['img_w'], 3))
        return {
            'phone_encoded': phone_encoded, 'txt': ph, 'ph': ph, 'vid2ph': None, 'vid': vid
        }

#######
# utils
#######
from skimage import io
import numpy as np
from skimage.exposure import is_low_contrast


def _process_lips_from_faces(vid_path, img_shape=(80, 160, 3)):
    vid_imgs_dir = vid_path
    if not os.path.exists(vid_imgs_dir):
        print(vid_imgs_dir, 'not exist!')
        return np.asarray([])
    vid_array = []

    img_file = "lip_000.png"
    img_path = os.path.join(vid_imgs_dir, img_file)  # path/vid_id/lips/lip_000.png
    if not os.path.exists(img_path):
        print('================> not exist lip: ', vid_imgs_dir)
    img = io.imread(img_path)   # range: [0, 255], type: np.uint8

    if is_low_contrast(img) or img.shape != img_shape:
        print('================> fatal error: ', vid_imgs_dir)
    else:
        img = img.astype(np.float32) / 255.  # range: [0, 1], type: float32
        vid_array.append(img)

    vid_array = np.asarray(vid_array)  # (seqlen, 80, 160, 3)
    return vid_array

if __name__ == "__main__":
    TimitProcessor().process()
