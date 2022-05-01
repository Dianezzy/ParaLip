import json
import os
from multiprocessing.pool import Pool
os.environ["OMP_NUM_THREADS"] = "1"
from utils.hparams import set_hparams, hparams
from utils.indexed_datasets import IndexedDatasetBuilder
from datasets.lipgen.utils import build_phone_encoder
import glob
set_hparams()
class GridHighProcessor:
    def __init__(self):
        super().__init__()
        raw_data_dir = hparams['raw_data_dir']
        self.raw_data_dir = raw_data_dir

        self.item2txt = {os.path.splitext(os.path.basename(v))[0]: v
                         for v in glob.glob(f"{raw_data_dir}/mfa_input/*/*.text")}

        self.item_names = sorted(list(self.item2txt.keys()))
        self.high_vid_data = os.path.join(hparams['high_vid_dir'])

    @property
    def train_item_names(self):
        raise NotImplementedError

    @property
    def valid_item_names(self):
        raise NotImplementedError

    @property
    def test_item_names(self):
        return self.valid_item_names

    def item_name2spk_id(self, item_name):
        return 0

    def _phone_encoder(self):
        raw_data_dir = hparams['raw_data_dir']
        ph_set = [x.split(' ')[0] for x in open(f'{raw_data_dir}/dict.txt').readlines()]
        json.dump(ph_set, open(f"{hparams['data_dir']}/phone_set.json", 'w'))
        print("| phone set: ", ph_set)
        return build_phone_encoder(hparams['data_dir'])


    def meta_data(self, prefix):
        raw_data_dir = hparams['raw_data_dir']
        if prefix == 'test':
            item_names = self.test_item_names
        else:
            raise NotImplementedError

        for item_name in item_names:
            if item_name not in self.item2txt:
                print(f"| Item not found. Skip {item_name}.")
                continue
            txt_fn = self.item2txt[item_name]
            group_id = txt_fn.split("/")[-2]
            ph_fn = f"{raw_data_dir}/mfa_input/{group_id}/{item_name}.ph"

            txt = self.item2txt[item_name]
            frame_fn = f"{self.high_vid_data}/{item_name}_done/"
            yield item_name, ph_fn, txt, frame_fn


    @staticmethod
    def process_item(ph_fn, txt_fn, frames_fn, encoder, hparams):
        vid = _high_process_lips(frames_fn, (80, 160, 3))
        ph = open(ph_fn).readlines()[0].strip()
        txt = open(txt_fn).readlines()[0].strip()
        ph = "| " + ph + " |"
        phone_encoded = encoder.encode(ph)

        return {
            'phone_encoded': phone_encoded, 'txt': txt, 'ph': ph, 'vid': vid
        }

    def process_data(self, prefix):
        data_dir = hparams['data_dir']
        futures = []
        p = Pool(int(os.getenv('N_PROC', os.cpu_count())))
        for m in self.meta_data(prefix):
            item_name, ph_fn, txt_fn, frame_fn = m
            futures.append([
                m, p.apply_async(self.process_item , args=(
                    ph_fn, txt_fn, frame_fn, self.phone_encoder, hparams))])
        p.close()

        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        all_keys = []
        lengths = []


        for f_id, future in enumerate((futures)):
            inp = future[0]
            res = future[1].get()
            if res is None:
                continue

            item_name = inp[0]
            if len(res['vid'].shape) < 4:
                print(f"| Skip {item_name}. ({res['txt']})")
                continue


            item = {
                'item_name': item_name,
                'txt': res['txt'],
                'phone': res['phone_encoded'],
                'vid2ph': None,
                'guide_face': res['vid'][0], #for inference
            }

            builder.add_item(item)
            lengths.append(res['vid'].shape[0])
            all_keys.append(item_name)

            futures[f_id] = None
        p.join()
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_all_keys.npy', all_keys)
        np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)


    @property
    def train_item_names(self):
        res_lst = []
        with open(os.path.join(hparams['raw_data_dir'], 'train_path_v250.txt')) as file_r:
            for file_path in file_r.readlines():
                dirs_ = file_path.split('/')
                utt_id = dirs_[-1].strip()[:-4]
                res_lst.append(utt_id)
        return res_lst

    @property
    def valid_item_names(self):
        res_lst = []
        with open(os.path.join(hparams['raw_data_dir'], 'test_path_v250.txt')) as file_r:
            for file_path in file_r.readlines():
                dirs_ = file_path.split('/')
                utt_id = dirs_[-1].strip()[:-4]
                res_lst.append(utt_id)
        return res_lst

    def process(self):
        os.makedirs(hparams['data_dir'], exist_ok=True)
        self.phone_encoder = self._phone_encoder()
        self.process_data('test')

#######
# utils
#######
from skimage import io
import numpy as np
from skimage.exposure import is_low_contrast
def _high_process_lips(vid_path, img_shape):
    vid_imgs_dir = vid_path
    if os.path.exists(vid_imgs_dir):
        img_files = [img_path for img_path in os.listdir(vid_imgs_dir) if img_path.startswith('lip_')]
    else:
        print(vid_imgs_dir, 'not exist!')
        img_files = []
    img_files.sort()
    # print(img_files)
    vid_array = []
    for img_id, img_file in enumerate(img_files):
        img_path = os.path.join(vid_imgs_dir, img_file)  # path/vid_id/img_id.png
        img = io.imread(img_path)   # range: [0, 255], type: np.uint8
        if img.shape != img_shape:
            continue

        if is_low_contrast(img):
            if img_id == 0:
                print('================> fatal error: ', vid_path, 'cnt: ', img_id)
                continue
            else:
                print('================> low contrast: ', vid_path, 'cnt: ', img_id)
                vid_array.append(vid_array[img_id - 1])  # fetch the last img-array
        else:
            img = img.astype(np.float32) / 255.  # range: [0, 1], type: float32
            vid_array.append(img)

    vid_array = np.asarray(vid_array)  # (~75, 60, 100)
    return vid_array


if __name__ == "__main__":
    GridHighProcessor().process()
