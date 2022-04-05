import os
import glob
import matplotlib

matplotlib.use('Agg')
from utils.pl_utils import data_loader
from multiprocessing.pool import Pool
from tqdm import tqdm

from modules.tts_modules import DurationPredictorLoss
from utils.hparams import hparams, set_hparams
from utils.indexed_datasets import IndexedDataset

import numpy as np

from modules.fslip import FastLip
from tasks.transformer_tts import TransformerTtsTask
from tasks.base_task import BaseDataset

import torch
import torch.optim
import torch.utils.data
import utils

set_hparams()


class GridDataset(BaseDataset):
    """A dataset that provides helpers for batching."""

    def __init__(self, data_dir, phone_encoder, prefix, hparams, shuffle=False):
        super().__init__(data_dir, prefix, hparams, shuffle)
        self.phone_encoder = phone_encoder
        self.data = None
        self.idx2key = np.load(f'{self.data_dir}/{self.prefix}_all_keys.npy')
        self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')

        if self.prefix == 'test':
            self.idx2key = self.idx2key[:hparams['cut_test_set']]
            self.sizes = self.sizes[:hparams['cut_test_set']]
            pass
        else:
            raise NotImplementedError

        self.indexed_ds = None



    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)

        vid2ph = None
        phone = torch.LongTensor(item['phone'][:hparams['max_input_tokens']])


        # for vid dataset
        # vid = torch.Tensor(item['vid'])    # [seq_len, h, w, c]  or  [seq_len, h, w]
        guide_face = torch.Tensor(item['guide_face'])   # [h, w, c] or [h, w]

        if len(guide_face.shape) == 2:
            guide_face = guide_face[:, :, None]  # [h, w, 1]



        sample = {
            "id": index,
            "utt_id": item['item_name'],
            "text": item['txt'],
            "source": phone,
            "vid2ph": vid2ph,
            "guide_face": guide_face
        # for vid dataset.
        #     "vid": vid,
        }

        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        pad_idx = self.phone_encoder.pad()
        id = torch.LongTensor([s['id'] for s in samples])
        utt_ids = [s['utt_id'] for s in samples]
        text = [s['text'] for s in samples]

        src_tokens = utils.collate_1d([s['source'] for s in samples], pad_idx)
        vid2ph = None

        #guide face
        face_lst = [s['guide_face'].reshape(-1) for s in samples] # [xxx]

        guide_face = utils.collate_1d(face_lst, 0.0)  # [B, xxx]
        guide_face = guide_face.reshape(guide_face.shape[0],
                        hparams['img_h'], hparams['img_w'], hparams['img_channel'])  # [B, h, w, channel]


        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        ntokens = sum(len(s['source']) for s in samples)

        batch = {
            'id': id,
            'utt_id': utt_ids,
            'nsamples': len(samples),
            'ntokens': ntokens,
            'text': text,
            'src_tokens': src_tokens,
            'vid2ph': vid2ph,
            'src_lengths': src_lengths,
            'guide_face': guide_face,
        }

        return batch


class FastLipGenTask(TransformerTtsTask):
    def __init__(self):
        super(FastLipGenTask, self).__init__()
        self.dur_loss_fn = DurationPredictorLoss()
        self.mse_loss_fn = torch.nn.MSELoss()
        self.ds_cls = GridDataset
        self.item2wav = {os.path.splitext(os.path.basename(v))[0]: v
                         for v in glob.glob('./data/grid/grid_wavs/*.wav')}

    @data_loader
    def test_dataloader(self):
        test_dataset = self.ds_cls(hparams['data_dir'], self.phone_encoder,
                                   hparams['test_set_name'], hparams, shuffle=False)
        return self.build_dataloader(test_dataset, False, self.max_eval_tokens, self.max_eval_sentences)

    def build_model(self):
        arch = self.arch
        model = FastLip(arch, self.phone_encoder)
        return model



    def test_step(self, sample, batch_idx):
        test_guide_face = sample['guide_face']   # uniformly use the first frame

        src_tokens = sample['src_tokens']
        if hparams['profile_infer']:
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            vid2ph = sample['vid2ph']
        else:
            vid2ph = None

        with utils.Timer('fs', print_time=hparams['profile_infer']):
            if hparams['vid_use_gt_dur']:
                outputs = self.model(src_tokens, sample['vid2ph'], test_guide_face)
            else:
                outputs = self.model(src_tokens, vid2ph, test_guide_face)
        sample['outputs'] = outputs['lip_out']
        sample['vid2ph_pred'] = outputs['vid2ph']

        return self.after_infer(sample)

    def after_infer(self, predictions):
        if self.saving_result_pool is None and not hparams['profile_infer']:
            self.saving_result_pool = Pool(8)
            # self.saving_results_futures = []
        predictions = utils.unpack_dict_to_list(predictions)
        t = tqdm(predictions)
        for num_predictions, prediction in enumerate(t):
            for k, v in prediction.items():
                if type(v) is torch.Tensor:
                    prediction[k] = v.cpu().numpy()

            utt_id = prediction.get('utt_id')
            text = prediction.get('text')

            # Lip related predictions
            outputs = self.remove_padding_3D(prediction["outputs"])

            # convert tensor to img for save
            outputs = self.tensor2img(outputs)

            vid2ph_pred = self.remove_padding(prediction.get("vid2ph_pred"))

            gen_dir = os.path.join(hparams['work_dir'],
                                   f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')

            if not hparams['profile_infer']:
                os.makedirs(gen_dir, exist_ok=True)
                self.saving_result_pool.apply_async(self.save_result, args=[
                        outputs, f'P', utt_id, text, gen_dir, self.item2wav])
                # self.save_result(outputs, f'P', utt_id, text, gen_dir, self.item2wav)

                t.set_description(f"Pred_shape: {outputs.shape}")
            else:
                if 'gen_wav_time' not in self.stats:
                    self.stats['gen_wav_time'] = 0
                self.stats['gen_lip_frames'] += outputs.shape[0]
                print('gen_lip_frames: ', self.stats['gen_lip_frames'])

        return {}

    def test_end(self, outputs):
        self.saving_result_pool.close()
        self.saving_result_pool.join()

        return {}

    def test_start(self):
        pass

    ##########
    # utils
    ##########
    def remove_padding_3D(self, x, padding_idx=0):
        if x is None:
            return None
        assert len(x.shape) in [4]   # [T, w, h, c]
        return x[np.abs(x).sum(-1).sum(-1).sum(-1) != padding_idx]

    @staticmethod
    def save_result(vid, prefix, utt_id, text, gen_dir, item2wav):

        from skimage import io
        from moviepy.editor import AudioFileClip, ImageSequenceClip
        base_fn = f'[{prefix}][{utt_id}]'
        if text is not None:
            TXT = text.replace(":", "%3A")[:80]
            base_fn += '_'.join(TXT.split(' '))
        os.makedirs(f'{gen_dir}/{base_fn}', exist_ok=True)
        destination_path = f'{gen_dir}/{base_fn}'
        # print("destination",destination_path)

        for idx, frame in enumerate(vid):
            io.imsave(destination_path + '/' + "{0:03d}.png".format(idx), frame)

        # save img_sequence and wav to video
        imgclip = ImageSequenceClip(list(vid), fps=hparams['vid_fps'])
        if utt_id in item2wav.keys() and os.path.exists(item2wav[utt_id]):
            wav_clip = AudioFileClip(item2wav[utt_id], fps=16000)
            imgclip = imgclip.set_audio(wav_clip)

        imgclip.write_videofile(gen_dir + f'/{prefix}+{utt_id}.avi', codec='png', fps=hparams['vid_fps'], audio_fps=16000, logger=None)

        return {}

    @staticmethod
    def tensor2img(vid):
        vid = (vid * 255.).clip(min=0., max=255.)
        vid = vid.astype(np.uint8)
        return vid

if __name__ == '__main__':
    FastLipGenTask.start()
