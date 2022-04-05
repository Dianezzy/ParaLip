import os
import glob
import matplotlib

matplotlib.use('Agg')

from tasks.fs2_lipgen_task import GridDataset, FastLipGenTask



class TimitDataset(GridDataset):
    """A dataset that provides helpers for batching."""

    def __init__(self, data_dir, phone_encoder, prefix, hparams, shuffle=False):
        super().__init__(data_dir, phone_encoder, prefix, hparams, shuffle)

        if self.prefix == 'test':
            self.idx2key = self.idx2key
            self.sizes = self.sizes
            pass
        else:
            raise NotImplementedError
        self.indexed_ds = None

class TimitFastLipTask(FastLipGenTask):
    def __init__(self):
        super(TimitFastLipTask, self).__init__()
        self.ds_cls = TimitDataset
        self.item2wav = {os.path.splitext(os.path.basename(v))[0]: v
                         for v in glob.glob('./data/timit/timit_wavs/*.wav')}


if __name__ == '__main__':
    TimitFastLipTask.start()
