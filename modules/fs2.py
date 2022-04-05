from modules.operations import *
from modules.transformer_tts import TransformerEncoder, Embedding
from modules.tts_modules import FastspeechDecoder, DurationPredictor, LengthRegulator, PitchPredictor, EnergyPredictor, \
    RefEncoder, ConvEmbedding, ConvEmbedding2, ConvEmbedding3



class FastSpeech2(nn.Module):
    def __init__(self, arch, dictionary, out_dims=None):
        super().__init__()
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()
        if isinstance(arch, str):
            self.arch = list(map(int, arch.strip().split()))
        else:
            assert isinstance(arch, (list, tuple))
            self.arch = arch
        self.enc_layers = hparams['enc_layers']
        self.dec_layers = hparams['dec_layers']
        self.enc_arch = self.arch[:self.enc_layers]
        self.dec_arch = self.arch[self.enc_layers:self.enc_layers + self.dec_layers]
        self.hidden_size = hparams['hidden_size']
        self.encoder_embed_tokens = self.build_embedding(self.dictionary, self.hidden_size)
        self.encoder = TransformerEncoder(self.enc_arch, self.encoder_embed_tokens)
        self.decoder = FastspeechDecoder(self.dec_arch) if hparams['dec_layers'] > 0 else None
        self.mel_out = Linear(self.hidden_size,
                              hparams['audio_num_mel_bins'] if out_dims is None else out_dims,
                              bias=True)
        if hparams['use_spk_id']:
            self.spk_embed_proj = Embedding(hparams['num_spk'], self.hidden_size)
        else:
            self.spk_embed_proj = Linear(256, self.hidden_size, bias=True)
        self.dur_predictor = DurationPredictor(
            self.hidden_size,
            n_chans=hparams['predictor_hidden'],
            dropout_rate=0.5, padding=hparams['ffn_padding'],
            kernel_size=hparams['dur_predictor_kernel'])
        self.length_regulator = LengthRegulator()
        if hparams['use_pitch_embed']:
            if hparams['pitch_embed_type'] == 1:
                self.pitch_embed = ConvEmbedding(300, self.hidden_size)
            elif hparams['pitch_embed_type'] == 2:
                self.pitch_embed = ConvEmbedding2(300, self.hidden_size)
            elif hparams['pitch_embed_type'] == 3:
                self.pitch_embed = ConvEmbedding3(300, self.hidden_size)
            else:
                self.pitch_embed = Embedding(300, self.hidden_size, self.padding_idx)
            self.pitch_predictor = PitchPredictor(
                self.hidden_size, n_chans=hparams['predictor_hidden'], dropout_rate=0.5,
                padding=hparams['ffn_padding'], odim=2)
            self.pitch_do = nn.Dropout(0.5)
        if hparams['use_energy_embed']:
            self.energy_embed = Embedding(256, self.hidden_size, self.padding_idx)
            self.energy_predictor = EnergyPredictor(
                self.hidden_size, n_chans=hparams['predictor_hidden'], dropout_rate=0.5, odim=1,
                padding=hparams['ffn_padding'])
            self.energy_do = nn.Dropout(0.5)
        if hparams['use_ref_enc']:
            self.ref_encoder = RefEncoder(hparams['audio_num_mel_bins'],
                                          hparams['ref_hidden_stride_kernel'],
                                          ref_norm_layer=hparams['ref_norm_layer'])

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        emb = Embedding(num_embeddings, embed_dim, self.padding_idx)
        return emb

    def forward(self, src_tokens, mel2ph=None, spk_embed=None,
                ref_mels=None, f0=None, uv=None, energy=None, skip_decoder=False):
        raise NotImplementedError


