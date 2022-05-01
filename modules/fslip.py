from modules.base_model import BaseModel
from modules.operations import *
from modules.lip_modules.img_modules import ImageEncoder, ImageDecoder2D, ImageDecoder3D

class FastLip(BaseModel):
    def __init__(self, arch, dictionary, out_dims=None):
        super().__init__(arch, dictionary, out_dims)
        self.img_encoder = ImageEncoder(hparams['img_channel'], [hparams['img_h'], hparams['img_w']]
                                        , hparams['hidden_size'], if_tanh=True, only_lip=True)
        if hparams['imgdecoder_2D']:
            self.img_decoder = ImageDecoder2D([hparams['img_h'], hparams['img_w']]
                                        , hparams['hidden_size'] * 2)   # hidden size for concat Z
        else:
            self.img_decoder = ImageDecoder3D(hparams['img_channel'], [hparams['img_h'], hparams['img_w']]
                                        , hparams['hidden_size'] * 2)

    def forward(self, src_tokens, vid2ph=None, guide_face=None,
                skip_decoder=False):
        """

        :param src_tokens: [B, T]
        :param vid2ph:
        :param guide_face: [B, img_h, img_w, C]
        :return: {
            'lip_out': [B, T_s, ?],
            'dur': [B, T_t],
            'w_st_pred': [heads, B, tokens], 'w_st': [heads, B, tokens],
            'encoder_out_noref': [B, T_t, H]
        }
        """
        ret = {}
        encoder_outputs = self.encoder(src_tokens)
        encoder_out = encoder_outputs['encoder_out']  # [T, B, C]
        src_nonpadding = (src_tokens > 0).float().permute(1, 0)[:, :, None]

        encoder_out = encoder_out * src_nonpadding  # [T, B, C]

        dur_input = encoder_out.transpose(0, 1)
        if hparams['predictor_sg']:
            dur_input = dur_input.detach()
        if vid2ph is None:
            dur = self.dur_predictor.inference(dur_input, src_tokens == 0)
            if not hparams['sep_dur_loss']:
                dur[src_tokens == self.dictionary.seg()] = 0
            vid2ph = self.length_regulator(dur, (src_tokens != 0).sum(-1))[..., 0]
        else:
            ret['dur'] = self.dur_predictor(dur_input, src_tokens == 0)
        ret['vid2ph'] = vid2ph
        # expand encoder out to make decoder inputs
        decoder_inp = F.pad(encoder_out, [0, 0, 0, 0, 1, 0])
        vid2ph_ = vid2ph.permute([1, 0])[..., None].repeat([1, 1, encoder_out.shape[-1]]).contiguous()
        decoder_inp = torch.gather(decoder_inp, 0, vid2ph_).transpose(0, 1)  # [B, T, H]
        ret['decoder_inp_origin'] = decoder_inp  # [B, T, H]


        # add guide face
        guide_face = guide_face.permute(0, 3, 1, 2)   # [B, h, w, c] -> [B, c, h, w]
        guide_face_embed, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4 = \
            self.img_encoder(guide_face)
        guide_face_embed = guide_face_embed[:, None, :]  # [B, 1, H]
        decoder_inp += guide_face_embed

        decoder_inp = decoder_inp * (vid2ph != 0).float()[:, :, None]
        ret['decoder_inp'] = decoder_inp
        if skip_decoder:
            return ret
        x = decoder_inp
        if hparams['dec_layers'] > 0:
            x = self.decoder(x)    # [B, seq_len, H]

        #  choose 2D or 3D  transConv
        if hparams['imgdecoder_2D']:

            B = x.shape[0]
            seq_len = x.shape[1]
            BB = B * seq_len
            # do concat
            # [B, seqlen, H * 2]
            concat_z = torch.cat([guide_face_embed.repeat([1, seq_len, 1]), x], dim=-1).reshape([BB, -1])
            # -> [B * seqlen, H * 2]

            # [B, seqlen, channel*, h, w]
            img_e_conv1 = img_e_conv1[:, None, :, :, :].repeat([1, seq_len, 1, 1, 1])
            img_e_conv1 = img_e_conv1.reshape([BB, img_e_conv1.shape[-3], img_e_conv1.shape[-2], img_e_conv1.shape[-1]])

            img_e_conv2 = img_e_conv2[:, None, :, :, :].repeat([1, seq_len, 1, 1, 1])
            img_e_conv2 = img_e_conv2.reshape([BB, img_e_conv2.shape[-3], img_e_conv2.shape[-2], img_e_conv2.shape[-1]])

            img_e_conv3 = img_e_conv3[:, None, :, :, :].repeat([1, seq_len, 1, 1, 1])
            img_e_conv3 = img_e_conv3.reshape([BB, img_e_conv3.shape[-3], img_e_conv3.shape[-2], img_e_conv3.shape[-1]])

            img_e_conv4 = img_e_conv4[:, None, :, :, :].repeat([1, seq_len, 1, 1, 1])
            img_e_conv4 = img_e_conv4.reshape([BB, img_e_conv4.shape[-3], img_e_conv4.shape[-2], img_e_conv4.shape[-1]])
            # -> [B * seqlen, channel*, h, w]

            # [BB, c, h, w]
            output_frames = self.img_decoder(concat_z, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4)
            output_frames = output_frames.reshape([B, seq_len, output_frames.shape[-3]
                                                      , output_frames.shape[-2], output_frames.shape[-1]])
            # -> [B, seqlen, c, h, w]
            x = output_frames.permute(0, 1, 3, 4, 2)  # [B, seqlen, h, w, c]
        else:
            seq_len = x.shape[1]
            # do concat
            concat_z = torch.cat([guide_face_embed.repeat([1, seq_len, 1]), x], dim=-1)  # [B, seqlen, H * 2]
            img_e_conv1 = img_e_conv1[:, :, None, :, :].repeat([1, 1, seq_len, 1, 1])  # [B, channel*, seqlen, h, w]
            img_e_conv2 = img_e_conv2[:, :, None, :, :].repeat([1, 1, seq_len, 1, 1])
            img_e_conv3 = img_e_conv3[:, :, None, :, :].repeat([1, 1, seq_len, 1, 1])
            img_e_conv4 = img_e_conv4[:, :, None, :, :].repeat([1, 1, seq_len, 1, 1])
            output_frames = self.img_decoder(concat_z, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4)

            x = output_frames.permute(0, 2, 3, 4, 1)  # [B, C, seqlen, h *, w *]  -> [B, seqlen, h *, w *, C]

        x = x * (vid2ph != 0).float()[:, :, None, None, None]
        ret['lip_out'] = x
        return ret
