from torch import nn
import torch.nn.functional as F
import torch
import math
from modules.lip_modules.lip_utils import *


class ImageEncoder(nn.Module):
    def __init__(self, img_channel, img_size, hidden_size, if_tanh=False, only_lip=False, norm_fn='none', acti_fn='relu'):
        super().__init__()
        self.if_tanh = if_tanh
        img_c = img_channel if only_lip else 6

        self.conv1 = Conv2dBlock(img_c, 16, 5, stride=2, padding=2, norm_fn=norm_fn, acti_fn=acti_fn)
        self.conv2 = Conv2dBlock(16, 32, 5, stride=2, padding=2, norm_fn=norm_fn, acti_fn=acti_fn)
        self.conv3 = Conv2dBlock(32, 64, 5, stride=2, padding=2, norm_fn=norm_fn, acti_fn=acti_fn)
        self.conv4 = Conv2dBlock(64, 128, 5, stride=2, padding=2, norm_fn=norm_fn, acti_fn=acti_fn)

        '''
        self.conv1=nn.Sequential(nn.Conv2d(img_c, 16, 5, stride=2, padding=2), nn.ReLu(inplace=True))
        self.conv2=nn.Sequential(nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.ReLu(inplace=True))
        self.conv3=nn.Sequential(nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLu(inplace=True))
        self.conv4=nn.Sequential(nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.ReLu(inplace=True))
        '''

        mini_map_h, mini_map_w = self.get_size(img_size, 4)
        self.fc = nn.Linear(mini_map_h * mini_map_w * 128, hidden_size)

    def get_size(self, img_size, num_layers):
        return tuple([math.ceil(_ / (2 ** num_layers)) for _ in img_size])

    def forward(self, inputs):
        # [16, 40, 80]
        img_e_conv1 = self.conv1(inputs)
        # [32, 20, 40]
        img_e_conv2 = self.conv2(img_e_conv1)
        # [64, 10, 20]
        img_e_conv3 = self.conv3(img_e_conv2)
        # [128, 5, 10]
        img_e_conv4 = self.conv4(img_e_conv3)
        # print('img_e_conv4: ', img_e_conv4.shape)
        img_e_fc_5 = img_e_conv4.contiguous().view(img_e_conv4.shape[0], -1)   # [BS, ]
        # print('img_e_fc5 before: ', img_e_fc_5.shape)
        img_e_fc_5 = self.fc(img_e_fc_5)

        if self.if_tanh:
            img_e_fc_5 = F.tanh(img_e_fc_5)
        return img_e_fc_5, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4


class ImageDecoder2D(nn.Module):
    def __init__(self, img_size, input_dim, norm_fn='none', acti_fn='relu'):
        super().__init__()
        self.mini_map_h, self.mini_map_w = self.get_size(img_size, 4)
        self.fc = nn.Linear(input_dim, self.mini_map_h * self.mini_map_w * 256)
        self.dconv1 = ConvTranspose2dBlock(384, 196, 5, stride=2, padding=2, output_padding=1, norm_fn=norm_fn,
                                           acti_fn=acti_fn)
        self.dconv2 = ConvTranspose2dBlock(260, 128, 5, stride=2, padding=2, output_padding=1, norm_fn=norm_fn,
                                           acti_fn=acti_fn)
        self.dconv3 = ConvTranspose2dBlock(160, 80, 5, stride=2, padding=2, output_padding=1, norm_fn=norm_fn,
                                           acti_fn=acti_fn)
        self.dconv4 = ConvTranspose2dBlock(96, 48, 5, stride=2, padding=2, output_padding=1, norm_fn=norm_fn,
                                           acti_fn=acti_fn)
        self.dconv5 = Conv2dBlock(48, 16, 5, stride=1, padding=2, norm_fn=norm_fn, acti_fn=acti_fn)
        self.dconv6 = nn.Conv2d(16, 3, 5, stride=1, padding=2)
        '''
        self.dconv1 = mn.Sequential(nn.ConvTranspose2d(384,196,5,stride = 2, padding = 2,output_padding - 1),nn.Rl(
            inplace=True))
        self.dconv2 = nn.Sequential(mn.convTranspose2d(260,128,5,stride = 2,, padding = 2,output_padding = 1),nn.ReLu(
            implace=Truel)
        self.dconv3 = nn.Sequential(nn.ConVvTranspose2d(160,80,5,stride = 2, padding = 2,output_padding - 1), nm.RelLu(
            inplace=rue)
        self.dconv4 = nn.Sequential(nn.CconvTranspose2d(96,48,5,stride = 2,padding = 2,output_padting = 1),mn.Relu(
            inplace=True)
        self.dconv5 = nn.Sequential(nn.conv2d(48,16,5,stride = 1,padding = 2),nn.ReLu(inplace=True))
        '''

    def get_size(self, img_size, num_layers):
        return tuple([math.ceil(_ / (2 ** num_layers)) for _ in img_size])

    def forward(self, concat_z, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4):
        out = self.fc(concat_z)
        # reshape 256 * 5 * 10
        out = out.contiguous().view(out.shape[0], 256, self.mini_map_h, self.mini_map_w)
        out = F.relu(out, inplace=True)
        # concate (256 + 128) * 5 * 10
        out = torch.cat([out, img_e_conv4], dim=1)
        out = self.dconv1(out)
        # concate (196 + 64) * 10 * 20
        out = torch.cat([out, img_e_conv3], dim=1)
        out = self.dconv2(out)
        # concate (128 + 32) * 20 * 40
        out = torch.cat([out, img_e_conv2], dim=1)
        out = self.dconv3(out)
        # concate (80 + 16) * 40 * 60
        out = torch.cat([out, img_e_conv1], dim=1)
        out = self.dconv4(out)
        out = self.dconv5(out)
        out = self.dconv6(out)
        return torch.tanh(out)


class ImageDecoder3D(nn.Module):
    def __init__(self, img_channel, img_size, input_dim, norm_fn='none', acti_fn='relu'):
        super().__init__()
        self.mini_map_h, self.mini_map_w = self.get_size(img_size, 4)
        self.fc = nn.Linear(input_dim, self.mini_map_h * self.mini_map_w * 256)
        self.dconv1 = ConvTranspose3dBlock(384, 196, (1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2),
                                           output_padding=(0, 1, 1), norm_fn=norm_fn, acti_fn=acti_fn)
        self.dconv2 = ConvTranspose3dBlock(260, 128, (1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2),
                                           output_padding=(0, 1, 1), norm_fn=norm_fn, acti_fn=acti_fn)
        self.dconv3 = ConvTranspose3dBlock(160, 80, (1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2),
                                           output_padding=(0, 1, 1), norm_fn=norm_fn, acti_fn=acti_fn)
        self.dconv4 = ConvTranspose3dBlock(96, 48, (1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2),
                                           output_padding=(0, 1, 1), norm_fn=norm_fn, acti_fn=acti_fn)
        self.dconv5 = Conv3dBlock(48, 16, (1, 5, 5), stride=(1, 1, 1),  padding=(0, 2, 2),
                                  norm_fn=norm_fn, acti_fn=acti_fn)
        self.dconv6 = nn.Conv3d(16, img_channel, (1, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2))

    def get_size(self, img_size, num_layers):
        return tuple([math.ceil(_ / (2 ** num_layers)) for _ in img_size])

    def forward(self, concat_z, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4):
        out = self.fc(concat_z)
        # print(out.shape)  # [B, seq_len, self.mini_map_h * self.mini_map_w * 256]
        # reshape 256 * 5 * 10
        # [B, seq-len, C, H, W]
        out = out.contiguous().view(out.shape[0], out.shape[1], 256, self.mini_map_h, self.mini_map_w)
        out = out.permute(0, 2, 1, 3, 4)  # [B, C, seq-len, H, W]

        out = F.relu(out, inplace=True)
        # concate (256 + 128) * 5 * 10
        out = torch.cat([out, img_e_conv4], dim=1)
        out = self.dconv1(out)
        # concate (196 + 64) * 10 * 20
        out = torch.cat([out, img_e_conv3], dim=1)
        out = self.dconv2(out)
        # concate (128 + 32) * 20 * 40
        out = torch.cat([out, img_e_conv2], dim=1)
        out = self.dconv3(out)
        # concate (80 + 16) * 40 * 80
        out = torch.cat([out, img_e_conv1], dim=1)
        out = self.dconv4(out)   # * origin size
        out = self.dconv5(out)
        out = self.dconv6(out)
        return torch.tanh(out)  # [B, C, D(seq_len), H, W]


if __name__ == '__main__':
    img_h = 80
    img_w = 160
    hidden_size = 256
    input_img = torch.randn(2, 1, img_h, img_w, device='cuda')
    img_encoder_2d = ImageEncoder(1, [img_h, img_w], hidden_size, only_lip=True).cuda()
    encoded_img, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4 = img_encoder_2d(input_img)
    print(encoded_img.shape)   # [2, hidden]

    seq_len = 21
    hidden = torch.randn(2, seq_len, hidden_size, device='cuda')   # [B, seq_len, H]


    # do concat z
    concat_z = torch.cat([(encoded_img[:, None, :]).repeat([1, seq_len, 1]), hidden], dim=-1)   # [B, seq_len, H * 2]
    print('concat_z shape: ', concat_z.shape)

    img_decoder_3d = ImageDecoder3D(1, [img_h, img_w], hidden_size * 2).cuda()

    # do concat feature:
    print(img_e_conv1.shape)  # [B, C, h*, w*] -> [B, C, 1 * seqlen, h*, w*]
    img_e_conv1 = img_e_conv1[:, :, None, :, :].repeat([1, 1, seq_len, 1, 1])
    img_e_conv2 = img_e_conv2[:, :, None, :, :].repeat([1, 1, seq_len, 1, 1])
    img_e_conv3 = img_e_conv3[:, :, None, :, :].repeat([1, 1, seq_len, 1, 1])
    img_e_conv4 = img_e_conv4[:, :, None, :, :].repeat([1, 1, seq_len, 1, 1])

    out_vid = img_decoder_3d(concat_z, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4)
    print('out_vid shape: ', out_vid.shape)

