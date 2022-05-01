from torch import nn


class Conv2dBlock(nn.Module):
    def __init__(self, idim, odim, kernel_size, stride, padding, norm_fn, acti_fn, dropout_rate=0.):
        super().__init__()
        self.conv_norm = nn.Sequential(
            nn.Conv2d(idim, odim, kernel_size, stride, padding),
            nn.BatchNorm2d(odim),
        )
        if acti_fn == 'relu':
            self.conv_norm.add_module('acti_relu', nn.ReLU(inplace=True))
        self.conv_norm.add_module('do', nn.Dropout(dropout_rate))

    def forward(self, inputs):
        return self.conv_norm(inputs)


class ConvTranspose2dBlock(nn.Module):
    def __init__(self, idim, odim, kernel_size, stride, padding, output_padding, norm_fn, acti_fn, dropout_rate=0.):
        super().__init__()
        self.trans_conv = nn.Sequential(
            nn.ConvTranspose2d(idim, odim, kernel_size, stride, padding, output_padding=output_padding),
            nn.BatchNorm2d(odim),
        )
        if acti_fn == 'relu':
            self.trans_conv.add_module('acti_relu', nn.ReLU(inplace=True))
        self.trans_conv.add_module('do', nn.Dropout(dropout_rate))

    def forward(self, inputs):
        return self.trans_conv(inputs)

class Conv3dBlock(nn.Module):
    def __init__(self, idim, odim, kernel_size, stride, padding, norm_fn, acti_fn, dropout_rate=0.):
        super().__init__()
        self.conv_norm = nn.Sequential(
            nn.Conv3d(idim, odim, kernel_size, stride, padding),
            nn.BatchNorm3d(odim),
        )
        if acti_fn == 'relu':
            self.conv_norm.add_module('acti_relu', nn.ReLU(inplace=True))
        self.conv_norm.add_module('do', nn.Dropout(dropout_rate))

    def forward(self, inputs):
        return self.conv_norm(inputs)

class ConvTranspose3dBlock(nn.Module):
    def __init__(self, idim, odim, kernel_size, stride, padding, output_padding, norm_fn, acti_fn, dropout_rate=0.):
        super().__init__()
        self.trans_conv = nn.Sequential(
            nn.ConvTranspose3d(idim, odim, kernel_size, stride, padding, output_padding=output_padding),
            nn.BatchNorm3d(odim),
        )
        if acti_fn == 'relu':
            self.trans_conv.add_module('acti_relu', nn.ReLU(inplace=True))
        self.trans_conv.add_module('do', nn.Dropout(dropout_rate))

    def forward(self, inputs):
        return self.trans_conv(inputs)
