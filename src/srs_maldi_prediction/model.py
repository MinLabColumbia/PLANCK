import functools
import torch
import torch.nn as nn

def get_norm_layer(norm_type="batch", num_groups=4):
    if norm_type == "batch":
        return nn.BatchNorm2d
    elif norm_type == "instance":
        return nn.InstanceNorm2d
    elif norm_type == "group":
        return lambda channels: nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        # Determine bias
        if norm_layer == nn.InstanceNorm2d:
            use_bias = True
        else:
            use_bias = False

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class HyperPix2pixGenerator(nn.Module):
    """HyperPix2pix generator (U-Net style) with switchable normalization."""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_type="batch", num_groups=4, use_dropout=False):
        super(HyperPix2pixGenerator, self).__init__()

        norm_layer = get_norm_layer(norm_type, num_groups=num_groups)

        # Construct Unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8,
                                             submodule=None, innermost=True,
                                             norm_layer=norm_layer)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8,
                                                 submodule=unet_block,
                                                 use_dropout=use_dropout,
                                                 norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc,
                                             submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.softplus = torch.nn.Softplus(threshold=10)

    def forward(self, input):
        out = self.model(input)
        out = self.softplus(out)
        return out


class HyperPix2pixDiscriminator(nn.Module):
    """HyperPix2pix PatchGAN discriminator."""

    def __init__(self, input_nc=307, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(HyperPix2pixDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
