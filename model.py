import torch
import torch.nn as nn


class _netG(nn.Module):
    def __init__(self, opt):
        super(_netG, self).__init__()
        self.ngpu = opt.ngpu
        layers = []
        layers.append(nn.Conv2d(opt.nc, opt.nef, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(opt.nef, opt.nef, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(opt.nef))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(opt.nef, opt.nef * 2, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(opt.nef * 2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(opt.nef * 2, opt.nef * 4, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(opt.nef * 4))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(opt.nef * 4, opt.nef * 8, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(opt.nef * 8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(opt.nef * 8, opt.nBottleneck, 4, bias=False))
        layers.append(nn.BatchNorm2d(opt.nBottleneck))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.ConvTranspose2d(opt.nBottleneck, opt.ngf * 8, 4, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(opt.ngf * 8))
        layers.append(nn.ReLU(True))

        layers.append(nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(opt.ngf * 4))
        layers.append(nn.ReLU(True))

        layers.append(nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(opt.ngf * 2))
        layers.append(nn.ReLU(True))

        layers.append(nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(opt.ngf))
        layers.append(nn.ReLU(True))

        if opt.masking == 'random-box' or opt.masking == 'random-crop' or opt.masking == 'stitch':
            layers.append(nn.ConvTranspose2d(opt.ngf, opt.ngf, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(opt.ngf))
            layers.append(nn.ReLU(True))

        layers.append(nn.ConvTranspose2d(opt.ngf, opt.nc, 4, 2, 1, bias=False))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netlocalD(nn.Module):
    def __init__(self, opt):
        super(_netlocalD, self).__init__()
        self.ngpu = opt.ngpu
        layers = []
        layers.append(nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        if opt.masking == 'random-box' or opt.masking == 'random-crop' or opt.masking == 'stitch':
            layers.append(nn.Conv2d(opt.ndf, opt.ndf, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(opt.ndf)) 
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(opt.ndf * 2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(opt.ndf * 4))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(opt.ndf * 8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False))

        if not opt.WGAN:
            layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1)

class _netlocalD_WGANGP(nn.Module):
    def __init__(self, opt):
        super(_netlocalD_WGANGP, self).__init__()
        self.ngpu = opt.ngpu
        layers = []
        layers.append(nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        if opt.masking == 'random-box' or opt.masking == 'random-crop' or opt.masking == 'stitch':
            layers.append(nn.Conv2d(opt.ndf, opt.ndf, 4, 2, 1, bias=False))
            # layers.append(nn.BatchNorm2d(opt.ndf))
            layers.append(nn.LayerNorm([opt.ndf, 32, 32]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False))
        # layers.append(nn.BatchNorm2d(opt.ndf * 2))
        layers.append(nn.LayerNorm([opt.ndf * 2, 16, 16]))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False))
        # layers.append(nn.BatchNorm2d(opt.ndf * 4))
        layers.append(nn.LayerNorm([opt.ndf * 4, 8, 8]))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False))
        # layers.append(nn.BatchNorm2d(opt.ndf * 8))
        layers.append(nn.LayerNorm([opt.ndf * 8, 4, 4]))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False))

        if not opt.WGAN:
            layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1)


