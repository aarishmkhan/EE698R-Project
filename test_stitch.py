from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from PIL import Image

from model import _netG

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  default='folder', help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot',  default='/scratch/krishnansh/Data/cifar10/test', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='model/lhq_256/normal_stitch/run_2/netG_final.pth', help="path to netG (to continue training)")
parser.add_argument('--netD', default='model/lhq_256/normal_stitch/run_2/netD_final.pth', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./out', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--nBottleneck', type=int,default=4000,help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred',type=int,default=4,help='overlapping edges')
parser.add_argument('--nef',type=int,default=64,help='of encoder filters in first conv layer')
parser.add_argument('--wtl2',type=float,default=0.999,help='0 means do not use else use with this weight')


parser.add_argument('--masking',type=str,default='random-crop',help='random-box | center-box | random-crop | custom')
parser.add_argument('--masksize',type=str,default='random',help='random | 0.25')

parser.add_argument('--device',type=str,default='cuda:0',help='cuda:0 | cuda:1 | cpu')

opt = parser.parse_args()
print(opt)

# writer = SummaryWriter(opt.outf)

test_image_dir = "./test_stitch"

netG = _netG(opt)
netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
netG.eval()

input_1 = torch.FloatTensor(1,3, opt.imageSize, opt.imageSize)
input_2 = torch.FloatTensor(1,3, opt.imageSize, opt.imageSize)
combined = torch.FloatTensor(1,3, opt.imageSize, opt.imageSize)
combined_masked = torch.FloatTensor(1,3, opt.imageSize, opt.imageSize)

mask = torch.FloatTensor(input_1.size()).fill_(1)

criterionMSE = nn.MSELoss()

if opt.cuda:
    netG.cuda(opt.device)
    input_1, input_2 = input_1.cuda(opt.device),input_2.cuda(opt.device)
    criterionMSE.cuda(opt.device)
    mask = mask.cuda(opt.device)
    combined = combined.cuda(opt.device)
    combined_masked = combined_masked.cuda(opt.device)

transform = transforms.Compose([transforms.Resize(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
img1 = Image.open(os.path.join(test_image_dir, "1.png"))
img1 = transform(img1)
img1 = img1.repeat(1, 1, 1, 1)
input_1.data.resize_(img1.size()).copy_(img1)

img2 = Image.open(os.path.join(test_image_dir, "2.png"))
img2 = transform(img2)
img2 = img2.repeat(1, 1, 1, 1)
input_2.data.resize_(img2.size()).copy_(img2)

combined = input_1.clone()
combined.data[:,:,:,opt.imageSize//2:opt.imageSize] = input_2.data[:,:,:,opt.imageSize//2:opt.imageSize]

w = opt.imageSize // 3
x1 = opt.imageSize // 2 - w//2
x2 = opt.imageSize // 2 + w//2
mask.data.fill_(1)
mask[:,:,:,x1:x2]=0

combined_masked = combined.clone()
combined_masked.data = combined.data*(mask)

fake = netG(combined_masked)
fake = fake*(1-mask) + combined*mask

vutils.save_image(fake[0],'%s/val_real_samples_2.png' % test_image_dir,normalize=True)