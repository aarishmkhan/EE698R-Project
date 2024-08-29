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
parser.add_argument('--netG', default='model/cifar10_outp_1/netG.pth', help="path to netG (to continue training)")
parser.add_argument('--netD', default='model/cifar10_outp_1/netlocalD.pth', help="path to netD (to continue training)")
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

writer = SummaryWriter(opt.outf)

netG = _netG(opt)
netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
netG.eval()

transform = transforms.Compose([transforms.Resize(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = dset.ImageFolder(root=opt.dataroot, transform=transform )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
real_center = torch.FloatTensor(opt.batchSize, 3, opt.imageSize//2, opt.imageSize//2)

mask = torch.FloatTensor(input_real.size()).fill_(0)

criterionMSE = nn.MSELoss()

if opt.cuda:
    netG.cuda(opt.device)
    input_real, input_cropped = input_real.cuda(opt.device),input_cropped.cuda(opt.device)
    criterionMSE.cuda(opt.device)
    real_center = real_center.cuda(opt.device)

input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
real_center = Variable(real_center)

dataiter = iter(dataloader)
psnr = []
l1 = []
l2 = []
for i, (real_cpu, _) in enumerate(dataiter):
    input_real.data.resize_(real_cpu.size()).copy_(real_cpu)
    input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)
    real_center_cpu = real_cpu[:,:,opt.imageSize/4:opt.imageSize/4+opt.imageSize/2,opt.imageSize/4:opt.imageSize/4+opt.imageSize/2]
    real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)

    if opt.masking == 'center-box':
        input_cropped.data[:,0,opt.imageSize/4+opt.overlapPred:opt.imageSize/4+opt.imageSize/2-opt.overlapPred,opt.imageSize/4+opt.overlapPred:opt.imageSize/4+opt.imageSize/2-opt.overlapPred] = 2*117.0/255.0 - 1.0
        input_cropped.data[:,1,opt.imageSize/4+opt.overlapPred:opt.imageSize/4+opt.imageSize/2-opt.overlapPred,opt.imageSize/4+opt.overlapPred:opt.imageSize/4+opt.imageSize/2-opt.overlapPred] = 2*104.0/255.0 - 1.0
        input_cropped.data[:,2,opt.imageSize/4+opt.overlapPred:opt.imageSize/4+opt.imageSize/2-opt.overlapPred,opt.imageSize/4+opt.overlapPred:opt.imageSize/4+opt.imageSize/2-opt.overlapPred] = 2*123.0/255.0 - 1.0
    elif opt.masking == 'random-box' or opt.masking == 'random-crop':
        x1, y1 = random.randint(opt.imageSize//5, opt.imageSize//4), random.randint(opt.imageSize//5, opt.imageSize//4)
        x2, y2 = random.randint(opt.imageSize//5, opt.imageSize//4), random.randint(opt.imageSize//5, opt.imageSize//4)
        x2 = opt.imageSize - x2
        y2 = opt.imageSize - y2
        with torch.no_grad():
            mask[:,:,x1:x2,y1:y2] = 1
            input_cropped.data = input_cropped.data*(mask)

    fake = netG(input_cropped)
    errG = criterionMSE(fake,real_center)

    if opt.masking == 'random-box' or opt.masking == 'random-crop':
        recon_image = fake*(1-mask) + input_real*mask
    else:
        recon_image = input_cropped.clone()
        recon_image.data[:,:,opt.imageSize/4:opt.imageSize/4+opt.imageSize/2,opt.imageSize/4:opt.imageSize/4+opt.imageSize/2] = fake.data

    vutils.save_image(real_cpu,'./result/test/val_real_samples_%d.png' % i,normalize=True)
    vutils.save_image(input_cropped.data,'./result/test/val_cropped_samples.png',normalize=True)
    vutils.save_image(recon_image.data,'./result/test/val_recon_samples.png',normalize=True)
    p=0
    l1=0
    l2=0
    fake = fake.data.numpy()
    real_center = real_center.data.numpy()
    from psnr import psnr
    import numpy as np

    if opt.masking == 'random-box' or opt.masking == 'random-crop':
        t = input_real - recon_image
        l2 = np.mean(np.square(t))
        l1 = np.mean(np.abs(t))
        for i in range(opt.batchSize):
            p = p + psnr(input_real[i].transpose(1,2,0) , fake[i].transpose(1,2,0))
    else:
        t = real_center - fake
        l2 = np.mean(np.square(t))
        l1 = np.mean(np.abs(t))
        real_center = (real_center+1)*127.5
        fake = (fake+1)*127.5
        for i in range(opt.batchSize):
            p = p + psnr(real_center[i].transpose(1,2,0) , fake[i].transpose(1,2,0))
    
    msg = '[%d/%d] : PSNR: %.4f, L1: %.4f, L2: %.4f' % (i*opt.batchSize,len(dataset),p/opt.batchSize,l1,l2)
    print(msg)

    writer.add_scalar('PSNR',p/opt.batchSize,i)
    writer.add_scalar('L1',l1,i)
    writer.add_scalar('L2',l2,i)
    # print(l2)

    # print(l1)

    # print(p/opt.batchSize)



