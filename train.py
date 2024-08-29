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
from torchvision.utils import make_grid

from torch.utils.tensorboard import SummaryWriter

from model import _netlocalD,_netG,_netlocalD_WGANGP
import utils

from custom_dset import CustomImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  default='custom', help='cifar10 | lsun | imagenet | folder | lfw | custom ')
parser.add_argument('--dataroot',  default='/scratch/krishnansh/Data/lhq_256/train', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=40)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')

parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./saves/lhq_256', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--nBottleneck', type=int,default=4000,help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred',type=int,default=7,help='overlapping edges')
parser.add_argument('--nef',type=int,default=64,help='of encoder filters in first conv layer')
parser.add_argument('--wtl2',type=float,default=1,help='0 means do not use else use with this weight')
parser.add_argument('--overlap_wt',type=float,default=10,help='overlapping pixel error weight')
parser.add_argument('--w_rec',type=float,default=0.998,help='weight for reconstruction loss')
# parser.add_argument('--wtlD',type=float,default=0.001,help='0 means do not use else use with this weight')

parser.add_argument('--masking',type=str,default='center-box',help='random-box | center-box | random-crop | custom | stitch')
parser.add_argument('--masksize',type=str,default='random',help='random | 0.25')

parser.add_argument('--device',type=str,default='cuda:0',help='cuda:0 | cuda:1 | cpu')

parser.add_argument('--save_freq',type=int,default=10,help='save frequency')
parser.add_argument('--save_dir',type=str,default='model/lhq_256/normal',help='save directory')

parser.add_argument('--WGAN',action='store_true',help='WGAN training')
parser.add_argument('--clip_value',type=float,default=0.01,help='clip value for WGAN')
parser.add_argument('--n_critic',type=int,default=1,help='number of critic updates per generator update')

parser.add_argument('--lamdagp',type=float,default=0,help='lambda for gradient penalty')


opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)

with open(os.path.join(opt.outf, 'config.txt'), 'w') as f:
    f.write(str(opt))

writer = SummaryWriter(opt.outf)

try:
    os.makedirs("result/train/cropped")
    os.makedirs("result/train/real")
    os.makedirs("result/train/recon")
    os.makedirs("model")
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:  
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
elif opt.dataset == 'streetview':
    transform = transforms.Compose([transforms.Resize(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = dset.ImageFolder(root=opt.dataroot, transform=transform )
elif opt.dataset == 'custom':
    transform = transforms.Compose([transforms.Resize(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CustomImageDataset(root=opt.dataroot, maskroot="", transforms_=transform, mode="train", masking=opt.masking)
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3
nef = int(opt.nef)
nBottleneck = int(opt.nBottleneck)
wtl2 = float(opt.wtl2)
overlapL2Weight = opt.overlap_wt

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


resume_epoch=0

netG = _netG(opt)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
print(netG)

if opt.WGAN and opt.lamdagp > 0:
    netD = _netlocalD_WGANGP(opt)
else:
    netD = _netlocalD(opt)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD,map_location=lambda storage, location: storage)['state_dict'])
print(netD)

criterion = nn.BCELoss()
criterionMSE = nn.MSELoss()

input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
label = torch.FloatTensor(opt.batchSize)
label.data.resize_(opt.batchSize)
real_label = 1
fake_label = 0

real_center = torch.FloatTensor(opt.batchSize, 3, opt.imageSize//2, opt.imageSize//2)

mask = torch.FloatTensor(input_real.size()).fill_(0)

if opt.cuda:
    netD.cuda(opt.device)
    netG.cuda(opt.device)
    criterion.cuda(opt.device)
    criterionMSE.cuda(opt.device)
    input_real, input_cropped,label = input_real.cuda(opt.device),input_cropped.cuda(opt.device), label.cuda(opt.device)
    real_center = real_center.cuda(opt.device)
    mask = mask.cuda(opt.device)


input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
label = Variable(label)

real_center = Variable(real_center)

# setup optimizer
if opt.WGAN:
    if opt.lamdagp == 0:
        optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lr)
        optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lr)
    else:
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
else:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr/2, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr*2, betas=(opt.beta1, 0.999))

def discriminator_loss(opt,output_real,output_fake,label):
    if opt.WGAN:
        errD_real = -1*torch.mean(output_real)
        errD_fake = torch.mean(output_fake)
    else:
        label.data.fill_(real_label)
        errD_real = criterion(output_real, label)
        label.data.fill_(fake_label)
        errD_fake = criterion(output_fake, label)
    return errD_real, errD_fake

def generator_loss(opt,output,label):
    if opt.WGAN:
        errG_D = -1*torch.mean(output)
    else:
        label.data.fill_(real_label)
        errG_D = criterion(output, label)
    return errG_D

def compute_gp(opt,D,real_samples,fake_samples,label):
    alpha = torch.rand(opt.batchSize, 1, 1, 1)
    if opt.cuda:
        alpha = alpha.cuda(opt.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    label.data.fill_(1.0)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=label,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

for epoch in range(resume_epoch,opt.niter):
    for i, data in enumerate(dataloader, 0):
        real_cpu, _ = data
        real_center_cpu = real_cpu[:,:,int(opt.imageSize/4):int(opt.imageSize/4)+int(opt.imageSize/2),int(opt.imageSize/4):int(opt.imageSize/4)+int(opt.imageSize/2)]
        batch_size = real_cpu.size(0)
        input_real.data.resize_(real_cpu.size()).copy_(real_cpu)
        input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)
        real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)
        if opt.masking == 'center-box':
            input_cropped.data[:,0,int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred),int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred)] = 2*117.0/255.0 - 1.0
            input_cropped.data[:,1,int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred),int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred)] = 2*104.0/255.0 - 1.0
            input_cropped.data[:,2,int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred),int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred)] = 2*123.0/255.0 - 1.0
        elif opt.masking == 'random-crop':
            x1, y1 = random.randint(opt.imageSize//6, opt.imageSize//5), random.randint(opt.imageSize//6, opt.imageSize//5)
            x2, y2 = random.randint(opt.imageSize//6, opt.imageSize//5), random.randint(opt.imageSize//6, opt.imageSize//5)
            x2 = opt.imageSize - x2
            y2 = opt.imageSize - y2
            with torch.no_grad():
                mask.data.fill_(0)
                x1_ = x1 + opt.overlapPred
                y1_ = y1 + opt.overlapPred
                x2_ = x2 - opt.overlapPred
                y2_ = y2 - opt.overlapPred
                mask[:,:,x1_:x2_,y1_:y2_] = 1
                input_cropped.data = input_cropped.data*(mask)

                wtl2Matrix = mask.clone()
                wtl2Matrix.data.fill_(wtl2)
                wtl2Matrix.data[:,:,x1:x2,y1:y2] = wtl2*overlapL2Weight
                # wtl2Matrix.data[:,:,x1+opt.overlapPred:x2-opt.overlapPred,y1+opt.overlapPred:y2-opt.overlapPred] = 0
        elif opt.masking == 'stitch':
            w = opt.imageSize // 2
            x1 = opt.imageSize // 2 - w//2
            x2 = opt.imageSize // 2 + w//2
            x1_ = x1 + opt.overlapPred
            x2_ = x2 - opt.overlapPred
            with torch.no_grad():
                mask.data.fill_(1)
                mask[:,:,:,x1_:x2_]=0
                input_cropped.data = input_cropped.data*(mask)

                wtl2Matrix = mask.clone()
                wtl2Matrix.data.fill_(wtl2)
                wtl2Matrix.data[:,:,:,x1:x2] = wtl2*overlapL2Weight

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ##########################
        netD.zero_grad()

        label.data.resize_(opt.batchSize)


        if opt.masking == 'random-box' or opt.masking  == 'random-crop' or opt.masking == 'stitch':
            output_real = netD(input_real)
        else:
            output_real = netD(real_center)

        label.data.fill_(real_label)
        if not opt.WGAN:
            errD_real = criterion(output_real, label)
        else:
            errD_real = -1*torch.mean(output_real)
        errD_real.backward()

        fake = netG(input_cropped)
        # fake = fake*(1-mask)+mask*(input_real)
        output_fake = netD(fake.detach())

        label.data.fill_(fake_label)
        if not opt.WGAN:
            errD_fake = criterion(output_fake, label)
        else:
            errD_fake = torch.mean(output_fake)
        errD_fake.backward()

        # errD_real, errD_fake = discriminator_loss(opt,output_real,output_fake,label)
        D_x = output_real.data.mean()
        D_G_z1 = output_fake.data.mean()

        if opt.WGAN:
            gradient_penalty = 0
            if opt.lamdagp > 0:
                gradient_penalty = compute_gp(opt,netD,input_real,fake,label)
            errD = errD_real + errD_fake + opt.lamdagp*gradient_penalty
        else:
            errD = errD_real + errD_fake
        # errD.backward()
            
        optimizerD.step()
        # Clip weights of discriminator if WGAN
        if opt.WGAN:
            for p in netD.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        if i % opt.n_critic == 0 or opt.WGAN == False:
            netG.zero_grad()

            # label.data.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)

            errG_D = generator_loss(opt,output,label)

            if opt.masking == 'center-box':
                wtl2Matrix = real_center.clone()
                wtl2Matrix.data.fill_(wtl2*overlapL2Weight)
                wtl2Matrix.data[:,:,int(opt.overlapPred):int(opt.imageSize/2 - opt.overlapPred),int(opt.overlapPred):int(opt.imageSize/2 - opt.overlapPred)] = wtl2
                
                errG_l2 = (fake-real_center).pow(2)
                errG_l2 = errG_l2 * wtl2Matrix
                errG_l2 = errG_l2.mean()
            elif opt.masking == 'random-box' or opt.masking == 'random-crop' or opt.masking == 'stitch':
                errG_l2 = (fake-input_real).pow(2)
                errG_l2 = errG_l2 * wtl2Matrix
                errG_l2 = errG_l2.mean()

            errG = (1-opt.w_rec) * errG_D + opt.w_rec * errG_l2

            errG.backward()

            D_G_z2 = output.data.mean()
            optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f l_D(x): %.4f l_D(G(z)): %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data, errG_D.data,errG_l2.data, D_x,D_G_z1, ))
        
        writer.add_scalar('Loss_D', errD.data, epoch*len(dataloader)+i)
        writer.add_scalar('Loss_G', errG.data, epoch*len(dataloader)+i)
        writer.add_scalar('Loss_G_D', errG_D.data, epoch*len(dataloader)+i)
        writer.add_scalar('Loss_G_l2', errG_l2.data, epoch*len(dataloader)+i)
        if opt.lamdagp > 0 and opt.WGAN:
            writer.add_scalar('Gradient_penalty', gradient_penalty, epoch*len(dataloader)+i)
        writer.add_scalar('D_x', D_x, epoch*len(dataloader)+i)
        writer.add_scalar('D_G_z1', D_G_z1, epoch*len(dataloader)+i)
        
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    'result/train/real/real_samples_epoch_%03d.png' % (epoch))
            vutils.save_image(input_cropped.data,
                    'result/train/cropped/cropped_samples_epoch_%03d.png' % (epoch))
            recon_image = input_cropped.clone()
            if opt.masking == 'center-box':
                recon_image.data[:,:,int(opt.imageSize/4):int(opt.imageSize/4+opt.imageSize/2),int(opt.imageSize/4):int(opt.imageSize/4+opt.imageSize/2)] = fake.data
            elif opt.masking == 'random-box' or opt.masking == 'random-crop' or opt.masking=='stitch':
                recon_image.data = fake.data
            vutils.save_image(recon_image.data,
                    'result/train/recon/recon_center_samples_epoch_%03d.png' % (epoch))

            save_img = torch.cat((input_cropped.data[:8], recon_image.data[:8], input_real.data[:8]), -2)
            image_grid = make_grid(save_img, nrow=8, normalize=True, scale_each=True)
            writer.add_image('train_samples', image_grid, epoch*len(dataloader)+i)


    # do checkpointing
    if epoch % opt.save_freq == 0:
        torch.save({'epoch':epoch+1,
                    'state_dict':netG.state_dict()},
                    '%s/netG_%d.pth' % (opt.save_dir, epoch))
        torch.save({'epoch':epoch+1,
                    'state_dict':netD.state_dict()},
                    '%s/netD_%d.pth' % (opt.save_dir, epoch))

torch.save({'epoch':opt.niter+1,
                    'state_dict':netG.state_dict()},
                    '%s/netG_final.pth' % opt.save_dir)
torch.save({'epoch':opt.niter+1,
            'state_dict':netD.state_dict()},
            '%s/netD_final.pth' % opt.save_dir)

writer.close()
