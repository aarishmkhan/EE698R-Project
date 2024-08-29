## EE698 Project: Image extension and stitching using inpainting models

This codebase contains two generative models for image inpainting implementations in pytorch:<br/>
Context Encoder[[Code](https://github.com/BoyuanJiang/context_encoder_pytorch/blob/master/train.py)][[Paper](https://arxiv.org/pdf/1604.07379.pdf)]<br/>
PDGAN[[Code](https://github.com/KumapowerLIU/PD-GAN/tree/main)][[Paper](https://arxiv.org/pdf/2105.02201.pdf)]

We have modified these codebase for image extension and stitching tasks. 
The context encoder contains implementation of generating, box mask, crop mask, and stitching masks, which could be specified using the --masking flag as is clear from the code. <br/>
To train the Context Encoder, use 
`
python3 train.py --cuda
`
<br/>
The masking is set to custom masks by default in PDGAN. Masks can be generated using mask_gen file. These custom masks can be specified to the PDGAN code by --mask_root flag
<br/>
To train the PDGAN code run <br/>
`
python3 train.py
`
<br/>
For diffusion pipline--<br/>
Download pre-trained model edm-imagenet-64x64-cond-adm.pkl from https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/ and in infinite_image_gen_diffusion directory run main.py to generate new samples.
![Alt text](/finalResult.png)
# EE698R-Project
