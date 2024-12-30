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


