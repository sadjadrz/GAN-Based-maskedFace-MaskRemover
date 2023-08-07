import torch
import torch.nn as nn
from torchvision.utils import save_image

import numpy as np
from PIL import Image
import cv2
from models import UNetSemantic, GatedGenerator
import argparse
from configs import Config

class Predictor():
    def __init__(self, cfg , checkpoint):
        self.cfg = cfg
        self.checkpoint = checkpoint
        self.device = torch.device('cuda:0' if cfg.cuda else 'cpu')

        self.inpaint = GatedGenerator().to(self.device)
        self.inpaint.load_state_dict(torch.load(f'weights/{self.checkpoint}.pth', map_location='cpu')['G'])
        self.inpaint.eval()

    def save_image(self, img_list, save_img_path, nrow):
        img_list  = [i.clone().cpu() for i in img_list]
        imgs = torch.stack(img_list, dim=1)
        imgs = imgs.view(-1, *list(imgs.size())[2:])
        save_image(imgs, save_img_path, nrow = nrow)
        print(f"Save image to {save_img_path}")

    def predict(self, image, num , outpath='sample/results.png'):
        outpath=f'output/results_{image}_{num}.png'
        image = 'sample/'+image
        img = cv2.imread(image+'_masked.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.cfg.img_size, self.cfg.img_size))
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        img = img.unsqueeze(0).to(self.device)

        img_binary = cv2.imread(image+'_binary.jpg', 0)
        img_binary[img_binary > 0] = 1.0
        img_binary = cv2.resize(img_binary, (self.cfg.img_size, self.cfg.img_size))
        img_binary = np.expand_dims(img_binary, axis=0)
        img_binary = np.expand_dims(img_binary, axis=0)
        img_binary = torch.from_numpy(img_binary.astype(np.float32)).contiguous()



        img_ori = cv2.imread(image+'.jpg')
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        img_ori = cv2.resize(img_ori, (self.cfg.img_size, self.cfg.img_size))
        img_ori = torch.from_numpy(img_ori.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        img_ori = img_ori.unsqueeze(0)
        with torch.no_grad():
            #outputs = self.masking(img)
            _, out = self.inpaint(img, img_binary)
            inpaint = img * (1 - img_binary) + out * img_binary
        masks = img * (1 - img_binary) + img_binary
        self.save_image([img, masks, inpaint, img_ori], outpath, nrow=4)

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training custom model')
    parser.add_argument('--image', default='003172',required=False , type=str, help='resume training')
    parser.add_argument('--config', default='facemask', required=False , type=str, help='config training')
    args = parser.parse_args() 

    config = Config(f'./configs/{args.config}.yaml')
    checkpoints = ['model_60_187500','model_64_199500' ,
                   'model_66_207000','model_67_210000','model_70_219000',
                   'model_71_222000','model_72_225000','model_73_228000',
                   'model_75_235500']
    img = ['003172','003171','003170','003169','003168']

    for i in range(len(checkpoints)):

        model = Predictor(config , checkpoints[i])
        for ii in range(len(img)):
            num = checkpoints[i]
            model.predict(img[ii] , checkpoints[i])