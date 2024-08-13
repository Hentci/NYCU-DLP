import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder
from torchvision.utils import save_image
from torch import stack

import imageio
from math import log10
from Trainer import VAE_Model
import glob
import pandas as pd


def get_key(fp):
    filename = fp.split('/')[-1]
    filename = filename.split('.')[0].replace('frame', '')
    return int(filename)

from torch.utils.data import DataLoader
from glob import glob
from torch.utils.data import Dataset as torchData
from torchvision.datasets.folder import default_loader as imgloader

class Dataset_Dance(torchData):
    """
        Args:
            root (str)      : The path of your Dataset
            transform       : Transformation to your dataset
            mode (str)      : train, val, test
            partial (float) : Percentage of your Dataset, may set to use part of the dataset
    """
    def __init__(self, root, transform, mode='train', video_len=7, partial=1.0):
        super().__init__()
        assert mode in ['train', 'val'], "There is no such mode !!!"
        if mode == 'train':
            self.img_folder     = sorted(glob(os.path.join(root, 'train/train_img/*.png')), key=get_key)
            self.prefix = 'train'
        elif mode == 'val':
            self.img_folder     = sorted(glob(os.path.join(root, 'val/val_img/*.png')), key=get_key)
            self.prefix = 'val'
        else:
            raise NotImplementedError
        
        self.transform = transform
        self.partial = partial
        self.video_len = video_len

    def __len__(self):
        return int(len(self.img_folder) * self.partial) // self.video_len

    def __getitem__(self, index):
        path = self.img_folder[index]
        
        imgs = []
        labels = []
        for i in range(self.video_len):
            label_list = self.img_folder[(index*self.video_len)+i].split('/')
            label_list[-2] = self.prefix + '_label'
            
            img_name    = self.img_folder[(index*self.video_len)+i]
            label_name = '/'.join(label_list)

            imgs.append(self.transform(imgloader(img_name)))
            labels.append(self.transform(imgloader(label_name)))
        return stack(imgs), stack(labels)


class Test_model(VAE_Model):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        
    def forward(self, img, label):
        pass     
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        psnr_list = []  # 用來存儲每一幀的PSNR值
        pred_seq_list = []
        for idx, (img, label) in enumerate(tqdm(val_loader, ncols=80)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            pred_seq, psnr = self.val_one_step(img, label, idx)
            pred_seq_list.append(pred_seq)
            psnr_list.extend(psnr)
        
        # submission.csv is the file you should submit to kaggle
        pred_to_int = (np.rint(torch.cat(pred_seq_list).numpy()*255)).astype(int)
        df = pd.DataFrame(pred_to_int)
        df.insert(0, 'id', range(0, len(df)))
        df.to_csv(os.path.join(self.args.save_root, f'submission.csv'), header=True, index=False)

        # 繪製PSNR圖表
        self.plot_psnr_curve(psnr_list)
            
    def plot_psnr_curve(self, psnr_list):
        plt.figure(figsize=(10, 5))
        plt.plot(psnr_list, label=f'Avg PSNR: {np.mean(psnr_list):.3f}')
        plt.title('Per frame Quality (PSNR)')
        plt.xlabel('Frame index')
        plt.ylabel('PSNR')
        plt.legend()
        plt.savefig(os.path.join(self.args.save_root, 'psnr_curve.png'))
        plt.close()
    
    def val_one_step(self, img, label, idx=0):
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        assert label.shape[0] == 630, "Testing pose seqence should be 630"
        # assert img.shape[0] == 1, "Testing video seqence should be 1"
        
        # decoded_frame_list is used to store the predicted frame seq
        # label_list is used to store the label seq
        # Both list will be used to make gif
        decoded_frame_list = [img[0]]
        label_list = []
        psnr_list = []

        # TODO
        for t in range(1, label.shape[0]):
            current_img = img[0] if t == 1 else decoded_frame_list[-1]
            current_label = label[t]
            
            # Forward pass
            frame_feature = self.frame_transformation(current_img)
            label_feature = self.label_transformation(current_label)
            
            # Conduct Posterior prediction in Encoder
            z, mu, logvar = self.Gaussian_Predictor(frame_feature, label_feature)
            
            # Decoder fusion and generative model
            decoder_feature = self.Decoder_Fusion(frame_feature, label_feature, torch.randn_like(z))
            output = self.Generator(decoder_feature)
            output = torch.nn.functional.sigmoid(output)
            
            decoded_frame_list.append(output)
            label_list.append(current_label)
            
            # 計算PSNR
            psnr = self.compute_psnr(output, current_img)
            psnr_list.append(psnr)
            
        # Please do not modify this part, it is used for visulization
        generated_frame = stack(decoded_frame_list).permute(1, 0, 2, 3, 4)
        label_frame = stack(label_list).permute(1, 0, 2, 3, 4)
        
        assert generated_frame.shape == (1, 630, 3, 32, 64), f"The shape of output should be (1, 630, 3, 32, 64), but your output shape is {generated_frame.shape}"
        
        self.make_gif(generated_frame[0], os.path.join(self.args.save_root, f'pred_seq{idx}.gif'))
        
        # Reshape the generated frame to (630, 3 * 64 * 32)
        generated_frame = generated_frame.reshape(630, -1)
        
        return generated_frame, psnr_list
    
    def compute_psnr(self, img1, img2, data_range=1.):
        mse = nn.functional.mse_loss(img1, img2)
        psnr = 20 * torch.log10(torch.tensor(data_range)) - 10 * torch.log10(mse)
        return psnr.item()
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=20, loop=0)
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 




def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    model = Test_model(args).to(args.device)
    model.load_checkpoint()
    model.eval()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--no_sanity',     action='store_true')
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--make_gif',      action='store_true')
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")

    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   

    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")

    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")




    args = parser.parse_args()

    main(args)