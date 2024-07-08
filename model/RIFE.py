import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.IFNet import *
from model.IFNet_m import *
import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
from model.refine import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
    

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss


class Model:
    def __init__(self, local_rank=-1, arbitrary=False):
        if arbitrary == True:
            self.flownet = IFNet_m()
        else:
            # self.flownet = IFNet()
            self.flownet = IFNet_Retouching()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3) # use large weight decay may avoid NaN loss
        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = SOBEL()
        self.vgg = VGGLoss()
        self.mse = torch.nn.MSELoss()
        print('local_rank: {}'.format(local_rank))
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)     # find_unused_parameters=True
            # self.vgg = DDP(self.vgg, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
            
        if rank <= 0:
            self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))))
        
    def save_model(self, path, epoch, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet_epoch{}.pkl'.format(path, epoch))

    def inference(self, img0, img1, scale=1, scale_list=[4, 2, 1], TTA=False, timestep=0.5):
        for i in range(3):
            scale_list[i] = scale_list[i] * 1.0 / scale
        imgs = torch.cat((img0, img1), 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        if TTA == False:
            return merged[2]
        else:
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2
    
    def update(self, real_A, real_B, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()
        
        merged_stu, merged_stu_11, merged_stu_22, merged_stu_33, merged_stu_44, merged_tea, merged_tea_11, merged_tea_22, merged_tea_33, merged_tea_44, flow_stu, flow_tea, loss_distill = self.flownet(real_A, real_B)
        loss_distill = loss_distill
        loss_l1 = (self.lap(merged_stu, real_B)).mean()
        loss_tea = (self.lap(merged_tea, real_B)).mean()
        # loss_l1_11 = (self.lap(merged_stu_11, real_B)).mean()
        # loss_l1_22 = (self.lap(merged_stu_22, real_B)).mean()
        # loss_l1_33 = (self.lap(merged_stu_33, real_B)).mean()
        # loss_l1_44 = (self.lap(merged_stu_44, real_B)).mean()
        # loss_tea_11 = (self.lap(merged_tea_11, real_B)).mean()
        # loss_tea_22 = (self.lap(merged_tea_22, real_B)).mean()
        # loss_tea_33 = (self.lap(merged_tea_33, real_B)).mean()
        # loss_tea_44 = (self.lap(merged_tea_44, real_B)).mean()
        loss_mse_stu = (self.mse(merged_stu, real_B)).mean() * 100
        loss_mse_stu_11 = (self.mse(merged_stu_11, real_B)).mean() * 100
        loss_mse_stu_22 = (self.mse(merged_stu_22, real_B)).mean() * 100
        loss_mse_stu_33 = (self.mse(merged_stu_33, real_B)).mean() * 100
        loss_mse_stu_44 = (self.mse(merged_stu_44, real_B)).mean() * 100
        loss_mse_tea = (self.mse(merged_tea, real_B)).mean() * 100
        loss_mse_tea_11 = (self.mse(merged_tea_11, real_B)).mean() * 100
        loss_mse_tea_22 = (self.mse(merged_tea_22, real_B)).mean() * 100
        loss_mse_tea_33 = (self.mse(merged_tea_33, real_B)).mean() * 100
        loss_mse_tea_44 = (self.mse(merged_tea_44, real_B)).mean() * 100
        loss_vgg_stu = self.vgg(merged_stu, real_B)
        loss_vgg_stu_11 = (self.lap(merged_stu_11, real_B)).mean()  # self.vgg(merged_stu_11, real_B)
        loss_vgg_stu_22 = (self.lap(merged_stu_22, real_B)).mean()  # self.vgg(merged_stu_22, real_B)
        loss_vgg_stu_33 = (self.lap(merged_stu_33, real_B)).mean()  # self.vgg(merged_stu_33, real_B)
        loss_vgg_stu_44 = (self.lap(merged_stu_44, real_B)).mean()  # self.vgg(merged_stu_44, real_B)
        loss_vgg_tea = self.vgg(merged_tea, real_B)
        loss_vgg_tea_11 = (self.lap(merged_tea_11, real_B)).mean()  # self.vgg(merged_tea_11, real_B)
        loss_vgg_tea_22 = (self.lap(merged_tea_22, real_B)).mean()  # self.vgg(merged_tea_22, real_B)
        loss_vgg_tea_33 = (self.lap(merged_tea_33, real_B)).mean()  # self.vgg(merged_tea_33, real_B)
        loss_vgg_tea_44 = (self.lap(merged_tea_44, real_B)).mean()  # self.vgg(merged_tea_44, real_B)

        if training:
            self.optimG.zero_grad()
            # loss_deep_supervision = loss_l1_11 + loss_l1_22 + loss_l1_33 + loss_l1_44 + loss_tea_11 + loss_tea_22 + loss_tea_33 + loss_tea_44
            # loss_G = loss_l1 + loss_tea + loss_distill * 0.01 + loss_deep_supervision   # when training RIFEm, the weight of loss_distill should be 0.005 or 0.002
            loss_deep_supervision = loss_mse_stu_11 + loss_mse_stu_22 + loss_mse_stu_33 + loss_mse_stu_44 + loss_mse_tea_11 + loss_mse_tea_22 + loss_mse_tea_33 + loss_mse_tea_44 + \
                                    loss_vgg_stu_11 + loss_vgg_stu_22 + loss_vgg_stu_33 + loss_vgg_stu_44 + loss_vgg_tea_11 + loss_vgg_tea_22 + loss_vgg_tea_33 + loss_vgg_tea_44
            loss_G = loss_l1 + loss_tea + loss_mse_stu + loss_mse_tea + loss_vgg_stu + loss_vgg_tea + loss_distill * 0.01 + loss_deep_supervision   # when training RIFEm, the weight of loss_distill should be 0.005 or 0.002
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow_tea
        return merged_stu, {
            'merged_tea': merged_tea,
            'flow': flow_stu,
            'flow_tea': flow_tea,
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_mse': loss_mse_stu,
            'loss_mse_tea': loss_mse_tea,
            'loss_mse_11': loss_mse_stu_11,
            'loss_mse_22': loss_mse_stu_22,
            'loss_mse_33': loss_mse_stu_33,
            'loss_mse_44': loss_mse_stu_44,
            'loss_mse_tea_11': loss_mse_tea_11,
            'loss_mse_tea_22': loss_mse_tea_22,
            'loss_mse_tea_33': loss_mse_tea_33,
            'loss_mse_tea_44': loss_mse_tea_44,
            'loss_vgg_stu': loss_vgg_stu,
            'loss_vgg_stu_11': loss_vgg_stu_11,
            'loss_vgg_stu_22': loss_vgg_stu_22,
            'loss_vgg_stu_33': loss_vgg_stu_33,
            'loss_vgg_stu_44': loss_vgg_stu_44,
            'loss_vgg_tea': loss_vgg_tea,
            'loss_vgg_tea_11': loss_vgg_tea_11,
            'loss_vgg_tea_22': loss_vgg_tea_22,
            'loss_vgg_tea_33': loss_vgg_tea_33,
            'loss_vgg_tea_44': loss_vgg_tea_44,
            'loss_distill': loss_distill * 0.01,
        }
