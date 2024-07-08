import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse

from model.RIFE import Model
from dataset import *
from aligned_dataset import AlignedDataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

device = torch.device("cuda")

log_path = 'train_log'

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
        return 1e-4 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (1e-4 - 3e-6) * mul + 3e-6

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def train(model, local_rank, args):
    if local_rank == 0:
        writer = SummaryWriter('train')
        writer_val = SummaryWriter('validate')
    else:
        writer = None
        writer_val = None
    step = 0
    nr_eval = 0
    # dataset = VimeoDataset('train')
    dataset = AlignedDataset(args)
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__()
    # dataset_val = VimeoDataset('validation')
    # val_data = DataLoader(dataset_val, batch_size=16, pin_memory=True, num_workers=8)
    print('training...')
    time_stamp = time.time()
    for epoch in range(args.epoch):
        sampler.set_epoch(epoch)
        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            # data_gpu, timestep = data
            # data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            # timestep = timestep.to(device, non_blocking=True)
            # imgs = data_gpu[:, :6]
            # gt = data_gpu[:, 6:9]
            A, B = data['A'].to(device, non_blocking=True), data['B'].to(device, non_blocking=True)
            learning_rate = get_learning_rate(step) * args.world_size / 4
            pred, info = model.update(A, B, learning_rate, training=True) # pass timestep if you are training RIFEm
            merged_stu, merged_tea = pred, info['merged_tea']
            flow_stu, flow_tea = info['flow'], info['flow_tea']
            # loss_l1, loss_l1_11, loss_l1_22, loss_l1_33, loss_l1_44 = info['loss_l1'], info['loss_l1_11'], info['loss_l1_22'], info['loss_l1_33'], info['loss_l1_44']
            # loss_tea, loss_tea_11, loss_tea_22, loss_tea_33, loss_tea_44 = info['loss_tea'], info['loss_tea_11'], info['loss_tea_22'], info['loss_tea_33'], info['loss_tea_44']
            # loss_vgg_stu, loss_vgg_stu_11, loss_vgg_stu_22, loss_vgg_stu_33, loss_vgg_stu_44 = info['loss_vgg_stu'], info['loss_vgg_stu_11'], info['loss_vgg_stu_22'], info['loss_vgg_stu_33'], info['loss_vgg_stu_44']
            # loss_vgg_tea, loss_vgg_tea_11, loss_vgg_tea_22, loss_vgg_tea_33, loss_vgg_tea_44 = info['loss_vgg_tea'], info['loss_vgg_tea_11'], info['loss_vgg_tea_22'], info['loss_vgg_tea_33'], info['loss_vgg_tea_44']
            loss_distill = info['loss_distill']
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 200 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/l1', info['loss_l1'], step)
                writer.add_scalar('loss/tea', info['loss_tea'], step)
                writer.add_scalar('loss/distill', info['loss_distill'], step)
                writer.add_scalar('loss/mse_stu', info['loss_mse'], step)
                writer.add_scalar('loss/mse_tea', info['loss_mse_tea'], step)
                writer.add_scalar('loss/mse_11', info['loss_mse_11'], step)
                writer.add_scalar('loss/mse_22', info['loss_mse_22'], step)
                writer.add_scalar('loss/mse_33', info['loss_mse_33'], step)
                writer.add_scalar('loss/mse_44', info['loss_mse_44'], step)
                writer.add_scalar('loss/tea_mse_11', info['loss_mse_tea_11'], step)
                writer.add_scalar('loss/tea_mse_22', info['loss_mse_tea_22'], step)
                writer.add_scalar('loss/tea_mse_33', info['loss_mse_tea_33'], step)
                writer.add_scalar('loss/tea_mse_44', info['loss_mse_tea_44'], step)
                writer.add_scalar('loss/vgg_stu', info['loss_vgg_stu'], step)
                writer.add_scalar('loss/vgg_tea', info['loss_vgg_tea'], step)
                writer.add_scalar('loss/vgg_stu_11', info['loss_vgg_stu_11'], step)
                writer.add_scalar('loss/vgg_stu_22', info['loss_vgg_stu_22'], step)
                writer.add_scalar('loss/vgg_stu_33', info['loss_vgg_stu_33'], step)
                writer.add_scalar('loss/vgg_stu_44', info['loss_vgg_stu_44'], step)
                writer.add_scalar('loss/vgg_tea_11', info['loss_vgg_tea_11'], step)
                writer.add_scalar('loss/vgg_tea_22', info['loss_vgg_tea_22'], step)
                writer.add_scalar('loss/vgg_tea_33', info['loss_vgg_tea_33'], step)
                writer.add_scalar('loss/vgg_tea_44', info['loss_vgg_tea_44'], step)
            if step % 100 == 1 and local_rank == 0:
                # gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                # mask = (torch.cat((info['mask'], info['mask_tea']), 3).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                # pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                # merged_img = (info['merged_tea'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                # flow0 = info['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                # flow1 = info['flow_tea'].permute(0, 2, 3, 1).detach().cpu().numpy()
                # for i in range(5):
                #     imgs = np.concatenate((merged_img[i], pred[i], gt[i]), 1)[:, :, ::-1]
                #     writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                #     writer.add_image(str(i) + '/flow', np.concatenate((flow2rgb(flow0[i]), flow2rgb(flow1[i])), 1), step, dataformats='HWC')
                #     writer.add_image(str(i) + '/mask', mask[i], step, dataformats='HWC')
                # writer.flush()
                A = (A.permute(0, 2, 3, 1).detach().cpu().numpy() * 255)[0].astype('uint8')
                B = (B.permute(0, 2, 3, 1).detach().cpu().numpy() * 255)[0].astype('uint8')
                flow_stu = (flow_stu.permute(0, 2, 3, 1).detach().cpu().numpy() * 255)[0].astype('uint8')
                flow_tea = (flow_tea.permute(0, 2, 3, 1).detach().cpu().numpy() * 255)[0].astype('uint8')
                merged_stu = (merged_stu.permute(0, 2, 3, 1).detach().cpu().numpy() * 255)[0].astype('uint8')
                merged_tea = (merged_tea.permute(0, 2, 3, 1).detach().cpu().numpy() * 255)[0].astype('uint8')
                cv2.imwrite(os.path.join(args.train_log, 'image/{}_A.png'.format(epoch)), cv2.cvtColor(A, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(args.train_log, 'image/{}_B.png'.format(epoch)), cv2.cvtColor(B, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(args.train_log, 'image/{}_flow_stu.png'.format(epoch)), cv2.cvtColor(flow_stu, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(args.train_log, 'image/{}_flow_tea.png'.format(epoch)), cv2.cvtColor(flow_tea, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(args.train_log, 'image/{}_merged_stu.png'.format(epoch)), cv2.cvtColor(merged_stu, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(args.train_log, 'image/{}_merged_tea.png'.format(epoch)), cv2.cvtColor(merged_tea, cv2.COLOR_RGB2BGR))

            if local_rank == 0 and step % 10 == 1:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} lr:{:.2e}, loss_l1:{:.2e}, loss_tea:{:.2e}, loss_mse:{:.2e}, loss_mse_tea:{:.2e}, loss_vgg_stu:{:.2e}, loss_vgg_tea:{:.2e}, loss_distill:{:.2e}'.format(
                    epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, learning_rate, info['loss_l1'], info['loss_tea'], info['loss_mse'], info['loss_mse_tea'], info['loss_vgg_stu'], info['loss_vgg_tea'], info['loss_distill']))
            step += 1
        nr_eval += 1
        # if nr_eval % 5 == 0:
        #     evaluate(model, val_data, step, local_rank, writer_val)
        model.save_model(log_path, local_rank, epoch)    
        dist.barrier()

def evaluate(model, val_data, nr_eval, local_rank, writer_val):
    loss_l1_list = []
    loss_distill_list = []
    loss_tea_list = []
    psnr_list = []
    psnr_list_teacher = []
    time_stamp = time.time()
    for i, data in enumerate(val_data):
        data_gpu, timestep = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.        
        imgs = data_gpu[:, :6]
        gt = data_gpu[:, 6:9]
        with torch.no_grad():
            pred, info = model.update(imgs, gt, training=False)
            merged_img = info['merged_tea']
        loss_l1_list.append(info['loss_l1'].cpu().numpy())
        loss_tea_list.append(info['loss_tea'].cpu().numpy())
        loss_distill_list.append(info['loss_distill'].cpu().numpy())
        for j in range(gt.shape[0]):
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
            psnr = -10 * math.log10(torch.mean((merged_img[j] - gt[j]) * (merged_img[j] - gt[j])).cpu().data)
            psnr_list_teacher.append(psnr)
        gt = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        merged_img = (merged_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        flow0 = info['flow'].permute(0, 2, 3, 1).cpu().numpy()
        flow1 = info['flow_tea'].permute(0, 2, 3, 1).cpu().numpy()
        if i == 0 and local_rank == 0:
            for j in range(10):
                imgs = np.concatenate((merged_img[j], pred[j], gt[j]), 1)[:, :, ::-1]
                writer_val.add_image(str(j) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
                writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j][:, :, ::-1]), nr_eval, dataformats='HWC')
    
    eval_time_interval = time.time() - time_stamp

    if local_rank != 0:
        return
    writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
    writer_val.add_scalar('psnr_teacher', np.array(psnr_list_teacher).mean(), nr_eval)
        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help='minibatch size')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    parser.add_argument('--train_log', default='train_log', type=str, help='train log dir')
    parser.add_argument('--dataroot', default='', type=str, help='dataset dir')
    parser.add_argument('--phase', default='train', type=str)
    parser.add_argument('--direction', default='BtoA', type=str)
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
    args = parser.parse_args()
    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(args.local_rank)
    train(model, args.local_rank, args)
        
