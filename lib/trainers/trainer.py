# Copyright (c) Ramy Mounir.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from lib.utils.file import checkdir
from lib.utils.tensorboard import get_writer, TBWriter
from lib.core.scheduler import cosine_scheduler
from lib.utils.distributed import MetricLogger
from glob import glob
import math
import numpy as np
import tifffile as tif
import sys
import re

import os

import torch

def get_three_slice(x):
    radius =int(x.shape[-1]//2)
    x_x = x[:,:,radius]
    x_y = x[:,radius,:]
    x_z = x[radius,:,:]
    return x_x, x_y, x_z

def center_crop_tensor(input_tensor, receptive_field):
    # Ensure that the input tensor has the shape B*C*D*H*W
    B, C, D, H, W = input_tensor.shape

    # Extract receptive field dimensions
    rD, rH, rW = receptive_field

    # Calculate the start and end indices for each dimension
    start_d = (D - rD) // 2
    end_d = start_d + rD
    start_h = (H - rH) // 2
    end_h = start_h + rH
    start_w = (W - rW) // 2
    end_w = start_w + rW

    # Perform the center cropping
    cropped_tensor = input_tensor[:, :, start_d:end_d, start_h:end_h, start_w:end_w]
    
    return cropped_tensor

def unnormalize(img):
    clip_low = 96
    clip_high = 2672
    return img *(clip_high - clip_low) + clip_low

class Trainer:

    def __init__(self, args, cfg, loader, model, loss, optimizer):

        self.args = args
        self.cfg = cfg
        self.train_gen = loader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.fp16_scaler = torch.GradScaler('cuda') if cfg.TRAINER.fp16 else None

        self.recon_img_dir = "{}/recon_img/{}".format(cfg.OUT, cfg.EXP_NAME)
        os.makedirs(self.recon_img_dir,exist_ok=True)

        # === TB writers === #
        if self.args.main:	

            self.writer = get_writer(args)
            self.lr_sched_writer = TBWriter(self.writer, 'scalar', 'Schedules/Learning Rate')			
            self.loss_writer = TBWriter(self.writer, 'scalar', 'Loss/total')

            checkdir("{}/weights/{}/".format(args.out, self.args.model), args.reset)


    def train_one_epoch(self, epoch, lr_schedule, save_recon_img_flag, MSE_loss:True):
        self.model.train()


        metric_logger = MetricLogger(delimiter="  ")
        header = 'Epoch: [{}/{}]'.format(epoch, self.cfg.TRAINER.epoch)

        for it, (input_data, labels) in enumerate(metric_logger.log_every(self.train_gen, 10, header)):

            # === Global Iteration === #
            it = len(self.train_gen) * epoch + it

            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]

            # === Inputs === #
            input_data, labels = input_data.cuda(non_blocking=True), labels.cuda(non_blocking=True)

            # === Forward pass === #
            if self.cfg.TRAINER.fp16:
                train_type=torch.float16
            else:
                train_type = torch.float32
            with torch.autocast('cuda',dtype=train_type):
                preds = self.model(input_data)
                loss = self.loss(preds, input_data)

            # Sanity Check
            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)
            
            # === Backward pass === #
            self.model.zero_grad()
            # for mix precision backward propogation
            if self.cfg.TRAINER.fp16:
                self.fp16_scaler.scale(loss).backward()
                self.fp16_scaler.step(self.optimizer)
                self.fp16_scaler.update()
            
            loss.backward()
            self.optimizer.step()


            # === Logging === #
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())

            if save_recon_img_flag:
                preds = preds.detach().cpu().numpy()
                preds = np.squeeze(preds)

                input_data = input_data.detach().cpu().numpy()
                input_data = np.squeeze(input_data)
                

                num=input_data.shape[0]
                for id in [0,1]:
                    x = input_data[id]
                    re_x = preds[id]

                    x_name = f"{epoch:04d}_{id}_x.tif"
                    re_x_name = f"{epoch:04d}_{id}_re_x.tif"
                    tif.imwrite(os.path.join(self.recon_img_dir,x_name) , x)
                    tif.imwrite(os.path.join(self.recon_img_dir,re_x_name) , re_x)

                    x_x,x_y,x_z=get_three_slice(x)
                    re_x_x, re_x_y, re_x_z = get_three_slice(re_x)
                    merged_x = np.concatenate((x_x, x_y, x_z), axis=1)  
                    merged_re_x = np.concatenate((re_x_x, re_x_y, re_x_z), axis=1)  
                    merged = np.concatenate((merged_x,merged_re_x), axis=0)
                    self.writer.add_image('x and re_x in 3 slice',merged,it,dataformats='HW')



            if self.args.main:
                self.loss_writer(metric_logger.meters['loss'].value, it)
                self.lr_sched_writer(self.optimizer.param_groups[0]["lr"], it)


        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)


    def fit(self):

        # === Resume === #
        self.load_if_available()

        # === Schedules === #
        if self.cfg.SOLVER.LR_SCHEDULER_NAME =='cosine':
            lr_schedule = cosine_scheduler(
                        base_value = self.args.lr_start * (self.cfg.DATASET.batch_per_gpu * self.args.world_size) / 256.,
                        final_value = self.args.lr_end,
                        epochs = self.cfg.TRAINER.epoch,
                        niter_per_ep = len(self.train_gen),
                        warmup_epochs= self.args.lr_warmup,
                        )           

        # === training loop === #
        for epoch in range(self.start_epoch, self.cfg.TRAINER.epoch):

            self.train_gen.sampler.set_epoch(epoch)

            save_recon_img_flag = ( epoch % 50 ==0)
            self.train_one_epoch(epoch, lr_schedule,save_recon_img_flag,MSE_loss=True)


            # === save model === #
            if self.args.main and (epoch+1)%self.cfg.TRAINER.save_every == 0:
                self.save(epoch)

    def load_if_available(self):

        ckpts = sorted(glob(f'{self.args.out}/weights/{self.args.model}/Epoch_*.pth'))

        if len(ckpts) >0:
            ckpts = sorted(
                    ckpts,
                    key=lambda x: int(re.search(r'Epoch_(\d+).pth', os.path.basename(x)).group(1))
                    )
            ckpt = torch.load(ckpts[-1], map_location='cpu')
            self.start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            if self.cfg.TRAINER.fp16: self.fp16_scaler.load_state_dict(ckpt['fp16_scaler'])
            print("Loaded ckpt: ", ckpts[-1])

        else:
            self.start_epoch = 0
            print("Starting from scratch")


    def save(self, epoch):

        if self.cfg.TRAINER.fp16:
            state = dict(epoch=epoch+1, 
                            model=self.model.state_dict(), 
                            optimizer=self.optimizer.state_dict(), 
                            fp16_scaler = self.fp16_scaler.state_dict(),
                            args = self.args
                        )
        else:
            state = dict(epoch=epoch+1, 
                            model=self.model.state_dict(), 
                            optimizer=self.optimizer.state_dict(),
                            args = self.args
                        )

        torch.save(state, "{}/weights/{}/Epoch_{}.pth".format(self.args.out, self.args.model, str(epoch+1).zfill(3) ))
