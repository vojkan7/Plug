from copy import deepcopy
from losses.poincare import poincare_loss
import math

import numpy as np
import torch
import torch.nn as nn
import random

class Optimization():
    def __init__(self, target_models, synthesis, discriminator, transformations, num_ws, config):
        self.synthesis = synthesis
        self.target = target_models
        self.discriminator = discriminator
        self.config = config
        self.transformations = transformations
        self.discriminator_weight = self.config.attack['discriminator_loss_weight']
        self.num_ws = num_ws
        self.clip = config.attack['clip']
        self.nr_of_target_models = config.attack['nr_of_target_models']
        self.ror=True
        
    def get_selcted_target_models(self,nr_of_target_models):

        new_target_models=[]
        targ=deepcopy(self.target)
        nr_of_models=nr_of_target_models
        
        while(nr_of_models)>0:
            mod=random.choice(targ)
            new_target_models.append(mod)
            nr_of_models=nr_of_models-1
            targ.remove(mod)


       
        
        return new_target_models


    def optimize(self, w_batch, targets_batch, num_epochs,nr_target_models):
        # Initialize attack
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)
        ######################################################################jedes mal nimmt
        #alte falsche target_models=new_target_models

        nr_models=nr_target_models
            
       
        

        # Start optimization
        for i in range(num_epochs):


            target_models=Optimization.get_selcted_target_models(self,nr_models)
            # synthesize imagesnd preprocess images
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)

            # compute discriminator loss
            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(
                    imgs)
            else:
                discriminator_loss = torch.tensor(0.0)

            # perform image transformations
            if self.clip:
                imgs = self.clip_images(imgs)
            if self.transformations:
                imgs = self.transformations(imgs)
                    

            # Compute target loss & combine losses and compute gradients
            loss_sum = torch.tensor(0.0)
            loss_sum=loss_sum.cuda()
            for target in target_models:
                outputs = target(imgs)
                target_loss = poincare_loss(
                    outputs, targets_batch).mean()
                loss_sum+=target_loss
            loss=loss_sum/len(target_models)
            loss.backward()
            optimizer.step()


            if scheduler:
                scheduler.step()

            # Log results
            if self.config.log_progress:
                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(
                        confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()

                if torch.cuda.current_device() == 0:
                    print(
                        f'iteration {i}: \t total_loss={loss:.4f} \t target_loss={target_loss:.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t mean_conf={mean_conf:.4f}'
                    )

        return w_batch.detach()

    def synthesize(self, w, num_ws):
        if w.shape[1] == 1:
            w_expanded = torch.repeat_interleave(w,
                                                 repeats=num_ws,
                                                 dim=1)
            imgs = self.synthesis(w_expanded,
                                  noise_mode='const',
                                  force_fp32=True)
        else:
            imgs = self.synthesis(w, noise_mode='const', force_fp32=True)
        return imgs

    def clip_images(self, imgs):
        lower_limit = torch.tensor(-1.0).float().to(imgs.device)
        upper_limit = torch.tensor(1.0).float().to(imgs.device)
        imgs = torch.where(imgs > upper_limit, upper_limit, imgs)
        imgs = torch.where(imgs < lower_limit, lower_limit, imgs)
        return imgs

    def compute_discriminator_loss(self, imgs):
        discriminator_logits = self.discriminator(imgs, None)
        discriminator_loss = nn.functional.softplus(
            -discriminator_logits).mean()
        return discriminator_loss
