import numpy as np
import torch
import argparse
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from utils import cuda, Weight_EMA_Update
from datasets.datasets import return_data
from model import ToyNet
from pathlib import Path
from EDGE_grad import EDGE

class Solver(object):

    def __init__(self, args):
        self.args = args

        self.cuda = (args.cuda and torch.cuda.is_available())
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.eps = 1e-9
        self.K = args.K
        self.beta = args.beta
        self.num_avg = args.num_avg
        self.global_iter = 0
        self.global_epoch = 0

        # Network & Optimizer
        self.toynet = cuda(ToyNet(self.K), self.cuda)
        self.toynet.weight_init()
        self.toynet_ema = Weight_EMA_Update(cuda(ToyNet(self.K), self.cuda),\
                self.toynet.state_dict(), decay=0.999)

        self.optim = optim.Adam(self.toynet.parameters(),lr=self.lr,betas=(0.5,0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optim,gamma=0.97)

        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.env_name)
        if not self.ckpt_dir.exists() : self.ckpt_dir.mkdir(parents=True,exist_ok=True)
        self.load_ckpt = args.load_ckpt
        if self.load_ckpt != '' : self.load_checkpoint(self.load_ckpt)

        # History
        self.history = dict()
        self.history['avg_acc']=0.
        self.history['info_loss']=0.
        self.history['class_loss']=0.
        self.history['total_loss']=0.
        self.history['epoch']=0
        self.history['iter']=0

        # Tensorboard
        self.tensorboard = args.tensorboard
        if self.tensorboard :
            self.env_name = args.env_name
            self.summary_dir = Path(args.summary_dir).joinpath(args.env_name)
            if not self.summary_dir.exists() : self.summary_dir.mkdir(parents=True,exist_ok=True)
            self.tf = SummaryWriter(log_dir=self.summary_dir)
            self.tf.add_text(tag='argument',text_string=str(args),global_step=self.global_epoch)

        # Dataset
        self.data_loader = return_data(args)

    def set_mode(self,mode='train'):
        if mode == 'train' :
            self.toynet.train()
            self.toynet_ema.model.train()
        elif mode == 'eval' :
            self.toynet.eval()
            self.toynet_ema.model.eval()
        else : raise('mode error. It should be either train or eval')

    def train(self):
        self.set_mode('train')
        for e in range(self.epoch) :
            print("epoch:", e)
            self.global_epoch += 1

            for idx, (images,labels) in enumerate(self.data_loader['train']):
                print(" Batch:", idx)
                self.global_iter += 1

                x = Variable(cuda(images, self.cuda))
                y = Variable(cuda(labels, self.cuda))
                (mu, std), logit = self.toynet(x)

                # Sample x and z to estimate the mutual information, by yuzeng
                N_x = x.shape[0]
                x = x.view(N_x, -1)
                D_x = x.shape[1]
                index_x = torch.rand(self.batch_size) * float(N_x)
                index_x = index_x.int()
                #x_sample = x[index_x.data.numpy().tolist()]
                x_sample = x
                z_sample = torch.normal(torch.zeros(mu.shape), torch.ones(std.shape))
                mu.retain_grad()
                std.retain_grad()
                logit.retain_grad()
                z_sample = z_sample * std + mu
                z_sample.retain_grad()
            

                # Normalize the sample matrix
                y_norm = y.float() / y.max().float()
                logit_norm = logit.float() / logit.max().float()
                y_norm.retain_grad()
                logit_norm.retain_grad()

                #x_sample = x_sample.float() / x_sample.max().float()
                #z_sample = z_sample.float() / z_sample.max().float()

                # Redefine the losses with MI estimation by yuzeng
                #class_loss = F.cross_entropy(logit,y).div(math.log(2))
                #info_size_0 = z_sample.shape[0] 
                #info_size_1 = z_sample.shape[1] 
                #class_size_0 = x.shape[0] 
                #class_size_1 = x.shape[1] 
                #info_edge = EDGE((info_size_0, info_size_1))
                #class_edge = EDGE((class_size_0, class_size_1))
                info_loss = EDGE.apply(z_sample, x_sample)
                class_loss = EDGE.apply(logit_norm, y_norm)
                #info_loss = -0.5*(1+2*std.log()-mu.pow(2)-std.pow(2)).sum(1).mean().div(math.log(2))
                total_loss = class_loss - self.beta*info_loss
                total_loss.retain_grad()
                info_loss.retain_grad()
                class_loss.retain_grad()

                izy_bound = - class_loss
                izx_bound = - info_loss
                
                izy_bound.retain_grad()
                izx_bound.retain_grad()

                self.optim.zero_grad()
                total_loss.cpu().backward()
                self.optim.step()
                self.toynet_ema.update(self.toynet.state_dict())

                prediction = F.softmax(logit,dim=1).max(1)[1]
                accuracy = torch.eq(prediction.float(),y.float()).float().mean()

                if self.num_avg != 0 :
                    _, avg_soft_logit = self.toynet(x,self.num_avg)
                    avg_prediction = avg_soft_logit.max(1)[1]
                    avg_accuracy = torch.eq(avg_prediction.float(), y.float()).float().mean()
                else : avg_accuracy = Variable(cuda(torch.zeros(accuracy.size()), self.cuda))

                if self.global_iter % 1 == 0 :
                    print('i:{} IZY:{:.2f} IZX:{:.2f}'
                            .format(idx+1, izy_bound.data[0], izx_bound.data[0]), end=' ')
                    print('acc:{:.4f} avg_acc:{:.4f}'
                            .format(accuracy.data[0], avg_accuracy.data[0]), end=' ')
                    print('err:{:.4f} avg_err:{:.4f}'
                            .format(1-accuracy.data[0], 1-avg_accuracy.data[0]))

                if self.global_iter % 10 == 0 :
                    if self.tensorboard :
                        self.tf.add_scalars(main_tag='performance/accuracy',
                                            tag_scalar_dict={
                                                'train_one-shot':accuracy.data[0],
                                                'train_multi-shot':avg_accuracy.data[0]},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/error',
                                            tag_scalar_dict={
                                                'train_one-shot':1-accuracy.data[0],
                                                'train_multi-shot':1-avg_accuracy.data[0]},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/cost',
                                            tag_scalar_dict={
                                                'train_one-shot_class':class_loss.data[0],
                                                'train_one-shot_info':info_loss.data[0],
                                                'train_one-shot_total':total_loss.data[0]},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='mutual_information/train',
                                            tag_scalar_dict={
                                                'I(Z;Y)':izy_bound.data[0],
                                                'I(Z;X)':izx_bound.data[0]},
                                            global_step=self.global_iter)


            if (self.global_epoch % 2) == 0 : self.scheduler.step()
            self.test()

        print(" [*] Training Finished!")

    def test(self, save_ckpt=True):
        self.set_mode('eval')

        class_loss = 0
        info_loss = 0
        total_loss = 0
        izy_bound = 0
        izx_bound = 0
        correct = 0
        avg_correct = 0
        total_num = 0
        for idx, (images,labels) in enumerate(self.data_loader['test']):

            x = Variable(cuda(images, self.cuda))
            y = Variable(cuda(labels, self.cuda))
            (mu, std), logit = self.toynet_ema.model(x)

            #class_loss += F.cross_entropy(logit,y,size_average=False).div(math.log(2))
            #info_loss += -0.5*(1+2*std.log()-mu.pow(2)-std.pow(2)).sum().div(math.log(2))
            ############################################
            # Sample x and z to estimate the mutual information, by yuzeng
            N_x = x.shape[0]
            x = x.view(N_x, -1)
            D_x = x.shape[1]
            index_x = torch.rand(self.batch_size) * float(N_x)
            index_x = index_x.int()
            #x_sample = x[index_x.data.numpy().tolist()]
            x_sample = x
            z_sample = torch.normal(torch.zeros(mu.shape), torch.ones(std.shape))
            mu.retain_grad()
            std.retain_grad()
            z_sample = z_sample * std + mu
            

            # Normalize the sample matrix
            y_norm = y.float() / y.max().float()
            logit_norm = logit.float() / logit.max().float()

            # Redefine the losses with MI estimation by yuzeng
            info_loss = EDGE.apply(z_sample, x_sample)
            class_loss = EDGE.apply(logit_norm, y_norm)


            ############################################

            total_loss += class_loss - self.beta*info_loss
            total_num += y.size(0)

            izy_bound += - class_loss
            izx_bound += - info_loss

            prediction = F.softmax(logit,dim=1).max(1)[1]
            correct += torch.eq(prediction,y).float().sum()

            if self.num_avg != 0 :
                _, avg_soft_logit = self.toynet_ema.model(x,self.num_avg)
                avg_prediction = avg_soft_logit.max(1)[1]
                avg_correct += torch.eq(avg_prediction,y).float().sum()
            else :
                avg_correct = Variable(cuda(torch.zeros(correct.size()), self.cuda))

        accuracy = correct/total_num
        avg_accuracy = avg_correct/total_num

        izy_bound /= total_num
        izx_bound /= total_num
        class_loss /= total_num
        info_loss /= total_num
        total_loss /= total_num

        print('[TEST RESULT]')
        print('e:{} IZY:{:.2f} IZX:{:.2f}'
                .format(self.global_epoch, izy_bound.data[0], izx_bound.data[0]), end=' ')
        print('acc:{:.4f} avg_acc:{:.4f}'
                .format(accuracy.data[0], avg_accuracy.data[0]), end=' ')
        print('err:{:.4f} avg_erra:{:.4f}'
                .format(1-accuracy.data[0], 1-avg_accuracy.data[0]))
        print()

        if self.history['avg_acc'] < avg_accuracy.data[0] :
            self.history['avg_acc'] = avg_accuracy.data[0]
            self.history['class_loss'] = class_loss.data[0]
            self.history['info_loss'] = info_loss.data[0]
            self.history['total_loss'] = total_loss.data[0]
            self.history['epoch'] = self.global_epoch
            self.history['iter'] = self.global_iter
            if save_ckpt : self.save_checkpoint('best_acc.tar')

        if self.tensorboard :
            self.tf.add_scalars(main_tag='performance/accuracy',
                                tag_scalar_dict={
                                    'test_one-shot':accuracy.data[0],
                                    'test_multi-shot':avg_accuracy.data[0]},
                                global_step=self.global_iter)
            self.tf.add_scalars(main_tag='performance/error',
                                tag_scalar_dict={
                                    'test_one-shot':1-accuracy.data[0],
                                    'test_multi-shot':1-avg_accuracy.data[0]},
                                global_step=self.global_iter)
            self.tf.add_scalars(main_tag='performance/cost',
                                tag_scalar_dict={
                                    'test_one-shot_class':class_loss.data[0],
                                    'test_one-shot_info':info_loss.data[0],
                                    'test_one-shot_total':total_loss.data[0]},
                                global_step=self.global_iter)
            self.tf.add_scalars(main_tag='mutual_information/test',
                                tag_scalar_dict={
                                    'I(Z;Y)':izy_bound.data[0],
                                    'I(Z;X)':izx_bound.data[0]},
                                global_step=self.global_iter)

        self.set_mode('train')

    def save_checkpoint(self, filename='best_acc.tar'):
        model_states = {
                'net':self.toynet.state_dict(),
                'net_ema':self.toynet_ema.model.state_dict(),
                }
        optim_states = {
                'optim':self.optim.state_dict(),
                }
        states = {
                'iter':self.global_iter,
                'epoch':self.global_epoch,
                'history':self.history,
                'args':self.args,
                'model_states':model_states,
                'optim_states':optim_states,
                }

        file_path = self.ckpt_dir.joinpath(filename)
        torch.save(states,file_path.open('wb+'))
        print("=> saved checkpoint '{}' (iter {})".format(file_path,self.global_iter))

    def load_checkpoint(self, filename='best_acc.tar'):
        file_path = self.ckpt_dir.joinpath(filename)
        if file_path.is_file():
            print("=> loading checkpoint '{}'".format(file_path))
            checkpoint = torch.load(file_path.open('rb'))
            self.global_epoch = checkpoint['epoch']
            self.global_iter = checkpoint['iter']
            self.history = checkpoint['history']

            self.toynet.load_state_dict(checkpoint['model_states']['net'])
            self.toynet_ema.model.load_state_dict(checkpoint['model_states']['net_ema'])

            print("=> loaded checkpoint '{} (iter {})'".format(
                file_path, self.global_iter))

        else:
            print("=> no checkpoint found at '{}'".format(file_path))
