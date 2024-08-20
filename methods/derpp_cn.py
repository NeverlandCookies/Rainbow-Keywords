import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio

from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torchaudio.transforms import MFCC
import torch.nn.functional as F

from utils.data_augmentation import mixup_data, spec_augmentation
from utils.data_loader import SpeechDataset

from utils.train_utils import select_model, select_optimizer
# from audiomentations import Compose, AddGaussianNoise, PitchShift, Shift, FrequencyMask, ClippingDistortion
from utils.data_loader import TimeMask
from methods.base import BaseMethod


from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch
import torch.nn as nn



class _CN(_BatchNorm):
    def __init__(self, target, eps = 1e-5, momentum = 0.1, affine=True):
        num_features = target.num_features
        super(_CN, self).__init__(num_features, eps, momentum, affine=True)
        self.running_mean = target.running_mean
        self.running_var = target.running_var
        
        self.weight = target.weight
        self.bias = target.bias

        self.N = num_features
        self.setG()

    def setG(self):
        pass

    def forward(self, input):
        out_gn = F.group_norm(input, self.G, None, None, self.eps)
        out = F.batch_norm(out_gn, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)
        return out
    
class CN4(_CN):
    def setG(self):
        self.G = 4

class CN8(_CN):
    def setG(self):
        self.G = 8
class CN(_CN):
    def setG(self):
        self.G = 4
def replace_bn(module, name, nl):
	for attr_str in dir(module):
		target_attr = getattr(module, attr_str)
		if type(target_attr) == torch.nn.BatchNorm2d:
			new_bn = nl(target_attr)
			setattr(module, attr_str, new_bn)
	for name, icm in module.named_children():
		if type(icm) == torch.nn.BatchNorm2d:
			new_bn = nl(icm)
			setattr(module, name, new_bn)
		replace_bn(icm, name, nl)    

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1

class Derpp(BaseMethod):
    def __init__(self, criterion, device, n_classes, **kwargs):
        super().__init__(criterion, device, n_classes, **kwargs)

        # Parameters for Model/Backbone——直接全分类头——DER++特色
        self.model = select_model(self.model_name, 30)

        # record current logit for buffer updata
        self.cur_logits = None

        # Parameters for Memory Updating
        # ---------------------------------------------------------------------------------------------------------------
        # 修改memory_list的用途，直接用来存sample
        self.memory_list = [None] * self.memory_size
        # ---------------------------------------------------------------------------------------------------------------
        self.mem_manage = kwargs["mem_manage"]
        if kwargs["mem_manage"] == "default":
            self.mem_manage = "reservoir"
        #---------------------------------------NEW---------------------------------------------------------------------
        # Buffer for logits
        self.mem_logits_list = [None] * self.memory_size
        # Buffer for labels
        self.mem_labels_list = [None] * self.memory_size

        self.minibatch_size = kwargs["minibatch_size"]

        self.attributes = ['examples', 'labels', 'logits']

        # DER++ hyperparameter
        self.alpha = 0.9
        self.beta = 0.3
        #---------------------------------------------------------------------------------------------------------------
        new_nl = CN
        replace_bn(self.model, 'model', new_nl)
    def update_memory(self, cur_iter, num_class=None):
        pass

    def before_task(self, datalist, init_model=False, init_opt=True):
        # logger.info("Apply before_task")
        # Confirm incoming classes
        incoming_classes = pd.DataFrame(datalist)["klass"].unique().tolist()
        self.exposed_classes = list(set(self.learned_classes + incoming_classes))
        self.num_learning_class = max(
            len(self.exposed_classes), self.num_learning_class
        )
        # in_features = self.model.tc_resnet.channels[-1]
        # out_features = self.model.tc_resnet.out_features
        # new_out_features = max(out_features, self.num_learning_class)
        # if init_model:
        #     # init model parameters in every iteration
        #     logger.info("Reset model parameters")
        #     self.model = select_model(self.model_name, new_out_features)
        # else:
        #     self.model.tc_resnet.linear = nn.Linear(in_features, new_out_features)
        # DER++一开始就是全分类头，因此不需要上述单独更新线性层的操作
        self.model = self.model.to(self.device)
        if init_opt:
            # reinitialize the optimizer and scheduler
            # logger.info("Reset the optimizer and scheduler states")
            self.optimizer, self.scheduler = select_optimizer(
                self.opt_name, self.lr, self.model, self.sched_name
            )

        # logger.info(f"Increasing the head of fc {out_features} -> {new_out_features}")
        # self.already_mem_update = False


    def after_task(self, cur_iter):
        # logger.info("Apply after_task")
        self.learned_classes = self.exposed_classes
        self.num_learned_class = self.num_learning_class
        # self.update_memory(cur_iter)


    def add_data(self, examples, labels=None, logits=None):
        for i in range(examples.shape[0]):
            index = reservoir(self.seen, self.memory_size)
            self.seen += 1
            if index >= 0:
                self.memory_list[index] = examples[i].to(self.device)
                if labels is not None:
                    self.mem_labels_list[index] = labels[i].to(self.device)
                if logits is not None:
                    self.mem_logits_list[index] = logits[i].to(self.device)


    def get_data_logit(self, size):
        if size > min(self.seen, len(self.memory_list)):
            size = min(self.seen, len(self.memory_list))
        choice = np.random.choice(min(self.seen, len(self.memory_list)), size, replace=False)
        # print(choice)
        if self.spec:
            # print('Logit specaugmentation enabled')
            ret_tuple = (torch.stack([self.memory_list[idx] for idx in choice]).to(self.device),)
            # ret_tuple = ([spec_augmentation(example) for example in self.memory_list[choice]].to(self.device),)
        else:
            # ret_tuple = ([example for example in self.memory_list[choice]].to(self.device),)
            ret_tuple = (torch.stack([self.memory_list[idx] for idx in choice]).to(self.device),)

        ret_tuple += (torch.tensor([self.mem_labels_list[idx] for idx in choice]).to(self.device),)
        ret_tuple += (torch.stack([self.mem_logits_list[idx] for idx in choice]).to(self.device),)
        # ret_tuple += (self.mem_labels_list[choice],)
        # ret_tuple += (self.mem_logits_list[choice],)

        return ret_tuple
    
    def get_data_label(self, size):
        if size > min(self.seen, len(self.memory_list)):
            size = min(self.seen, len(self.memory_list))
        choice = np.random.choice(min(self.seen, len(self.memory_list)), size, replace=False)
        # print(choice)
        if self.spec:
            print('Rehearsal specaugmentation enabled')
            ret_tuple = (torch.stack([spec_augmentation(self.memory_list[idx]) for idx in choice]).to(self.device),)
            # ret_tuple = ([spec_augmentation(example) for example in self.memory_list[choice]].to(self.device),)
        else:
            # ret_tuple = ([example for example in self.memory_list[choice]].to(self.device),)
            ret_tuple = (torch.stack([self.memory_list[idx] for idx in choice]).to(self.device),)

        ret_tuple += (torch.tensor([self.mem_labels_list[idx] for idx in choice]).to(self.device),)
        ret_tuple += (torch.stack([self.mem_logits_list[idx] for idx in choice]).to(self.device),)
        # ret_tuple += (self.mem_labels_list[choice],)
        # ret_tuple += (self.mem_logits_list[choice],)

        return ret_tuple


    def train(self, cur_iter, n_epoch, batch_size, n_worker, n_passes=1):
        
        train_list = self.streamed_list
        random.shuffle(train_list)
        test_list = self.test_list
        train_loader, test_loader = self.get_dataloader(
            batch_size, n_worker, train_list, test_list
        )

        # logger.info(f"Streamed samples: {len(self.streamed_list)}")
        # logger.info(f"In-memory samples: {len(self.memory_list)}")
        # logger.info(f"Train samples: {len(train_list)}")
        # logger.info(f"Test samples: {len(test_list)}")

        # TRAIN
        best_acc = 0.0
        eval_dict = dict()
        for epoch in range(n_epoch):
            print(f"Epoch {epoch} start")
            # https://github.com/drimpossible/GDumb/blob/master/src/main.py
            # initialize for each task
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * 0.1
            elif epoch == 1:  # Then set to max lr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
            else:  # And go!
                self.scheduler.step()

            train_loss, train_acc = self._train(
                train_loader=train_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                epoch=epoch,
                total_epochs=n_epoch,
                n_passes=n_passes,
            )

            eval_dict = self.evaluation(
                test_loader=test_loader, criterion=self.criterion
            )

            # writer.add_scalar(f"task{cur_iter}/train/loss", train_loss, epoch)
            # writer.add_scalar(f"task{cur_iter}/train/acc", train_acc, epoch)
            # writer.add_scalar(f"task{cur_iter}/test/loss", eval_dict["avg_loss"], epoch)
            # writer.add_scalar(f"task{cur_iter}/test/acc", eval_dict["avg_acc"], epoch)
            # writer.add_scalar(
            #     f"task{cur_iter}/train/lr", self.optimizer.param_groups[0]["lr"], epoch
            # )

            print(
                f"Task {cur_iter} | Epoch {epoch + 1}/{n_epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )

            best_acc = max(best_acc, eval_dict["avg_acc"])

        return best_acc, eval_dict


    def _train(
            self, train_loader, optimizer, criterion, epoch, total_epochs, n_passes=1
    ):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        self.model.train()
        for i, data in enumerate(train_loader):
            for pass_ in range(n_passes):
                # print(f"batch {i} start")
                x = data["waveform"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()

                if self.mix:
                    xi, labels_a, labels_b, lam = mixup_data(x=x, y=y, alpha=0.5)
                    logit = self.model(xi)
                    #------------------------------------collecting buffer data-----------------------------------------
                    self.add_data(data["non_aug_waveform"], data["label"], logit.data)
                    #---------------------------------------------------------------------------------------------------
                    loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(
                        logit, labels_b
                    )
                    #-----------------------------------DER++ Part------------------------------------------------------
                    if len(self.memory_list) > 0:
                        buf_samples, buf_labels, buf_logits = self.get_data_logit(self.minibatch_size)
                        buf_x, labels_a, labels_b, lam = mixup_data(x=buf_samples, y=buf_labels, alpha=0.5)
                        buf_outputs = self.model(buf_x)
                        #-----------------------------------------------------------------------------------------------
                        # buf_outputs_list = [None] * self.minibatch_size
                        # loss_logit = 0
                        # for i in range(self.minibatch_size):
                        #     buf_outputs_list[i] = buf_outputs[i][:buf_logits[i].shape[-1]]
                        #     loss_logit += F.mse_loss(buf_outputs_list[i], buf_logits[i])
                        # loss += self.alpha * (loss_logit / self.minibatch_size)
                        # MSE loss
                        # loss += self.alpha * F.mse_loss(buf_outputs, buf_logits)
                         # KLD loss
                        # loss += self.alpha * F.kl_div(F.log_softmax(buf_outputs, dim=-1), F.softmax(buf_logits, dim=-1))
                         # cosine similarity loss
                        # loss += self.alpha * F.cosine_similarity(F.normalize(buf_outputs), F.normalize(buf_logits))
                        #-----------------------------------------------------------------------------------------------

                        # ！！！？？？存在比较多问题！！！？？？
                        buf_sample, buf_labels, buf_logits = self.get_data_label(self.minibatch_size)
                        buf_xi, labels_a, labels_b, lam = mixup_data(x=buf_sample, y=buf_labels, alpha=0.5)
                        buf_outputs = self.model(buf_xi)
                        # ----------------------------logit维度要对应（其他还没改）------------------------------------------
                        # buf_outputs_list = [None] * self.minibatch_size
                        # loss_label = 0
                        # for i in range(self.minibatch_size):
                        #     buf_outputs_list[i] = buf_outputs[i][:buf_logits[i].shape[-1]]
                        #     loss_label += (lam * criterion(buf_outputs_list[i], labels_a) + (1 - lam) * criterion(buf_outputs_list[i], labels_b))
                        # loss += self.beta * (loss_label / self.minibatch_size)
                        # loss += self.beta * criterion(buf_outputs, buf_labels)
                        # ----------------------------------------------------------------------------------------------
                        loss += self.beta * (lam * criterion(buf_outputs, labels_a) + (1 - lam) * criterion(buf_outputs, labels_b))




                else:
                    # print("No mixup")
                    logit = self.model(x)
                    #---------------------------------------------------------------------------------------------------
                    self.add_data(data["non_aug_waveform"], data["label"], logit.data)
                    #---------------------------------------------------------------------------------------------------
                    loss = criterion(logit, y)
                    #----------------------------------DER++------------------------------------------------------------
                    if len(self.memory_list) > 0:
                        buf_samples, buf_labels, buf_logits = self.get_data_logit(self.minibatch_size)
                        buf_outputs = self.model(buf_samples)
                        #----------------------------logit维度要对应（其他还没改）------------------------------------------
                        # for i in range(self.minibatch_size):
                        #     buf_outputs_list[i] = buf_outputs[i][:buf_logits[i].shape[-1]]
                        #     loss_logit += F.mse_loss(buf_outputs_list[i], buf_logits[i])
                        # loss += self.alpha * (loss_logit / self.minibatch_size)
                        # MSE loss
                        loss += self.alpha * F.mse_loss(buf_outputs, buf_logits)
                        #-----------------------------------------------------------------------------------------------

                        buf_samples, buf_labels, buf_logits = self.get_data_logit(self.minibatch_size)
                        buf_outputs = self.model(buf_samples)
                        # ----------------------------logit维度要对应（其他还没改）------------------------------------------
                        # buf_outputs_list = [None] * self.minibatch_size
                        # loss_label = 0
                        # for i in range(self.minibatch_size):
                        #     buf_outputs_list[i] = buf_outputs[i][:buf_logits[i].shape[-1]]
                        #     loss_label += criterion(buf_outputs_list[i], buf_labels[i])
                        # loss += self.beta * (loss_label / self.minibatch_size)
                        loss += self.beta * criterion(buf_outputs, buf_labels)
                        # -----------------------------------------------------------------------------------------------


                _, preds = logit.topk(self.topk, 1, True, True)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += torch.sum(preds == y.unsqueeze(1)).item()
                num_data += y.size(0)

        n_batches = len(train_loader)

        return total_loss / n_batches, correct / num_data


    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["waveform"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit = self.model(x)

                loss = criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)

                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

        return ret