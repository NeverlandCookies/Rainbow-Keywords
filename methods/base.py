"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/2/7 下午5:07
@Author  : Yang "Jan" Xiao 
@Description : base
"""
# When we make a new one, we should inherit the BaseMethod class.
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio

from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from torchaudio.transforms import MFCC
import torch.nn.functional as F

from utils.data_augmentation import mixup_data
from utils.data_loader import SpeechDataset

from utils.train_utils import select_model, select_optimizer
from audiomentations import Compose, AddGaussianNoise, PitchShift, Shift, ClippingDistortion

from audiomentations import SpecFrequencyMask as FrequencyMask
from utils.data_loader import TimeMask

logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")


class ICaRLNet(nn.Module):
    def __init__(self, model, feature_size, n_class):
        super().__init__()
        self.model = model
        self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)
        self.ReLU = nn.ReLU()
        self.fc = nn.Linear(feature_size, n_class, bias=False)

    def forward(self, x):
        x = self.model(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.fc(x)
        return x


class BaseMethod:
    def __init__(
            self, criterion, device, n_classes, **kwargs
    ):
        # Parameters for Dataloader
        self.num_learned_class = 0
        self.num_learning_class = kwargs["n_init_cls"]
        self.n_classes = n_classes
        self.learned_classes = []
        self.class_mean = [None] * n_classes
        self.exposed_classes = []
        self.seen = 0
        self.topk = kwargs["topk"]
        self.dataset = kwargs["dataset"]

        # Parameters for Trainer
        self.device = device
        self.criterion = criterion
        self.opt_name = kwargs["opt_name"]
        self.sched_name = kwargs["sched_name"]
        self.lr = kwargs["lr"]
        self.optimizer, self.scheduler = None, None
        self.criterion = self.criterion.to(self.device)

        # Parameters for Model
        self.model_name = kwargs["model_name"]
        self.model = select_model(self.model_name, kwargs["n_init_cls"])
        self.model = self.model.to(self.device)

        # Parameters for Prototype Sampler
        self.feature_size = kwargs["feature_size"]
        self.feature_extractor = None
        self.mfcc = MFCC(sample_rate=16000, n_mfcc=40, log_mels=True)
        self.sample_length = 16000

        # Parameters for Augmentation
        self.transform = str(kwargs["transforms"])
        self.mix = "mixup" in self.transform
        logger.info(f"Take the transforms from {self.transform}")
        logger.info(f"mix:{self.mix}")
        self.spec = "specaug" in self.transform
        logger.info(f"specaug:{self.spec}")

        # Parameters for Memory Updating
        self.prev_streamed_list = []
        self.streamed_list = []
        self.test_list = []
        self.memory_list = []
        self.memory_size = kwargs["memory_size"]
        self.mem_manage = kwargs["mem_manage"]
        if kwargs["mem_manage"] == "default":
            self.mem_manage = "random"
        self.already_mem_update = False
        self.mode = kwargs["mode"]
        self.uncert_metric = kwargs["uncert_metric"]

    def set_current_dataset(self, train_datalist, test_datalist):
        random.shuffle(train_datalist)
        self.prev_streamed_list = self.streamed_list
        self.streamed_list = train_datalist
        self.test_list = test_datalist

    def before_task(self, datalist, init_model=False, init_opt=True):
        logger.info("Apply before_task")
        # Confirm incoming classes
        incoming_classes = pd.DataFrame(datalist)["klass"].unique().tolist()
        self.exposed_classes = list(set(self.learned_classes + incoming_classes))
        self.num_learning_class = max(
            len(self.exposed_classes), self.num_learning_class
        )
        if self.mem_manage == "prototype":
            self.model.tc_resnet.linear = nn.Linear(self.model.tc_resnet.channels[-1], self.feature_size)
            self.feature_extractor = self.model
            self.model = ICaRLNet(
                self.feature_extractor, self.feature_size, self.num_learning_class
            )
        in_features = self.model.tc_resnet.channels[-1]
        out_features = self.model.tc_resnet.out_features
        # To care the case of decreasing head
        new_out_features = max(out_features, self.num_learning_class)
        if init_model:
            # init model parameters in every iteration
            logger.info("Reset model parameters")
            self.model = select_model(self.model_name, new_out_features)
        else:
            self.model.tc_resnet.linear = nn.Linear(in_features, new_out_features)
        self.model = self.model.to(self.device)
        if init_opt:
            # reinitialize the optimizer and scheduler
            logger.info("Reset the optimizer and scheduler states")
            self.optimizer, self.scheduler = select_optimizer(
                self.opt_name, self.lr, self.model, self.sched_name
            )

        logger.info(f"Increasing the head of fc {out_features} -> {new_out_features}")

        self.already_mem_update = False

    def after_task(self, cur_iter):
        logger.info("Apply after_task")
        self.learned_classes = self.exposed_classes
        self.num_learned_class = self.num_learning_class
        self.update_memory(cur_iter)

    def update_memory(self, cur_iter, num_class=None):
        if num_class is None:
            num_class = self.num_learning_class

        if not self.already_mem_update:
            logger.info(f"Update memory over {num_class} classes by {self.mem_manage}")
            candidates = self.streamed_list + self.memory_list
            if len(candidates) <= self.memory_size:
                self.memory_list = candidates
                self.seen = len(candidates)
                logger.warning("Candidates < Memory size")
            else:
                if self.mem_manage == "random":
                    self.memory_list = self.rnd_sampling(candidates)
                elif self.mem_manage == "reservoir":
                    self.reservoir_sampling(self.streamed_list)
                elif self.mem_manage == "prototype":
                    self.memory_list = self.mean_feature_sampling(
                        exemplars=self.memory_list,
                        samples=self.streamed_list,
                        num_class=num_class,
                    )
                elif self.mem_manage == "uncertainty":
                    if cur_iter == 0:
                        self.memory_list = self.equal_class_sampling(
                            candidates, num_class
                        )
                    else:
                        self.memory_list = self.uncertainty_sampling(
                            candidates,
                            num_class=num_class,
                        )
                else:
                    logger.error("Not implemented memory management")
                    raise NotImplementedError

            assert len(self.memory_list) <= self.memory_size
            logger.info("Memory statistic")
            memory_df = pd.DataFrame(self.memory_list)
            if len(self.memory_list) > 0:
                logger.info(f"\n{memory_df.klass.value_counts(sort=True)}")
            # memory update happens only once per task iteratin.
            self.already_mem_update = True
        else:
            logger.warning(f"Already updated the memory during this iter ({cur_iter})")

    def get_dataloader(self, batch_size, n_worker, train_list, test_list):
        # Loader
        train_loader = None
        test_loader = None
        if train_list is not None and len(train_list) > 0:
            train_dataset = SpeechDataset(
                pd.DataFrame(train_list),
                dataset=self.dataset,
                is_training=True,
                transform=self.transform
            )
            # drop last becasue of BatchNorm1D in IcarlNet
            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=n_worker,
                drop_last=True,
            )

        if test_list is not None:
            test_dataset = SpeechDataset(
                pd.DataFrame(test_list),
                dataset=self.dataset,
                is_training=False
            )
            test_loader = DataLoader(
                test_dataset, shuffle=False, batch_size=batch_size, num_workers=n_worker
            )

        return train_loader, test_loader

    def train(self, cur_iter, n_epoch, batch_size, n_worker, n_passes=1):

        train_list = self.streamed_list + self.memory_list
        random.shuffle(train_list)
        test_list = self.test_list
        train_loader, test_loader = self.get_dataloader(
            batch_size, n_worker, train_list, test_list
        )

        logger.info(f"Streamed samples: {len(self.streamed_list)}")
        logger.info(f"In-memory samples: {len(self.memory_list)}")
        logger.info(f"Train samples: {len(train_list)}")
        logger.info(f"Test samples: {len(test_list)}")

        # TRAIN
        best_acc = 0.0
        eval_dict = dict()
        for epoch in range(n_epoch):
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

            #writer.add_scalar(f"task{cur_iter}/train/loss", train_loss, epoch)
            #writer.add_scalar(f"task{cur_iter}/train/acc", train_acc, epoch)
            #writer.add_scalar(f"task{cur_iter}/test/loss", eval_dict["avg_loss"], epoch)
            #writer.add_scalar(f"task{cur_iter}/test/acc", eval_dict["avg_acc"], epoch)
            #writer.add_scalar(
                #f"task{cur_iter}/train/lr", self.optimizer.param_groups[0]["lr"], epoch
            #)

            logger.info(
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
                x = data["waveform"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()

                if self.mix:
                    x, labels_a, labels_b, lam = mixup_data(x=x, y=y, alpha=0.5)
                    logit = self.model(x)
                    loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(
                        logit, labels_b
                    )
                else:
                    logit = self.model(x)
                    loss = criterion(logit, y)
                _, preds = logit.topk(self.topk, 1, True, True)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += torch.sum(preds == y.unsqueeze(1)).item()
                num_data += y.size(0)

        n_batches = len(train_loader)

        return total_loss / n_batches, correct / num_data

    def evaluation_ext(self, test_list):
        # evaluation from out of class
        test_dataset = SpeechDataset(
            pd.DataFrame(test_list),
            dataset=self.dataset,
            is_training=False
        )
        test_loader = DataLoader(
            test_dataset, shuffle=False, batch_size=32, num_workers=2
        )
        eval_dict = self.evaluation(test_loader, self.criterion)

        return eval_dict

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

    def _interpret_pred(self, y, pred):
        # xlable is batch
        ret_num_data = torch.zeros(self.n_classes)
        ret_corrects = torch.zeros(self.n_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        correct_xlabel = y.masked_select(y == pred)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects

    def rnd_sampling(self, samples):
        random.shuffle(samples)
        return samples[: self.memory_size]

    def reservoir_sampling(self, samples):
        for sample in samples:
            if len(self.memory_list) < self.memory_size:
                self.memory_list += [sample]
            else:
                j = np.random.randint(0, self.seen)
                if j < self.memory_size:
                    self.memory_list[j] = sample
            self.seen += 1

    def mean_feature_sampling(self, exemplars, samples, num_class):
        """Prototype sampling

        Args:
            features ([Tensor]): [features corresponding to the samples]
            samples ([Datalist]): [datalist for a class]

        Returns:
            [type]: [Sampled datalist]
        """

        def _reduce_exemplar_sets(exemplars, mem_per_cls):
            if len(exemplars) == 0:
                return exemplars

            exemplar_df = pd.DataFrame(exemplars)
            ret = []
            for y in range(self.num_learned_class):
                cls_df = exemplar_df[exemplar_df["label"] == y]
                ret += cls_df.sample(n=min(mem_per_cls, len(cls_df))).to_dict(
                    orient="records"
                )

            num_dups = pd.DataFrame(ret).duplicated().sum()
            if num_dups > 0:
                logger.warning(f"Duplicated samples in memory: {num_dups}")

            return ret

        mem_per_cls = self.memory_size // num_class
        exemplars = _reduce_exemplar_sets(exemplars, mem_per_cls)
        old_exemplar_df = pd.DataFrame(exemplars)

        new_exemplar_set = []
        sample_df = pd.DataFrame(samples)
        for y in range(self.num_learning_class):
            cls_samples = []
            cls_exemplars = []
            if len(sample_df) != 0:
                cls_samples = sample_df[sample_df["label"] == y].to_dict(
                    orient="records"
                )
            if len(old_exemplar_df) != 0:
                cls_exemplars = old_exemplar_df[old_exemplar_df["label"] == y].to_dict(
                    orient="records"
                )

            if len(cls_exemplars) >= mem_per_cls:
                new_exemplar_set += cls_exemplars
                continue

            # Assign old exemplars to the samples
            cls_samples += cls_exemplars
            if len(cls_samples) <= mem_per_cls:
                new_exemplar_set += cls_samples
                continue

            features = []
            self.feature_extractor.eval()
            with torch.no_grad():
                for data in cls_samples:
                    audio_path = os.path.join("/home/xiaoyang/Dev/kws-efficient-cl/dataset/data", data["file_name"])
                    waveform, sample_rate = torchaudio.load(audio_path)
                    if waveform.shape[1] < self.sample_length:
                        # padding if the audio length is smaller than samping length.
                        waveform = F.pad(waveform, [0, self.sample_length - waveform.shape[1]])
                    waveform = self.mfcc(waveform)
                    waveform = waveform.to(self.device)
                    feature = (
                        self.feature_extractor(waveform.unsqueeze(0)).detach().cpu().numpy()
                    )
                    feature = feature / np.linalg.norm(feature, axis=1)  # Normalize
                    features.append(feature.squeeze())

            features = np.array(features)
            logger.debug(f"[Prototype] features: {features.shape}")

            # do not replace the existing class mean
            if self.class_mean[y] is None:
                cls_mean = np.mean(features, axis=0)
                cls_mean /= np.linalg.norm(cls_mean)
                self.class_mean[y] = cls_mean
            else:
                cls_mean = self.class_mean[y]
            assert cls_mean.ndim == 1

            phi = features
            mu = cls_mean
            # select exemplars from the scratch
            exemplar_features = []
            num_exemplars = min(mem_per_cls, len(cls_samples))
            for j in range(num_exemplars):
                S = np.sum(exemplar_features, axis=0)
                mu_p = 1.0 / (j + 1) * (phi + S)
                mu_p = mu_p / np.linalg.norm(mu_p, axis=1, keepdims=True)

                dist = np.sqrt(np.sum((mu - mu_p) ** 2, axis=1))
                i = np.argmin(dist)

                new_exemplar_set.append(cls_samples[i])
                exemplar_features.append(phi[i])

                # Avoid to sample the duplicated one.
                del cls_samples[i]
                phi = np.delete(phi, i, 0)

        return new_exemplar_set

    def uncertainty_sampling(self, samples, num_class):
        """uncertainty based sampling

        Args:
            samples ([list]): [training_list + memory_list]
        """
        self.montecarlo(samples, uncert_metric=self.uncert_metric)

        sample_df = pd.DataFrame(samples)
        mem_per_cls = self.memory_size // num_class  # kc: the number of the samples of each class

        ret = []
        """
        Sampling class by class
        """
        for i in range(num_class):
            cls_df = sample_df[sample_df["label"] == i]
            if len(cls_df) <= mem_per_cls:
                ret += cls_df.to_dict(orient="records")
            else:
                jump_idx = len(cls_df) // mem_per_cls
                uncertain_samples = cls_df.sort_values(by="uncertainty")[::jump_idx]
                ret += uncertain_samples[:mem_per_cls].to_dict(orient="records")

        num_rest_slots = self.memory_size - len(ret)
        if num_rest_slots > 0:
            logger.warning("Fill the unused slots by breaking the equilibrium.")
            ret += (
                sample_df[~sample_df.file_name.isin(pd.DataFrame(ret).file_name)]
                    .sample(n=num_rest_slots)
                    .to_dict(orient="records")
            )

        num_dups = pd.DataFrame(ret).file_name.duplicated().sum()
        if num_dups > 0:
            logger.warning(f"Duplicated samples in memory: {num_dups}")

        return ret

    def _compute_uncert(self, infer_list, infer_transform, uncert_name):
        batch_size = 128
        infer_df = pd.DataFrame(infer_list)
        infer_dataset = SpeechDataset(
            infer_df, dataset=self.dataset, transform=infer_transform, is_training=False
        )
        infer_loader = DataLoader(
            infer_dataset, shuffle=False, batch_size=batch_size, num_workers=8
        )

        self.model.eval()
        with torch.no_grad():
            for n_batch, data in enumerate(infer_loader):
                x = data["waveform"]
                x = x.to(self.device)
                logit = self.model(x)
                logit = logit.detach().cpu()

                for i, cert_value in enumerate(logit):
                    sample = infer_list[batch_size * n_batch + i]
                    sample[uncert_name] = 1 - cert_value

    def montecarlo(self, candidates, uncert_metric="vr"):
        transform_cands = []
        logger.info(f"Compute uncertainty by {uncert_metric}!")
        if uncert_metric == "vr":
            transform_cands = [
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1),
                PitchShift(min_semitones=-4, max_semitones=4, p=1),
                Shift(min_fraction=-0.5, max_fraction=0.5, p=1),
                TimeMask(min_band_part=0, max_band_part=0.1),
                FrequencyMask(min_frequency_band=0, max_frequency_band=0.1, p=1),
                ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=10, p=1)
            ]
        elif uncert_metric == "vr_timemask":
            transform_cands = [TimeMask(min_band_part=0, max_band_part=0.1)] * 12

        n_transforms = len(transform_cands)

        for idx, tr in enumerate(transform_cands):
            _tr = Compose([tr])
            self._compute_uncert(candidates, _tr, uncert_name=f"uncert_{str(idx)}")

        for sample in candidates:
            self.variance_ratio(sample, n_transforms)

    def variance_ratio(self, sample, cand_length):
        vote_counter = torch.zeros(sample["uncert_0"].size(0))
        for i in range(cand_length):
            top_class = int(torch.argmin(sample[f"uncert_{i}"]))  # uncert argmin.
            vote_counter[top_class] += 1
        assert vote_counter.sum() == cand_length
        sample["uncertainty"] = (1 - vote_counter.max() / cand_length).item()

    def equal_class_sampling(self, samples, num_class):
        mem_per_cls = self.memory_size // num_class
        sample_df = pd.DataFrame(samples)
        # Warning: assuming the classes were ordered following task number.
        ret = []
        for y in range(self.num_learning_class):
            cls_df = sample_df[sample_df["label"] == y]
            ret += cls_df.sample(n=min(mem_per_cls, len(cls_df))).to_dict(
                orient="records"
            )

        num_rest_slots = self.memory_size - len(ret)
        if num_rest_slots > 0:
            logger.warning("Fill the unused slots by breaking the equilibrium.")
            ret += (
                sample_df[~sample_df.file_name.isin(pd.DataFrame(ret).file_name)]
                    .sample(n=num_rest_slots)
                    .to_dict(orient="records")
            )

        num_dups = pd.DataFrame(ret).file_name.duplicated().sum()
        if num_dups > 0:
            logger.warning(f"Duplicated samples in memory: {num_dups}")

        return ret
