# coding: utf-8
import os

import torch
import torch.distributed as dist
from torch.nn import Module
from torch.nn.functional import normalize, linear
from torch.nn.parameter import Parameter

dist.init_process_group("gloo", init_method="tcp://127.0.0.1:1234", rank=0, world_size=1)


class PartialFC(Module):
    """
    Author: {Xiang An, Yang Xiao, XuHan Zhu} in DeepGlint,
    Partial FC: Training 10 Million Identities on a Single Machine
    See the original paper:
    https://arxiv.org/abs/2010.05222
    """

    @torch.no_grad()
    def __init__(self, rank, local_rank, world_size, batch_size, resume,
                 margin_softmax, num_classes, sample_rate=1.0, embedding_size=512, prefix="./",
                 test_weight_name_path="", test_weight_mom_name_path=""):
        super(PartialFC, self).__init__()
        #
        self.num_classes = num_classes
        self.rank = rank
        self.local_rank = local_rank
        self.device = torch.device("cuda:{}".format(self.local_rank))
        self.world_size = world_size
        self.batch_size = batch_size
        self.margin_softmax = margin_softmax
        self.sample_rate = sample_rate
        self.embedding_size = embedding_size
        self.prefix = prefix
        self.num_local = num_classes
        self.class_start = 0
        self.num_sample = int(self.sample_rate * self.num_local)

        self.weight_name = os.path.join(self.prefix, "rank_{}_softmax_weight.pt".format(self.rank))
        self.weight_mom_name = os.path.join(self.prefix, "rank_{}_softmax_weight_mom.pt".format(self.rank))

        self.weight = torch.normal(0, 0.01, (self.num_local, self.embedding_size), device=self.device)
        self.weight_mom = torch.zeros_like(self.weight)
        self.stream = torch.cuda.Stream(local_rank)

        if resume:
            self.weight = torch.load(test_weight_name_path)
            self.weight_mom = torch.load(test_weight_mom_name_path)

        self.index = None
        if int(self.sample_rate) == 1:
            self.update = lambda: 0
            self.sub_weight = Parameter(self.weight)
            self.sub_weight_mom = self.weight_mom
        else:
            self.sub_weight = Parameter(torch.empty((0, 0)).cuda(local_rank))
    
    def save_params(self, weight_name_path, weight_mom_name):
        torch.save(self.weight.data, weight_name_path)
        torch.save(self.weight_mom, weight_mom_name)

    @torch.no_grad()
    def sample(self, total_label):
        index_positive = (self.class_start <= total_label) & (total_label < self.class_start + self.num_local)
        total_label[~index_positive] = -1
        total_label[index_positive] -= self.class_start

    def forward(self, total_features, norm_weight):
        torch.cuda.current_stream().wait_stream(self.stream)
        logits = linear(total_features, norm_weight)
        return logits

    @torch.no_grad()
    def update(self):
        self.weight_mom[self.index] = self.sub_weight_mom
        self.weight[self.index] = self.sub_weight

    def prepare(self, label, optimizer):
        with torch.cuda.stream(self.stream):
            total_label = torch.zeros(
                size=[self.batch_size * self.world_size], device=self.device, dtype=torch.long)
            dist.all_gather(list(total_label.chunk(self.world_size, dim=0)), label)
            self.sample(total_label)
            optimizer.state.pop(optimizer.param_groups[-1]['params'][0], None)
            optimizer.param_groups[-1]['params'][0] = self.sub_weight
            optimizer.state[self.sub_weight]['momentum_buffer'] = self.sub_weight_mom
            norm_weight = normalize(self.sub_weight)
            return total_label, norm_weight
    
    def prepare_in_test(self, optimizer):
        with torch.cuda.stream(self.stream):
            optimizer.state.pop(optimizer.param_groups[-1]['params'][0], None)
            optimizer.param_groups[-1]['params'][0] = self.sub_weight
            optimizer.state[self.sub_weight]['momentum_buffer'] = self.sub_weight_mom
            norm_weight = normalize(self.sub_weight)
            return norm_weight

    def forward_backward(self, label, features, optimizer):
        total_label, norm_weight = self.prepare(label, optimizer)
        total_features = torch.zeros(
            size=[self.batch_size * self.world_size, self.embedding_size], device=self.device)
        dist.all_gather(list(total_features.chunk(self.world_size, dim=0)), features.data)
        total_features.requires_grad = True

        logits = self.forward(total_features, norm_weight)
        logits = self.margin_softmax(logits, total_label)
        feature_0820 = logits

        with torch.no_grad():
            max_fc = torch.max(logits, dim=1, keepdim=True)[0]  
            dist.all_reduce(max_fc, dist.ReduceOp.MAX)

            # calculate exp(logits) and all-reduce
            # for numerical stability , this is a exp normalised implementation
            logits_exp = torch.exp(logits - max_fc)
            logits_sum_exp = logits_exp.sum(dim=1, keepdims=True)
            dist.all_reduce(logits_sum_exp, dist.ReduceOp.SUM)

            # calculate prob
            logits_exp.div_(logits_sum_exp)

            # get one-hot
            grad = logits_exp
            index = torch.where(total_label != -1)[0]
            one_hot = torch.zeros(size=[index.size()[0], grad.size()[1]], device=grad.device)
            one_hot.scatter_(1, total_label[index, None], 1)

            # calculate loss
            loss = torch.zeros(grad.size()[0], 1, device=grad.device)
            loss[index] = grad[index].gather(1, total_label[index, None])
            dist.all_reduce(loss, dist.ReduceOp.SUM)
            loss_v = loss.clamp_min_(1e-30).log_().mean() * (-1)

            # calculate grad
            grad[index] -= one_hot
            grad.div_(self.batch_size * self.world_size)

        logits.backward(grad)
        if total_features.grad is not None:
            total_features.grad.detach_()
        x_grad = torch.zeros_like(features, requires_grad=True)
        # feature gradient all-reduce
        x_grad = list(total_features.grad.chunk(self.world_size, dim=0))
        x_grad = x_grad * self.world_size
        # backward backbone
        return x_grad, loss_v, feature_0820
    
    def forward_in_test(self, features):
        norm_weight = normalize(self.sub_weight)
        total_features = torch.zeros(size=[self.batch_size * self.world_size, self.embedding_size], device=self.device)
        dist.all_gather(list(total_features.chunk(self.world_size, dim=0)), features.data)
        logits = self.forward(total_features, norm_weight)
        # print("logits:", logits)
        return logits
    
