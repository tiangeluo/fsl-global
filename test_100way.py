import argparse
import os.path as osp

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.autograd import Variable

from mini_imagenet import MiniImageNet
from mini_imagenet_drop500 import MiniImageNet2
from convnet import Convnet
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric
from IPython import embed


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=5)
    parser.add_argument('--query_val', type=int, default=15)
    parser.add_argument('--n_base_class', type=int, default=80)
    parser.add_argument('--train-way', type=int, default=20)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./save/proto-5-gen-fus-5-au')
    parser.add_argument('--gpu', default='1')

    args = parser.parse_args()
    logname = 'baseline_test'
    logfile = open(osp.join(args.save_path, logname + '.txt'), 'w+')
    pprint(vars(args))

    set_gpu(args.gpu)

    valset = MiniImageNet2('trainvaltest')
    val_loader = DataLoader(dataset=valset, batch_size = 128,
                            num_workers=8, pin_memory=True)
    valset2 = MiniImageNet2('trainval')
    val_loader2 = DataLoader(dataset=valset2, batch_size = 128,
                            num_workers=8, pin_memory=True)
    valset3 = MiniImageNet2('test')
    val_loader3 = DataLoader(dataset=valset3, batch_size = 128,
                            num_workers=8, pin_memory=True)

    model_cnn = Convnet().cuda()
    model_cnn.load_state_dict(torch.load('./100way_pn_basenovel.pth'))
    global_proto = torch.load('./global_proto_basenovel_PN_5shot_500.pth')
    global_base =global_proto[:args.n_base_class,:]
    global_novel = global_proto[args.n_base_class:,:]
    global_base = [Variable(global_base.cuda(),requires_grad=True)]
    global_novel = [Variable(global_novel.cuda(),requires_grad=True)]

    def log(out_str):
        print(out_str)
        logfile.write(out_str+'\n')
        logfile.flush()

    model_cnn.eval()
    for epoch in range(1, args.max_epoch + 1):

        for i, batch in enumerate(val_loader, 1):
            data, lab = [_.cuda() for _ in batch]

            data_shot = data[:, 3:, :]
            proto = model_cnn(data_shot)
            global_set=torch.cat([global_base[0],global_novel[0]])
            logits = euclidean_metric(proto, global_set)
            loss = F.cross_entropy(logits, lab)
            acc = count_acc(logits, lab)

            vl.add(loss.item())
            va.add(acc)
            proto = None; logits = None; loss = None

        vl = vl.item()
        va = va.item()
        log('both epoch {}, val, loss={:.4f} acc={:.4f}'.format(i, vl, va))

        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader2, 1):
            data, lab = [_.cuda() for _ in batch]

            data_shot = data[:, 3:, :]
            proto = model_cnn(data_shot)
            global_set=torch.cat([global_base[0],global_novel[0]])
            logits = euclidean_metric(proto, global_set)
            loss = F.cross_entropy(logits, lab)
            acc = count_acc(logits, lab)
            #logits = euclidean_metric(proto, global_base[0])
            #loss = F.cross_entropy(logits, lab)
            #acc = count_acc(logits, lab)

            vl.add(loss.item())
            va.add(acc)
            proto = None; logits = None; loss = None

        vl = vl.item()
        va = va.item()
        log('base epoch {}, val, loss={:.4f} acc={:.4f}'.format(i, vl, va))

        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader3, 1):
            data, lab = [_.cuda() for _ in batch]

            data_shot = data[:, 3:, :]
            proto = model_cnn(data_shot)
            global_set=torch.cat([global_base[0],global_novel[0]])
            logits = euclidean_metric(proto, global_set)
            loss = F.cross_entropy(logits, lab+80)
            acc = count_acc(logits, lab+80)

            vl.add(loss.item())
            va.add(acc)
            proto = None; logits = None; loss = None

        vl = vl.item()
        va = va.item()
        log('novel {}, val, loss={:.4f} acc={:.4f}'.format(i, vl, va))
