import argparse
import os.path as osp

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.autograd import Variable

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler_train_100way, CategoriesSampler_val_100way
from convnet import Convnet, Registrator, Hallucinator
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
    parser.add_argument('--save-path', default='./save/proto-5shot')
    parser.add_argument('--gpu', default='1')

    args = parser.parse_args()
    logname = 'baseline'
    logfile = open(osp.join(args.save_path, logname + '.txt'), 'w+')
    pprint(vars(args))

    set_gpu(args.gpu)

    trainset = MiniImageNet('trainvaltest')
    train_sampler = CategoriesSampler_train_100way(trainset.label, 100,
                                      args.train_way, args.shot, args.query,args.n_base_class)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

    #valset = MiniImageNet('test')
    valset = MiniImageNet('trainvaltest')
    val_sampler = CategoriesSampler_val_100way(valset.label, 400,
                                    args.test_way, args.shot, args.query_val)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    model_cnn = Convnet().cuda()
    model_reg = Registrator().cuda()
    model_cnn.load_state_dict(torch.load('./iterative_G3_trainval_lr150epoch_dataaugumentation_2epoch-175_backbone.pth'))

    noise_dim = 128
    model_gen = Hallucinator(noise_dim).cuda()
    model_gen.load_state_dict(torch.load('./iterative_G3_trainval_lr150epoch_dataaugumentation_2epoch-175_gen.pth'))

    global_proto = torch.load('./global_proto_all_new.pth')
    global_base =global_proto[:args.n_base_class,:]
    global_novel = global_proto[args.n_base_class:,:]

    global_base = [Variable(global_base.cuda(),requires_grad=True)]
    global_novel = [Variable(global_novel.cuda(),requires_grad=True)]

    learning_rate=0.001
    optimizer_cnn = torch.optim.SGD(model_cnn.parameters(), lr=learning_rate,momentum=0.9)
    optimizer_atten = torch.optim.SGD(model_reg.parameters(), lr=learning_rate,momentum=0.9)
    optimizer_gen = torch.optim.SGD(model_gen.parameters(), lr=learning_rate,momentum=0.9)
    optimizer_global1 = torch.optim.SGD(global_base, lr=learning_rate,momentum=0.9)
    optimizer_global2 = torch.optim.SGD(global_novel, lr=learning_rate,momentum=0.9)

    lr_scheduler_cnn = torch.optim.lr_scheduler.MultiStepLR(optimizer_cnn, milestones=[30,60], gamma=0.1)
    lr_scheduler_atten = torch.optim.lr_scheduler.MultiStepLR(optimizer_atten, milestones=[30,60], gamma=0.1)
    lr_scheduler_gen = torch.optim.lr_scheduler.MultiStepLR(optimizer_gen, milestones=[30,60], gamma=0.1)
    lr_scheduler_global1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_global1, milestones=[30,60], gamma=0.1)
    lr_scheduler_global2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_global2, milestones=[30,60], gamma=0.1)


    def save_model(name):
        torch.save(model_cnn.state_dict(), osp.join(args.save_path, name + '_model_cnn.pth'))
        torch.save(model_reg.state_dict(), osp.join(args.save_path, name + '_model_reg.pth'))
        torch.save(model_gen.state_dict(), osp.join(args.save_path, name + '_model_gen.pth'))
        torch.save(torch.cat([global_base[0], global_novel[0]]), osp.join(args.save_path, 'global.pth'))

    def log(out_str):
        print(out_str)
        logfile.write(out_str+'\n')
        logfile.flush()

    eps = 0.1
    def label_smooth(pred, gold):
        gold = gold.contiguous().view(-1)
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1)
        return torch.mean(loss)

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler_cnn.step()
        lr_scheduler_atten.step()
        lr_scheduler_gen.step()
        lr_scheduler_global1.step()
        lr_scheduler_global2.step()

        model_cnn.train()
        model_reg.train()
        model_gen.train()

        tl1 = Averager()
        tl2 = Averager()
        ta1 = Averager()
        ta2 = Averager()

        for i, batch in enumerate(train_loader, 1):
            data,lab = [_.cuda() for _ in batch]

            p = args.shot * args.train_way
            data_shot = data[:p]
            data_query = data[p:]
            data_shot = data_shot[:,:3,:]
            data_query = data_query[:,3:,:]
            train_gt = lab[:p].reshape(args.shot, args.train_way)[0,:]


            #data_query = data_query[:,:3,:]

            proto = model_cnn(data_shot)
            proto = proto.reshape(args.shot, args.train_way, -1)


            which_novel = torch.gt(train_gt,79)
            which_novel = args.train_way-torch.numel(train_gt[which_novel])



            if which_novel < args.train_way:
                proto_base = proto[:,:which_novel,:]
                proto_novel = proto[:,which_novel:,:]
                noise = torch.cuda.FloatTensor((args.train_way-which_novel)*args.shot, noise_dim).normal_()
                proto_novel_gen = model_gen(proto_novel.reshape(args.shot*(args.train_way-which_novel),-1), noise)
                proto_novel_gen = proto_novel_gen.reshape(args.shot, args.train_way-which_novel, -1)
                proto_novel_wgen = torch.cat([proto_novel,proto_novel_gen])
                ind_gen = torch.randperm(2*args.shot)
                train_num = np.random.randint(1, args.shot)
                proto_novel_f = proto_novel_wgen[ind_gen[:train_num],:,:]
                weight_arr = np.random.rand(train_num)
                weight_arr = weight_arr/np.sum(weight_arr)
                proto_novel_f = (torch.from_numpy(weight_arr.reshape(-1,1,1)).type(torch.float).cuda()*proto_novel_f).sum(dim=0)
                proto_base = proto_base.mean(dim=0)
                proto_final = torch.cat([proto_base, proto_novel_f],0)
            else:
                proto_final = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)

            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)
            global_new, proto_new = model_reg(support_set=torch.cat([global_base[0],global_novel[0]]), query_set=proto_final)
            logits2 = euclidean_metric(proto_new, global_new)
            loss2 = F.cross_entropy(logits2, train_gt)

            similarity = F.softmax(logits2)
            feature = torch.matmul(similarity, torch.cat([global_base[0],global_novel[0]]))
            logits = euclidean_metric(model_cnn(data_query), feature)
            loss1 = F.cross_entropy(logits, label)

            acc1 = count_acc(logits, label)
            acc2 = count_acc(similarity, train_gt)

            tl1.add(loss1.item())
            tl2.add(loss2.item())
            ta1.add(acc1)
            ta2.add(acc2)

            optimizer_gen.zero_grad()
            optimizer_cnn.zero_grad()
            optimizer_atten.zero_grad()
            optimizer_global1.zero_grad()
            optimizer_global2.zero_grad()
            total_loss = loss1 + loss2
            #loss.backward()
            total_loss.backward()


            if epoch > 45:
                optimizer_gen.step()
                optimizer_cnn.step()
            optimizer_atten.step()
            optimizer_global1.step()
            optimizer_global2.step()


            proto = None; proto_final = None; logits = None; loss = None

        tl1 = tl1.item()
        tl2 = tl2.item()
        ta1 = ta1.item()
        ta2 = ta2.item()
        #log('epoch {}, train, loss={:.4f} acc={:.4f}'.format(epoch, tl, ta))
        log('epoch {}, train, loss1={:.4f} loss2={:.4f} acc1={:.4f} acc2={:.4f}'.format(epoch, tl1, tl2, ta1, ta2))

        model_cnn.eval()
        model_reg.eval()

        vl1 = Averager()
        vl2 = Averager()
        va1 = Averager()
        va2 = Averager()

        for i, batch in enumerate(val_loader, 1):
            data, lab = [_.cuda() for _ in batch]

            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]
            data_shot = data_shot[:, 3:,:]
            data_query = data_query[:, 3:,:]



            proto = model_cnn(data_shot)
            proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

            label = torch.arange(args.test_way).repeat(args.query_val)
            label = label.type(torch.cuda.LongTensor)

            # global_proto = torch.load(osp.join(args.save_path,'global_proto.pth'))
            #logits = euclidean_metric(model_cnn(data_query), model_reg(support_set=torch.cat([global_base[0],global_novel[0]]),query_set=proto))

            #feature, similarity,_ = model_reg(support_set=torch.cat([global_base[0],global_novel[0]]),query_set=proto)

            #logits = euclidean_metric(model_cnn(data_query), feature)
            #loss1 = F.cross_entropy(logits, label)

            #val_gt = lab[:p].reshape(args.shot, args.test_way)[0,:]
            #loss2 = F.cross_entropy(similarity, val_gt)

            global_new, proto_new = model_reg(support_set=torch.cat([global_base[0],global_novel[0]]), query_set=proto)
            logits2 = euclidean_metric(proto_new, global_new)

            val_gt = lab[:p].reshape(args.shot, args.test_way)[0,:]
            loss2 = F.cross_entropy(logits2, val_gt)

            similarity = F.softmax(logits2)
            feature = torch.matmul(similarity, torch.cat([global_base[0],global_novel[0]]))
            logits = euclidean_metric(model_cnn(data_query), feature)
            loss1 = F.cross_entropy(logits, label)

            acc1 = count_acc(logits, label)
            acc2 = count_acc(similarity, val_gt)

            vl1.add(loss1.item())
            vl2.add(loss2.item())
            va1.add(acc1)
            va2.add(acc2)

            proto = None; logits = None; loss = None

        vl1 = vl1.item()
        vl2 = vl2.item()
        va1 = va1.item()
        va2 = va2.item()
        # print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
        log('epoch {}, val, loss1={:.4f} loss2={:.4f} acc1={:.4f} acc2={:.4f}'.format(epoch, vl1, vl2, va1, va2))
        if va1 > trlog['max_acc']:
            trlog['max_acc'] = va1
            save_model('max-acc')

        trlog['train_loss'].append(tl1+tl2)
        trlog['train_acc'].append(ta1)
        trlog['val_loss'].append(vl1+vl2)
        trlog['val_acc'].append(va1)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last')

        if epoch % args.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        # print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
