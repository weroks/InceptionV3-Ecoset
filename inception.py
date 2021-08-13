#!/usr/bin/env python3

import os
import random
import shutil
import time
import warnings
import logging

import horovod.torch as hvd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

#import warnings
#warnings.filterwarnings('ignore')
#from torch.utils.tensorboard import SummaryWriter

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch', dest='batch_size', action='store', help='batch size', type=int, required=True)
parser.add_argument('--lr', dest='initial_lr', action='store', help='learning rate', type=float, required=True)

args = parser.parse_args()
batch_size = args.batch_size
initial_lr = args.initial_lr

data_dir = '/u/hannesh/ecoset'
working_dir = os.getcwd()

best_acc1 = 0

hvd.init()

# Logging
#writer = SummaryWriter('%s/runs/' % working_dir)
if hvd.rank() == 0:
    logging.basicConfig(filename='%s/train.log' % working_dir, level=logging.INFO)

num_classes = 565

def create_model():
    logging.info('create model')
    model = models.inception_v3(pretrained=False, num_classes=num_classes)
    num_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if hvd.rank() == 0:
        torch.save(state, filename)
        if is_best:
            shutil.copyfile('%s/%s' % (working_dir, filename),
'model_best.pth.tar')

def log_after_epoch(losses, top1, top5, mode, epoch, time_needed=0):
    if hvd.rank() == 0:
        logging.info('%s at epoch %s -> mean results: Acc-1: %s Acc-5: %s, Loss: %s -> took %s min' % (mode, str(epoch), str(top1.avg), str(top5.avg), str(losses.avg),
                        str(time_needed/60)))

def train(train_loader, model, criterion, optimizer, epoch):
    logging.info('Start training')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    model.train()

    epoch_start_time = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        batch_start_time = time.time()

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output, aux_output = model(images)

        loss1 = criterion(output, target)
        loss2 = criterion(aux_output, target)
        loss = loss1 + 0.4 * loss2

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        end = time.time()
        #logging.info('Batch %s of %s at epoch %s done in %s seconds' % (str(i), str(len(train_loader)), str(epoch), str(end - batch_start_time)))
        if i % 50 == 0:
            if hvd.rank() == 0:
                logging.info('Batch %s of %s at epoch %s:' % (str(i),
                                str(len(train_loader)), str(epoch)))
                logging.info('mean results: Acc-1: %s Acc-5: %s, Loss: %s' %
                            (str(top1.avg), str(top5.avg), str(losses.avg)))

    end_time = time.time()
    epoch_time = end_time - epoch_start_time
    log_after_epoch(losses, top1, top5, 'train', epoch, epoch_time)

def run():
    model = create_model()
    epochs = 90
    start_epoch = 0
    complete_batch_size = batch_size
    logging.info('Use total batch size: %s' % str(complete_batch_size))
    lr = initial_lr # * hvd.size() ?
    print('Initial learning rate: %s' % str(lr))
    lr_decay_rate = 0.94
    resume = '%s/checkpoint.pth.tar' % working_dir
    weight_decay = 0.9
    eps = 1.0
    momentum = 0.9
    data = data_dir
    workers = 0
    global best_acc1

    torch.cuda.set_device(hvd.local_rank())
    model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.RMSprop(model.parameters(), lr, eps=eps,
                                momentum=momentum,
                                weight_decay=weight_decay)

    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay_rate)

    # optionally resume from a checkpoint
    if resume and hvd.rank() == 0:
        if os.path.isfile(resume):
            logging.info("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            logging.info('Start at epoch: %s' % str(start_epoch))
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(resume))



    # Data loading code
    traindir = os.path.join(data, 'train')
    valdir = os.path.join(data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=complete_batch_size, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=complete_batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    start_epoch = hvd.broadcast(torch.tensor(start_epoch), root_rank=0, name='start_epoch').item()
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    for epoch in range(start_epoch, epochs):
        #adjust_learning_rate(optimizer, epoch, lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # adjust learning rate
        if epoch % 2 == 0:
            lr_scheduler.step()

        save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'inception_v3',
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
        }, is_best)


def validate(val_loader, model, criterion, epoch):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    log_after_epoch(losses, top1, top5, 'val', epoch)

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def adjust_learning_rate(optimizer, epoch, lr):
    """Decays learning rate by exponential rate of 0.94 every 2 epochs."""
    lr = lr * (0.94 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == "__main__":
    run()
