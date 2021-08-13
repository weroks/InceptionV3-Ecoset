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

args = parser.parse_args()

batch_size = args.batch_size
working_dir = os.getcwd()

best_acc1 = 0

# Logging
logging.basicConfig(filename='%s/train_nohvd.log' % working_dir, level=logging.INFO)

num_classes = 565

def create_model():
    logging.info('create model')
    model = models.inception_v3(pretrained=False)
    num_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def save_checkpoint(state, is_best, filename='checkpoint_nohvd.pth.tar'):
    #if hvd.rank() == 0:
    torch.save(state, filename)
    if is_best:
        shutil.copyfile('%s/%s' % (working_dir, filename),'model_best_nohvd.pth.tar')

def log_after_epoch(losses, top1, top5, mode, epoch, time_needed=0):
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
        logging.info('Batch %s of %s at epoch %s done in %s seconds' % (str(i), str(len(train_loader)), str(epoch), str(end - batch_start_time)))

    end_time = time.time()
    epoch_time = end_time - epoch_start_time
    log_after_epoch(losses, top1, top5, 'train', epoch, epoch_time)


def run():
    model = create_model()
    epochs = 90
    start_epoch = 0
    complete_batch_size = batch_size
    logging.info('Use total batch size: %s' % str(complete_batch_size))
    lr = 0.018
    lr_decay_rate = 0.94
    resume = '%s/checkpoint_nohvd.pth.tar' % working_dir
    weight_decay = 0.9
    eps = 1.0
    momentum = 0.9
    data = '/u/hannesh/ecoset'
    workers = 4
    global best_acc1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info('check GPU count')
    amount_gpus = torch.cuda.device_count()
    logging.info('%s GPUs avaiable' % str(amount_gpus))
    #if amount_gpus > 1:
        #logging.info("Use all GPUs")
        #model = nn.DataParallel(model)

    model.to(device)


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.RMSprop(model.parameters(), lr, eps=eps,
                                momentum=momentum,
                                weight_decay=weight_decay)

    #optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay_rate)

    # optionally resume from a checkpoint
    if resume:
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

#    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=complete_batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=complete_batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

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
                'arch': 'vgg16_bn',
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
        }, is_best)
        logging.info('LR at epoch {}: {}'.format(epoch + 1, lr_scheduler.get_lr()))


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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
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
