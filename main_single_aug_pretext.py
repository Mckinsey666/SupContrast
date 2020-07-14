from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, MultiTwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, LinearClassifier
from losses import SupConLoss

from aug import GaussianBlur
from custom_data import CIFAR10_PretextRotate

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--accum_grad', type = int, default = 8, help = 'Accumulate grad')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='dataset')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument('--aug_type', type=str, default='crop',
                        choices=['crop', 'jitter', 'affine', 'blur'])
    parser.add_argument('--aug_levels', type=int, default=5,
                        help='Augmentation intensity levels.')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    # add single augmentation to model
    opt.model_name = '{}_aug_{}_{}'.format(opt.model_name, opt.aug_type, opt.aug_levels)
    
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size = 3, min = 0.1, max = 2),
        transforms.ToTensor(),
        normalize,
    ])
    """
    s = np.linspace(0, 1, opt.aug_levels)
    train_transform_list = []

    if opt.aug_type == 'crop':
        # crop from 0.08 (SimCLR) ~ 0.58
        for aug_strength in s:
            scale_low = -0.5 * aug_strength + 0.58
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale=(scale_low, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
            train_transform_list.append(train_transform)

    elif opt.aug_type == 'blur':
        # sigma from 1 ~ 5
        kernel_size = 3
        for aug_strength in s:
            sigma = aug_strength * 4 + 1 
            train_transform = transforms.Compose([
                GaussianBlur(kernel_size, min = 0.5, max = sigma),
                transforms.ToTensor(),
                normalize,
            ])
            train_transform_list.append(train_transform)

    elif opt.aug_type == 'jitter':
        # Jitter scale from 0.125 ~ 2.5
        for aug_strength in s:
            scale = 2.375 * aug_strength + 0.125
            train_transform = transforms.Compose([
                transforms.ColorJitter(0.4 * scale, 0.4 * scale, 0.4 * scale, 0.1 * scale),
                transforms.ToTensor(),
                normalize,
            ])
            train_transform_list.append(train_transform)

    if opt.dataset == 'cifar10':
        train_dataset = CIFAR10_PretextRotate(root=opt.data_folder,
                                              transform=MultiTwoCropTransform(train_transform_list))

    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    model = SupConResNet(name=opt.model)
    contrast_criterion = SupConLoss(temperature=opt.temp)

    classifier = LinearClassifier(name=opt.model, num_classes=4) # rotation pretext = 4 labels
    cls_criterion = torch.nn.CrossEntropyLoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        contrast_criterion = contrast_criterion.cuda()
        classifier = classifier.cuda()
        cls_criterion = cls_criterion.cuda()
        cudnn.benchmark = True

    return model, contrast_criterion, classifier, cls_criterion


def train(train_loader, model, contrast_criterion, classifier, cls_criterion, p, optimizers, epoch, opt):
    """one epoch training"""
    
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses = AverageMeter()

    end = time.time()
    for idx, (images, class_labels, pretext_labels) in enumerate(train_loader):
        # images is multi-two crop transform

        # split multitransform into two crops
        v1 = torch.stack(images[::2]).to(device)
        v2 = torch.stack(images[1::2]).to(device)

        # sample one strength of augmentation
        sample = F.gumbel_softmax(p[0], hard=True, tau=0.8).view(opt.aug_levels, 1, 1, 1, 1).to(device)
        # filter out one augmentation
        v1 = torch.sum(v1 * sample, dim=0)
        v2 = torch.sum(v2 * sample, dim=0)
        images = [v1, v2]

        
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        #print(images.shape)
        images = images.to(device) #cuda(non_blocking=True)
        class_labels = class_labels.to(device) #cuda(non_blocking=True)
        pretext_labels = pretext_labels.to(device)
        bsz = class_labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizers['model'])
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizers['cls'])
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizers['p'])

        # compute loss
        features = model(images)
        hidden_features = model.hidden_feat
        
        predict_pretext_labels = classifier(hidden_features)

        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = contrast_criterion(features, class_labels)
        elif opt.method == 'SimCLR':
            loss = contrast_criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        pretext_labels = pretext_labels.repeat(2)
        cls_loss = cls_criterion(predict_pretext_labels, pretext_labels)
        
        f_loss = loss # minimize InfoNCE = maximize MI
        gc_loss = -loss + cls_loss #+ p_norm_loss # maxmimize InfoNCE + minimize cls + regularize norm = minimize MI and cls
        # update metric
        losses.update(loss.item(), bsz)
        cls_losses.update(cls_loss.item(), bsz)

        # SGD: update all params
        # update feature extractor
        optimizers['model'].zero_grad()
        f_loss.backward(retain_graph = True)
        optimizers['model'].step()
        
        #update p and cls
        optimizers['p'].zero_grad()
        optimizers['cls'].zero_grad()
        gc_loss.backward()
        print(optimizers['p'].param_groups[0]['params'][0].grad)
        #optimizers['p'].step()
        print('a', optimizers['p'].param_groups[0])
        print('b', p[0])
        optimizers['cls'].step()


        batch_time.update(time.time() - end)
        end = time.time()
        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'f_loss {f_loss.val:.3f} ({f_loss.avg:.3f})\t'
                  'cls_loss {cls_loss.val:.3f} ({cls_loss.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, f_loss=losses, cls_loss=cls_losses))
            sys.stdout.flush()

    return losses.avg, cls_losses.avg


def main():
    p_evolve = []
    cls_losses = []

    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, contrast_criterion, classifier, cls_criterion = set_model(opt)
    # augmentation probabilities
    p = Variable(torch.zeros(opt.aug_levels, device = device), requires_grad = True)

    # build optimizer'SSS
    optimizers = {}
    optimizers['model'] = set_optimizer(opt, model)
    optimizers['cls'] = set_optimizer(opt, classifier)
    optimizers['p'] = optim.SGD([p], lr=0.1, momentum=0.9)

    # ANOTHER optimzier!!

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(opt, optimizers['model'], epoch)
        adjust_learning_rate(opt, optimizers['cls'], epoch)
        adjust_learning_rate(opt, optimizers['p'], epoch)


        # train for one epoch
        time1 = time.time()
        loss, cls_loss = train(train_loader, model, contrast_criterion, classifier, cls_criterion, [p], optimizers, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        #p = optimizers['p'].param_groups[0]['params'][0]
        print("P: {}".format(p.data))
        p_evolve.append(p.data.cpu().unsqueeze(-1).numpy())
        cls_losses.append(cls_loss)
        np.save('pretext_{}_p.npy'.format(opt.aug_type), np.concatenate(p_evolve, axis=1))
        np.save('cls_loss.npy', np.array(cls_losses))
        
        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('cls_loss', cls_loss, epoch)
        #logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizers['model'], opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizers['model'], opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
